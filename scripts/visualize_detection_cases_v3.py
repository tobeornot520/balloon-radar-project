#!/usr/bin/env python3
"""Visualize key H/V/HV detection cases.

The script always creates raw RD diagnostic figures with CSV label/prediction overlays.
It also attempts to load SimpleFCN checkpoints and create model heatmaps. If the local
model class/checkpoint schema differs, it falls back cleanly and records the reason.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

GROUP_FILES = {
    "rescued": "hv_rescued_vs_h.csv",
    "regressed": "hv_regressed_vs_h.csv",
    "common_hard": "common_hard_positive_samples.csv",
    "false_alarm": "false_alarm_samples.csv",
}
EXPERIMENTS = {
    "H": "detection_h_baseline_v2",
    "V": "detection_v_baseline_v2",
    "HV": "detection_hv_baseline_v2",
}


def load_iq(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    try:
        data = loadmat(path)
        h = np.asarray(data["local_data_H"])
        v = np.asarray(data["local_data_V"])
        return h, v
    except NotImplementedError:
        import h5py
        with h5py.File(path, "r") as f:
            h = np.asarray(f["local_data_H"])
            v = np.asarray(f["local_data_V"])
            if h.dtype.fields and set(h.dtype.fields) >= {"real", "imag"}:
                h = h["real"] + 1j * h["imag"]
            if v.dtype.fields and set(v.dtype.fields) >= {"real", "imag"}:
                v = v["real"] + 1j * v["imag"]
            return h, v


def rd_normalized(iq: np.ndarray) -> np.ndarray:
    iq = np.asarray(iq)
    if iq.ndim != 2:
        raise ValueError(f"IQ数组应为二维，实际为 {iq.shape}")
    window = np.hanning(iq.shape[0])[:, None]
    rd = np.fft.fftshift(np.fft.fft(iq * window, axis=0), axes=0)
    db = 20.0 * np.log10(np.abs(rd) + 1e-8)
    finite = db[np.isfinite(db)]
    if finite.size == 0:
        return np.zeros_like(db, dtype=np.float32)
    lo, hi = np.percentile(finite, [1.0, 99.0])
    if hi <= lo:
        return np.zeros_like(db, dtype=np.float32)
    return np.clip((db - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def remove_prefixes(
    state: Dict[str, torch.Tensor],
    prefixes: Tuple[str, ...],
) -> Dict[str, torch.Tensor]:
    """Remove at most one matching wrapper prefix from every state-dict key."""
    cleaned: Dict[str, torch.Tensor] = {}
    for original_key, value in state.items():
        key = original_key
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        cleaned[key] = value
    return cleaned


def extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    """Extract the saved state dict without guessing the model wrapper name."""
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model", "network"):
            value = checkpoint.get(key)
            if isinstance(value, dict) and value and all(torch.is_tensor(v) for v in value.values()):
                return dict(value)
        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return dict(checkpoint)
    raise RuntimeError("无法从checkpoint提取state_dict。")


def state_dict_variants(
    state: Dict[str, torch.Tensor],
) -> Iterable[Tuple[str, Dict[str, torch.Tensor]]]:
    """Yield common wrapper-prefix variants for robust checkpoint loading."""
    seen = set()

    def emit(name: str, candidate: Dict[str, torch.Tensor]):
        signature = tuple(candidate.keys())
        if signature not in seen:
            seen.add(signature)
            return name, candidate
        return None

    candidates = []
    candidates.append(("raw", dict(state)))
    no_parallel = remove_prefixes(state, ("module.",))
    candidates.append(("strip_module", no_parallel))
    no_outer = remove_prefixes(no_parallel, ("model.",))
    candidates.append(("strip_module_model", no_outer))
    no_network = remove_prefixes(no_outer, ("network.",))
    candidates.append(("strip_all_wrappers", no_network))
    candidates.append(("add_network", {f"network.{k}": v for k, v in no_network.items()}))

    for name, candidate in candidates:
        item = emit(name, candidate)
        if item is not None:
            yield item


def candidate_model_classes() -> Iterable[type]:
    modules = ("models.simple_fcn", "models.detection.simple_fcn")
    preferred = ("SimpleFCN", "FCNDetector", "DetectionFCN", "RadarFCN")
    for module_name in modules:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        yielded = set()
        for name in preferred:
            cls = getattr(module, name, None)
            if inspect.isclass(cls) and issubclass(cls, torch.nn.Module):
                yielded.add(cls)
                yield cls
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls not in yielded and issubclass(cls, torch.nn.Module) and cls is not torch.nn.Module:
                yield cls


def instantiate_model(in_channels: int) -> torch.nn.Module:
    errors = []
    kwargs_candidates = (
        {"in_channels": in_channels},
        {"input_channels": in_channels},
        {"channels": in_channels},
        {},
    )
    for cls in candidate_model_classes():
        for kwargs in kwargs_candidates:
            try:
                model = cls(**kwargs)
                return model
            except Exception as exc:
                errors.append(f"{cls.__module__}.{cls.__name__}{kwargs}: {exc}")
    raise RuntimeError("无法实例化检测模型。尝试记录：\n" + "\n".join(errors[-12:]))


def load_model(mode: str, experiments_root: Path, device: torch.device) -> torch.nn.Module:
    checkpoint_path = experiments_root / EXPERIMENTS[mode] / "checkpoints" / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = extract_state_dict(checkpoint)
    model = instantiate_model(2 if mode == "HV" else 1)

    attempts = []
    best = None
    for variant_name, candidate in state_dict_variants(state):
        result = model.load_state_dict(candidate, strict=False)
        mismatch = len(result.missing_keys) + len(result.unexpected_keys)
        attempts.append((variant_name, mismatch, list(result.missing_keys), list(result.unexpected_keys)))
        if best is None or mismatch < best[0]:
            best = (mismatch, variant_name, candidate, result)
        if mismatch == 0:
            break

    assert best is not None
    mismatch, variant_name, candidate, result = best
    # Reload the best variant because later trial loads may have partially overwritten parameters.
    result = model.load_state_dict(candidate, strict=False)
    if mismatch > 4:
        details = "; ".join(f"{name}: mismatch={count}" for name, count, _, _ in attempts)
        raise RuntimeError(
            "checkpoint与模型结构不匹配。"
            f" 最佳键变体={variant_name}, missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}; 尝试={details}"
        )

    print(f"{mode} checkpoint加载成功：键变体={variant_name}, mismatch={mismatch}")
    model.to(device).eval()
    return model


def forward_heatmap(model: torch.nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch.from_numpy(x[None, ...]).float().to(device)
    with torch.no_grad():
        output = model(tensor)
        if isinstance(output, dict):
            for key in ("logits", "heatmap", "output", "pred"):
                if key in output:
                    output = output[key]
                    break
        if isinstance(output, (tuple, list)):
            output = output[0]
        if output.ndim == 3:
            output = output[:, None, ...]
        heatmap = torch.sigmoid(output)[0, 0].detach().cpu().numpy()
    return heatmap


def point(row: pd.Series, prefix: str) -> Optional[Tuple[float, float]]:
    r = pd.to_numeric(row.get(f"{prefix}_pred_range_index"), errors="coerce")
    v = pd.to_numeric(row.get(f"{prefix}_pred_velocity_index"), errors="coerce")
    if pd.isna(r) or pd.isna(v):
        return None
    return float(r), float(v)


def draw_marker(ax: Any, xy: Optional[Tuple[float, float]], marker: str, label: str) -> None:
    if xy is None:
        return
    ax.scatter([xy[0]], [xy[1]], marker=marker, s=90, facecolors="none", linewidths=1.7, label=label)


def true_point(row: pd.Series) -> Optional[Tuple[float, float]]:
    present = int(pd.to_numeric(row.get("target_present", 0), errors="coerce") or 0)
    if present != 1:
        return None
    r = pd.to_numeric(row.get("true_range_index"), errors="coerce")
    v = pd.to_numeric(row.get("true_velocity_index"), errors="coerce")
    if pd.isna(r) or pd.isna(v):
        return None
    return float(r), float(v)


def title_for(row: pd.Series, group: str) -> str:
    return (
        f"{row['sample_id']} | {group} | beam={row.get('beam_layer', '')} "
        f"az={row.get('azimuth_deg', '')}"
    )


def plot_case(
    row: pd.Series,
    group: str,
    h_rd: np.ndarray,
    v_rd: np.ndarray,
    model_heatmaps: Dict[str, np.ndarray],
    output: Path,
) -> Dict[str, Any]:
    use_models = len(model_heatmaps) == 3
    fig, axes = plt.subplots(2 if use_models else 1, 3, figsize=(15, 9 if use_models else 5.5))
    if not use_models:
        axes = np.asarray([axes])
    inputs = {"H": h_rd, "V": v_rd, "HV": (h_rd + v_rd) / 2.0}
    points = {mode: point(row, mode) for mode in ("H", "V", "HV")}
    truth = true_point(row)

    peak_report: Dict[str, Any] = {}
    for col, mode in enumerate(("H", "V", "HV")):
        ax = axes[0, col]
        ax.imshow(inputs[mode], origin="lower", aspect="auto")
        draw_marker(ax, truth, "o", "True")
        draw_marker(ax, points[mode], "x", f"{mode} pred")
        ax.set_title(f"{mode} normalized RD")
        ax.set_xlabel("Range gate")
        ax.set_ylabel("Velocity bin")
        ax.legend(loc="upper right", fontsize=8)

        if use_models:
            hm = model_heatmaps[mode]
            ax2 = axes[1, col]
            ax2.imshow(hm, origin="lower", aspect="auto", vmin=0.0, vmax=1.0)
            draw_marker(ax2, truth, "o", "True")
            draw_marker(ax2, points[mode], "x", f"CSV pred")
            peak_v, peak_r = np.unravel_index(int(np.argmax(hm)), hm.shape)
            draw_marker(ax2, (float(peak_r), float(peak_v)), "+", "Recomputed peak")
            ax2.set_title(f"{mode} model heatmap | max={hm.max():.4f}")
            ax2.set_xlabel("Range gate")
            ax2.set_ylabel("Velocity bin")
            ax2.legend(loc="upper right", fontsize=7)
            csv_point = points[mode]
            peak_report[f"{mode}_recomputed_peak_range"] = int(peak_r)
            peak_report[f"{mode}_recomputed_peak_velocity"] = int(peak_v)
            peak_report[f"{mode}_matches_csv_peak"] = bool(
                csv_point is not None and int(csv_point[0]) == int(peak_r) and int(csv_point[1]) == int(peak_v)
            )

    fig.suptitle(title_for(row, group), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return peak_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, default=Path("results/experiments/detection_ablation_analysis_v2"))
    parser.add_argument("--experiments-root", type=Path, default=Path("results/experiments"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/experiments/detection_case_visualization_v3"))
    parser.add_argument("--groups", nargs="+", choices=list(GROUP_FILES), default=list(GROUP_FILES))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-model-heatmaps", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    models: Dict[str, torch.nn.Module] = {}
    model_error = ""
    if not args.skip_model_heatmaps:
        try:
            models = {mode: load_model(mode, args.experiments_root, device) for mode in ("H", "V", "HV")}
            print(f"模型加载成功，设备：{device}")
        except Exception as exc:
            model_error = str(exc)
            models = {}
            print("警告：模型热图加载失败，将生成RD输入与CSV预测叠加图。")
            print(model_error)

    records = []
    for group in args.groups:
        csv_path = args.analysis_dir / GROUP_FILES[group]
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        group_dir = args.output_dir / group
        group_dir.mkdir(exist_ok=True)
        for _, row in df.iterrows():
            mat_path = Path(str(row["mat_path"]))
            if not mat_path.exists():
                raise FileNotFoundError(
                    f"CSV中的MAT路径不存在：{mat_path}\n"
                    "请在原项目WSL环境运行本脚本，或修正CSV中的mat_path。"
                )
            h_iq, v_iq = load_iq(mat_path)
            h_rd, v_rd = rd_normalized(h_iq), rd_normalized(v_iq)
            heatmaps: Dict[str, np.ndarray] = {}
            if models:
                heatmaps["H"] = forward_heatmap(models["H"], h_rd[None, ...], device)
                heatmaps["V"] = forward_heatmap(models["V"], v_rd[None, ...], device)
                heatmaps["HV"] = forward_heatmap(models["HV"], np.stack([h_rd, v_rd], axis=0), device)
            output = group_dir / f"{row['sample_id']}.png"
            peak_report = plot_case(row, group, h_rd, v_rd, heatmaps, output)
            record = {
                "group": group,
                "sample_id": row["sample_id"],
                "mat_path": str(mat_path),
                "figure_path": str(output),
                "model_heatmaps_generated": bool(models),
                **peak_report,
            }
            records.append(record)

    manifest = pd.DataFrame(records)
    manifest.to_csv(args.output_dir / "case_visualization_manifest.csv", index=False, encoding="utf-8-sig")
    summary = {
        "groups": args.groups,
        "case_count": len(records),
        "model_heatmaps_generated": bool(models),
        "model_heatmap_error": model_error,
        "important_note": (
            "CSV prediction points remain the official stored results. "
            "Recomputed heatmap peaks are valid only when they match CSV peaks; mismatch indicates preprocessing/model loading incompatibility."
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 78)
    print(f"关键样本可视化完成：{len(records)} 个文件")
    print(f"模型热图：{'已生成' if models else '未生成，仅输出RD叠加图'}")
    print(f"结果目录：{args.output_dir.resolve()}")
    print("=" * 78)


if __name__ == "__main__":
    main()
