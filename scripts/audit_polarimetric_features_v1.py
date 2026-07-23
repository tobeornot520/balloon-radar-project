#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.io import loadmat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.detection_dataset_v2 import DetectionGeometry
from features.polarimetric_rd import (
    PolarimetricConfig,
    explicit_polarimetric_rd,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit explicit H/V polarimetric RD features before "
            "adding a polarimetric network branch."
        )
    )
    parser.add_argument(
        "--manifest-path",
        default=(
            "results/data_audit/dataset_v4_multifold/"
            "fold_01_manifest.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "results/data_audit/"
            "polarimetric_feature_audit_v1"
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=30,
        help=(
            "Maximum samples for each split/target class. "
            "Use 0 to process all samples."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--velocity-window", type=int, default=5)
    parser.add_argument("--range-window", type=int, default=3)
    parser.add_argument("--roi-velocity-radius", type=int, default=2)
    parser.add_argument("--roi-range-radius", type=int, default=2)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def recover_mat_path(path_text: str) -> Path:
    candidate = Path(str(path_text)).expanduser()
    if candidate.is_file():
        return candidate.resolve()

    parts = candidate.parts
    if "data" in parts:
        index = parts.index("data")
        local = PROJECT_ROOT.joinpath(*parts[index:])
        if local.is_file():
            return local.resolve()

    matches = list(
        (PROJECT_ROOT / "data").rglob(candidate.name)
    )
    if len(matches) == 1:
        return matches[0].resolve()
    if not matches:
        raise FileNotFoundError(
            f"Cannot locate MAT file: {path_text}"
        )
    raise RuntimeError(
        f"Ambiguous MAT basename {candidate.name}: {matches[:5]}"
    )


def load_iq_pair(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = loadmat(path)
    missing = {"local_data_H", "local_data_V"} - set(data)
    if missing:
        raise KeyError(
            f"{path} missing fields: {sorted(missing)}"
        )
    h = np.asarray(data["local_data_H"])
    v = np.asarray(data["local_data_V"])
    if h.shape != v.shape:
        raise ValueError(
            f"H/V shape mismatch in {path.name}: {h.shape}/{v.shape}"
        )
    if h.ndim != 2:
        raise ValueError(
            f"Expected 2-D IQ, got {h.shape} in {path.name}"
        )
    if not np.iscomplexobj(h) or not np.iscomplexobj(v):
        raise TypeError(f"{path.name} is not complex H/V IQ")
    if not np.isfinite(h).all() or not np.isfinite(v).all():
        raise ValueError(f"{path.name} contains NaN or Inf")
    return h, v


def bounded_roi(
    array: np.ndarray,
    velocity_index: int,
    range_index: int,
    velocity_radius: int,
    range_radius: int,
) -> np.ndarray:
    v0 = max(0, int(velocity_index) - velocity_radius)
    v1 = min(
        array.shape[0],
        int(velocity_index) + velocity_radius + 1,
    )
    r0 = max(0, int(range_index) - range_radius)
    r1 = min(
        array.shape[1],
        int(range_index) + range_radius + 1,
    )
    return array[v0:v1, r0:r1]


def circular_resultant(
    phi_cos: np.ndarray,
    phi_sin: np.ndarray,
) -> float:
    return float(
        np.hypot(
            np.mean(phi_cos),
            np.mean(phi_sin),
        )
    )


def safe_stat(
    values: np.ndarray,
    function,
) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan
    return float(function(values))


def summarize_region(
    prefix: str,
    features: dict[str, np.ndarray],
    velocity_index: int,
    range_index: int,
    velocity_radius: int,
    range_radius: int,
) -> dict[str, float]:
    region = {
        key: bounded_roi(
            value,
            velocity_index,
            range_index,
            velocity_radius,
            range_radius,
        )
        for key, value in features.items()
        if isinstance(value, np.ndarray)
        and value.ndim == 2
    }
    return {
        f"{prefix}_zdr_mean": safe_stat(
            region["zdr_like_db"],
            np.mean,
        ),
        f"{prefix}_zdr_median": safe_stat(
            region["zdr_like_db"],
            np.median,
        ),
        f"{prefix}_zdr_std": safe_stat(
            region["zdr_like_db"],
            np.std,
        ),
        f"{prefix}_rho_mean": safe_stat(
            region["rho_hv_local"],
            np.mean,
        ),
        f"{prefix}_rho_median": safe_stat(
            region["rho_hv_local"],
            np.median,
        ),
        f"{prefix}_rho_p10": safe_stat(
            region["rho_hv_local"],
            lambda x: np.quantile(x, 0.10),
        ),
        f"{prefix}_phase_resultant": circular_resultant(
            region["phi_cos"],
            region["phi_sin"],
        ),
        f"{prefix}_phase_cos_mean": safe_stat(
            region["phi_cos"],
            np.mean,
        ),
        f"{prefix}_phase_sin_mean": safe_stat(
            region["phi_sin"],
            np.mean,
        ),
        f"{prefix}_stokes_s1_mean": safe_stat(
            region["stokes_s1"],
            np.mean,
        ),
        f"{prefix}_stokes_s2_mean": safe_stat(
            region["stokes_s2"],
            np.mean,
        ),
        f"{prefix}_stokes_s3_mean": safe_stat(
            region["stokes_s3"],
            np.mean,
        ),
    }


def balanced_sample(
    frame: pd.DataFrame,
    splits: Iterable[str],
    max_per_class: int,
    seed: int,
) -> pd.DataFrame:
    frame = frame.loc[
        frame["new_split"].isin(list(splits))
    ].copy()
    if max_per_class <= 0:
        return frame.reset_index(drop=True)

    pieces = []
    for _, group in frame.groupby(
        ["new_split", "target_present"],
        sort=False,
    ):
        n = min(len(group), max_per_class)
        pieces.append(
            group.sample(
                n=n,
                random_state=seed,
                replace=False,
            )
        )
    return (
        pd.concat(pieces, ignore_index=True)
        .sort_values(
            ["new_split", "target_present", "sample_id"]
        )
        .reset_index(drop=True)
    )


def standardized_effect(
    positive: np.ndarray,
    background: np.ndarray,
) -> float:
    positive = positive[np.isfinite(positive)]
    background = background[np.isfinite(background)]
    if len(positive) < 2 or len(background) < 2:
        return math.nan
    pooled = math.sqrt(
        (
            (len(positive) - 1) * float(np.var(positive, ddof=1))
            + (len(background) - 1)
            * float(np.var(background, ddof=1))
        )
        / (len(positive) + len(background) - 2)
    )
    if pooled <= 1e-12:
        return 0.0
    return float(
        (np.mean(positive) - np.mean(background)) / pooled
    )


def univariate_auc(
    labels: np.ndarray,
    values: np.ndarray,
) -> tuple[float, float]:
    mask = np.isfinite(values)
    labels = labels[mask]
    values = values[mask]
    if len(np.unique(labels)) != 2:
        return math.nan, math.nan
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        return math.nan, math.nan
    auc = float(roc_auc_score(labels, values))
    return auc, max(auc, 1.0 - auc)


def make_plots(
    frame: pd.DataFrame,
    effect_frame: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return [f"Matplotlib unavailable: {exc!r}"]

    warnings: list[str] = []
    top = (
        effect_frame.sort_values(
            "orientation_free_auc",
            ascending=False,
        )
        .head(6)["feature"]
        .tolist()
    )
    for feature in top:
        try:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for label, name in ((0, "Background"), (1, "Target")):
                values = frame.loc[
                    frame["target_present"] == label,
                    feature,
                ].dropna()
                ax.hist(
                    values,
                    bins=30,
                    alpha=0.55,
                    density=True,
                    label=name,
                )
            ax.set_title(feature)
            ax.set_xlabel("Feature value")
            ax.set_ylabel("Density")
            ax.legend()
            fig.tight_layout()
            fig.savefig(
                output_dir / f"hist_{feature}.png",
                dpi=180,
            )
            plt.close(fig)
        except Exception as exc:
            warnings.append(f"{feature}: {exc!r}")
    return warnings


def main() -> None:
    args = parse_args()
    manifest_path = resolve_path(args.manifest_path)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path)
    required = {
        "new_split",
        "target_present",
        "sample_id",
        "mat_path",
        "distance_m",
        "velocity_mps",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(
            f"Manifest missing columns: {sorted(missing)}"
        )

    selected = balanced_sample(
        manifest,
        args.splits,
        args.max_per_class,
        args.seed,
    )
    geometry = DetectionGeometry()
    config = PolarimetricConfig(
        velocity_window=args.velocity_window,
        range_window=args.range_window,
    )

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    observed_shapes: dict[str, int] = {}

    for position, record in selected.iterrows():
        sample_id = str(record["sample_id"])
        try:
            path = recover_mat_path(str(record["mat_path"]))
            h, v = load_iq_pair(path)
            observed_shapes[str(tuple(h.shape))] = (
                observed_shapes.get(str(tuple(h.shape)), 0) + 1
            )
            features = dict(
                explicit_polarimetric_rd(
                    h,
                    v,
                    config,
                )
            )
            combined = (
                features["local_power_h"]
                + features["local_power_v"]
            )
            peak_flat = int(np.argmax(combined))
            peak_v, peak_r = np.unravel_index(
                peak_flat,
                combined.shape,
            )

            row: dict[str, Any] = {
                "sample_id": sample_id,
                "split": str(record["new_split"]),
                "target_present": int(record["target_present"]),
                "mat_path": str(path),
                "iq_shape": str(tuple(h.shape)),
                "peak_velocity_index": int(peak_v),
                "peak_range_index": int(peak_r),
                "global_zdr_median": safe_stat(
                    features["zdr_like_db"],
                    np.median,
                ),
                "global_zdr_iqr": safe_stat(
                    features["zdr_like_db"],
                    lambda x: np.quantile(x, 0.75)
                    - np.quantile(x, 0.25),
                ),
                "global_rho_mean": safe_stat(
                    features["rho_hv_local"],
                    np.mean,
                ),
                "global_rho_median": safe_stat(
                    features["rho_hv_local"],
                    np.median,
                ),
                "global_phase_resultant": circular_resultant(
                    features["phi_cos"],
                    features["phi_sin"],
                ),
                "finite_fraction": float(
                    np.mean(
                        np.isfinite(
                            features["zdr_like_db"]
                        )
                        & np.isfinite(
                            features["rho_hv_local"]
                        )
                        & np.isfinite(
                            features["phi_dp_like_rad"]
                        )
                    )
                ),
            }
            row.update(
                summarize_region(
                    "peak",
                    features,
                    peak_v,
                    peak_r,
                    args.roi_velocity_radius,
                    args.roi_range_radius,
                )
            )

            if int(record["target_present"]) == 1:
                true_r = geometry.range_to_index(
                    float(record["distance_m"])
                )
                true_v = geometry.velocity_to_index(
                    float(record["velocity_mps"])
                )
                row["true_range_index"] = int(true_r)
                row["true_velocity_index"] = int(true_v)
                row.update(
                    summarize_region(
                        "target",
                        features,
                        true_v,
                        true_r,
                        args.roi_velocity_radius,
                        args.roi_range_radius,
                    )
                )
            else:
                row["true_range_index"] = -1
                row["true_velocity_index"] = -1
            rows.append(row)
        except Exception as exc:
            errors.append(
                {
                    "sample_id": sample_id,
                    "error": repr(exc),
                }
            )
        if (position + 1) % 25 == 0:
            print(
                f"processed {position + 1}/{len(selected)}"
            )

    feature_frame = pd.DataFrame(rows)
    feature_path = output_dir / "sample_features.csv"
    feature_frame.to_csv(
        feature_path,
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(errors).to_csv(
        output_dir / "errors.csv",
        index=False,
        encoding="utf-8-sig",
    )

    numeric_features = [
        column
        for column in feature_frame.columns
        if (
            column.startswith("peak_")
            or column.startswith("global_")
        )
        and pd.api.types.is_numeric_dtype(
            feature_frame[column]
        )
        and column
        not in {
            "peak_velocity_index",
            "peak_range_index",
        }
    ]

    effects = []
    if not feature_frame.empty:
        labels = feature_frame[
            "target_present"
        ].to_numpy(dtype=int)
        for feature in numeric_features:
            values = feature_frame[
                feature
            ].to_numpy(dtype=float)
            positive = values[labels == 1]
            background = values[labels == 0]
            auc, orientation_free_auc = univariate_auc(
                labels,
                values,
            )
            effects.append(
                {
                    "feature": feature,
                    "background_mean": safe_stat(
                        background,
                        np.mean,
                    ),
                    "target_mean": safe_stat(
                        positive,
                        np.mean,
                    ),
                    "cohens_d_target_minus_background": (
                        standardized_effect(
                            positive,
                            background,
                        )
                    ),
                    "auc_target_high": auc,
                    "orientation_free_auc": (
                        orientation_free_auc
                    ),
                }
            )
    effect_frame = pd.DataFrame(effects).sort_values(
        "orientation_free_auc",
        ascending=False,
        na_position="last",
    )
    effect_frame.to_csv(
        output_dir / "univariate_separability.csv",
        index=False,
        encoding="utf-8-sig",
    )

    group_columns = [
        column
        for column in numeric_features
        if column in feature_frame.columns
    ]
    if not feature_frame.empty and group_columns:
        grouped = feature_frame.groupby(
            ["split", "target_present"],
            dropna=False,
        )[group_columns].agg(["count", "mean", "std", "median"])
        grouped.columns = [
            f"{feature}_{stat}"
            for feature, stat in grouped.columns
        ]
        grouped.reset_index().to_csv(
            output_dir / "group_statistics.csv",
            index=False,
            encoding="utf-8-sig",
        )

    plot_warnings: list[str] = []
    if (
        not args.no_plots
        and not feature_frame.empty
        and not effect_frame.empty
    ):
        plot_warnings = make_plots(
            feature_frame,
            effect_frame,
            output_dir,
        )

    summary = {
        "manifest_path": str(manifest_path),
        "selected_sample_count": int(len(selected)),
        "processed_sample_count": int(len(feature_frame)),
        "error_count": int(len(errors)),
        "splits": list(args.splits),
        "max_per_class": args.max_per_class,
        "neighborhood": {
            "velocity_window": args.velocity_window,
            "range_window": args.range_window,
        },
        "observed_iq_shapes": observed_shapes,
        "all_complex_iq": bool(len(errors) == 0),
        "minimum_finite_fraction": (
            float(feature_frame["finite_fraction"].min())
            if not feature_frame.empty
            else math.nan
        ),
        "explicit_polarimetric_status": (
            "ready_for_exploratory_ablation"
            if len(feature_frame) > 0 and len(errors) == 0
            else "requires_data_or_interface_fix"
        ),
        "absolute_calibration_status": (
            "unknown; use relative ZDR-like and relative phase naming"
        ),
        "micro_doppler_status": (
            "not established by this 2-D 128-pulse snapshot audit; "
            "requires ordered continuous slow-time windows, PRF, and "
            "target-aligned sequence construction"
        ),
        "plot_warnings": plot_warnings,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    top_effects = effect_frame.head(10)
    lines = [
        "# 显式极化特征审计 V1",
        "",
        f"- Manifest：`{manifest_path}`",
        f"- 选取样本：{len(selected)}",
        f"- 成功处理：{len(feature_frame)}",
        f"- 错误：{len(errors)}",
        f"- IQ形状：{observed_shapes}",
        "",
        "## 已计算的显式极化量",
        "",
        "- H/V复数RD与功率；",
        "- 局部功率比 `ZDR-like`；",
        "- 邻域估计的共极化相关系数幅值 `rho_HV`；",
        "- 相对差分相位的正弦/余弦表示；",
        "- 归一化 Stokes-like S1/S2/S3。",
        "",
        "注意：尚未提供通道幅相标定信息，因此只能使用"
        "“relative ZDR-like”和“relative differential phase”命名，"
        "不能直接宣称为绝对气象雷达ZDR或PhiDP。",
        "",
        "## 单特征区分度前10项",
        "",
    ]
    if top_effects.empty:
        lines.append("没有足够数据计算。")
    else:
        lines.append(
            top_effects.to_markdown(
                index=False,
                floatfmt=".4f",
            )
        )
    lines.extend(
        [
            "",
            "## 多域路线判定",
            "",
            "1. 当前DPG/BC-DPG只使用H/V功率RD与网络隐特征，"
            "没有显式极化分支；",
            "2. 分类数据集中的旧`polar5`只是早期逐点特征，"
            "其`corr`实质更接近相位差余弦，不是真正邻域"
            "相关系数；",
            "3. 本审计通过后，可依次比较 Power2、RI4、"
            "Polar6、RI8；",
            "4. 时频/微多普勒分支不能从单帧128脉冲数据"
            "直接宣称完成，需要连续慢时间序列、PRF及目标"
            "距离门对齐。",
        ]
    )
    (output_dir / "README_极化特征审计.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("=" * 82)
    print("Explicit polarimetric feature audit complete")
    print(f"features : {feature_path}")
    print(f"effects  : {output_dir / 'univariate_separability.csv'}")
    print(f"summary  : {output_dir / 'summary.json'}")
    print(
        f"report   : "
        f"{output_dir / 'README_极化特征审计.md'}"
    )
    print("=" * 82)


if __name__ == "__main__":
    main()
