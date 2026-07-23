#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.detection_dataset_v2 import _load_iq_pair
from features.polarimetric_rd import PolarimetricConfig, explicit_polarimetric_rd
from features.polarimetric_gated_rd import (
    PolarimetricGateConfig,
    gated_explicit_channels,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit ungated and gated explicit polarimetric RD channels")
    p.add_argument("--folds", nargs="+", type=int, default=[1, 4])
    p.add_argument("--per-split-class", type=int, default=16)
    p.add_argument("--gate-low-percentile", type=float, default=50.0)
    p.add_argument("--gate-high-percentile", type=float, default=99.0)
    p.add_argument("--gate-gamma", type=float, default=1.5)
    p.add_argument("--zdr-clip-db", type=float, default=20.0)
    return p.parse_args()


def recover_path(value: str) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_file():
        return path.resolve()
    parts = path.parts
    if "data" in parts:
        candidate = PROJECT_ROOT.joinpath(*parts[parts.index("data"):])
        if candidate.is_file():
            return candidate.resolve()
    matches = list((PROJECT_ROOT / "data").rglob(path.name))
    if len(matches) == 1:
        return matches[0].resolve()
    if not matches:
        raise FileNotFoundError(f"Cannot locate data file: {value}")
    raise RuntimeError(f"Ambiguous data basename {path.name}: {matches[:5]}")


def summarize(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).ravel()
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_q01": float(np.quantile(values, 0.01)),
        f"{prefix}_q50": float(np.quantile(values, 0.50)),
        f"{prefix}_q99": float(np.quantile(values, 0.99)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
    }


def main() -> None:
    args = parse_args()
    if args.per_split_class <= 0:
        raise ValueError("per-split-class must be positive")
    polar_cfg = PolarimetricConfig(zdr_clip_db=args.zdr_clip_db)
    gate_cfg = PolarimetricGateConfig(
        low_percentile=args.gate_low_percentile,
        high_percentile=args.gate_high_percentile,
        gamma=args.gate_gamma,
    )
    rows: list[dict] = []

    for fold in args.folds:
        manifest_path = PROJECT_ROOT / f"results/data_audit/dataset_v4_multifold/fold_{fold:02d}_manifest.csv"
        manifest = pd.read_csv(manifest_path)
        required = {"new_split", "target_present", "sample_id", "mat_path", "source_file"}
        if not required.issubset(manifest.columns):
            raise ValueError(f"Manifest missing columns: {sorted(required - set(manifest.columns))}")
        sampled = (
            manifest.sort_values("sample_id")
            .groupby(["new_split", "target_present"], group_keys=False)
            .head(args.per_split_class)
        )
        for record in sampled.to_dict("records"):
            h, v = _load_iq_pair(recover_path(record["mat_path"]))
            features = explicit_polarimetric_rd(h, v, polar_cfg)
            confidence, gated_zdr, gated_rho, gated_cos, gated_sin = gated_explicit_channels(
                features,
                zdr_clip_db=args.zdr_clip_db,
                gate_config=gate_cfg,
            )
            zdr_scaled = np.clip(features["zdr_like_db"] / args.zdr_clip_db, -1.0, 1.0)
            rho = np.clip(features["rho_hv_local"], 0.0, 1.0)
            low_mask = confidence < 0.1
            high_mask = confidence >= 0.5
            phase_gated_amplitude = np.sqrt(gated_cos ** 2 + gated_sin ** 2)
            row = {
                "fold": fold,
                "split": record["new_split"],
                "target_present": int(record["target_present"]),
                "class_label": "target" if int(record["target_present"]) else "background",
                "sample_id": record["sample_id"],
                "source_file": record["source_file"],
                "confidence_lt_0_1_fraction": float(np.mean(low_mask)),
                "confidence_ge_0_5_fraction": float(np.mean(high_mask)),
                "zdr_saturation_fraction": float(np.mean(np.abs(zdr_scaled) >= 0.999)),
                "ungated_phase_full_amplitude_fraction": 1.0,
                "gated_phase_amplitude_lt_0_1_fraction": float(np.mean(phase_gated_amplitude < 0.1)),
                "gated_zdr_abs_mean_low_conf": float(np.mean(np.abs(gated_zdr[low_mask]))) if np.any(low_mask) else 0.0,
                "ungated_zdr_abs_mean_low_conf": float(np.mean(np.abs(zdr_scaled[low_mask]))) if np.any(low_mask) else 0.0,
                "gated_rho_mean_low_conf": float(np.mean(gated_rho[low_mask])) if np.any(low_mask) else 0.0,
                "ungated_rho_mean_low_conf": float(np.mean(rho[low_mask])) if np.any(low_mask) else 0.0,
            }
            row.update(summarize(confidence, "confidence"))
            row.update(summarize(zdr_scaled, "zdr_scaled"))
            row.update(summarize(rho, "rho"))
            row.update(summarize(gated_zdr, "gated_zdr"))
            row.update(summarize(gated_rho, "gated_rho"))
            row.update(summarize(phase_gated_amplitude, "gated_phase_amplitude"))
            rows.append(row)

    detail = pd.DataFrame(rows)
    out = PROJECT_ROOT / "results/data_audit/polarimetric_channel_audit_v1"
    out.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out / "channel_audit_sample_detail.csv", index=False, encoding="utf-8-sig")
    numeric = [
        c for c in detail.columns
        if c not in {"fold", "split", "target_present", "class_label", "sample_id", "source_file"}
    ]
    aggregate = detail.groupby(["fold", "split", "class_label"], as_index=False)[numeric].mean()
    aggregate.to_csv(out / "channel_audit_aggregate.csv", index=False, encoding="utf-8-sig")
    overall = detail.groupby(["class_label"], as_index=False)[numeric].mean()
    overall.to_csv(out / "channel_audit_overall.csv", index=False, encoding="utf-8-sig")
    status = {
        "status": "PASS",
        "folds": args.folds,
        "sample_count": int(len(detail)),
        "per_split_class": int(args.per_split_class),
        "gate": {
            "low_percentile": args.gate_low_percentile,
            "high_percentile": args.gate_high_percentile,
            "gamma": args.gate_gamma,
        },
        "interpretation": (
            "The audit quantifies low-power suppression and channel saturation. "
            "It does not establish absolute polarimetric calibration."
        ),
    }
    (out / "latest_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 显式极化通道稳定性审计", "",
        "本审计检查低功率区域、ZDR类通道饱和、局部相关性和门控后的相位幅度。", "",
        overall.to_markdown(index=False, floatfmt=".4f"), "",
        f"sample_count={len(detail)}",
    ]
    (out / "README_channel_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("=" * 82)
    print("polarimetric channel audit")
    print(f"samples : {len(detail)}")
    print(f"output  : {out}")
    print("=" * 82)


if __name__ == "__main__":
    main()
