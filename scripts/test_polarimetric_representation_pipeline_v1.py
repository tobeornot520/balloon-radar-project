#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.polarimetric_detection_dataset_v1 import (
    PolarimetricDetectionDatasetV1,
    representation_channels,
)
from features.polarimetric_rd import PolarimetricConfig, explicit_polarimetric_rd, make_power2, make_ri4, make_polar6, make_ri8
from models.polarimetric_representation_fcn import PolarimetricRepresentationFCN, count_parameters


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest-path", default="results/data_audit/dataset_v4_multifold/fold_01_manifest.csv")
    p.add_argument("--output-dir", default="results/data_audit/polarimetric_representation_pipeline_v1")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output = PROJECT_ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    h = (rng.normal(size=(128, 100)) + 1j * rng.normal(size=(128, 100))).astype(np.complex64)
    v = (0.7 * h + 0.3 * (rng.normal(size=(128, 100)) + 1j * rng.normal(size=(128, 100)))).astype(np.complex64)
    features = explicit_polarimetric_rd(h, v, PolarimetricConfig())
    synthetic = {
        "power2": make_power2(features),
        "ri4": make_ri4(features),
        "polar6": make_polar6(features),
        "ri8": make_ri8(features),
    }
    rows = []
    for mode, array in synthetic.items():
        if array.shape != (len(representation_channels(mode)), 128, 100):
            raise RuntimeError(f"Unexpected synthetic shape for {mode}: {array.shape}")
        if not np.isfinite(array).all():
            raise RuntimeError(f"Non-finite values for {mode}")
        dataset = PolarimetricDetectionDatasetV1(
            manifest_path=args.manifest_path, split="train", input_mode=mode, max_samples=1
        )
        sample = dataset[0]
        if tuple(sample["input"].shape) != (8, 128, 100):
            raise RuntimeError(f"Canonical shape failure for {mode}")
        rows.append({
            "mode": mode,
            "active_channel_count": len(representation_channels(mode)),
            "active_channels": list(representation_channels(mode)),
            "canonical_shape": list(sample["input"].shape),
            "finite": bool(np.isfinite(sample["input"].numpy()).all()),
            "min": float(sample["input"].min()),
            "max": float(sample["input"].max()),
        })
    model = PolarimetricRepresentationFCN()
    report = {
        "status": "PASS",
        "same_architecture_across_modes": True,
        "parameter_count": count_parameters(model),
        "modes": rows,
        "warning": "ZDR-like and relative phase are uncalibrated relative features.",
    }
    path = output / "pipeline_test.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 82)
    print("polarimetric representation pipeline test: PASS")
    print(f"parameter count: {count_parameters(model):,}")
    for row in rows:
        print(f"{row['mode']:8s}: active={row['active_channel_count']} canonical={row['canonical_shape']}")
    print(f"report: {path}")
    print("=" * 82)


if __name__ == "__main__":
    main()
