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

from datasets.polarimetric_detection_dataset_v2 import (
    PolarimetricDetectionDatasetV2,
    representation_channels,
)
from models.polarimetric_representation_fcn import PolarimetricRepresentationFCN, count_parameters


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--manifest-path",
        default="results/data_audit/dataset_v4_multifold/fold_01_manifest.csv",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    modes = ("power2", "ri4", "polar6_gated", "ri8_gated")
    rows = []
    tensors = {}
    for mode in modes:
        dataset = PolarimetricDetectionDatasetV2(
            manifest_path=args.manifest_path,
            split="train",
            input_mode=mode,
            max_samples=2,
        )
        item = dataset[0]
        array = item["input"].numpy()
        tensors[mode] = array
        rows.append({
            "mode": mode,
            "shape": list(array.shape),
            "finite": bool(np.isfinite(array).all()),
            "minimum": float(array.min()),
            "maximum": float(array.max()),
            "active_channels": list(representation_channels(mode)),
        })
        if array.shape != (8, 128, 100):
            raise AssertionError(f"Unexpected shape for {mode}: {array.shape}")
        if not np.isfinite(array).all():
            raise AssertionError(f"Non-finite values for {mode}")
    if not np.allclose(tensors["power2"][:2], tensors["polar6_gated"][:2], atol=1e-6):
        raise AssertionError("Polar6-gated must preserve the Power2 channels")
    if not np.allclose(tensors["ri4"][:4], tensors["ri8_gated"][:4], atol=1e-6):
        raise AssertionError("RI8-gated must preserve the RI4 channels")
    for mode in ("polar6_gated", "ri8_gated"):
        extra = tensors[mode][2:6] if mode == "polar6_gated" else tensors[mode][4:8]
        if np.max(np.abs(extra)) > 1.000001:
            raise AssertionError(f"Gated explicit channels out of bounds for {mode}")
    model = PolarimetricRepresentationFCN()
    report = {
        "status": "PASS",
        "manifest_path": str(args.manifest_path),
        "modes": rows,
        "same_architecture": True,
        "parameter_count": count_parameters(model),
        "invariants": {
            "power2_preserved_in_polar6_gated": True,
            "ri4_preserved_in_ri8_gated": True,
            "gated_channels_bounded": True,
            "sample_independent": True,
            "scan_context": False,
        },
    }
    out = PROJECT_ROOT / "results/data_audit/polarimetric_gated_pipeline_v2"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "pipeline_test.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"written: {path}")


if __name__ == "__main__":
    main()
