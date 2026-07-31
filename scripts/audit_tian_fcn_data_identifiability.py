#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.detection_dataset_v3 import DetectionGeometry
from models.tian_fcn import TianFastUAVFCN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether a manifest can identify Tian FCN output locations"
    )
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def main() -> int:
    args = parse_args()
    manifest_path = resolve_path(args.manifest_path)
    output_dir = resolve_path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(manifest_path, encoding="utf-8-sig")
    split_column = "new_split" if "new_split" in frame.columns else "split"
    required = {
        split_column,
        "sample_id",
        "source_file",
        "target_present",
        "distance_m",
        "velocity_mps",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    targets = frame.loc[frame["target_present"].eq(1)].copy()
    if targets.empty:
        raise ValueError("manifest contains no target samples")

    geometry = DetectionGeometry()
    stride_y, stride_x = TianFastUAVFCN.output_stride
    targets["range_index"] = targets["distance_m"].map(geometry.range_to_index)
    targets["velocity_index"] = targets["velocity_mps"].map(
        geometry.velocity_to_index
    )
    targets["output_grid_x"] = targets["range_index"] // stride_x
    targets["output_grid_y"] = targets["velocity_index"] // stride_y

    split_summary = (
        targets.groupby(split_column, observed=True)
        .agg(
            target_count=("sample_id", "size"),
            source_count=("source_file", "nunique"),
            range_index_min=("range_index", "min"),
            range_index_max=("range_index", "max"),
            output_grid_x_unique=("output_grid_x", "nunique"),
            output_grid_y_unique=("output_grid_y", "nunique"),
        )
        .reset_index()
    )
    grid_coverage = (
        targets.groupby(
            [split_column, "output_grid_x", "output_grid_y"], observed=True
        )
        .size()
        .rename("target_count")
        .reset_index()
    )
    source_velocity = (
        targets.groupby([split_column, "source_file"], observed=True)
        .agg(
            target_count=("sample_id", "size"),
            velocity_value_count=("velocity_mps", "nunique"),
            output_grid_y_count=("output_grid_y", "nunique"),
            velocity_min_mps=("velocity_mps", "min"),
            velocity_max_mps=("velocity_mps", "max"),
        )
        .reset_index()
    )

    range_columns = sorted(int(value) for value in targets["output_grid_x"].unique())
    summary = {
        "status": "BLOCKED" if len(range_columns) == 1 else "REVIEW",
        "evidence_role": "tian_fcn_data_identifiability_pretraining_audit",
        "manifest_path": str(manifest_path),
        "target_count": int(len(targets)),
        "source_count": int(targets["source_file"].nunique()),
        "output_stride": [stride_y, stride_x],
        "output_grid_x_values": range_columns,
        "all_targets_share_one_output_range_column": len(range_columns) == 1,
        "single_velocity_source_count": int(
            source_velocity["velocity_value_count"].eq(1).sum()
        ),
        "source_group_count": int(len(source_velocity)),
        "interpretation": (
            "single output range column cannot test range-column generalization; "
            "source/velocity coupling can support fixed location templates"
            if len(range_columns) == 1
            else "multiple output range columns are present; inspect split coverage"
        ),
    }

    targets.to_csv(output_dir / "target_grid_assignments.csv", index=False)
    split_summary.to_csv(output_dir / "split_grid_summary.csv", index=False)
    grid_coverage.to_csv(output_dir / "split_grid_coverage.csv", index=False)
    source_velocity.to_csv(output_dir / "source_velocity_coverage.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
