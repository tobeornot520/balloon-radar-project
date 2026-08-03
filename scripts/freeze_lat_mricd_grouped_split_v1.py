#!/usr/bin/env python3
"""Freeze metadata-only LAT-MRICD X-band batch assignments before feature extraction."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_lat_mricd_grouped_baseline_v1 import (
    DEFAULT_CONFIG,
    build_batch_split_manifest,
    build_grouped_fold_assignments,
    load_config,
    resolve_path,
    sha256_file,
    single_public_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze balanced batch groups using only four metadata columns."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def freeze_split(
    *,
    config_path: Path,
    dataset_root: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    config_path = resolve_path(config_path)
    config = load_config(config_path)
    dataset_root = resolve_path(dataset_root or config["dataset_root"])
    manifest_path = resolve_path(config["split_manifest"])
    summary_path = resolve_path(config["split_summary"])
    existing = [path for path in (manifest_path, summary_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"frozen split output already exists: {existing}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_frames: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []
    for task in config["tasks"]:
        path = dataset_root / task["relative_path"]
        matrix = single_public_matrix(path)
        expected_columns = 1028 if task["representation"] == "Narrow" else 504
        if matrix.ndim != 2 or matrix.shape[1] != expected_columns:
            raise ValueError(f"{task['task_id']}: unexpected matrix shape {matrix.shape}")
        metadata = np.rint(matrix[:, :4]).astype(np.int64)
        if not np.allclose(matrix[:, :4], metadata):
            raise ValueError(f"{task['task_id']}: metadata must contain integer codes")
        if set(metadata[:, 0]) != {int(task["band_code"])}:
            raise ValueError(f"{task['task_id']}: band code mismatch")
        if set(metadata[:, 1]) != {int(code) for code in task["class_codes"]}:
            raise ValueError(f"{task['task_id']}: category coverage mismatch")

        # No signal column is passed to the assignment optimizer.
        assignments = build_grouped_fold_assignments(
            metadata[:, 1],
            metadata[:, 3],
            n_splits=int(config["n_splits"]),
            random_state=int(config["random_state"]),
        )
        task_manifest = build_batch_split_manifest(
            task=task,
            metadata=metadata,
            assignments=assignments,
        )
        manifest_frames.append(task_manifest)
        for fold, group in task_manifest.groupby("heldout_fold", observed=True):
            row: dict[str, Any] = {
                "task_id": task["task_id"],
                "heldout_fold": int(fold),
                "batch_count": int(len(group)),
                "record_count": int(group["record_count"].sum()),
            }
            for code in [int(value) for value in task["class_codes"]]:
                category = {1: "uav", 2: "bird", 3: "weather"}[code]
                row[f"{category}_batch_count"] = int(group[f"{category}_present"].sum())
                row[f"{category}_record_count"] = int(
                    group[f"{category}_record_count"].sum()
                )
            coverage_rows.append(row)
        source_files.append(
            {
                "task_id": task["task_id"],
                "relative_path": task["relative_path"],
                "sha256": sha256_file(path),
                "record_count": int(len(metadata)),
                "signal_column_count": int(matrix.shape[1] - 4),
                "signal_columns_used_for_assignment": False,
            }
        )

    manifest = pd.concat(manifest_frames, ignore_index=True).sort_values(
        ["task_id", "batch_code"]
    )
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    summary = {
        "status": "FROZEN_METADATA_ONLY_BATCH_SPLIT",
        "splitter": config["splitter"],
        "group_key": config["group_key"],
        "n_splits": int(config["n_splits"]),
        "config_sha256": sha256_file(config_path),
        "split_manifest_sha256": sha256_file(manifest_path),
        "signal_columns_used_for_assignment": False,
        "aggregate_files_loaded_only": True,
        "detail_files_loaded": False,
        "batch_group_count": int(len(manifest)),
        "source_files": source_files,
        "fold_coverage": coverage_rows,
        "claim_scope": "metadata-only conservative batch-code grouping",
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = freeze_split(
        config_path=args.config,
        dataset_root=args.dataset_root,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
