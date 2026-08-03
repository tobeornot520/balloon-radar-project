#!/usr/bin/env python3
"""Freeze sanitized aggregate evidence from the LAT-MRICD grouped baseline run."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = (
    PROJECT_ROOT / "results/experiments/lat_mricd_grouped_baseline_v1"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "results/final_evidence/lat_mricd_grouped_baselines_v1"
)
CONFIG_PATH = PROJECT_ROOT / "configs/lat_mricd_grouped_baseline_v1.json"
IMPLEMENTATION_PATH = PROJECT_ROOT / "scripts/run_lat_mricd_grouped_baseline_v1.py"
FROZEN_SPLIT_PATH = PROJECT_ROOT / "data/splits/lat_mricd_x_batch_grouped_v1.csv"

TABLE_FILES = (
    "split_manifest.csv",
    "fold_coverage.csv",
    "fold_metrics.csv",
    "aggregate_metrics.csv",
    "batch_class_metrics.csv",
    "batch_class_distribution.csv",
    "confusion_matrices.csv",
    "feature_definitions.csv",
    "feature_summary_by_category.csv",
    "feature_importance.csv",
    "cluster_bootstrap_intervals.csv",
    "subtype_pressure.csv",
    "claim_boundaries.csv",
)
EXCLUDED_SOURCE_FILES = (
    "oof_predictions.csv",
    "feature_importance_by_fold.csv",
)
EXPECTED_TASK_RECORDS = {
    "narrow_x_category": 8715,
    "hrrp_x_category": 3648,
}
EXPECTED_MODELS = {
    "dummy_prior",
    "logistic_batch_balanced",
    "random_forest_batch_balanced",
}
SENSITIVE_MARKERS = ("/home/", "tobeornot8259748", "C:\\Users\\")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a formal grouped run and publish aggregate evidence only."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def commit_is_ancestor(commit: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
    )
    return result.returncode == 0


def validate_oof_predictions(
    predictions: pd.DataFrame,
    split_manifest: pd.DataFrame,
    *,
    expected_task_records: dict[str, int] = EXPECTED_TASK_RECORDS,
    expected_models: set[str] = EXPECTED_MODELS,
) -> None:
    required = {
        "task_id",
        "source_row_index",
        "model_id",
        "batch_code",
        "heldout_fold",
        "category_code",
        "predicted_category_code",
        "probability_uav",
        "probability_bird",
        "probability_weather",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"OOF predictions missing columns: {sorted(missing)}")
    if predictions.duplicated(["task_id", "model_id", "source_row_index"]).any():
        raise ValueError("OOF predictions contain duplicate task/model/source rows")
    if set(predictions["model_id"].unique()) != expected_models:
        raise ValueError("OOF model set does not match frozen configuration")
    for task_id, expected_count in expected_task_records.items():
        selected = predictions.loc[predictions["task_id"].eq(task_id)]
        counts = selected.groupby("model_id", observed=True).size()
        if set(counts.index) != expected_models or not counts.eq(expected_count).all():
            raise ValueError(f"{task_id}: incomplete OOF coverage")
        for _, model_rows in selected.groupby("model_id", observed=True):
            expected_indices = np.arange(expected_count, dtype=np.int64)
            actual_indices = np.sort(
                model_rows["source_row_index"].to_numpy(dtype=np.int64)
            )
            if not np.array_equal(actual_indices, expected_indices):
                raise ValueError(f"{task_id}: source row coverage is not exact")

    probabilities = predictions[
        ["probability_uav", "probability_bird", "probability_weather"]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all():
        raise ValueError("OOF probabilities contain NaN or Inf")
    if np.any(probabilities < -1e-12) or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-9
    ):
        raise ValueError("OOF probabilities are not valid class probabilities")
    if not set(predictions["category_code"].unique()) <= {1, 2, 3}:
        raise ValueError("OOF true labels contain unknown categories")
    if not set(predictions["predicted_category_code"].unique()) <= {1, 2, 3}:
        raise ValueError("OOF predictions contain unknown categories")

    split_lookup = split_manifest.set_index(["task_id", "batch_code"])[
        "heldout_fold"
    ]
    if not split_lookup.index.is_unique:
        raise ValueError("split manifest has duplicate task/batch rows")
    expected_folds = np.asarray(
        [
            int(split_lookup.loc[(task, int(batch))])
            for task, batch in zip(
                predictions["task_id"], predictions["batch_code"], strict=True
            )
        ]
    )
    if not np.array_equal(
        expected_folds,
        predictions["heldout_fold"].to_numpy(dtype=np.int64),
    ):
        raise ValueError("OOF prediction fold does not match frozen batch manifest")


def validate_source_run(source_dir: Path) -> dict[str, Any]:
    required = {"REPORT.md", "summary.json", "oof_predictions.csv", *TABLE_FILES}
    missing = sorted(name for name in required if not (source_dir / name).is_file())
    if missing:
        raise FileNotFoundError(f"formal run is missing files: {missing}")
    summary = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
    expected = {
        "status": "COMPLETE_GROUPED_PUBLIC_DATA_BASELINE",
        "group_leakage_detected": False,
        "all_rows_heldout_once": True,
        "all_folds_cover_all_classes": True,
        "hyperparameter_search_performed": False,
        "raw_data_in_output": False,
        "model_checkpoints_saved": False,
        "physical_frequency_hz_reported": False,
        "primary_group_metric": "batch_class_macro_accuracy",
    }
    for field, value in expected.items():
        if summary.get(field) != value:
            raise ValueError(f"unexpected formal summary {field}: {summary.get(field)!r}")
    if summary.get("config_sha256") != sha256_file(CONFIG_PATH):
        raise ValueError("formal run config hash is stale")
    if summary.get("implementation_sha256") != sha256_file(IMPLEMENTATION_PATH):
        raise ValueError("formal run implementation hash is stale")
    if summary.get("frozen_split_manifest_sha256") != sha256_file(FROZEN_SPLIT_PATH):
        raise ValueError("formal run split hash is stale")
    implementation_commit = str(summary.get("implementation_commit", ""))
    if not implementation_commit or not commit_is_ancestor(implementation_commit):
        raise ValueError("formal implementation commit is not an ancestor of HEAD")

    predictions = pd.read_csv(source_dir / "oof_predictions.csv", encoding="utf-8-sig")
    split_manifest = pd.read_csv(source_dir / "split_manifest.csv", encoding="utf-8-sig")
    validate_oof_predictions(predictions, split_manifest)
    aggregate = pd.read_csv(source_dir / "aggregate_metrics.csv", encoding="utf-8-sig")
    expected_pairs = {
        (task, model)
        for task in EXPECTED_TASK_RECORDS
        for model in EXPECTED_MODELS
    }
    actual_pairs = set(zip(aggregate["task_id"], aggregate["model_id"], strict=True))
    if actual_pairs != expected_pairs:
        raise ValueError("aggregate metric task/model coverage is incomplete")
    if not np.isfinite(
        aggregate.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    ).all():
        raise ValueError("aggregate metric table contains NaN or Inf")
    return summary


def audit_publication(output_dir: Path) -> None:
    forbidden_name_parts = ("oof", "prediction", "sample", "checkpoint", "raw")
    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        lowered = path.name.lower()
        if any(part in lowered for part in forbidden_name_parts):
            raise ValueError(f"publication contains forbidden sample-level file: {path.name}")
        if path.suffix.lower() in {".md", ".csv", ".json", ".txt"}:
            text = path.read_text(encoding="utf-8-sig")
            for marker in SENSITIVE_MARKERS:
                if marker in text:
                    raise ValueError(f"publication contains sensitive marker {marker!r}")


def build_evidence(
    *, source_dir: Path, output_dir: Path, overwrite: bool = False
) -> dict[str, Any]:
    source_dir = resolve_path(source_dir)
    output_dir = resolve_path(output_dir)
    if output_dir in {PROJECT_ROOT, source_dir}:
        raise ValueError("evidence output must be separate from project and experiment roots")
    summary = validate_source_run(source_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"evidence directory is nonempty: {output_dir}")
        shutil.rmtree(output_dir)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_dir / "REPORT.md", output_dir / "REPORT.md")
    for name in TABLE_FILES:
        shutil.copyfile(source_dir / name, tables_dir / name)

    records: list[dict[str, Any]] = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            records.append(
                {
                    "relative_path": path.relative_to(output_dir).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    manifest = {
        "status": "FROZEN_SANITIZED_AGGREGATE_EVIDENCE",
        "evidence_id": "lat_mricd_grouped_baselines_v1",
        "source_experiment": summary["experiment_id"],
        "source_implementation_commit": summary["implementation_commit"],
        "evidence_builder_commit": current_commit(),
        "source_summary_sha256": sha256_file(source_dir / "summary.json"),
        "sample_predictions_validated": True,
        "sample_predictions_included": False,
        "raw_data_included": False,
        "model_checkpoints_included": False,
        "excluded_source_files": list(EXCLUDED_SOURCE_FILES),
        "published_file_count_excluding_manifest": len(records),
        "files": records,
        "claim_scope": summary["claim_scope"],
    }
    (output_dir / "evidence_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    audit_publication(output_dir)
    return manifest


def main() -> None:
    args = parse_args()
    manifest = build_evidence(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
