#!/usr/bin/env python3
"""Audit the public LAT-MRICD-1.0 dataset before model development."""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data/raw/external/LAT-MRICD-1.0"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/data_audit/lat_mricd_v1"

BAND_NAMES = {1: "S", 2: "X", 3: "Ku"}
CATEGORY_NAMES = {1: "UAV", 2: "bird", 3: "weather"}
MODEL_NAMES = {
    1: "Mavic 2",
    2: "Phantom 4",
    3: "Air 3S",
    4: "M30T",
    5: "racing drone",
    6: "self-built UAV",
    7: "pigeon",
    8: "goose",
    9: "weather clutter",
    10: "unspecified UAV",
}
MODEL_CATEGORY = {
    1: 1,
    2: 1,
    3: 1,
    4: 1,
    5: 1,
    6: 1,
    7: 2,
    8: 2,
    9: 3,
    10: 1,
}


@dataclass(frozen=True)
class AggregateSpec:
    relative_path: str
    representation: str
    band_code: int
    column_count: int


EXPECTED_AGGREGATES = (
    AggregateSpec("HRRP/S波段/data_hrrp_S.mat", "HRRP", 1, 504),
    AggregateSpec("HRRP/X波段/data_hrrp_X.mat", "HRRP", 2, 504),
    AggregateSpec("HRRP/Ku波段/data_hrrp_Ku.mat", "HRRP", 3, 504),
    AggregateSpec("Narrow/S波段/data_narrow_S.mat", "Narrow", 1, 1028),
    AggregateSpec("Narrow/X波段/data_narrow_X.mat", "Narrow", 2, 1028),
    AggregateSpec("Narrow/Ku波段/data_narrow_Ku.mat", "Narrow", 3, 1028),
)

# The release intentionally has no S-band HRRP aggregate.
REQUIRED_AGGREGATES = tuple(
    spec for spec in EXPECTED_AGGREGATES if spec.relative_path != "HRRP/S波段/data_hrrp_S.mat"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate LAT-MRICD schemas and audit batch/class confounding."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _single_public_matrix(path: Path) -> tuple[str, np.ndarray]:
    payload = loadmat(path)
    variables = [(key, np.asarray(value)) for key, value in payload.items() if not key.startswith("__")]
    matrices = [item for item in variables if item[1].ndim == 2 and np.issubdtype(item[1].dtype, np.number)]
    if len(matrices) != 1:
        names = [key for key, _ in variables]
        raise ValueError(f"{path}: expected one public numeric matrix, found {names}")
    return matrices[0]


def reconstruct_narrow_iq(matrix: np.ndarray) -> np.ndarray:
    """Return 512 complex slow-time samples from alternating I/Q columns."""
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[1] != 1028:
        raise ValueError("narrowband matrix must have shape (records, 1028)")
    signal = matrix[:, 4:]
    return signal[:, 0::2] + 1j * signal[:, 1::2]


def _validate_matrix(
    matrix: np.ndarray, *, path: Path, representation: str, band_code: int
) -> dict[str, Any]:
    expected_columns = 504 if representation == "HRRP" else 1028
    if matrix.ndim != 2 or matrix.shape[1] != expected_columns:
        raise ValueError(
            f"{path}: expected (*, {expected_columns}), got {tuple(matrix.shape)}"
        )
    if matrix.shape[0] == 0:
        raise ValueError(f"{path}: matrix contains no records")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{path}: matrix contains NaN or Inf")

    metadata = matrix[:, :4]
    rounded = np.rint(metadata)
    if not np.allclose(metadata, rounded):
        raise ValueError(f"{path}: metadata columns must contain integer codes")
    metadata_int = rounded.astype(np.int64)
    bands = set(metadata_int[:, 0].tolist())
    categories = set(metadata_int[:, 1].tolist())
    models = set(metadata_int[:, 2].tolist())
    batches = metadata_int[:, 3]
    if bands != {band_code}:
        raise ValueError(f"{path}: band codes {sorted(bands)} do not match {band_code}")
    if not categories <= set(CATEGORY_NAMES):
        raise ValueError(f"{path}: unknown category codes {sorted(categories)}")
    if not models <= set(MODEL_NAMES):
        raise ValueError(f"{path}: unknown model codes {sorted(models)}")
    if np.any((batches < 1) | (batches > 9999)):
        raise ValueError(f"{path}: batch codes must be positive integers below 10000")
    mismatched = [
        (int(category), int(model))
        for category, model in metadata_int[:, 1:3]
        if MODEL_CATEGORY[int(model)] != int(category)
    ]
    if mismatched:
        raise ValueError(f"{path}: category/model mismatch {mismatched[0]}")

    signal = matrix[:, 4:]
    if representation == "HRRP":
        if np.any(signal < 0):
            raise ValueError(f"{path}: HRRP amplitude sequence contains negative values")
        signal_sample_count = int(signal.shape[1])
    else:
        signal_sample_count = int(reconstruct_narrow_iq(matrix).shape[1])

    return {
        "record_count": int(matrix.shape[0]),
        "column_count": int(matrix.shape[1]),
        "dtype": str(matrix.dtype),
        "finite": True,
        "signal_sample_count": signal_sample_count,
        "batch_count": int(np.unique(batches).size),
    }


def _spec_by_path() -> dict[str, AggregateSpec]:
    return {spec.relative_path: spec for spec in EXPECTED_AGGREGATES}


def _infer_file_spec(relative_path: Path) -> tuple[str, int]:
    if len(relative_path.parts) < 3 or relative_path.parts[0] not in {"HRRP", "Narrow"}:
        raise ValueError(f"unexpected MAT location: {relative_path}")
    representation = relative_path.parts[0]
    band_by_directory = {"S波段": 1, "X波段": 2, "Ku波段": 3}
    band_directory = relative_path.parts[1]
    if band_directory not in band_by_directory:
        raise ValueError(f"unexpected band directory: {relative_path}")
    return representation, band_by_directory[band_directory]


def _majority_fraction(series: pd.Series) -> float:
    counts = series.value_counts()
    return float(counts.iloc[0] / counts.sum())


def _make_report(summary: dict[str, Any]) -> str:
    return f"""# LAT-MRICD-1.0 Data Audit V1

Status: `{summary['status']}`  
Records: `{summary['record_count']}`  
Aggregate files: `{summary['aggregate_file_count']}`  
MAT files checked: `{summary['mat_file_count']}`

All released matrices passed shape, finite-value, metadata-code and category/model checks.
Narrowband rows reconstruct to 512 complex samples from alternating I/Q columns.

## Split decision

- Random row splitting is forbidden.
- The conservative group key is `(representation, band_code, batch_code)`.
- Raw batch codes collide across models/categories, and their acquisition semantics are not
  independently documented. Use `batch_code_collisions.csv` before freezing a split.
- Category-level grouped baseline coverage is sufficient in every released aggregate, but some
  fine-grained models have fewer than three independent batch codes. See
  `category_split_readiness.csv` and `model_split_readiness.csv`.

## Claim boundary

This dataset supports HRRP, normalized-frequency narrowband micro-motion, category/model
classification baselines and cross-band transfer studies. It contains no H/V paired channels.
Without verified PRF, timestamps and continuous-session metadata, do not report physical
micro-Doppler in Hz, causal timing, or same-event cross-band pairing.
"""


def audit_dataset(
    *, dataset_root: Path, output_dir: Path, overwrite: bool = False
) -> dict[str, Any]:
    dataset_root = resolve_path(dataset_root)
    output_dir = resolve_path(output_dir)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {dataset_root}")
    if output_dir in {PROJECT_ROOT, dataset_root}:
        raise ValueError("output directory must be separate from project and dataset roots")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is nonempty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    missing = [
        spec.relative_path
        for spec in REQUIRED_AGGREGATES
        if not (dataset_root / spec.relative_path).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing required aggregate files: {missing}")

    aggregate_specs = _spec_by_path()
    inventory_rows: list[dict[str, Any]] = []
    aggregate_frames: list[pd.DataFrame] = []
    detail_counts: dict[tuple[str, int], int] = {}
    aggregate_counts: dict[tuple[str, int], int] = {}

    mat_paths = sorted(dataset_root.rglob("*.mat"))
    if not mat_paths:
        raise FileNotFoundError(f"no MAT files found under {dataset_root}")
    for path in mat_paths:
        relative = path.relative_to(dataset_root)
        relative_text = relative.as_posix()
        representation, band_code = _infer_file_spec(relative)
        variable_name, matrix = _single_public_matrix(path)
        checks = _validate_matrix(
            matrix,
            path=relative,
            representation=representation,
            band_code=band_code,
        )
        is_aggregate = relative_text in aggregate_specs
        key = (representation, band_code)
        if is_aggregate:
            aggregate_counts[key] = checks["record_count"]
            metadata = np.rint(matrix[:, :4]).astype(np.int64)
            frame = pd.DataFrame(
                metadata,
                columns=["band_code", "category_code", "model_code", "batch_code"],
            )
            frame.insert(0, "representation", representation)
            aggregate_frames.append(frame)
        else:
            detail_counts[key] = detail_counts.get(key, 0) + checks["record_count"]
        inventory_rows.append(
            {
                "relative_path": relative_text,
                "representation": representation,
                "band_code": band_code,
                "band": BAND_NAMES[band_code],
                "variable_name": variable_name,
                "is_aggregate": is_aggregate,
                **checks,
            }
        )

    for key, aggregate_count in aggregate_counts.items():
        if key in detail_counts and detail_counts[key] != aggregate_count:
            raise ValueError(
                f"detail rows {detail_counts[key]} do not match aggregate rows "
                f"{aggregate_count} for {key}"
            )

    records = pd.concat(aggregate_frames, ignore_index=True)
    records["band"] = records["band_code"].map(BAND_NAMES)
    records["category"] = records["category_code"].map(CATEGORY_NAMES)
    records["model"] = records["model_code"].map(MODEL_NAMES)

    label_counts = (
        records.groupby(
            [
                "representation",
                "band_code",
                "band",
                "category_code",
                "category",
                "model_code",
                "model",
            ],
            observed=True,
        )
        .agg(record_count=("batch_code", "size"), batch_count=("batch_code", "nunique"))
        .reset_index()
    )
    batch_groups = (
        records.groupby(["representation", "band_code", "band", "batch_code"], observed=True)
        .agg(
            record_count=("category_code", "size"),
            category_count=("category_code", "nunique"),
            model_count=("model_code", "nunique"),
            majority_category_fraction=("category_code", _majority_fraction),
        )
        .reset_index()
    )
    batch_groups["category_pure"] = batch_groups["category_count"].eq(1)
    batch_collisions = batch_groups.loc[
        batch_groups["category_count"].gt(1) | batch_groups["model_count"].gt(1)
    ].copy()

    category_readiness = (
        records.groupby(
            ["representation", "band_code", "band", "category_code", "category"],
            observed=True,
        )
        .agg(record_count=("batch_code", "size"), batch_count=("batch_code", "nunique"))
        .reset_index()
    )
    category_readiness["minimum_three_batches"] = category_readiness["batch_count"].ge(3)
    model_readiness = (
        records.groupby(
            [
                "representation",
                "band_code",
                "band",
                "category_code",
                "category",
                "model_code",
                "model",
            ],
            observed=True,
        )
        .agg(record_count=("batch_code", "size"), batch_count=("batch_code", "nunique"))
        .reset_index()
    )
    model_readiness["minimum_three_batches"] = model_readiness["batch_count"].ge(3)

    grouped_category_minimum_met = bool(category_readiness["minimum_three_batches"].all())
    summary = {
        "status": (
            "READY_FOR_PREREGISTERED_GROUPED_BASELINE"
            if grouped_category_minimum_met
            else "BLOCKED_INSUFFICIENT_CATEGORY_BATCHES"
        ),
        "dataset": "LAT-MRICD-1.0",
        "record_count": int(len(records)),
        "hrrp_record_count": int(records["representation"].eq("HRRP").sum()),
        "narrow_record_count": int(records["representation"].eq("Narrow").sum()),
        "aggregate_file_count": int(len(aggregate_frames)),
        "mat_file_count": int(len(mat_paths)),
        "schema_valid": True,
        "category_grouped_split_minimum_met": grouped_category_minimum_met,
        "fine_model_classes_below_three_batches": int(
            (~model_readiness["minimum_three_batches"]).sum()
        ),
        "batch_group_count": int(len(batch_groups)),
        "batch_code_collision_count": int(len(batch_collisions)),
        "category_pure_batch_fraction": float(batch_groups["category_pure"].mean()),
        "recommended_group_key": ["representation", "band_code", "batch_code"],
        "random_row_split_allowed": False,
        "batch_semantics_verified": False,
        "h_v_polarimetric_channels_available": False,
        "physical_micro_doppler_hz_allowed": False,
        "same_event_cross_band_pairing_verified": False,
    }

    pd.DataFrame(inventory_rows).to_csv(output_dir / "file_inventory.csv", index=False)
    label_counts.to_csv(output_dir / "label_counts.csv", index=False)
    batch_groups.to_csv(output_dir / "batch_group_summary.csv", index=False)
    batch_collisions.to_csv(output_dir / "batch_code_collisions.csv", index=False)
    category_readiness.to_csv(output_dir / "category_split_readiness.csv", index=False)
    model_readiness.to_csv(output_dir / "model_split_readiness.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "REPORT.md").write_text(_make_report(summary), encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    summary = audit_dataset(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
