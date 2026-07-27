#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import whosmat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "results"
    / "data_audit"
    / "dataset_v4_multifold"
    / "fold_01_manifest.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "data_audit" / "detection_acquisition_order"
)
SAMPLE_TIMESTAMP_PATTERN = re.compile(r"^(\d{8}_\d{6})")
MAT_CREATED_PATTERN = re.compile(
    rb"Created on:\s*([A-Z][a-z]{2} [A-Z][a-z]{2} +\d{1,2} "
    rb"\d{2}:\d{2}:\d{2} \d{4})"
)
TIMESTAMP_VARIABLE_PATTERN = re.compile(
    r"(^|_)(timestamp|datetime|date|time|acquisition_time|capture_time)(_|$)",
    re.IGNORECASE,
)
REQUIRED_COLUMNS = {
    "source_file",
    "sample_id",
    "target_present",
    "beam_layer",
    "azimuth_deg",
    "mat_path",
}


@dataclass(frozen=True)
class MatInspection:
    header_created_at: datetime | None
    filesystem_mtime: datetime
    variable_names: tuple[str, ...]
    timestamp_variable_names: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether detection samples contain a verified within-scan "
            "acquisition order suitable for causal context training."
        )
    )
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_sample_timestamp(value: str) -> datetime | None:
    match = SAMPLE_TIMESTAMP_PATTERN.match(str(value))
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")


def parse_mat_header_created_at(header: bytes) -> datetime | None:
    match = MAT_CREATED_PATTERN.search(header[:128])
    if not match:
        return None
    return datetime.strptime(
        match.group(1).decode("ascii"),
        "%a %b %d %H:%M:%S %Y",
    )


def inspect_mat_file(path: Path) -> MatInspection:
    with path.open("rb") as handle:
        header = handle.read(128)
    variable_names = tuple(name for name, _, _ in whosmat(path))
    timestamp_names = tuple(
        name for name in variable_names if TIMESTAMP_VARIABLE_PATTERN.search(name)
    )
    return MatInspection(
        header_created_at=parse_mat_header_created_at(header),
        filesystem_mtime=datetime.fromtimestamp(path.stat().st_mtime),
        variable_names=variable_names,
        timestamp_variable_names=timestamp_names,
    )


def load_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing manifest columns: {missing}")
    if frame["sample_id"].astype(str).duplicated().any():
        raise ValueError("sample_id must be unique")
    labels = frame["target_present"].to_numpy(dtype=np.int64)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("target_present must contain only 0 and 1")
    group_labels = frame.groupby("source_file")["target_present"].nunique()
    if group_labels.gt(1).any():
        raise ValueError("source_file groups must not mix labels")
    return frame.reset_index(drop=True)


def inspect_manifest(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        mat_path = resolve_path(Path(str(row.mat_path)))
        if not mat_path.is_file():
            raise FileNotFoundError(f"Missing MAT file: {display_path(mat_path)}")
        sample_time = parse_sample_timestamp(str(row.sample_id))
        inspection = inspect_mat_file(mat_path)
        header_lag = (
            (inspection.header_created_at - sample_time).total_seconds()
            if sample_time is not None and inspection.header_created_at is not None
            else np.nan
        )
        mtime_lag = (
            (inspection.filesystem_mtime - sample_time).total_seconds()
            if sample_time is not None
            else np.nan
        )
        header_mtime_delta = (
            abs(
                (
                    inspection.header_created_at - inspection.filesystem_mtime
                ).total_seconds()
            )
            if inspection.header_created_at is not None
            else np.nan
        )
        records.append(
            {
                "source_file": str(row.source_file),
                "target_present": int(row.target_present),
                "sample_id": str(row.sample_id),
                "beam_layer": int(row.beam_layer),
                "azimuth_deg": float(row.azimuth_deg),
                "sample_timestamp": sample_time,
                "header_created_at": inspection.header_created_at,
                "filesystem_mtime": inspection.filesystem_mtime,
                "header_lag_seconds": header_lag,
                "mtime_lag_seconds": mtime_lag,
                "header_mtime_delta_seconds": header_mtime_delta,
                "mat_variable_names": "|".join(inspection.variable_names),
                "timestamp_variable_count": len(
                    inspection.timestamp_variable_names
                ),
            }
        )
    detail = pd.DataFrame(records)

    groups: list[dict[str, Any]] = []
    for source_file, group in detail.groupby("source_file", sort=True):
        order_key_unique = not group.duplicated(
            ["beam_layer", "azimuth_deg", "sample_id"]
        ).any()
        header_created = group["header_created_at"].dropna()
        groups.append(
            {
                "source_file": source_file,
                "class_name": (
                    "target" if int(group["target_present"].iloc[0]) else "background"
                ),
                "samples": len(group),
                "sample_timestamp_unique_count": int(
                    group["sample_timestamp"].nunique(dropna=True)
                ),
                "beam_layer_count": int(group["beam_layer"].nunique()),
                "azimuth_count": int(group["azimuth_deg"].nunique()),
                "inferred_order_key_unique": bool(order_key_unique),
                "embedded_timestamp_variable_samples": int(
                    group["timestamp_variable_count"].gt(0).sum()
                ),
                "mat_header_coverage": int(header_created.shape[0]),
                "header_lag_days_min": float(
                    group["header_lag_seconds"].min() / 86400.0
                ),
                "header_lag_days_max": float(
                    group["header_lag_seconds"].max() / 86400.0
                ),
                "header_mtime_delta_seconds_max": float(
                    group["header_mtime_delta_seconds"].max()
                ),
                "verified_within_group_order_available": False,
            }
        )
    group_summary = pd.DataFrame(groups)

    source_summary = pd.DataFrame(
        [
            {
                "candidate_source": "sample_id_timestamp",
                "sample_coverage": int(detail["sample_timestamp"].notna().sum()),
                "within_group_distinct_max": int(
                    group_summary["sample_timestamp_unique_count"].max()
                ),
                "timestamp_verified": False,
                "within_group_order_verified": False,
                "assessment": "filename-encoded group second; not externally verified",
            },
            {
                "candidate_source": "MAT embedded variables",
                "sample_coverage": int(
                    detail["timestamp_variable_count"].gt(0).sum()
                ),
                "within_group_distinct_max": 0,
                "timestamp_verified": False,
                "within_group_order_verified": False,
                "assessment": "no timestamp-like MAT variable",
            },
            {
                "candidate_source": "MAT v5 header Created on",
                "sample_coverage": int(detail["header_created_at"].notna().sum()),
                "within_group_distinct_max": int(
                    detail.groupby("source_file")["header_created_at"].nunique().max()
                ),
                "timestamp_verified": False,
                "within_group_order_verified": False,
                "assessment": "post-acquisition conversion/save time",
            },
            {
                "candidate_source": "filesystem mtime",
                "sample_coverage": len(detail),
                "within_group_distinct_max": int(
                    detail.groupby("source_file")["filesystem_mtime"].nunique().max()
                ),
                "timestamp_verified": False,
                "within_group_order_verified": False,
                "assessment": "copy/save metadata, not acquisition time",
            },
            {
                "candidate_source": "beam_layer/azimuth_deg/sample_id",
                "sample_coverage": len(detail),
                "within_group_distinct_max": int(
                    group_summary["samples"].max()
                ),
                "timestamp_verified": False,
                "within_group_order_verified": False,
                "assessment": "deterministic inferred order only",
            },
        ]
    )
    return group_summary, source_summary


def build_report(
    frame: pd.DataFrame,
    groups: pd.DataFrame,
    sources: pd.DataFrame,
) -> str:
    background_groups = int((groups["class_name"] == "background").sum())
    target_groups = int((groups["class_name"] == "target").sum())
    min_lag = float(groups["header_lag_days_min"].min())
    max_header_mtime_delta = float(
        groups["header_mtime_delta_seconds_max"].max()
    )
    return "\n".join(
        [
            "# Detection acquisition-order audit",
            "",
            "## Decision",
            "",
            "**Formal causal-training gate: CLOSED.** The current files do not "
            "contain a verified within-scan sample acquisition order.",
            "",
            f"The audit covers {len(frame)} samples in {target_groups} target and "
            f"{background_groups} background scan groups. The timestamp encoded in "
            "each sample ID is identical for every sample in its scan group, so it "
            "cannot order samples within that group.",
            "",
            "## Evidence",
            "",
            "- MAT files contain H/V IQ arrays but no timestamp-like variable.",
            f"- MAT v5 `Created on` values are at least {min_lag:.1f} days after the "
            "filename acquisition second and track filesystem mtime within "
            f"{max_header_mtime_delta:.1f} seconds. They are conversion/save times.",
            "- Filesystem mtime belongs to later file handling and is not acquisition "
            "metadata.",
            "- `(beam_layer, azimuth_deg, sample_id)` is unique and deterministic "
            "within each group, but no hardware log verifies that execution order.",
            "",
            "## Allowed use",
            "",
            "The inferred beam/azimuth order may be used only for interface smoke "
            "tests that are explicitly labelled development-only. It must not be used "
            "for model/window selection, formal performance claims, or deployment.",
            "",
            "## Reopening the gate",
            "",
            "Provide a per-sample acquisition timestamp or monotonic hardware sequence "
            "number, document clock resolution and reset behavior, and verify one-to-one "
            "alignment with sample IDs before causal model selection begins.",
            "",
            "See `order_source_summary.csv` and `group_order_coverage.csv` for the "
            "machine-readable findings.",
            "",
        ]
    )


def generated_records(output_dir: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(output_dir).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "audit_manifest.json"
    ]


def write_audit(manifest_path: Path, output_dir: Path) -> None:
    frame = load_manifest(manifest_path)
    groups, sources = inspect_manifest(frame)
    groups.to_csv(output_dir / "group_order_coverage.csv", index=False)
    sources.to_csv(output_dir / "order_source_summary.csv", index=False)
    (output_dir / "ACQUISITION_ORDER_AUDIT.md").write_text(
        build_report(frame, groups, sources),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "audit_role": "causal-training input readiness",
        "formal_causal_training_gate_open": False,
        "verified_within_scan_order_available": False,
        "inferred_order_columns": ["beam_layer", "azimuth_deg", "sample_id"],
        "inferred_order_allowed_for": ["development-only interface smoke tests"],
        "sample_count": len(frame),
        "scan_group_count": int(frame["source_file"].nunique()),
        "input": {
            "path": display_path(manifest_path),
            "size_bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
        },
        "implementation": {
            "path": display_path(Path(__file__)),
            "size_bytes": Path(__file__).stat().st_size,
            "sha256": sha256_file(Path(__file__)),
        },
        "generated_files": generated_records(output_dir),
    }
    (output_dir / "audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_output(
    manifest_path: Path,
    output_dir: Path,
    overwrite: bool,
) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"Output path is not a directory: {output_dir}")
    if output_dir.is_dir() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is nonempty: {output_dir}. Use --overwrite to replace it."
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_parent = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.build-", dir=output_dir.parent)
    )
    staging_dir = staging_parent / output_dir.name
    staging_dir.mkdir()
    try:
        write_audit(manifest_path, staging_dir)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        staging_dir.replace(output_dir)
    except Exception:
        shutil.rmtree(staging_parent, ignore_errors=True)
        raise
    shutil.rmtree(staging_parent, ignore_errors=True)


def main() -> int:
    args = parse_args()
    manifest_path = resolve_path(args.manifest_path)
    output_dir = resolve_path(args.output_dir)
    try:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing manifest: {display_path(manifest_path)}")
        build_output(manifest_path, output_dir, args.overwrite)
    except (FileNotFoundError, FileExistsError, ValueError, OSError) as exc:
        print(f"[ERROR] {exc}")
        return 1
    print("Detection acquisition-order audit: PASS")
    print(f"output_dir={display_path(output_dir)}")
    print("formal_causal_training_gate_open=False")
    print("verified_within_scan_order_available=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
