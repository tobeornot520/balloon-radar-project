#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCAN_PATTERN = re.compile(r"(?<!\d)(\d{8}_\d{6})(?!\d)")
BEAM_PATTERN = re.compile(r"_beam\d+", re.IGNORECASE)
AZIMUTH_PATTERN = re.compile(r"_az\d+", re.IGNORECASE)

COLUMN_ALIASES = {
    "split": (
        "new_split",
        "split",
        "subset",
        "partition",
        "set",
        "data_split",
    ),
    "sample_id": (
        "sample_id",
        "id",
        "sample",
        "record_id",
        "uid",
    ),
    "mat_path": (
        "mat_path",
        "iq_path",
        "data_path",
        "input_path",
        "file_path",
        "filepath",
    ),
    "label_path": (
        "label_path",
        "annotation_path",
        "target_path",
    ),
    "source_file": (
        "source_file",
        "source_path",
        "raw_file",
        "origin_file",
    ),
    "target": (
        "target_present",
        "is_target",
        "is_uav",
        "label",
        "class",
        "target",
    ),
}

CANONICAL_SPLITS = {
    "training": "train",
    "train": "train",
    "tr": "train",
    "validation": "val",
    "valid": "val",
    "val": "val",
    "dev": "val",
    "testing": "test",
    "test": "test",
    "te": "test",
}

SEVERITY_ORDER = {
    "INFO": 0,
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
    "CRITICAL": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit radar cross-validation manifests for split leakage, "
            "scan-environment overlap, duplicate files, cross-fold rotation "
            "errors, and effective environmental sample size."
        )
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5, 6],
    )
    parser.add_argument(
        "--manifest-template",
        default=(
            "results/data_audit/dataset_v4_multifold/"
            "fold_{fold:02d}_manifest.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="results/data_audit/overfitting_audit",
    )
    parser.add_argument(
        "--hash-files",
        action="store_true",
        help=(
            "Compute SHA256 for existing data/label files. This is slower "
            "but catches identical files copied to different paths."
        ),
    )
    parser.add_argument(
        "--hash-chunk-size",
        type=int,
        default=1024 * 1024,
    )
    parser.add_argument(
        "--test-feedback-known",
        action="store_true",
        help=(
            "Record that test-fold outcomes influenced model design. "
            "This does not indicate file leakage, but makes current test "
            "metrics development estimates rather than final blind estimates."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code when critical leakage is detected.",
    )
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def detect_column(
    frame: pd.DataFrame,
    logical_name: str,
    *,
    required: bool = False,
) -> str | None:
    lower_map = {
        str(column).strip().lower(): str(column)
        for column in frame.columns
    }
    for alias in COLUMN_ALIASES[logical_name]:
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    if required:
        raise ValueError(
            f"Could not find required '{logical_name}' column. "
            f"Available columns: {list(frame.columns)}"
        )
    return None


def normalize_split(value: Any) -> str:
    text = str(value).strip().lower()
    return CANONICAL_SPLITS.get(text, text)


def normalized_path(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return str(Path(text).expanduser())


def basename_key(value: Any) -> str:
    text = normalized_path(value)
    return Path(text).name.lower() if text else ""


def stem_key(value: Any) -> str:
    text = normalized_path(value)
    return Path(text).stem.lower() if text else ""


def strip_sample_suffixes(value: str) -> str:
    result = BEAM_PATTERN.sub("", value)
    result = AZIMUTH_PATTERN.sub("", result)
    return result.rstrip("_- ")


def derive_scan_group(
    sample_id: str,
    mat_path: str,
    label_path: str,
    source_file: str,
) -> str:
    candidates = (
        sample_id,
        Path(mat_path).stem if mat_path else "",
        Path(label_path).stem if label_path else "",
        Path(source_file).stem if source_file else "",
    )
    for candidate in candidates:
        match = SCAN_PATTERN.search(str(candidate))
        if match:
            return match.group(1)

    for candidate in candidates:
        if candidate:
            return strip_sample_suffixes(str(candidate))
    return "UNKNOWN"


def derive_sample_key(
    sample_id: str,
    mat_path: str,
    source_file: str,
    row_index: int,
) -> str:
    if sample_id:
        return sample_id.strip().lower()
    if mat_path:
        return stem_key(mat_path)
    if source_file:
        return stem_key(source_file)
    return f"row_{row_index}"


def sha256_file(path_text: str, chunk_size: int) -> str:
    if not path_text:
        return ""
    path = Path(path_text).expanduser()
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def prepare_manifest(
    fold: int,
    path: Path,
    *,
    hash_files: bool,
    hash_chunk_size: int,
) -> tuple[pd.DataFrame, dict[str, str | None]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path, encoding="utf-8-sig")
    if frame.empty:
        raise ValueError(f"Manifest is empty: {path}")

    columns = {
        name: detect_column(
            frame,
            name,
            required=(name == "split"),
        )
        for name in COLUMN_ALIASES
    }

    prepared = pd.DataFrame(index=frame.index)
    prepared["fold"] = int(fold)
    prepared["manifest_path"] = str(path)
    prepared["manifest_row"] = np.arange(len(frame), dtype=int)
    prepared["split"] = frame[columns["split"]].map(
        normalize_split
    )

    for name in ("sample_id", "mat_path", "label_path", "source_file"):
        column = columns[name]
        if column is None:
            prepared[name] = ""
        else:
            prepared[name] = frame[column].map(normalized_path)

    target_column = columns["target"]
    if target_column is None:
        prepared["target"] = np.nan
    else:
        prepared["target"] = frame[target_column]

    prepared["sample_key"] = [
        derive_sample_key(sample_id, mat_path, source_file, row_index)
        for sample_id, mat_path, source_file, row_index in zip(
            prepared["sample_id"],
            prepared["mat_path"],
            prepared["source_file"],
            prepared["manifest_row"],
        )
    ]
    prepared["scan_group"] = [
        derive_scan_group(
            sample_id,
            mat_path,
            label_path,
            source_file,
        )
        for sample_id, mat_path, label_path, source_file in zip(
            prepared["sample_id"],
            prepared["mat_path"],
            prepared["label_path"],
            prepared["source_file"],
        )
    ]

    prepared["mat_basename"] = prepared["mat_path"].map(
        basename_key
    )
    prepared["mat_stem"] = prepared["mat_path"].map(stem_key)
    prepared["label_basename"] = prepared["label_path"].map(
        basename_key
    )
    prepared["source_basename"] = prepared["source_file"].map(
        basename_key
    )

    if hash_files:
        print(f"Hashing Fold {fold} files...")
        prepared["mat_sha256"] = [
            sha256_file(value, hash_chunk_size)
            for value in prepared["mat_path"]
        ]
        prepared["label_sha256"] = [
            sha256_file(value, hash_chunk_size)
            for value in prepared["label_path"]
        ]
    else:
        prepared["mat_sha256"] = ""
        prepared["label_sha256"] = ""

    allowed = {"train", "val", "test"}
    unknown = sorted(set(prepared["split"]) - allowed)
    if unknown:
        raise ValueError(
            f"Fold {fold} contains unsupported split values: {unknown}"
        )

    return prepared, columns


def nonempty_unique(values: Iterable[Any]) -> list[str]:
    result = sorted(
        {
            str(value)
            for value in values
            if value is not None
            and not pd.isna(value)
            and str(value).strip()
        }
    )
    return result


def overlap_records(
    frame: pd.DataFrame,
    *,
    fold: int,
    key_column: str,
    key_type: str,
    severity: str,
) -> list[dict[str, Any]]:
    if key_column not in frame.columns:
        return []

    subset = frame.loc[
        frame[key_column].astype(str).str.len() > 0
    ].copy()
    records: list[dict[str, Any]] = []

    for key_value, group in subset.groupby(
        key_column,
        sort=False,
    ):
        splits = nonempty_unique(group["split"])
        if len(splits) <= 1:
            continue

        records.append(
            {
                "fold": int(fold),
                "severity": severity,
                "key_type": key_type,
                "key_value": str(key_value),
                "splits": ",".join(splits),
                "row_count": int(len(group)),
                "train_count": int((group["split"] == "train").sum()),
                "val_count": int((group["split"] == "val").sum()),
                "test_count": int((group["split"] == "test").sum()),
                "sample_examples": " | ".join(
                    nonempty_unique(group["sample_id"])[:5]
                ),
                "path_examples": " | ".join(
                    nonempty_unique(group["mat_path"])[:3]
                ),
            }
        )
    return records


def within_split_duplicates(
    frame: pd.DataFrame,
    fold: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for split, split_frame in frame.groupby("split"):
        for key_type, key_column in (
            ("sample_key", "sample_key"),
            ("mat_path", "mat_path"),
            ("label_path", "label_path"),
        ):
            values = split_frame[key_column].astype(str)
            duplicated = (
                values.str.len().gt(0)
                & values.duplicated(keep=False)
            )
            for key_value, group in split_frame.loc[
                duplicated
            ].groupby(key_column):
                records.append(
                    {
                        "fold": int(fold),
                        "split": split,
                        "key_type": key_type,
                        "key_value": str(key_value),
                        "row_count": int(len(group)),
                        "sample_examples": " | ".join(
                            nonempty_unique(
                                group["sample_id"]
                            )[:5]
                        ),
                    }
                )
    return records


def split_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (fold, split), part in frame.groupby(
        ["fold", "split"],
        sort=True,
    ):
        target_values = part["target"]
        target_numeric = pd.to_numeric(
            target_values,
            errors="coerce",
        )
        rows.append(
            {
                "fold": int(fold),
                "split": split,
                "row_count": int(len(part)),
                "unique_sample_count": int(
                    part["sample_key"].nunique()
                ),
                "unique_scan_group_count": int(
                    part["scan_group"].nunique()
                ),
                "largest_scan_group_size": int(
                    part.groupby("scan_group").size().max()
                ),
                "median_scan_group_size": float(
                    part.groupby("scan_group").size().median()
                ),
                "positive_count_if_numeric": (
                    int((target_numeric > 0).sum())
                    if target_numeric.notna().any()
                    else math.nan
                ),
                "background_count_if_numeric": (
                    int((target_numeric == 0).sum())
                    if target_numeric.notna().any()
                    else math.nan
                ),
                "existing_mat_file_count": int(
                    sum(
                        Path(value).expanduser().is_file()
                        for value in part["mat_path"]
                        if value
                    )
                ),
                "existing_label_file_count": int(
                    sum(
                        Path(value).expanduser().is_file()
                        for value in part["label_path"]
                        if value
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def group_role_table(
    all_rows: pd.DataFrame,
    key_column: str,
    key_name: str,
    fold_count: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key_value, group in all_rows.groupby(
        key_column,
        sort=False,
    ):
        role_by_fold = (
            group.groupby("fold")["split"]
            .agg(lambda values: ",".join(nonempty_unique(values)))
            .to_dict()
        )
        role_counts = Counter(group["split"])

        row = {
            "key_type": key_name,
            "key_value": str(key_value),
            "fold_presence_count": int(group["fold"].nunique()),
            "train_count": int(role_counts.get("train", 0)),
            "val_count": int(role_counts.get("val", 0)),
            "test_count": int(role_counts.get("test", 0)),
            "expected_train_count_if_complete_cv": max(
                fold_count - 2,
                0,
            ),
            "expected_val_count_if_complete_cv": 1,
            "expected_test_count_if_complete_cv": 1,
        }
        for fold in sorted(all_rows["fold"].unique()):
            row[f"fold_{int(fold):02d}_role"] = role_by_fold.get(
                fold,
                ""
            )
        rows.append(row)
    return pd.DataFrame(rows)


def add_rotation_flags(
    table: pd.DataFrame,
    fold_count: int,
) -> pd.DataFrame:
    result = table.copy()
    expected_train = max(fold_count - 2, 0)
    complete_presence = (
        result["fold_presence_count"] == fold_count
    )
    result["complete_fold_presence"] = complete_presence
    result["rotation_ok"] = (
        complete_presence
        & (result["train_count"] == expected_train)
        & (result["val_count"] == 1)
        & (result["test_count"] == 1)
    )
    return result


def environment_risk_rows(
    summary: pd.DataFrame,
    all_rows: pd.DataFrame,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for fold in sorted(summary["fold"].unique()):
        fold_summary = summary.loc[summary["fold"] == fold]
        union_groups = all_rows.loc[
            all_rows["fold"] == fold,
            "scan_group",
        ].nunique()

        minimum_split_groups = int(
            fold_summary["unique_scan_group_count"].min()
        )
        if minimum_split_groups <= 1:
            severity = "CRITICAL"
            message = (
                "At least one split contains only one independent scan "
                "environment; sample-level metrics will substantially "
                "overstate environmental generalization."
            )
        elif minimum_split_groups <= 2:
            severity = "HIGH"
            message = (
                "At least one split contains only two independent scan "
                "environments; environmental effective sample size is very "
                "small."
            )
        elif minimum_split_groups < 5:
            severity = "MEDIUM"
            message = (
                "Some splits contain fewer than five scan environments; "
                "confidence intervals should be reported by scan group."
            )
        else:
            severity = "LOW"
            message = (
                "Each split contains at least five scan environments."
            )

        records.append(
            {
                "fold": int(fold),
                "severity": severity,
                "risk_type": "environment_effective_sample_size",
                "unique_scan_groups_in_fold": int(union_groups),
                "minimum_scan_groups_in_one_split": (
                    minimum_split_groups
                ),
                "message": message,
            }
        )

    return records


def highest_severity(records: list[dict[str, Any]]) -> str:
    if not records:
        return "LOW"
    return max(
        (str(record["severity"]) for record in records),
        key=lambda value: SEVERITY_ORDER[value],
    )


def main() -> None:
    args = parse_args()

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared_frames: list[pd.DataFrame] = []
    detected_columns: dict[int, dict[str, str | None]] = {}

    for fold in args.folds:
        manifest_path = resolve_path(
            args.manifest_template.format(fold=fold)
        )
        print(f"Reading Fold {fold}: {manifest_path}")
        prepared, columns = prepare_manifest(
            fold,
            manifest_path,
            hash_files=args.hash_files,
            hash_chunk_size=args.hash_chunk_size,
        )
        prepared_frames.append(prepared)
        detected_columns[int(fold)] = columns

    all_rows = pd.concat(
        prepared_frames,
        ignore_index=True,
    )

    all_rows.to_csv(
        output_dir / "normalized_manifest_rows.csv",
        index=False,
        encoding="utf-8-sig",
    )

    overlap_specs = [
        ("sample_key", "exact_sample_identity", "CRITICAL"),
        ("mat_path", "exact_mat_path", "CRITICAL"),
        ("label_path", "exact_label_path", "CRITICAL"),
        ("source_file", "exact_source_file", "HIGH"),
        ("mat_sha256", "identical_mat_content_sha256", "CRITICAL"),
        (
            "label_sha256",
            "identical_label_content_sha256",
            "CRITICAL",
        ),
        ("scan_group", "scan_environment", "HIGH"),
        ("mat_basename", "same_mat_basename", "MEDIUM"),
        ("mat_stem", "same_mat_stem", "MEDIUM"),
        (
            "source_basename",
            "same_source_basename",
            "MEDIUM",
        ),
    ]

    overlap_rows: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []

    for fold, frame in all_rows.groupby("fold", sort=True):
        for key_column, key_type, severity in overlap_specs:
            overlap_rows.extend(
                overlap_records(
                    frame,
                    fold=int(fold),
                    key_column=key_column,
                    key_type=key_type,
                    severity=severity,
                )
            )
        duplicate_rows.extend(
            within_split_duplicates(frame, int(fold))
        )

    overlaps = pd.DataFrame(overlap_rows)
    if overlaps.empty:
        overlaps = pd.DataFrame(
            columns=[
                "fold",
                "severity",
                "key_type",
                "key_value",
                "splits",
                "row_count",
                "train_count",
                "val_count",
                "test_count",
                "sample_examples",
                "path_examples",
            ]
        )
    overlaps.to_csv(
        output_dir / "within_fold_split_overlaps.csv",
        index=False,
        encoding="utf-8-sig",
    )

    duplicates = pd.DataFrame(duplicate_rows)
    duplicates.to_csv(
        output_dir / "within_split_duplicates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = split_summary(all_rows)
    summary.to_csv(
        output_dir / "split_environment_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    sample_rotation = add_rotation_flags(
        group_role_table(
            all_rows,
            "sample_key",
            "sample",
            len(args.folds),
        ),
        len(args.folds),
    )
    sample_rotation.to_csv(
        output_dir / "cross_fold_sample_rotation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    group_rotation = add_rotation_flags(
        group_role_table(
            all_rows,
            "scan_group",
            "scan_group",
            len(args.folds),
        ),
        len(args.folds),
    )
    group_rotation.to_csv(
        output_dir / "cross_fold_scan_group_rotation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    risk_records = environment_risk_rows(summary, all_rows)

    critical_overlap_count = int(
        (
            overlaps["severity"].astype(str) == "CRITICAL"
        ).sum()
    )
    high_overlap_count = int(
        (
            overlaps["severity"].astype(str) == "HIGH"
        ).sum()
    )

    if critical_overlap_count:
        risk_records.append(
            {
                "fold": "all",
                "severity": "CRITICAL",
                "risk_type": "exact_split_leakage",
                "unique_scan_groups_in_fold": math.nan,
                "minimum_scan_groups_in_one_split": math.nan,
                "message": (
                    f"Detected {critical_overlap_count} exact identity/path/"
                    "content overlaps across train/val/test."
                ),
            }
        )

    scan_overlap_count = int(
        (
            overlaps["key_type"].astype(str)
            == "scan_environment"
        ).sum()
    )
    if scan_overlap_count:
        risk_records.append(
            {
                "fold": "all",
                "severity": "HIGH",
                "risk_type": "scan_environment_leakage",
                "unique_scan_groups_in_fold": math.nan,
                "minimum_scan_groups_in_one_split": math.nan,
                "message": (
                    f"Detected {scan_overlap_count} scan groups spanning "
                    "multiple splits within the same fold."
                ),
            }
        )

    sample_rotation_bad = int(
        (~sample_rotation["rotation_ok"]).sum()
    )
    group_rotation_bad = int(
        (~group_rotation["rotation_ok"]).sum()
    )

    if sample_rotation_bad:
        risk_records.append(
            {
                "fold": "all",
                "severity": "MEDIUM",
                "risk_type": "cross_fold_sample_rotation_anomaly",
                "unique_scan_groups_in_fold": math.nan,
                "minimum_scan_groups_in_one_split": math.nan,
                "message": (
                    f"{sample_rotation_bad} sample identities do not follow "
                    "the expected train=(F-2), val=1, test=1 rotation. "
                    "This can be legitimate only if fold universes differ."
                ),
            }
        )

    if group_rotation_bad:
        risk_records.append(
            {
                "fold": "all",
                "severity": "MEDIUM",
                "risk_type": "cross_fold_scan_rotation_anomaly",
                "unique_scan_groups_in_fold": math.nan,
                "minimum_scan_groups_in_one_split": math.nan,
                "message": (
                    f"{group_rotation_bad} scan groups do not follow the "
                    "expected train=(F-2), val=1, test=1 rotation."
                ),
            }
        )

    if args.test_feedback_known:
        risk_records.append(
            {
                "fold": "all",
                "severity": "HIGH",
                "risk_type": "test_feedback_overfitting",
                "unique_scan_groups_in_fold": math.nan,
                "minimum_scan_groups_in_one_split": math.nan,
                "message": (
                    "Test-fold outcomes influenced v1/v2/v3 design. Current "
                    "six-fold test metrics are development estimates and "
                    "require a new blind scan set or nested group CV for a "
                    "final unbiased claim."
                ),
            }
        )

    risk_table = pd.DataFrame(risk_records)
    risk_table.to_csv(
        output_dir / "overfitting_risk_register.csv",
        index=False,
        encoding="utf-8-sig",
    )

    overall_severity = highest_severity(risk_records)

    result_summary = {
        "folds": list(args.folds),
        "manifest_template": args.manifest_template,
        "hash_files_enabled": bool(args.hash_files),
        "test_feedback_known": bool(args.test_feedback_known),
        "detected_columns": detected_columns,
        "row_count": int(len(all_rows)),
        "unique_sample_count": int(
            all_rows["sample_key"].nunique()
        ),
        "unique_scan_group_count": int(
            all_rows["scan_group"].nunique()
        ),
        "within_fold_overlap_count": int(len(overlaps)),
        "critical_overlap_count": critical_overlap_count,
        "high_overlap_count": high_overlap_count,
        "scan_environment_overlap_count": scan_overlap_count,
        "within_split_duplicate_count": int(len(duplicates)),
        "sample_rotation_anomaly_count": sample_rotation_bad,
        "scan_group_rotation_anomaly_count": group_rotation_bad,
        "overall_risk_severity": overall_severity,
        "deployment_note": (
            "The current v3 group features aggregate all samples from a scan "
            "group. This is valid for offline complete-scan processing. For "
            "real-time deployment, recompute group features causally using "
            "only current and previously observed beams/azimuths."
        ),
        "interpretation": {
            "no_critical_overlap": (
                critical_overlap_count == 0
            ),
            "scan_groups_disjoint_within_folds": (
                scan_overlap_count == 0
            ),
            "current_test_is_blind": (
                not args.test_feedback_known
            ),
            "recommended_final_validation": (
                "Freeze v3 and evaluate once on newly collected scan groups. "
                "If new data are unavailable, run nested leave-one-scan-group-"
                "out validation with all model choices made only inside the "
                "inner folds."
            ),
        },
        "output_files": {
            "normalized_rows": str(
                output_dir / "normalized_manifest_rows.csv"
            ),
            "split_overlaps": str(
                output_dir / "within_fold_split_overlaps.csv"
            ),
            "within_split_duplicates": str(
                output_dir / "within_split_duplicates.csv"
            ),
            "environment_summary": str(
                output_dir / "split_environment_summary.csv"
            ),
            "sample_rotation": str(
                output_dir / "cross_fold_sample_rotation.csv"
            ),
            "scan_rotation": str(
                output_dir / "cross_fold_scan_group_rotation.csv"
            ),
            "risk_register": str(
                output_dir / "overfitting_risk_register.csv"
            ),
        },
    }

    (output_dir / "summary.json").write_text(
        json.dumps(
            result_summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 92)
    print("SPLIT / ENVIRONMENT SUMMARY")
    print("=" * 92)
    print(
        summary[
            [
                "fold",
                "split",
                "row_count",
                "unique_sample_count",
                "unique_scan_group_count",
                "largest_scan_group_size",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 92)
    print("WITHIN-FOLD LEAKAGE SUMMARY")
    print("=" * 92)
    if overlaps.empty:
        print("No cross-split overlap detected.")
    else:
        counts = (
            overlaps.groupby(["severity", "key_type"])
            .size()
            .reset_index(name="count")
            .sort_values(
                ["severity", "count"],
                ascending=[False, False],
            )
        )
        print(counts.to_string(index=False))

    print("\n" + "=" * 92)
    print("CROSS-FOLD ROTATION")
    print("=" * 92)
    print(
        f"Sample rotation anomalies    : {sample_rotation_bad}"
    )
    print(
        f"Scan-group rotation anomalies: {group_rotation_bad}"
    )

    print("\n" + "=" * 92)
    print("OVERFITTING RISK REGISTER")
    print("=" * 92)
    print(
        risk_table[
            ["fold", "severity", "risk_type", "message"]
        ].to_string(index=False)
    )

    print("\n" + "=" * 92)
    print(f"Overall risk severity: {overall_severity}")
    print(f"Saved outputs to: {output_dir}")
    print("=" * 92)

    if args.strict and critical_overlap_count:
        sys.exit(2)


if __name__ == "__main__":
    main()
