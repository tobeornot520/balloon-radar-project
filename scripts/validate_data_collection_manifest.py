#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = PROJECT_ROOT / "configs" / "data_collection_contract_v1.json"
PROFILE_ORDER = ("capture", "causal", "locked_evaluation")
ALL_GATES = (
    "schema",
    "row_integrity",
    "event_timing",
    "causal_order",
    "channel_integrity",
    "partition_isolation",
    "same_condition_class_control",
)
PROFILE_GATES = {
    "capture": ALL_GATES[:3],
    "causal": ALL_GATES[:5],
    "locked_evaluation": ALL_GATES,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a new radar collection manifest against the versioned "
            "capture, causal, or locked-evaluation data contract."
        )
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--profile",
        choices=PROFILE_ORDER,
        default="capture",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("schema_version") != 1:
        raise ValueError("Unsupported collection contract schema_version")
    profiles = tuple(contract.get("profiles", []))
    if profiles != PROFILE_ORDER:
        raise ValueError(f"Contract profiles must be {PROFILE_ORDER}")
    columns = contract.get("columns")
    if not isinstance(columns, list) or not columns:
        raise ValueError("Contract columns must be a nonempty list")
    names = [item.get("name") for item in columns]
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Every contract column needs a nonempty name")
    if len(names) != len(set(names)):
        raise ValueError("Contract column names must be unique")
    return contract


def nonempty_mask(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype(str).str.strip().ne("")


def issue(
    issues: list[dict[str, Any]],
    *,
    code: str,
    gate: str,
    message: str,
    rows: pd.Index | list[int] | None = None,
) -> None:
    row_numbers: list[int] = []
    if rows is not None:
        row_numbers = [int(value) + 2 for value in list(rows)[:10]]
    issues.append(
        {
            "severity": "ERROR",
            "code": code,
            "gate": gate,
            "message": message,
            "row_examples": row_numbers,
        }
    )


def parse_utc_datetime(value: Any) -> datetime | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        return None
    return parsed.astimezone(timezone.utc)


def parse_iso_date(value: Any) -> date | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def parse_boolean_series(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "1": True,
        "false": False,
        "0": False,
    }
    return normalized.map(mapping)


def validate_column(
    frame: pd.DataFrame,
    specification: dict[str, Any],
    issues: list[dict[str, Any]],
) -> None:
    name = str(specification["name"])
    series = frame[name]
    present = nonempty_mask(series)
    if not specification.get("nullable", False) and not present.all():
        issue(
            issues,
            code="NULL_REQUIRED_VALUE",
            gate="schema",
            message=f"{name} contains {int((~present).sum())} empty required values",
            rows=frame.index[~present],
        )
    values = series.loc[present]
    if values.empty:
        return

    value_type = specification.get("type")
    invalid = pd.Series(False, index=values.index)
    numeric: pd.Series | None = None
    if value_type == "string":
        invalid = values.astype(str).str.strip().eq("")
    elif value_type in {"integer", "number"}:
        numeric = pd.to_numeric(values, errors="coerce")
        invalid = numeric.isna() | ~np.isfinite(numeric)
        if value_type == "integer":
            invalid |= numeric.mod(1).ne(0)
    elif value_type == "boolean":
        invalid = parse_boolean_series(values).isna()
    elif value_type == "date":
        invalid = values.map(parse_iso_date).isna()
    elif value_type == "datetime_utc":
        invalid = values.map(parse_utc_datetime).isna()
    elif value_type == "relative_path":
        invalid_values = []
        for value in values.astype(str):
            path = Path(value.strip())
            invalid_values.append(path.is_absolute() or ".." in path.parts)
        invalid = pd.Series(invalid_values, index=values.index)
    elif value_type == "enum":
        invalid = ~values.astype(str).isin(specification.get("allowed", []))
    else:
        raise ValueError(f"Unsupported contract column type: {value_type}")

    if invalid.any():
        issue(
            issues,
            code="INVALID_COLUMN_VALUE",
            gate="schema",
            message=f"{name} contains {int(invalid.sum())} invalid values",
            rows=invalid.index[invalid],
        )

    if numeric is not None:
        valid_numeric = numeric.loc[~invalid]
        minimum = specification.get("minimum")
        maximum = specification.get("maximum")
        maximum_exclusive = specification.get("maximum_exclusive")
        if minimum is not None:
            bad = valid_numeric.lt(float(minimum))
            if bad.any():
                issue(
                    issues,
                    code="VALUE_BELOW_MINIMUM",
                    gate="schema",
                    message=f"{name} contains values below {minimum}",
                    rows=bad.index[bad],
                )
        if maximum is not None:
            bad = valid_numeric.gt(float(maximum))
            if bad.any():
                issue(
                    issues,
                    code="VALUE_ABOVE_MAXIMUM",
                    gate="schema",
                    message=f"{name} contains values above {maximum}",
                    rows=bad.index[bad],
                )
        if maximum_exclusive is not None:
            bad = valid_numeric.ge(float(maximum_exclusive))
            if bad.any():
                issue(
                    issues,
                    code="VALUE_AT_OR_ABOVE_EXCLUSIVE_MAXIMUM",
                    gate="schema",
                    message=(
                        f"{name} contains values at or above {maximum_exclusive}"
                    ),
                    rows=bad.index[bad],
                )

    allowed = specification.get("allowed")
    if allowed is not None and value_type != "enum":
        comparable = pd.to_numeric(values, errors="coerce")
        bad = ~comparable.isin(allowed)
        if bad.any():
            issue(
                issues,
                code="VALUE_NOT_ALLOWED",
                gate="schema",
                message=f"{name} contains values outside {allowed}",
                rows=bad.index[bad],
            )


def build_column_coverage(
    frame: pd.DataFrame,
    contract: dict[str, Any],
) -> pd.DataFrame:
    records = []
    for specification in contract["columns"]:
        name = str(specification["name"])
        present = name in frame.columns
        count = int(nonempty_mask(frame[name]).sum()) if present else 0
        records.append(
            {
                "column": name,
                "type": specification["type"],
                "nullable": bool(specification.get("nullable", False)),
                "present": present,
                "nonempty_rows": count,
                "row_count": len(frame),
                "coverage": float(count / len(frame)) if len(frame) else 0.0,
            }
        )
    return pd.DataFrame(records)


def validate_row_integrity(
    frame: pd.DataFrame,
    issues: list[dict[str, Any]],
    *,
    data_root: Path | None,
    check_files: bool,
) -> None:
    duplicate = frame["sample_id"].astype(str).duplicated(keep=False)
    if duplicate.any():
        issue(
            issues,
            code="DUPLICATE_SAMPLE_ID",
            gate="row_integrity",
            message=f"sample_id has {int(duplicate.sum())} duplicated rows",
            rows=frame.index[duplicate],
        )

    labels = pd.to_numeric(frame["target_present"], errors="coerce")
    classes = frame["target_class"].astype(str)
    background = labels.eq(0)
    target = labels.eq(1)
    mismatch = (background & classes.ne("background")) | (
        target & classes.eq("background")
    )
    if mismatch.any():
        issue(
            issues,
            code="TARGET_CLASS_MISMATCH",
            gate="row_integrity",
            message="target_class is inconsistent with target_present",
            rows=frame.index[mismatch],
        )

    for column in (
        "label_path",
        "flight_id",
        "platform_id",
        "distance_m",
        "velocity_mps",
    ):
        missing = target & ~nonempty_mask(frame[column])
        if missing.any():
            issue(
                issues,
                code="MISSING_TARGET_METADATA",
                gate="row_integrity",
                message=f"Target rows require {column}",
                rows=frame.index[missing],
            )
    missing_background_type = background & ~nonempty_mask(frame["background_type"])
    if missing_background_type.any():
        issue(
            issues,
            code="MISSING_BACKGROUND_TYPE",
            gate="row_integrity",
            message="Background rows require background_type",
            rows=frame.index[missing_background_type],
        )

    loaded_balloon = classes.eq("balloon_loaded")
    missing_payload = loaded_balloon & ~nonempty_mask(frame["payload_class"])
    if missing_payload.any():
        issue(
            issues,
            code="MISSING_BALLOON_PAYLOAD_CLASS",
            gate="row_integrity",
            message="balloon_loaded rows require payload_class",
            rows=frame.index[missing_payload],
        )
    balloon = classes.isin({"balloon_unloaded", "balloon_loaded"})
    missing_motion = balloon & ~nonempty_mask(frame["motion_state"])
    if missing_motion.any():
        issue(
            issues,
            code="MISSING_BALLOON_MOTION_STATE",
            gate="row_integrity",
            message="Balloon rows require motion_state",
            rows=frame.index[missing_motion],
        )

    timestamps = frame["acquisition_timestamp_utc"].map(parse_utc_datetime)
    collection_dates = frame["collection_date"].map(parse_iso_date)
    comparable = timestamps.notna() & collection_dates.notna()
    wrong_date = comparable & pd.Series(
        [
            timestamp.date() != collection_date_value
            if timestamp is not None and collection_date_value is not None
            else False
            for timestamp, collection_date_value in zip(
                timestamps, collection_dates, strict=True
            )
        ],
        index=frame.index,
    )
    if wrong_date.any():
        issue(
            issues,
            code="COLLECTION_DATE_MISMATCH",
            gate="row_integrity",
            message="collection_date must equal the UTC acquisition date",
            rows=frame.index[wrong_date],
        )

    if check_files:
        if data_root is None:
            issue(
                issues,
                code="DATA_ROOT_REQUIRED",
                gate="row_integrity",
                message="--check-files requires --data-root",
            )
        else:
            for column in ("iq_path", "label_path"):
                required = nonempty_mask(frame[column])
                missing_rows = []
                for index, value in frame.loc[required, column].items():
                    if not (data_root / str(value)).is_file():
                        missing_rows.append(int(index))
                if missing_rows:
                    issue(
                        issues,
                        code="MISSING_REFERENCED_FILE",
                        gate="row_integrity",
                        message=(
                            f"{column} has {len(missing_rows)} paths missing under "
                            "the supplied data root"
                        ),
                        rows=missing_rows,
                    )


def validate_event_timing(
    frame: pd.DataFrame,
    issues: list[dict[str, Any]],
) -> None:
    acquired = frame["acquisition_timestamp_utc"].map(parse_utc_datetime)
    starts = frame["event_start_utc"].map(parse_utc_datetime)
    ends = frame["event_end_utc"].map(parse_utc_datetime)
    valid = acquired.notna() & starts.notna() & ends.notna()
    invalid_bounds = valid & pd.Series(
        [
            not (start <= timestamp <= end)
            if timestamp is not None and start is not None and end is not None
            else False
            for timestamp, start, end in zip(acquired, starts, ends, strict=True)
        ],
        index=frame.index,
    )
    if invalid_bounds.any():
        issue(
            issues,
            code="SAMPLE_OUTSIDE_EVENT_BOUNDS",
            gate="event_timing",
            message="Acquisition timestamp must lie inside event bounds",
            rows=frame.index[invalid_bounds],
        )
    reversed_bounds = starts.notna() & ends.notna() & pd.Series(
        [
            start > end if start is not None and end is not None else False
            for start, end in zip(starts, ends, strict=True)
        ],
        index=frame.index,
    )
    if reversed_bounds.any():
        issue(
            issues,
            code="REVERSED_EVENT_BOUNDS",
            gate="event_timing",
            message="event_start_utc must not be later than event_end_utc",
            rows=frame.index[reversed_bounds],
        )
    for event_id, group in frame.groupby("event_id", sort=False):
        if not event_id:
            continue
        for column in ("event_start_utc", "event_end_utc", "observation_id"):
            if group[column].astype(str).nunique(dropna=False) != 1:
                issue(
                    issues,
                    code="INCONSISTENT_EVENT_METADATA",
                    gate="event_timing",
                    message=f"event_id must map to one {column}",
                    rows=group.index,
                )


def validate_causal_order(
    frame: pd.DataFrame,
    issues: list[dict[str, Any]],
) -> None:
    verified = parse_boolean_series(frame["order_verified"])
    bad_verified = verified.ne(True)
    if bad_verified.any():
        issue(
            issues,
            code="ORDER_NOT_VERIFIED",
            gate="causal_order",
            message="Causal profile requires order_verified=true for every row",
            rows=frame.index[bad_verified],
        )
    bad_source = frame["order_source"].astype(str).ne("hardware_sequence")
    if bad_source.any():
        issue(
            issues,
            code="UNVERIFIED_ORDER_SOURCE",
            gate="causal_order",
            message="Causal profile requires hardware_sequence order source",
            rows=frame.index[bad_source],
        )

    for scan_id, group in frame.groupby("scan_id", sort=False):
        if not scan_id:
            continue
        for column in (
            "session_id",
            "clock_id",
            "clock_reset_counter",
            "evaluation_partition",
            "outer_group_id",
        ):
            if group[column].astype(str).nunique(dropna=False) != 1:
                issue(
                    issues,
                    code="INCONSISTENT_SCAN_METADATA",
                    gate="causal_order",
                    message=f"scan_id must map to one {column}",
                    rows=group.index,
                )
        scan_sequence = pd.to_numeric(group["scan_sequence"], errors="coerce")
        hardware_sequence = pd.to_numeric(
            group["hardware_sequence"], errors="coerce"
        )
        if scan_sequence.isna().any() or hardware_sequence.isna().any():
            continue
        ordered = group.assign(
            _scan_sequence=scan_sequence,
            _hardware_sequence=hardware_sequence,
            _timestamp=group["acquisition_timestamp_utc"].map(parse_utc_datetime),
        ).sort_values("_scan_sequence")
        expected = np.arange(len(ordered), dtype=np.int64)
        actual = ordered["_scan_sequence"].to_numpy(dtype=np.int64)
        if not np.array_equal(actual, expected):
            issue(
                issues,
                code="NONCONTIGUOUS_SCAN_SEQUENCE",
                gate="causal_order",
                message=(
                    "scan_sequence must start at zero and be contiguous within "
                    f"scan_id; failed scan has {len(group)} rows"
                ),
                rows=ordered.index,
            )
        hardware = ordered["_hardware_sequence"].to_numpy(dtype=np.int64)
        if len(hardware) > 1 and not (np.diff(hardware) > 0).all():
            issue(
                issues,
                code="NONMONOTONIC_HARDWARE_SEQUENCE",
                gate="causal_order",
                message="hardware_sequence must be unique and strictly increasing",
                rows=ordered.index,
            )
        timestamps = ordered["_timestamp"]
        if timestamps.isna().any():
            continue
        timestamp_values = timestamps.tolist()
        time_decreases = any(
            later < earlier
            for earlier, later in zip(
                timestamp_values[:-1],
                timestamp_values[1:],
                strict=True,
            )
        )
        if time_decreases:
            issue(
                issues,
                code="NONMONOTONIC_ACQUISITION_TIME",
                gate="causal_order",
                message="UTC acquisition timestamps must be nondecreasing",
                rows=ordered.index,
            )


def validate_channels(
    frame: pd.DataFrame,
    issues: list[dict[str, Any]],
) -> None:
    h_valid = parse_boolean_series(frame["h_channel_valid"])
    v_valid = parse_boolean_series(frame["v_channel_valid"])
    bad = h_valid.ne(True) | v_valid.ne(True)
    if bad.any():
        issue(
            issues,
            code="INVALID_POLARIZATION_CHANNEL",
            gate="channel_integrity",
            message="Causal profile requires valid H and V channels",
            rows=frame.index[bad],
        )


def validate_locked_evaluation(
    frame: pd.DataFrame,
    issues: list[dict[str, Any]],
) -> None:
    partitions = frame["evaluation_partition"].astype(str)
    locked = partitions.eq("locked_test")
    if not locked.any():
        issue(
            issues,
            code="MISSING_LOCKED_TEST",
            gate="partition_isolation",
            message="locked_evaluation profile requires locked_test rows",
        )
    for column in ("session_id", "outer_group_id"):
        partition_counts = frame.groupby(column)["evaluation_partition"].nunique()
        leaked_values = partition_counts.loc[partition_counts.gt(1)].index
        if len(leaked_values):
            leaked = frame[column].isin(leaked_values)
            issue(
                issues,
                code="PARTITION_GROUP_LEAKAGE",
                gate="partition_isolation",
                message=f"{column} must map to exactly one evaluation partition",
                rows=frame.index[leaked],
            )

    labels = pd.to_numeric(frame["target_present"], errors="coerce")
    if locked.any() and set(labels.loc[locked].dropna().astype(int)) != {0, 1}:
        issue(
            issues,
            code="LOCKED_TEST_MISSING_CLASS",
            gate="same_condition_class_control",
            message="locked_test must contain target and background rows",
            rows=frame.index[locked],
        )

    condition_columns = [
        "collection_date",
        "site_id",
        "radar_config_id",
        "calibration_id",
    ]
    for _, group in frame.groupby(condition_columns, dropna=False, sort=False):
        if set(pd.to_numeric(group["target_present"], errors="coerce").dropna()) != {
            0,
            1,
        }:
            issue(
                issues,
                code="CONDITION_CLASS_CONFOUNDING",
                gate="same_condition_class_control",
                message=(
                    "Each date-site-radar-calibration condition must contain "
                    "target and background rows"
                ),
                rows=group.index,
            )

    target = labels.eq(1)
    missing_snr = target & ~nonempty_mask(frame["snr_db"])
    if missing_snr.any():
        issue(
            issues,
            code="MISSING_TARGET_SNR",
            gate="same_condition_class_control",
            message="locked_evaluation requires SNR for every target row",
            rows=frame.index[missing_snr],
        )


def gate_statuses(
    profile: str,
    issues: list[dict[str, Any]],
) -> dict[str, str]:
    evaluated = set(PROFILE_GATES[profile])
    failed = {item["gate"] for item in issues}
    schema_blocked = "schema" in failed
    return {
        gate: (
            "NOT_EVALUATED"
            if gate not in evaluated
            else "FAIL"
            if gate in failed
            else "BLOCKED"
            if schema_blocked
            else "PASS"
        )
        for gate in ALL_GATES
    }


def validate_collection_frame(
    frame: pd.DataFrame,
    contract: dict[str, Any],
    profile: str,
    *,
    data_root: Path | None = None,
    check_files: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if profile not in PROFILE_ORDER:
        raise ValueError(f"Unknown validation profile: {profile}")
    issues: list[dict[str, Any]] = []
    coverage = build_column_coverage(frame, contract)
    expected_columns = [str(item["name"]) for item in contract["columns"]]
    missing_columns = [name for name in expected_columns if name not in frame.columns]
    unexpected_columns = sorted(set(frame.columns).difference(expected_columns))
    if missing_columns:
        issue(
            issues,
            code="MISSING_COLUMNS",
            gate="schema",
            message=f"Manifest is missing columns: {missing_columns}",
        )
    if unexpected_columns:
        issue(
            issues,
            code="UNEXPECTED_COLUMNS",
            gate="schema",
            message=f"Manifest contains unexpected columns: {unexpected_columns}",
        )
    if frame.empty:
        issue(
            issues,
            code="EMPTY_MANIFEST",
            gate="schema",
            message="Manifest contains no data rows",
        )

    if not missing_columns and not frame.empty:
        working = frame.loc[:, expected_columns].copy()
        for specification in contract["columns"]:
            validate_column(working, specification, issues)
        schema_failed = any(item["gate"] == "schema" for item in issues)
        if not schema_failed:
            validate_row_integrity(
                working,
                issues,
                data_root=data_root,
                check_files=check_files,
            )
            validate_event_timing(working, issues)
            if profile in {"causal", "locked_evaluation"}:
                validate_causal_order(working, issues)
                validate_channels(working, issues)
            if profile == "locked_evaluation":
                validate_locked_evaluation(working, issues)

    gates = gate_statuses(profile, issues)
    evaluated_statuses = [gates[gate] for gate in PROFILE_GATES[profile]]
    status = "PASS" if evaluated_statuses and set(evaluated_statuses) == {"PASS"} else "FAIL"
    if "target_present" in frame.columns:
        labels = pd.to_numeric(frame["target_present"], errors="coerce")
        target_rows = int(labels.eq(1).sum())
        background_rows = int(labels.eq(0).sum())
    else:
        target_rows = 0
        background_rows = 0
    report = {
        "schema_version": 1,
        "contract_name": contract["contract_name"],
        "contract_schema_version": contract["schema_version"],
        "profile": profile,
        "status": status,
        "formal_causal_training_gate_open": bool(
            profile in {"causal", "locked_evaluation"} and status == "PASS"
        ),
        "locked_evaluation_gate_open": bool(
            profile == "locked_evaluation" and status == "PASS"
        ),
        "row_count": len(frame),
        "target_rows": target_rows,
        "background_rows": background_rows,
        "missing_columns": missing_columns,
        "unexpected_columns": unexpected_columns,
        "gates": gates,
        "issue_count": len(issues),
        "issues": issues,
        "check_files": bool(check_files),
    }
    return report, coverage


def render_report(report: dict[str, Any]) -> str:
    lines = [
        "# Data collection manifest precheck",
        "",
        "## Decision",
        "",
        f"**Profile `{report['profile']}`: {report['status']}.**",
        "",
        f"Rows: {report['row_count']}; targets: {report['target_rows']}; "
        f"backgrounds: {report['background_rows']}.",
        "",
        f"Formal causal-training gate open: "
        f"`{str(report['formal_causal_training_gate_open']).lower()}`.",
        f"Locked-evaluation gate open: "
        f"`{str(report['locked_evaluation_gate_open']).lower()}`.",
        "",
        "## Gates",
        "",
        "| Gate | Status |",
        "|---|---|",
    ]
    lines.extend(
        f"| {gate} | {status} |" for gate, status in report["gates"].items()
    )
    lines.extend(["", "## Issues", ""])
    if report["issues"]:
        for item in report["issues"]:
            rows = (
                f" Rows: {item['row_examples']}." if item["row_examples"] else ""
            )
            lines.append(
                f"- `{item['code']}` [{item['gate']}]: {item['message']}.{rows}"
            )
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A passing capture profile establishes record completeness only. A passing "
            "causal profile establishes that the manifest can support verified ordered "
            "context construction. Only a passing locked_evaluation profile supports "
            "opening the external locked-evaluation gate, after model and metric choices "
            "are frozen.",
            "",
        ]
    )
    return "\n".join(lines)


def ensure_output_available(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"Output path is not a directory: {output_dir}")
    if output_dir.is_dir() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is nonempty: {output_dir}. Use --overwrite to replace it."
        )


def write_precheck_output(
    output_dir: Path,
    report: dict[str, Any],
    coverage: pd.DataFrame,
    *,
    manifest_path: Path,
    contract_path: Path,
    overwrite: bool,
) -> None:
    ensure_output_available(output_dir, overwrite)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_parent = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.build-", dir=output_dir.parent)
    )
    staging_dir = staging_parent / output_dir.name
    staging_dir.mkdir()
    try:
        payload = dict(report)
        payload["input"] = {
            "name": manifest_path.name,
            "size_bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
        }
        payload["contract"] = {
            "path": display_path(contract_path),
            "size_bytes": contract_path.stat().st_size,
            "sha256": sha256_file(contract_path),
        }
        payload["implementation"] = {
            "path": display_path(Path(__file__)),
            "size_bytes": Path(__file__).stat().st_size,
            "sha256": sha256_file(Path(__file__)),
        }
        (staging_dir / "preflight.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        coverage.to_csv(staging_dir / "column_coverage.csv", index=False)
        (staging_dir / "PRECHECK_REPORT.md").write_text(
            render_report(payload),
            encoding="utf-8",
        )
        checksum_paths = sorted(path for path in staging_dir.iterdir() if path.is_file())
        (staging_dir / "SHA256SUMS.txt").write_text(
            "\n".join(
                f"{sha256_file(path)}  {path.name}" for path in checksum_paths
            )
            + "\n",
            encoding="utf-8",
        )
        if output_dir.exists():
            shutil.rmtree(output_dir)
        staging_dir.replace(output_dir)
    except Exception:
        shutil.rmtree(staging_parent, ignore_errors=True)
        raise
    shutil.rmtree(staging_parent, ignore_errors=True)


def main() -> int:
    args = parse_args()
    manifest_path = resolve_path(args.manifest)
    contract_path = resolve_path(args.contract)
    output_dir = resolve_path(args.output_dir)
    data_root = resolve_path(args.data_root) if args.data_root is not None else None
    try:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing manifest: {display_path(manifest_path)}")
        if not contract_path.is_file():
            raise FileNotFoundError(f"Missing contract: {display_path(contract_path)}")
        contract = load_contract(contract_path)
        frame = pd.read_csv(manifest_path, encoding="utf-8-sig")
        report, coverage = validate_collection_frame(
            frame,
            contract,
            args.profile,
            data_root=data_root,
            check_files=args.check_files,
        )
        write_precheck_output(
            output_dir,
            report,
            coverage,
            manifest_path=manifest_path,
            contract_path=contract_path,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError, OSError) as exc:
        print(f"[ERROR] {exc}")
        return 2
    print(f"Data collection manifest precheck: {report['status']}")
    print(f"profile={args.profile}")
    print(f"rows={report['row_count']}")
    print(f"formal_causal_training_gate_open={report['formal_causal_training_gate_open']}")
    print(f"locked_evaluation_gate_open={report['locked_evaluation_gate_open']}")
    print(f"output_dir={display_path(output_dir)}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
