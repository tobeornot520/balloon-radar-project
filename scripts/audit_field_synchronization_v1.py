#!/usr/bin/env python3
"""Audit paired radar/video/truth synchronization events for field readiness."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = PROJECT_ROOT / "configs/field_sync_event_contract_v1.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/data_audit/field_synchronization_v1"
PAIR_DEFINITIONS = (
    ("radar_video", "radar_timestamp_utc", "video_timestamp_utc"),
    ("radar_truth", "radar_timestamp_utc", "truth_timestamp_utc"),
    ("video_truth", "video_timestamp_utc", "truth_timestamp_utc"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit radar/video/truth synchronization event timestamps."
    )
    parser.add_argument("events", type=Path)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    path = resolve_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"missing synchronization contract: {path}")
    contract = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "contract_id",
        "minimum_accepted_events",
        "minimum_accepted_events_per_session",
        "maximum_event_uncertainty_ms",
        "radar_video_limits_ms",
        "allowed_event_methods",
        "required_columns",
        "claim_boundaries",
    }
    missing = required - set(contract)
    if missing:
        raise ValueError(f"synchronization contract missing keys: {sorted(missing)}")
    if contract["schema_version"] != 1:
        raise ValueError("only synchronization schema_version=1 is supported")
    if int(contract["minimum_accepted_events"]) < 5:
        raise ValueError("minimum_accepted_events must be at least five")
    if int(contract["minimum_accepted_events_per_session"]) < 1:
        raise ValueError("minimum accepted events per session must be positive")
    if float(contract["maximum_event_uncertainty_ms"]) < 0:
        raise ValueError("maximum event uncertainty must be nonnegative")
    limits = contract["radar_video_limits_ms"]
    if set(limits) != {"absolute_error_p95_maximum", "absolute_error_maximum"}:
        raise ValueError("radar-video limits do not match the V1 schema")
    if float(limits["absolute_error_p95_maximum"]) <= 0 or float(
        limits["absolute_error_maximum"]
    ) <= 0:
        raise ValueError("synchronization error limits must be positive")
    columns = contract["required_columns"]
    if not isinstance(columns, list) or len(columns) != len(set(columns)):
        raise ValueError("required_columns must be a unique list")
    if not contract["allowed_event_methods"]:
        raise ValueError("allowed_event_methods must be nonempty")
    expected_claims = {
        "all_radar_frames_timestamped_established",
        "hardware_sequence_integrity_established",
        "clock_mapping_provenance_established",
        "formal_synchronization_gate_opened_by_numeric_audit_alone",
        "model_training_allowed_by_sync_audit",
    }
    if set(contract["claim_boundaries"]) != expected_claims:
        raise ValueError("synchronization claim boundaries do not match V1")
    if any(contract["claim_boundaries"].values()):
        raise ValueError("all V1 synchronization claim boundaries must remain false")
    return contract


def parse_utc(value: Any) -> datetime | None:
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


def parse_boolean(series: pd.Series) -> pd.Series:
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    return normalized.map({"true": True, "1": True, "false": False, "0": False})


def issue(
    issues: list[dict[str, Any]],
    code: str,
    message: str,
    rows: pd.Index | list[int] | None = None,
) -> None:
    row_examples = [] if rows is None else [int(value) + 2 for value in list(rows)[:10]]
    issues.append(
        {
            "severity": "ERROR",
            "code": code,
            "message": message,
            "row_examples": row_examples,
        }
    )


def validate_events(
    frame: pd.DataFrame, contract: dict[str, Any]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    required = list(contract["required_columns"])
    if list(frame.columns) != required:
        raise ValueError(f"event columns must exactly match the contract: {required}")
    if frame.empty:
        raise ValueError("synchronization event table must not be empty")
    working = frame.fillna("").copy()
    issues: list[dict[str, Any]] = []
    for column in (
        "session_id",
        "event_id",
        "radar_clock_id",
        "video_clock_id",
        "truth_clock_id",
        "timestamp_mapping_id",
        "event_method",
    ):
        empty = working[column].astype(str).str.strip().eq("")
        if empty.any():
            issue(issues, "MISSING_REQUIRED_VALUE", f"{column} contains empty values", working.index[empty])
        working[column] = working[column].astype(str).str.strip()

    duplicates = working["event_id"].duplicated(keep=False)
    if duplicates.any():
        issue(issues, "DUPLICATE_EVENT_ID", "event_id must be globally unique", working.index[duplicates])
    methods = set(working["event_method"]) - set(contract["allowed_event_methods"])
    if methods:
        bad = ~working["event_method"].isin(contract["allowed_event_methods"])
        issue(issues, "INVALID_EVENT_METHOD", f"unsupported event methods: {sorted(methods)}", working.index[bad])

    working["_accepted"] = parse_boolean(working["accepted"])
    invalid_boolean = working["_accepted"].isna()
    if invalid_boolean.any():
        issue(issues, "INVALID_ACCEPTED_FLAG", "accepted must be true/false or 1/0", working.index[invalid_boolean])
    rejected_without_reason = working["_accepted"].eq(False) & working[
        "rejection_reason"
    ].astype(str).str.strip().eq("")
    if rejected_without_reason.any():
        issue(issues, "MISSING_REJECTION_REASON", "rejected events require a reason", working.index[rejected_without_reason])

    event_index = pd.to_numeric(working["event_index"], errors="coerce")
    invalid_index = event_index.isna() | ~np.isfinite(event_index) | event_index.mod(1).ne(0) | event_index.lt(0)
    if invalid_index.any():
        issue(issues, "INVALID_EVENT_INDEX", "event_index must be a nonnegative integer", working.index[invalid_index])
    working["_event_index"] = event_index

    uncertainty = pd.to_numeric(working["event_uncertainty_ms"], errors="coerce")
    invalid_uncertainty = uncertainty.isna() | ~np.isfinite(uncertainty) | uncertainty.lt(0)
    if invalid_uncertainty.any():
        issue(issues, "INVALID_EVENT_UNCERTAINTY", "event uncertainty must be finite and nonnegative", working.index[invalid_uncertainty])
    too_uncertain = working["_accepted"].eq(True) & uncertainty.gt(
        float(contract["maximum_event_uncertainty_ms"])
    )
    if too_uncertain.any():
        issue(issues, "EVENT_UNCERTAINTY_ABOVE_MAXIMUM", "event uncertainty exceeds the frozen limit", working.index[too_uncertain])
    working["_event_uncertainty_ms"] = uncertainty

    timestamp_columns = (
        "radar_timestamp_utc",
        "video_timestamp_utc",
        "truth_timestamp_utc",
    )
    for column in timestamp_columns:
        parsed = working[column].map(parse_utc)
        invalid = parsed.isna()
        if invalid.any():
            issue(issues, "INVALID_UTC_TIMESTAMP", f"{column} must contain UTC timestamps", working.index[invalid])
        working[f"_{column}"] = parsed

    for session_id, group in working.groupby("session_id", sort=False):
        if not session_id or group["_event_index"].isna().any():
            continue
        ordered = group.sort_values("_event_index")
        actual = ordered["_event_index"].to_numpy(dtype=np.int64)
        expected = np.arange(len(ordered), dtype=np.int64)
        if not np.array_equal(actual, expected):
            issue(issues, "NONCONTIGUOUS_EVENT_INDEX", "event_index must start at zero and be contiguous per session", ordered.index)
        for column in timestamp_columns:
            values = ordered[f"_{column}"]
            if values.isna().any():
                continue
            seconds = np.array([value.timestamp() for value in values], dtype=np.float64)
            if len(seconds) > 1 and not np.all(np.diff(seconds) > 0):
                issue(issues, "NONMONOTONIC_EVENT_TIME", f"{column} must increase with event_index", ordered.index)
        for column in (
            "radar_clock_id",
            "video_clock_id",
            "truth_clock_id",
            "timestamp_mapping_id",
        ):
            if group[column].nunique(dropna=False) != 1:
                issue(issues, "INCONSISTENT_SESSION_CLOCK_MAPPING", f"session_id must map to one {column}", group.index)
    return working, issues


def add_pair_errors(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for pair_name, left, right in PAIR_DEFINITIONS:
        output[f"{pair_name}_signed_error_ms"] = [
            (left_value - right_value).total_seconds() * 1000.0
            if left_value is not None and right_value is not None
            else np.nan
            for left_value, right_value in zip(
                output[f"_{left}"], output[f"_{right}"], strict=True
            )
        ]
        output[f"{pair_name}_absolute_error_ms"] = output[
            f"{pair_name}_signed_error_ms"
        ].abs()
    return output


def metric_summary(accepted: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pair_name, _, _ in PAIR_DEFINITIONS:
        signed = accepted[f"{pair_name}_signed_error_ms"].astype(float)
        absolute = signed.abs()
        rows.append(
            {
                "pair": pair_name,
                "event_count": int(len(signed)),
                "signed_error_mean_ms": float(signed.mean()),
                "signed_error_std_ms": float(signed.std(ddof=0)),
                "absolute_error_median_ms": float(absolute.median()),
                "absolute_error_p95_ms": float(np.quantile(absolute, 0.95)),
                "absolute_error_max_ms": float(absolute.max()),
            }
        )
    return pd.DataFrame(rows)


def anonymous_session(value: str) -> str:
    token = ("field-sync-session-v1:" + value).encode("utf-8")
    return "session_" + hashlib.sha256(token).hexdigest()[:12]


def session_summary(accepted: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for session_id, group in accepted.groupby("session_id", sort=True):
        row: dict[str, Any] = {
            "session_alias": anonymous_session(str(session_id)),
            "accepted_event_count": int(len(group)),
            "event_span_s": float(
                (
                    group["_truth_timestamp_utc"].max()
                    - group["_truth_timestamp_utc"].min()
                ).total_seconds()
            ),
            "maximum_event_uncertainty_ms": float(
                group["_event_uncertainty_ms"].max()
            ),
        }
        for pair_name, _, _ in PAIR_DEFINITIONS:
            absolute = group[f"{pair_name}_absolute_error_ms"].astype(float)
            row[f"{pair_name}_p95_ms"] = float(np.quantile(absolute, 0.95))
            row[f"{pair_name}_max_ms"] = float(absolute.max())
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate(
    frame: pd.DataFrame,
    issues: list[dict[str, Any]],
    contract: dict[str, Any],
) -> tuple[
    dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    enriched = add_pair_errors(frame)
    accepted = enriched[enriched["_accepted"].eq(True)].copy()
    minimum = int(contract["minimum_accepted_events"])
    if len(accepted) < minimum:
        issue(issues, "INSUFFICIENT_ACCEPTED_EVENTS", f"at least {minimum} accepted events are required")
    per_session = accepted.groupby("session_id").size().reindex(
        sorted(set(enriched["session_id"])), fill_value=0
    )
    minimum_per_session = int(contract["minimum_accepted_events_per_session"])
    if len(per_session) and per_session.lt(minimum_per_session).any():
        issue(issues, "INSUFFICIENT_SESSION_EVENTS", "each represented session needs the frozen minimum event count")
    if accepted.empty:
        metrics = pd.DataFrame(
            columns=[
                "pair",
                "event_count",
                "signed_error_mean_ms",
                "signed_error_std_ms",
                "absolute_error_median_ms",
                "absolute_error_p95_ms",
                "absolute_error_max_ms",
            ]
        )
        sessions = pd.DataFrame()
        p95 = math.nan
        maximum = math.nan
    else:
        metrics = metric_summary(accepted)
        sessions = session_summary(accepted)
        primary = metrics[metrics["pair"].eq("radar_video")].iloc[0]
        p95 = float(primary["absolute_error_p95_ms"])
        maximum = float(primary["absolute_error_max_ms"])
    limits = contract["radar_video_limits_ms"]
    if math.isfinite(p95) and p95 > float(limits["absolute_error_p95_maximum"]):
        issue(issues, "RADAR_VIDEO_P95_ABOVE_MAXIMUM", "radar-video absolute error P95 exceeds the frozen limit")
    if math.isfinite(maximum) and maximum > float(limits["absolute_error_maximum"]):
        issue(issues, "RADAR_VIDEO_MAX_ABOVE_MAXIMUM", "radar-video maximum absolute error exceeds the frozen limit")

    status = "PASS_NUMERIC_LIMITS_ONLY" if not issues else "FAIL"
    measurements = pd.DataFrame(
        [
            {
                "item_id": "SYNC_EVENT_REPEATS",
                "measured_value": int(len(accepted)),
                "unit": "count",
                "recommended_status": "pass" if len(accepted) >= minimum else "fail",
            },
            {
                "item_id": "SYNC_P95_ERROR",
                "measured_value": p95,
                "unit": "ms",
                "recommended_status": (
                    "pass"
                    if math.isfinite(p95)
                    and p95 <= float(limits["absolute_error_p95_maximum"])
                    else "fail"
                ),
            },
            {
                "item_id": "SYNC_MAX_ERROR",
                "measured_value": maximum,
                "unit": "ms",
                "recommended_status": (
                    "pass"
                    if math.isfinite(maximum)
                    and maximum <= float(limits["absolute_error_maximum"])
                    else "fail"
                ),
            },
        ]
    )
    summary = {
        "schema_version": 1,
        "audit_id": "field_synchronization_numeric_audit_v1",
        "status": status,
        "event_count": int(len(enriched)),
        "accepted_event_count": int(len(accepted)),
        "rejected_event_count": int(enriched["_accepted"].eq(False).sum()),
        "session_count": int(accepted["session_id"].nunique()),
        "radar_video_absolute_error_p95_ms": p95,
        "radar_video_absolute_error_max_ms": maximum,
        "numeric_readiness_items_supported": [
            "SYNC_EVENT_REPEATS",
            "SYNC_P95_ERROR",
            "SYNC_MAX_ERROR",
        ],
        "non_numeric_readiness_items_still_required": [
            "SYNC_COMMON_TIMEBASE",
            "SYNC_RADAR_TIMESTAMP",
            "SYNC_VIDEO_TIMESTAMP",
        ],
        "formal_synchronization_gate_open": False,
        "issues": issues,
        "claim_boundaries": contract["claim_boundaries"],
    }
    return summary, enriched, metrics, sessions, measurements


def make_readme(summary: dict[str, Any]) -> str:
    return f"""# Field Synchronization Numeric Audit V1

Status: `{summary['status']}`. Accepted events:
{summary['accepted_event_count']} across {summary['session_count']} sessions.

The audit computes paired radar/video/truth timestamp errors and supports only
the three numeric readiness rows listed in `readiness_measurements.csv`. It does
not by itself pass the full synchronization gate. Per-event IDs and timestamps
remain in `event_audit_local.csv`; shareable tables are aggregate-only.
"""


def audit_synchronization(
    *, events_path: Path, contract_path: Path, output_dir: Path, overwrite: bool
) -> dict[str, Any]:
    events_path = resolve_path(events_path)
    contract_path = resolve_path(contract_path)
    output_dir = resolve_path(output_dir)
    if not events_path.is_file():
        raise FileNotFoundError(f"missing synchronization events: {events_path}")
    if output_dir == PROJECT_ROOT:
        raise ValueError("output directory must not be the project root")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is nonempty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    contract = load_contract(contract_path)
    events = pd.read_csv(events_path, encoding="utf-8-sig", keep_default_na=False)
    validated, issues = validate_events(events, contract)
    summary, enriched, metrics, sessions, measurements = evaluate(
        validated, issues, contract
    )
    summary["input"] = {
        "events_name": events_path.name,
        "events_sha256": sha256_file(events_path),
        "contract_name": contract_path.name,
        "contract_sha256": sha256_file(contract_path),
        "script_sha256": sha256_file(Path(__file__)),
        "absolute_paths_published": False,
    }
    local_columns = [
        "session_id",
        "event_id",
        "event_index",
        "radar_timestamp_utc",
        "video_timestamp_utc",
        "truth_timestamp_utc",
        "radar_clock_id",
        "video_clock_id",
        "truth_clock_id",
        "timestamp_mapping_id",
        "event_method",
        "event_uncertainty_ms",
        "accepted",
        "rejection_reason",
        "notes",
        *[
            f"{pair_name}_{suffix}_error_ms"
            for pair_name, _, _ in PAIR_DEFINITIONS
            for suffix in ("signed", "absolute")
        ],
    ]
    enriched[local_columns].to_csv(
        output_dir / "event_audit_local.csv", index=False, encoding="utf-8-sig"
    )
    metrics.to_csv(
        output_dir / "pair_metric_summary.csv", index=False, encoding="utf-8-sig"
    )
    sessions.to_csv(
        output_dir / "session_sync_summary.csv", index=False, encoding="utf-8-sig"
    )
    measurements.to_csv(
        output_dir / "readiness_measurements.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "README.md").write_text(make_readme(summary), encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    summary = audit_synchronization(
        events_path=args.events,
        contract_path=args.contract,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print("Field synchronization numeric audit: COMPLETE")
    print(f"status={summary['status']}")
    print(f"accepted_events={summary['accepted_event_count']}")
    print(f"radar_video_p95_ms={summary['radar_video_absolute_error_p95_ms']}")
    print(f"radar_video_max_ms={summary['radar_video_absolute_error_max_ms']}")
    return 0 if summary["status"] == "PASS_NUMERIC_LIMITS_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
