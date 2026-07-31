from __future__ import annotations

import csv
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = PROJECT_ROOT / "results" / "experiment_ledger" / "experiments.csv"

LEDGER_COLUMNS = (
    "experiment_id",
    "parent_experiment_id",
    "started_at_utc",
    "completed_at_utc",
    "run_status",
    "decision_status",
    "purpose",
    "evidence_role",
    "data_manifest",
    "split_scope",
    "test_policy",
    "test_split_loaded",
    "config_path",
    "summary_path",
    "artifact_dir",
    "git_commit",
    "git_dirty_at_start",
    "seed",
    "fold",
    "channel",
    "primary_metric_name",
    "primary_metric_value",
    "exit_code",
    "command_json",
    "notes",
)

RUN_STATUSES = {
    "RUNNING",
    "COMPLETED",
    "FAILED",
    "ABORTED",
    "POLICY_VIOLATION",
    "LOST",
}
DECISION_STATUSES = {
    "PENDING_REVIEW",
    "RETAIN",
    "REJECT",
    "DIAGNOSTIC_ONLY",
    "LOST",
}
TEST_POLICIES = {"forbidden", "deferred", "allowed", "unknown"}
TRISTATE = {"true", "false", "unknown"}
EXPERIMENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def relative_path(value: str | Path | None) -> str:
    if value is None or str(value).strip() == "":
        return ""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def git_state() -> tuple[str, str]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown", "unknown"
    return commit or "unknown", "true" if status.strip() else "false"


def empty_record(experiment_id: str) -> dict[str, str]:
    return {column: experiment_id if column == "experiment_id" else "" for column in LEDGER_COLUMNS}


def normalize_record(record: dict[str, Any]) -> dict[str, str]:
    unknown = set(record) - set(LEDGER_COLUMNS)
    if unknown:
        raise ValueError(f"unknown ledger columns: {sorted(unknown)}")
    normalized = empty_record(str(record.get("experiment_id", "")).strip())
    for key, value in record.items():
        if value is None:
            normalized[key] = ""
        elif isinstance(value, bool):
            normalized[key] = str(value).lower()
        elif isinstance(value, float) and not math.isfinite(value):
            normalized[key] = ""
        else:
            normalized[key] = str(value)
    if not normalized["experiment_id"]:
        raise ValueError("experiment_id is required")
    if not EXPERIMENT_ID_PATTERN.fullmatch(normalized["experiment_id"]):
        raise ValueError(
            "experiment_id must contain only letters, digits, dot, underscore, or dash"
        )
    if normalized["run_status"] not in RUN_STATUSES:
        raise ValueError(f"invalid run_status: {normalized['run_status']}")
    if normalized["decision_status"] not in DECISION_STATUSES:
        raise ValueError(f"invalid decision_status: {normalized['decision_status']}")
    if normalized["test_policy"] not in TEST_POLICIES:
        raise ValueError(f"invalid test_policy: {normalized['test_policy']}")
    if normalized["test_split_loaded"] not in TRISTATE:
        raise ValueError(
            f"invalid test_split_loaded: {normalized['test_split_loaded']}"
        )
    return normalized


def read_ledger(path: Path = DEFAULT_LEDGER) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != LEDGER_COLUMNS:
            raise ValueError("experiment ledger schema does not match LEDGER_COLUMNS")
        rows = [dict(row) for row in reader]
    identifiers = [row["experiment_id"] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("experiment ledger contains duplicate experiment_id values")
    return rows


def write_ledger(rows: Iterable[dict[str, Any]], path: Path = DEFAULT_LEDGER) -> None:
    normalized = [normalize_record(row) for row in rows]
    identifiers = [row["experiment_id"] for row in normalized]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("experiment ledger contains duplicate experiment_id values")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_COLUMNS)
        writer.writeheader()
        writer.writerows(normalized)
    temporary.replace(path)


def upsert_record(
    record: dict[str, Any],
    path: Path = DEFAULT_LEDGER,
    *,
    overwrite: bool = False,
) -> None:
    normalized = normalize_record(record)
    rows = read_ledger(path)
    matches = [
        index
        for index, row in enumerate(rows)
        if row["experiment_id"] == normalized["experiment_id"]
    ]
    if matches and not overwrite:
        raise FileExistsError(
            f"experiment_id already exists: {normalized['experiment_id']}"
        )
    if matches:
        rows[matches[0]] = normalized
    else:
        rows.append(normalized)
    write_ledger(rows, path)


def dotted_value(payload: dict[str, Any], dotted_path: str) -> Any:
    value: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"summary field not found: {dotted_path}")
        value = value[part]
    return value


def snapshot_summary(
    source: Path,
    experiment_id: str,
    ledger_path: Path = DEFAULT_LEDGER,
    *,
    overwrite: bool = False,
) -> Path:
    if not EXPERIMENT_ID_PATTERN.fullmatch(experiment_id):
        raise ValueError("invalid experiment_id for summary snapshot")
    payload = json.loads(source.read_text(encoding="utf-8"))
    destination = ledger_path.parent / "summaries" / f"{experiment_id}.json"
    if destination.exists() and not overwrite:
        raise FileExistsError(f"summary snapshot already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def record_from_summary(
    summary_path: Path,
    *,
    experiment_id: str | None = None,
    purpose: str,
    decision_status: str,
    test_policy: str,
    primary_metric_path: str | None = None,
    notes: str = "",
    config_path: str | Path | None = None,
) -> dict[str, str]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    identifier = experiment_id or payload.get("experiment_name")
    if not identifier:
        raise ValueError("experiment_id is absent from arguments and summary")
    loaded = payload.get("test_split_loaded", "unknown")
    if loaded not in (True, False, "unknown"):
        loaded = "unknown"
    metric_value: Any = ""
    if primary_metric_path:
        metric_value = dotted_value(payload, primary_metric_path)
    completed = datetime.fromtimestamp(
        summary_path.stat().st_mtime, timezone.utc
    ).replace(microsecond=0).isoformat()
    return normalize_record(
        {
            "experiment_id": identifier,
            "parent_experiment_id": "",
            "started_at_utc": "unknown",
            "completed_at_utc": completed,
            "run_status": "COMPLETED" if payload.get("status") == "PASS" else "FAILED",
            "decision_status": decision_status,
            "purpose": purpose,
            "evidence_role": payload.get("evidence_role", "unknown"),
            "data_manifest": relative_path(payload.get("manifest_path")),
            "split_scope": payload.get("scope", "unknown"),
            "test_policy": test_policy,
            "test_split_loaded": loaded,
            "config_path": relative_path(config_path),
            "summary_path": relative_path(summary_path),
            "artifact_dir": relative_path(summary_path.parent.parent),
            "git_commit": "unknown",
            "git_dirty_at_start": "unknown",
            "seed": payload.get("seed", ""),
            "fold": payload.get("fold_id", ""),
            "channel": payload.get("channel", ""),
            "primary_metric_name": primary_metric_path or "",
            "primary_metric_value": metric_value,
            "exit_code": 0 if payload.get("status") == "PASS" else "unknown",
            "command_json": "",
            "notes": notes,
        }
    )


def validate_ledger(rows: Iterable[dict[str, str]]) -> list[str]:
    issues: list[str] = []
    for index, raw in enumerate(rows, start=2):
        try:
            row = normalize_record(raw)
        except ValueError as exc:
            issues.append(f"row {index}: {exc}")
            continue
        if row["test_policy"] == "forbidden" and row["test_split_loaded"] == "true":
            issues.append(
                f"row {index}: forbidden test split was loaded for {row['experiment_id']}"
            )
        if row["run_status"] == "RUNNING" and row["completed_at_utc"]:
            issues.append(f"row {index}: RUNNING record has completed_at_utc")
        if row["run_status"] == "LOST" and row["decision_status"] != "LOST":
            issues.append(f"row {index}: LOST run must have LOST decision")
    return issues
