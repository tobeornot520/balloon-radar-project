#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.experiment_ledger import (
    DEFAULT_LEDGER,
    empty_record,
    git_state,
    read_ledger,
    relative_path,
    snapshot_summary,
    upsert_record,
    utc_now,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one command and persist its experiment-level provenance"
    )
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--purpose", required=True)
    parser.add_argument("--evidence-role", required=True)
    parser.add_argument("--data-manifest")
    parser.add_argument("--split-scope", required=True)
    parser.add_argument(
        "--test-policy",
        choices=("forbidden", "deferred", "allowed", "unknown"),
        required=True,
    )
    parser.add_argument("--config-path")
    parser.add_argument("--summary-path")
    parser.add_argument("--artifact-dir")
    parser.add_argument("--parent-experiment-id", default="")
    parser.add_argument("--seed", default="")
    parser.add_argument("--fold", default="")
    parser.add_argument("--channel", default="")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def load_summary(path: Path | None) -> dict[str, object]:
    if path is None or not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("a command is required after --")
    ledger = args.ledger.expanduser().resolve()
    if any(row["experiment_id"] == args.experiment_id for row in read_ledger(ledger)):
        raise FileExistsError(f"experiment_id already exists: {args.experiment_id}")

    commit, dirty = git_state()
    record = empty_record(args.experiment_id)
    record.update(
        {
            "parent_experiment_id": args.parent_experiment_id,
            "started_at_utc": utc_now(),
            "run_status": "RUNNING",
            "decision_status": "PENDING_REVIEW",
            "purpose": args.purpose,
            "evidence_role": args.evidence_role,
            "data_manifest": relative_path(args.data_manifest),
            "split_scope": args.split_scope,
            "test_policy": args.test_policy,
            "test_split_loaded": "unknown",
            "config_path": relative_path(args.config_path),
            "summary_path": relative_path(args.summary_path),
            "artifact_dir": relative_path(args.artifact_dir),
            "git_commit": commit,
            "git_dirty_at_start": dirty,
            "seed": args.seed,
            "fold": args.fold,
            "channel": args.channel,
            "command_json": json.dumps(command, ensure_ascii=False),
        }
    )
    upsert_record(record, ledger)

    exit_code = 1
    try:
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        exit_code = int(completed.returncode)
        record["run_status"] = "COMPLETED" if exit_code == 0 else "FAILED"
    except KeyboardInterrupt:
        record["run_status"] = "ABORTED"
        exit_code = 130
    except OSError as exc:
        record["run_status"] = "FAILED"
        record["notes"] = str(exc)
        exit_code = 127

    summary_path = (
        None
        if args.summary_path is None
        else (PROJECT_ROOT / args.summary_path).resolve()
    )
    summary = load_summary(summary_path)
    loaded = summary.get("test_split_loaded", "unknown")
    if loaded in (True, False):
        record["test_split_loaded"] = str(loaded).lower()
    if args.test_policy == "forbidden" and loaded is True:
        record["run_status"] = "POLICY_VIOLATION"
        record["notes"] = "test split was loaded despite forbidden policy"
        exit_code = 3
    if summary_path is not None and summary:
        snapshot = snapshot_summary(summary_path, args.experiment_id, ledger)
        record["summary_path"] = relative_path(snapshot)
    record["completed_at_utc"] = utc_now()
    record["exit_code"] = str(exit_code)
    upsert_record(record, ledger, overwrite=True)
    print(f"ledger: {ledger}")
    print(f"experiment: {args.experiment_id} -> {record['run_status']}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
