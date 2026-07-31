#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.experiment_ledger import (
    DEFAULT_LEDGER,
    empty_record,
    read_ledger,
    record_from_summary,
    relative_path,
    snapshot_summary,
    upsert_record,
    utc_now,
    validate_ledger,
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Manage the project experiment ledger")
    root.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    subparsers = root.add_subparsers(dest="action", required=True)

    lost = subparsers.add_parser("add-lost")
    lost.add_argument("--experiment-id", required=True)
    lost.add_argument("--purpose", required=True)
    lost.add_argument("--notes", required=True)

    imported = subparsers.add_parser("import-summary")
    imported.add_argument("summary", type=Path)
    imported.add_argument("--experiment-id")
    imported.add_argument("--purpose", required=True)
    imported.add_argument(
        "--decision-status",
        choices=("PENDING_REVIEW", "RETAIN", "REJECT", "DIAGNOSTIC_ONLY"),
        required=True,
    )
    imported.add_argument(
        "--test-policy",
        choices=("forbidden", "deferred", "allowed", "unknown"),
        required=True,
    )
    imported.add_argument("--primary-metric-path")
    imported.add_argument("--config-path")
    imported.add_argument("--notes", default="")
    imported.add_argument("--overwrite", action="store_true")

    update = subparsers.add_parser("update")
    update.add_argument("--experiment-id", required=True)
    update.add_argument(
        "--decision-status",
        choices=("PENDING_REVIEW", "RETAIN", "REJECT", "DIAGNOSTIC_ONLY"),
        required=True,
    )
    update.add_argument("--notes")

    aborted = subparsers.add_parser("mark-aborted")
    aborted.add_argument("--experiment-id", required=True)
    aborted.add_argument("--notes", required=True)

    subparsers.add_parser("verify")
    return root


def main() -> int:
    args = parser().parse_args()
    ledger = args.ledger.expanduser().resolve()
    if args.action == "add-lost":
        record = empty_record(args.experiment_id)
        record.update(
            {
                "started_at_utc": "unknown",
                "completed_at_utc": "unknown",
                "run_status": "LOST",
                "decision_status": "LOST",
                "purpose": args.purpose,
                "evidence_role": "historical_untracked_exploration",
                "split_scope": "unknown",
                "test_policy": "unknown",
                "test_split_loaded": "unknown",
                "git_commit": "unknown",
                "git_dirty_at_start": "unknown",
                "exit_code": "unknown",
                "notes": args.notes,
            }
        )
        upsert_record(record, ledger)
    elif args.action == "import-summary":
        summary = args.summary.expanduser().resolve()
        record = record_from_summary(
            summary,
            experiment_id=args.experiment_id,
            purpose=args.purpose,
            decision_status=args.decision_status,
            test_policy=args.test_policy,
            primary_metric_path=args.primary_metric_path,
            notes=args.notes,
            config_path=args.config_path,
        )
        snapshot = snapshot_summary(
            summary,
            record["experiment_id"],
            ledger,
            overwrite=args.overwrite,
        )
        record["summary_path"] = relative_path(snapshot)
        upsert_record(record, ledger, overwrite=args.overwrite)
    elif args.action == "update":
        rows = read_ledger(ledger)
        matches = [row for row in rows if row["experiment_id"] == args.experiment_id]
        if not matches:
            raise KeyError(f"experiment_id not found: {args.experiment_id}")
        record = matches[0]
        record["decision_status"] = args.decision_status
        if args.notes is not None:
            record["notes"] = args.notes
        upsert_record(record, ledger, overwrite=True)
    elif args.action == "mark-aborted":
        rows = read_ledger(ledger)
        matches = [row for row in rows if row["experiment_id"] == args.experiment_id]
        if not matches:
            raise KeyError(f"experiment_id not found: {args.experiment_id}")
        record = matches[0]
        if record["run_status"] != "RUNNING":
            raise ValueError("only a RUNNING experiment can be marked ABORTED")
        record["run_status"] = "ABORTED"
        record["completed_at_utc"] = utc_now()
        record["exit_code"] = "130"
        record["notes"] = args.notes
        upsert_record(record, ledger, overwrite=True)
    else:
        rows = read_ledger(ledger)
        issues = validate_ledger(rows)
        print(f"records: {len(rows)}")
        for issue in issues:
            print(f"[FAIL] {issue}")
        if issues:
            return 1
        print("[PASS] experiment ledger is valid")
        return 0
    print(f"ledger: {ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
