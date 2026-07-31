from __future__ import annotations

import json
from pathlib import Path

import pytest

from utils.experiment_ledger import (
    empty_record,
    read_ledger,
    record_from_summary,
    snapshot_summary,
    upsert_record,
    validate_ledger,
)


def complete_record(experiment_id: str) -> dict[str, str]:
    record = empty_record(experiment_id)
    record.update(
        {
            "run_status": "LOST",
            "decision_status": "LOST",
            "test_policy": "unknown",
            "test_split_loaded": "unknown",
        }
    )
    return record


def test_ledger_rejects_duplicate_without_overwrite(tmp_path: Path) -> None:
    ledger = tmp_path / "experiments.csv"
    upsert_record(complete_record("lost_history"), ledger)
    with pytest.raises(FileExistsError):
        upsert_record(complete_record("lost_history"), ledger)
    assert read_ledger(ledger)[0]["experiment_id"] == "lost_history"


def test_summary_import_preserves_test_provenance_and_metric(tmp_path: Path) -> None:
    summary = tmp_path / "experiment" / "tables" / "summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(
        json.dumps(
            {
                "status": "PASS",
                "experiment_name": "fold01_diagnostic",
                "evidence_role": "validation_only",
                "manifest_path": "data/metadata/fold01.csv",
                "scope": "diagnostic",
                "test_split_loaded": False,
                "seed": 42,
                "fold_id": 1,
                "channel": "H",
                "validation": {"joint_pd": 0.25},
            }
        ),
        encoding="utf-8",
    )
    record = record_from_summary(
        summary,
        purpose="test import",
        decision_status="DIAGNOSTIC_ONLY",
        test_policy="forbidden",
        primary_metric_path="validation.joint_pd",
    )
    assert record["test_split_loaded"] == "false"
    assert record["primary_metric_value"] == "0.25"
    assert validate_ledger([record]) == []


def test_ledger_flags_forbidden_test_access() -> None:
    record = complete_record("policy_violation")
    record.update(
        {
            "run_status": "COMPLETED",
            "decision_status": "REJECT",
            "test_policy": "forbidden",
            "test_split_loaded": "true",
        }
    )
    assert "forbidden test split" in validate_ledger([record])[0]


def test_summary_snapshot_survives_source_cleanup(tmp_path: Path) -> None:
    source = tmp_path / "ignored" / "summary.json"
    source.parent.mkdir()
    source.write_text('{"status": "PASS"}', encoding="utf-8")
    ledger = tmp_path / "tracked" / "experiments.csv"
    snapshot = snapshot_summary(source, "experiment_01", ledger)
    source.unlink()
    assert json.loads(snapshot.read_text(encoding="utf-8"))["status"] == "PASS"
