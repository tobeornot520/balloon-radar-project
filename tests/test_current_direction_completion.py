from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.check_current_direction_completion_v1 import (
    inspect_evidence,
    summarize_items,
    validate_config,
)


def item(
    identifier: str, status: str, evidence: str = "evidence.txt"
) -> dict[str, object]:
    return {
        "id": identifier,
        "ledger_ids": ["D01"],
        "title": identifier,
        "owner": "user",
        "status": status,
        "evidence": evidence,
        "next_action": "next",
    }


def test_completion_requires_every_item(tmp_path: Path) -> None:
    (tmp_path / "evidence.txt").write_text("evidence\n", encoding="utf-8")
    summary, table = summarize_items(
        inspect_evidence(
            [item("CD01", "complete"), item("CD02", "pending_user")],
            evidence_root=tmp_path,
        )
    )

    assert summary["status"] == "IN_PROGRESS"
    assert summary["milestone_complete"] is False
    assert len(table) == 2


def test_external_blocker_is_reported_without_false_completion(tmp_path: Path) -> None:
    (tmp_path / "evidence.txt").write_text("evidence\n", encoding="utf-8")
    summary, _ = summarize_items(
        inspect_evidence(
            [item("CD01", "complete"), item("CD02", "blocked_external")],
            evidence_root=tmp_path,
        )
    )

    assert summary["status"] == "BLOCKED_EXTERNAL"
    assert summary["blocked_external_count"] == 1
    assert summary["milestone_complete"] is False


def test_missing_complete_evidence_remains_visible_with_external_blocker(
    tmp_path: Path,
) -> None:
    (tmp_path / "evidence.txt").write_text("evidence\n", encoding="utf-8")
    summary, table = summarize_items(
        inspect_evidence(
            [
                item("CD01", "complete", "missing.txt"),
                item("CD02", "blocked_external"),
            ],
            evidence_root=tmp_path,
        )
    )

    assert summary["status"] == "IN_PROGRESS"
    assert summary["completed_count"] == 0
    assert summary["remaining_count"] == 2
    assert summary["configured_complete_but_missing_evidence_count"] == 1
    assert set(table.loc[~table["is_complete"], "id"]) == {"CD01", "CD02"}


def test_invalid_completion_status_is_rejected() -> None:
    payload = {"schema_version": 1, "items": [item("CD01", "done")]}

    with pytest.raises(ValueError, match="invalid completion status"):
        validate_config(payload)


def test_missing_evidence_prevents_configured_completion(tmp_path: Path) -> None:
    inspected = inspect_evidence(
        [item("CD01", "complete", "missing.txt")], evidence_root=tmp_path
    )
    summary, table = summarize_items(inspected)

    assert summary["status"] == "IN_PROGRESS"
    assert summary["completed_count"] == 0
    assert summary["missing_evidence_count"] == 1
    assert summary["configured_complete_but_missing_evidence_count"] == 1
    assert table.iloc[0]["evidence_complete"] == False  # noqa: E712
    assert table.iloc[0]["missing_evidence_paths"] == "missing.txt"


def test_existing_file_and_nonempty_directory_are_valid_evidence(tmp_path: Path) -> None:
    (tmp_path / "report.md").write_text("report\n", encoding="utf-8")
    package = tmp_path / "package"
    package.mkdir()
    (package / "MANIFEST.json").write_text(
        json.dumps({"source_commit": "abc"}), encoding="utf-8"
    )

    inspected = inspect_evidence(
        [item("CD01", "complete", "report.md; package")],
        evidence_root=tmp_path,
    )
    summary, table = summarize_items(inspected)

    assert summary["status"] == "COMPLETE"
    assert summary["missing_evidence_count"] == 0
    assert table.iloc[0]["evidence_path_count"] == 2
    assert table.iloc[0]["evidence_complete"] == True  # noqa: E712


def test_empty_evidence_file_or_directory_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "empty.txt").touch()
    (tmp_path / "empty-dir").mkdir()

    inspected = inspect_evidence(
        [item("CD01", "complete", "empty.txt; empty-dir")],
        evidence_root=tmp_path,
    )
    summary, table = summarize_items(inspected)

    assert summary["completed_count"] == 0
    assert summary["missing_evidence_count"] == 2
    assert table.iloc[0]["missing_evidence_paths"] == "empty.txt; empty-dir"


@pytest.mark.parametrize("evidence", ["/tmp/outside.txt", "../outside.txt"])
def test_evidence_must_stay_inside_repository(tmp_path: Path, evidence: str) -> None:
    with pytest.raises(ValueError, match="evidence path"):
        inspect_evidence(
            [item("CD01", "complete", evidence)], evidence_root=tmp_path
        )
