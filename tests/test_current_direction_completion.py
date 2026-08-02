from __future__ import annotations

import pytest

from scripts.check_current_direction_completion_v1 import summarize_items, validate_config


def item(identifier: str, status: str) -> dict[str, object]:
    return {
        "id": identifier,
        "ledger_ids": ["D01"],
        "title": identifier,
        "owner": "user",
        "status": status,
        "evidence": "evidence",
        "next_action": "next",
    }


def test_completion_requires_every_item() -> None:
    summary, table = summarize_items(
        [item("CD01", "complete"), item("CD02", "pending_user")]
    )

    assert summary["status"] == "IN_PROGRESS"
    assert summary["milestone_complete"] is False
    assert len(table) == 2


def test_external_blocker_is_reported_without_false_completion() -> None:
    summary, _ = summarize_items(
        [item("CD01", "complete"), item("CD02", "blocked_external")]
    )

    assert summary["status"] == "BLOCKED_EXTERNAL"
    assert summary["blocked_external_count"] == 1
    assert summary["milestone_complete"] is False


def test_invalid_completion_status_is_rejected() -> None:
    payload = {"schema_version": 1, "items": [item("CD01", "done")]}

    with pytest.raises(ValueError, match="invalid completion status"):
        validate_config(payload)
