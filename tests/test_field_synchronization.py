from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from scripts.audit_field_synchronization_v1 import (
    DEFAULT_CONTRACT,
    audit_synchronization,
    load_contract,
    validate_events,
)
from scripts.audit_field_readiness_v1 import DEFAULT_CHECKLIST, load_checklist


def timestamp(base: datetime, seconds: float) -> str:
    value = base + timedelta(seconds=seconds)
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def valid_events(errors_ms: list[float] | None = None) -> pd.DataFrame:
    contract = load_contract(DEFAULT_CONTRACT)
    base = datetime(2026, 8, 12, tzinfo=timezone.utc)
    errors = errors_ms or [10.0, -20.0, 30.0, -40.0, 50.0]
    rows = []
    for index, error_ms in enumerate(errors):
        truth_seconds = float(index * 10)
        rows.append(
            {
                "session_id": "sync-bench-01",
                "event_id": f"event-{index}",
                "event_index": index,
                "radar_timestamp_utc": timestamp(base, truth_seconds),
                "video_timestamp_utc": timestamp(
                    base, truth_seconds - error_ms / 1000.0
                ),
                "truth_timestamp_utc": timestamp(base, truth_seconds + 0.005),
                "radar_clock_id": "radar-clock-01",
                "video_clock_id": "video-clock-01",
                "truth_clock_id": "truth-clock-01",
                "timestamp_mapping_id": "mapping-v1",
                "event_method": "visible_marker",
                "event_uncertainty_ms": 10.0,
                "accepted": True,
                "rejection_reason": "",
                "notes": "",
            }
        )
    return pd.DataFrame(rows, columns=contract["required_columns"])


def write_events(path: Path, frame: pd.DataFrame) -> Path:
    frame.to_csv(path, index=False)
    return path


def test_five_events_pass_numeric_limits_only(tmp_path: Path) -> None:
    events = write_events(tmp_path / "events.csv", valid_events())

    summary = audit_synchronization(
        events_path=events,
        contract_path=DEFAULT_CONTRACT,
        output_dir=tmp_path / "output",
        overwrite=False,
    )
    measurements = pd.read_csv(
        tmp_path / "output/readiness_measurements.csv", encoding="utf-8-sig"
    ).set_index("item_id")

    assert summary["status"] == "PASS_NUMERIC_LIMITS_ONLY"
    assert summary["formal_synchronization_gate_open"] is False
    assert summary["radar_video_absolute_error_p95_ms"] == pytest.approx(48.0)
    assert summary["radar_video_absolute_error_max_ms"] == pytest.approx(50.0)
    assert measurements.loc["SYNC_EVENT_REPEATS", "measured_value"] == 5
    assert set(measurements["recommended_status"]) == {"pass"}


def test_p95_limit_failure_is_reported(tmp_path: Path) -> None:
    events = write_events(
        tmp_path / "events.csv", valid_events([60.0, 60.0, 60.0, 60.0, 60.0])
    )

    summary = audit_synchronization(
        events_path=events,
        contract_path=DEFAULT_CONTRACT,
        output_dir=tmp_path / "output",
        overwrite=False,
    )

    assert summary["status"] == "FAIL"
    assert "RADAR_VIDEO_P95_ABOVE_MAXIMUM" in {
        item["code"] for item in summary["issues"]
    }


def test_too_few_events_fails(tmp_path: Path) -> None:
    events = write_events(tmp_path / "events.csv", valid_events()[:4])

    summary = audit_synchronization(
        events_path=events,
        contract_path=DEFAULT_CONTRACT,
        output_dir=tmp_path / "output",
        overwrite=False,
    )

    assert summary["status"] == "FAIL"
    assert "INSUFFICIENT_ACCEPTED_EVENTS" in {
        item["code"] for item in summary["issues"]
    }


def test_nonmonotonic_video_time_fails_validation() -> None:
    frame = valid_events()
    frame.loc[2, "video_timestamp_utc"] = frame.loc[0, "video_timestamp_utc"]
    _, issues = validate_events(frame, load_contract(DEFAULT_CONTRACT))

    assert "NONMONOTONIC_EVENT_TIME" in {item["code"] for item in issues}


def test_rejected_event_requires_reason() -> None:
    frame = valid_events()
    frame.loc[0, "accepted"] = False
    _, issues = validate_events(frame, load_contract(DEFAULT_CONTRACT))

    assert "MISSING_REJECTION_REASON" in {item["code"] for item in issues}


def test_documented_high_uncertainty_rejection_is_excluded(tmp_path: Path) -> None:
    frame = valid_events([10.0, 15.0, 20.0, 25.0, 30.0, 35.0])
    frame.loc[5, "accepted"] = False
    frame.loc[5, "event_uncertainty_ms"] = 100.0
    frame.loc[5, "rejection_reason"] = "marker occluded"
    events = write_events(tmp_path / "events.csv", frame)

    summary = audit_synchronization(
        events_path=events,
        contract_path=DEFAULT_CONTRACT,
        output_dir=tmp_path / "output",
        overwrite=False,
    )

    assert summary["status"] == "PASS_NUMERIC_LIMITS_ONLY"
    assert summary["accepted_event_count"] == 5
    assert summary["rejected_event_count"] == 1


def test_session_without_accepted_event_fails(tmp_path: Path) -> None:
    frame = valid_events([10.0] * 6)
    frame.loc[5, "session_id"] = "sync-bench-empty"
    frame.loc[5, "event_index"] = 0
    frame.loc[5, "accepted"] = False
    frame.loc[5, "rejection_reason"] = "trigger not visible"
    events = write_events(tmp_path / "events.csv", frame)

    summary = audit_synchronization(
        events_path=events,
        contract_path=DEFAULT_CONTRACT,
        output_dir=tmp_path / "output",
        overwrite=False,
    )

    assert summary["status"] == "FAIL"
    assert "INSUFFICIENT_SESSION_EVENTS" in {
        item["code"] for item in summary["issues"]
    }


def test_summary_omits_event_ids_and_private_directory(tmp_path: Path) -> None:
    events = write_events(tmp_path / "private_events.csv", valid_events())
    output = tmp_path / "output"

    audit_synchronization(
        events_path=events,
        contract_path=DEFAULT_CONTRACT,
        output_dir=output,
        overwrite=False,
    )
    text = (output / "summary.json").read_text(encoding="utf-8")
    sessions = pd.read_csv(output / "session_sync_summary.csv", encoding="utf-8-sig")

    assert "event-0" not in text
    assert str(tmp_path) not in text
    assert "sync-bench-01" not in sessions.to_csv(index=False)


def test_template_matches_contract() -> None:
    contract = load_contract(DEFAULT_CONTRACT)
    template = pd.read_csv(
        DEFAULT_CONTRACT.with_name("field_sync_event_template_v1.csv")
    )

    assert template.empty
    assert template.columns.tolist() == contract["required_columns"]


def test_sync_limits_match_field_readiness_checklist() -> None:
    contract = load_contract(DEFAULT_CONTRACT)
    checklist = load_checklist(DEFAULT_CHECKLIST)
    items = {
        item["item_id"]: item["measurement"]
        for item in checklist["items"]
        if item["item_id"] in {
            "SYNC_EVENT_REPEATS",
            "SYNC_P95_ERROR",
            "SYNC_MAX_ERROR",
        }
    }
    assert items == {
        "SYNC_EVENT_REPEATS": {
            "minimum": contract["minimum_accepted_events"],
            "unit": "count",
        },
        "SYNC_P95_ERROR": {
            "maximum": contract["radar_video_limits_ms"][
                "absolute_error_p95_maximum"
            ],
            "unit": "ms",
        },
        "SYNC_MAX_ERROR": {
            "maximum": contract["radar_video_limits_ms"]["absolute_error_maximum"],
            "unit": "ms",
        },
    }
