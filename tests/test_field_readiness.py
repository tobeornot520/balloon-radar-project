from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.audit_field_readiness_v1 import (
    DEFAULT_CHECKLIST,
    DEFAULT_EVIDENCE_TEMPLATE,
    EVIDENCE_COLUMNS,
    audit_readiness,
    load_checklist,
    pending_evidence_frame,
    write_output,
)


def passing_evidence(checklist: dict) -> pd.DataFrame:
    rows = []
    for item in checklist["items"]:
        measurement = item.get("measurement", {})
        if "minimum" in measurement:
            value: str | float = measurement["minimum"]
        elif "maximum" in measurement:
            value = measurement["maximum"]
        else:
            value = ""
        rows.append(
            {
                "item_id": item["item_id"],
                "status": "pass",
                "evidence_path": f"evidence/{item['item_id'].lower()}.json",
                "measured_value": value,
                "unit": measurement.get("unit", ""),
                "verified_at_utc": "2026-08-01T00:00:00Z",
                "verified_by": "field-lead",
                "notes": "",
            }
        )
    return pd.DataFrame(rows, columns=EVIDENCE_COLUMNS)


def test_checklist_and_empty_template_are_aligned() -> None:
    checklist = load_checklist(DEFAULT_CHECKLIST)
    assert checklist["gate_order"] == [
        "capability",
        "synchronization",
        "polarimetric_calibration",
        "dry_run",
        "pilot",
    ]
    assert len(checklist["items"]) == len(
        {item["item_id"] for item in checklist["items"]}
    )
    template = pd.read_csv(DEFAULT_EVIDENCE_TEMPLATE)
    assert template.empty
    assert tuple(template.columns) == EVIDENCE_COLUMNS


def test_pending_readiness_is_blocked_not_passed() -> None:
    checklist = load_checklist(DEFAULT_CHECKLIST)
    evidence = pending_evidence_frame(checklist)
    report, items = audit_readiness(checklist, evidence, "capability")
    assert report["status"] == "BLOCKED"
    assert report["formal_pilot_gate_open"] is False
    assert report["gate_status"]["capability"] == "BLOCKED"
    assert report["gate_status"]["synchronization"] == "NOT_EVALUATED"
    assert items.loc[items["gate"].eq("capability"), "assessment"].eq(
        "BLOCKED"
    ).all()


def test_complete_evidence_opens_pilot_gate() -> None:
    checklist = load_checklist(DEFAULT_CHECKLIST)
    report, _ = audit_readiness(
        checklist,
        passing_evidence(checklist),
        "pilot",
    )
    assert report["status"] == "PASS"
    assert report["formal_pilot_gate_open"] is True
    assert set(report["gate_status"].values()) == {"PASS"}


def test_failed_numeric_limit_closes_its_gate() -> None:
    checklist = load_checklist(DEFAULT_CHECKLIST)
    evidence = passing_evidence(checklist)
    evidence.loc[
        evidence["item_id"].eq("SYNC_P95_ERROR"), "measured_value"
    ] = "50.1"
    report, items = audit_readiness(checklist, evidence, "synchronization")
    assert report["status"] == "FAIL"
    assert report["gate_status"]["capability"] == "PASS"
    assert report["gate_status"]["synchronization"] == "FAIL"
    assert "MEASUREMENT_ABOVE_MAXIMUM" in {
        issue["code"] for issue in report["issues"]
    }
    row = items.loc[items["item_id"].eq("SYNC_P95_ERROR")].iloc[0]
    assert row["assessment"] == "FAIL"


def test_absolute_evidence_path_is_rejected() -> None:
    checklist = load_checklist(DEFAULT_CHECKLIST)
    evidence = passing_evidence(checklist)
    evidence.loc[evidence.index[0], "evidence_path"] = "/private/device.json"
    report, _ = audit_readiness(checklist, evidence, "capability")
    assert report["status"] == "FAIL"
    assert "INVALID_EVIDENCE_PATH" in {
        issue["code"] for issue in report["issues"]
    }


def test_readiness_report_omits_private_input_directory(tmp_path: Path) -> None:
    checklist = load_checklist(DEFAULT_CHECKLIST)
    evidence = passing_evidence(checklist)
    evidence_path = tmp_path / "private_evidence.csv"
    checklist_path = tmp_path / "private_checklist.json"
    evidence.to_csv(evidence_path, index=False)
    checklist_path.write_text(json.dumps(checklist), encoding="utf-8")
    report, table = audit_readiness(checklist, evidence, "pilot")
    output = tmp_path / "report"
    write_output(
        output,
        report,
        table,
        evidence_path=evidence_path,
        checklist_path=checklist_path,
        overwrite=False,
    )
    text = (output / "readiness_audit.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in text
    assert (output / "READINESS_REPORT.md").is_file()
    assert (output / "SHA256SUMS.txt").is_file()
