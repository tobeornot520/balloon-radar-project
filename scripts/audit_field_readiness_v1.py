#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKLIST = PROJECT_ROOT / "configs" / "field_readiness_checklist_v1.json"
DEFAULT_EVIDENCE_TEMPLATE = (
    PROJECT_ROOT / "configs" / "field_readiness_evidence_template_v1.csv"
)
EVIDENCE_COLUMNS = (
    "item_id",
    "status",
    "evidence_path",
    "measured_value",
    "unit",
    "verified_at_utc",
    "verified_by",
    "notes",
)
VALID_STATUSES = {"pending", "pass", "fail", "not_applicable"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit pre-collection field readiness through a selected gate"
    )
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--checklist", type=Path, default=DEFAULT_CHECKLIST)
    parser.add_argument(
        "--through",
        choices=(
            "capability",
            "synchronization",
            "polarimetric_calibration",
            "dry_run",
            "pilot",
        ),
        default="pilot",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--check-files", action="store_true")
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


def parse_utc(value: str) -> datetime | None:
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


def load_checklist(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported readiness checklist schema_version")
    gates = payload.get("gate_order")
    items = payload.get("items")
    if not isinstance(gates, list) or not gates or len(gates) != len(set(gates)):
        raise ValueError("gate_order must be a nonempty unique list")
    if not isinstance(items, list) or not items:
        raise ValueError("items must be a nonempty list")
    item_ids = [item.get("item_id") for item in items]
    if any(not isinstance(item_id, str) or not item_id for item_id in item_ids):
        raise ValueError("every readiness item needs an item_id")
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("readiness item_id values must be unique")
    for item in items:
        if item.get("gate") not in gates:
            raise ValueError(f"unknown gate for {item.get('item_id')}")
        measurement = item.get("measurement")
        if measurement is not None:
            if not isinstance(measurement, dict) or not measurement.get("unit"):
                raise ValueError(f"invalid measurement rule for {item['item_id']}")
            if "minimum" not in measurement and "maximum" not in measurement:
                raise ValueError(f"measurement rule has no bound: {item['item_id']}")
    return payload


def pending_evidence_frame(checklist: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "item_id": item["item_id"],
                "status": "pending",
                "evidence_path": "",
                "measured_value": "",
                "unit": item.get("measurement", {}).get("unit", ""),
                "verified_at_utc": "",
                "verified_by": "",
                "notes": "",
            }
            for item in checklist["items"]
        ],
        columns=EVIDENCE_COLUMNS,
    )


def _issue(
    issues: list[dict[str, Any]],
    code: str,
    message: str,
    *,
    item_id: str = "",
    severity: str = "ERROR",
) -> None:
    issues.append(
        {
            "severity": severity,
            "code": code,
            "item_id": item_id,
            "message": message,
        }
    )


def audit_readiness(
    checklist: dict[str, Any],
    evidence: pd.DataFrame,
    through: str,
    *,
    evidence_root: Path | None = None,
    check_files: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame]:
    gates = list(checklist["gate_order"])
    if through not in gates:
        raise ValueError(f"unknown readiness gate: {through}")
    evaluated_gates = set(gates[: gates.index(through) + 1])
    issues: list[dict[str, Any]] = []
    actual_columns = tuple(evidence.columns)
    if actual_columns != EVIDENCE_COLUMNS:
        _issue(
            issues,
            "EVIDENCE_SCHEMA_MISMATCH",
            f"evidence columns must exactly match {list(EVIDENCE_COLUMNS)}",
        )
        evidence = pd.DataFrame(columns=EVIDENCE_COLUMNS)
    else:
        evidence = evidence.fillna("").astype(str)

    duplicated = evidence["item_id"].duplicated(keep=False)
    for item_id in sorted(set(evidence.loc[duplicated, "item_id"])):
        _issue(
            issues,
            "DUPLICATE_ITEM",
            "item_id appears more than once",
            item_id=item_id,
        )
    expected_ids = {item["item_id"] for item in checklist["items"]}
    for item_id in sorted(set(evidence["item_id"]) - expected_ids):
        _issue(
            issues,
            "UNKNOWN_ITEM",
            "item_id is not defined by the checklist",
            item_id=item_id,
        )

    evidence_by_id: dict[str, pd.Series] = {}
    for _, row in evidence.loc[~duplicated].iterrows():
        item_id = row["item_id"].strip()
        if item_id in expected_ids:
            evidence_by_id[item_id] = row

    item_rows: list[dict[str, Any]] = []
    for specification in checklist["items"]:
        item_id = specification["item_id"]
        gate = specification["gate"]
        evaluated = gate in evaluated_gates
        row = evidence_by_id.get(item_id)
        status = "missing" if row is None else row["status"].strip().lower()
        assessment = "NOT_EVALUATED" if not evaluated else "BLOCKED"
        measured_value: float | None = None
        evidence_path = "" if row is None else row["evidence_path"].strip()

        if evaluated:
            if row is None:
                _issue(
                    issues,
                    "MISSING_ITEM",
                    "required readiness item has no evidence row",
                    item_id=item_id,
                    severity="BLOCKER",
                )
            elif status not in VALID_STATUSES:
                assessment = "FAIL"
                _issue(
                    issues,
                    "INVALID_STATUS",
                    f"status must be one of {sorted(VALID_STATUSES)}",
                    item_id=item_id,
                )
            elif status == "pending":
                _issue(
                    issues,
                    "PENDING_ITEM",
                    "readiness item is still pending",
                    item_id=item_id,
                    severity="BLOCKER",
                )
            elif status == "fail":
                assessment = "FAIL"
                _issue(
                    issues,
                    "DECLARED_FAILURE",
                    "readiness evidence declares this item failed",
                    item_id=item_id,
                )
            elif status == "not_applicable" and specification.get("required", True):
                assessment = "FAIL"
                _issue(
                    issues,
                    "REQUIRED_ITEM_NOT_APPLICABLE",
                    "required readiness items cannot be marked not_applicable",
                    item_id=item_id,
                )
            elif status == "not_applicable":
                assessment = "PASS"
            else:
                assessment = "PASS"
                if specification.get("evidence_required", False) and not evidence_path:
                    assessment = "FAIL"
                    _issue(
                        issues,
                        "MISSING_EVIDENCE_PATH",
                        "passing this item requires a relative evidence path",
                        item_id=item_id,
                    )
                if evidence_path:
                    path = Path(evidence_path)
                    if path.is_absolute() or ".." in path.parts:
                        assessment = "FAIL"
                        _issue(
                            issues,
                            "INVALID_EVIDENCE_PATH",
                            "evidence_path must be storage-root relative",
                            item_id=item_id,
                        )
                    elif check_files:
                        if evidence_root is None:
                            assessment = "FAIL"
                            _issue(
                                issues,
                                "EVIDENCE_ROOT_REQUIRED",
                                "--check-files requires --evidence-root",
                                item_id=item_id,
                            )
                        elif not (evidence_root / path).is_file():
                            assessment = "FAIL"
                            _issue(
                                issues,
                                "MISSING_EVIDENCE_FILE",
                                "referenced evidence file does not exist",
                                item_id=item_id,
                            )
                if not row["verified_by"].strip():
                    assessment = "FAIL"
                    _issue(
                        issues,
                        "MISSING_VERIFIER",
                        "passing evidence requires verified_by",
                        item_id=item_id,
                    )
                if parse_utc(row["verified_at_utc"]) is None:
                    assessment = "FAIL"
                    _issue(
                        issues,
                        "INVALID_VERIFICATION_TIME",
                        "passing evidence requires a UTC verification timestamp",
                        item_id=item_id,
                    )

                measurement = specification.get("measurement")
                if measurement is not None:
                    try:
                        measured_value = float(row["measured_value"])
                    except ValueError:
                        assessment = "FAIL"
                        _issue(
                            issues,
                            "INVALID_MEASUREMENT",
                            "passing evidence requires a numeric measured_value",
                            item_id=item_id,
                        )
                    if row["unit"].strip() != measurement["unit"]:
                        assessment = "FAIL"
                        _issue(
                            issues,
                            "MEASUREMENT_UNIT_MISMATCH",
                            f"measurement unit must be {measurement['unit']}",
                            item_id=item_id,
                        )
                    if measured_value is not None and not math.isfinite(measured_value):
                        assessment = "FAIL"
                        _issue(
                            issues,
                            "NONFINITE_MEASUREMENT",
                            "measured_value must be finite",
                            item_id=item_id,
                        )
                    if measured_value is not None and "minimum" in measurement:
                        if measured_value < float(measurement["minimum"]):
                            assessment = "FAIL"
                            _issue(
                                issues,
                                "MEASUREMENT_BELOW_MINIMUM",
                                f"measured value is below {measurement['minimum']} {measurement['unit']}",
                                item_id=item_id,
                            )
                    if measured_value is not None and "maximum" in measurement:
                        if measured_value > float(measurement["maximum"]):
                            assessment = "FAIL"
                            _issue(
                                issues,
                                "MEASUREMENT_ABOVE_MAXIMUM",
                                f"measured value is above {measurement['maximum']} {measurement['unit']}",
                                item_id=item_id,
                            )

        item_rows.append(
            {
                "item_id": item_id,
                "gate": gate,
                "description": specification["description"],
                "required": bool(specification.get("required", True)),
                "reported_status": status,
                "assessment": assessment,
                "evidence_path": evidence_path,
                "measured_value": measured_value,
                "expected_unit": specification.get("measurement", {}).get("unit", ""),
            }
        )

    item_table = pd.DataFrame(item_rows)
    schema_failed = any(
        issue["code"] in {"EVIDENCE_SCHEMA_MISMATCH", "DUPLICATE_ITEM", "UNKNOWN_ITEM"}
        for issue in issues
    )
    gate_status: dict[str, str] = {}
    for gate in gates:
        if gate not in evaluated_gates:
            gate_status[gate] = "NOT_EVALUATED"
            continue
        assessments = item_table.loc[item_table["gate"].eq(gate), "assessment"]
        if schema_failed or assessments.eq("FAIL").any():
            gate_status[gate] = "FAIL"
        elif assessments.eq("BLOCKED").any():
            gate_status[gate] = "BLOCKED"
        elif assessments.eq("PASS").all():
            gate_status[gate] = "PASS"
        else:
            gate_status[gate] = "BLOCKED"

    evaluated_status = [gate_status[gate] for gate in gates if gate in evaluated_gates]
    if schema_failed or "FAIL" in evaluated_status:
        overall = "FAIL"
    elif evaluated_status and set(evaluated_status) == {"PASS"}:
        overall = "PASS"
    else:
        overall = "BLOCKED"
    highest_passed_gate = None
    for gate in gates:
        if gate_status[gate] != "PASS":
            break
        highest_passed_gate = gate

    report = {
        "schema_version": 1,
        "checklist_name": checklist["checklist_name"],
        "through_gate": through,
        "status": overall,
        "highest_consecutive_passed_gate": highest_passed_gate,
        "formal_pilot_gate_open": bool(through == "pilot" and overall == "PASS"),
        "gate_status": gate_status,
        "item_count": len(item_table),
        "passed_item_count": int(item_table["assessment"].eq("PASS").sum()),
        "blocked_item_count": int(item_table["assessment"].eq("BLOCKED").sum()),
        "failed_item_count": int(item_table["assessment"].eq("FAIL").sum()),
        "issues": issues,
    }
    return report, item_table


def write_output(
    output_dir: Path,
    report: dict[str, Any],
    item_table: pd.DataFrame,
    *,
    evidence_path: Path,
    checklist_path: Path,
    overwrite: bool,
) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"output directory is not empty: {output_dir}; use --overwrite"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        **report,
        "input": {
            "evidence_name": evidence_path.name,
            "evidence_sha256": sha256_file(evidence_path),
            "checklist_name": checklist_path.name,
            "checklist_sha256": sha256_file(checklist_path),
        },
    }
    report_path = output_dir / "readiness_audit.json"
    item_path = output_dir / "item_status.csv"
    markdown_path = output_dir / "READINESS_REPORT.md"
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    item_table.to_csv(item_path, index=False, encoding="utf-8-sig")
    lines = [
        "# Field readiness audit",
        "",
        f"- Status: `{report['status']}`",
        f"- Audited through: `{report['through_gate']}`",
        f"- Highest consecutive passed gate: `{report['highest_consecutive_passed_gate']}`",
        f"- Formal pilot gate open: `{report['formal_pilot_gate_open']}`",
        "",
        "## Gates",
        "",
        "| Gate | Status |",
        "|---|---|",
    ]
    lines.extend(
        f"| `{gate}` | `{status}` |"
        for gate, status in report["gate_status"].items()
    )
    lines.extend(["", "## Issues", ""])
    if report["issues"]:
        lines.extend(
            f"- `{item['severity']}` `{item['code']}` "
            f"{item['item_id']}: {item['message']}"
            for item in report["issues"]
        )
    else:
        lines.append("- None")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    checksum_lines = []
    for path in (report_path, item_path, markdown_path):
        checksum_lines.append(f"{sha256_file(path)}  {path.name}")
    (output_dir / "SHA256SUMS.txt").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    checklist_path = resolve_path(args.checklist)
    evidence_path = resolve_path(args.evidence)
    output_dir = resolve_path(args.output_dir)
    evidence_root = (
        resolve_path(args.evidence_root) if args.evidence_root is not None else None
    )
    if not checklist_path.is_file():
        raise FileNotFoundError(f"checklist not found: {checklist_path}")
    if not evidence_path.is_file():
        raise FileNotFoundError(f"evidence table not found: {evidence_path}")
    checklist = load_checklist(checklist_path)
    evidence = pd.read_csv(evidence_path, keep_default_na=False, dtype=str)
    report, item_table = audit_readiness(
        checklist,
        evidence,
        args.through,
        evidence_root=evidence_root,
        check_files=args.check_files,
    )
    write_output(
        output_dir,
        report,
        item_table,
        evidence_path=evidence_path,
        checklist_path=checklist_path,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
