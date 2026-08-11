from __future__ import annotations

import csv
from pathlib import Path

from scripts.check_project_contracts import (
    collect_violations,
    joint_pd_violations,
)


def test_current_project_contracts_pass() -> None:
    assert collect_violations() == []


def test_joint_pd_contract_rejects_invalid_structured_row(tmp_path: Path) -> None:
    table = tmp_path / "results/final_evidence/example.csv"
    table.parent.mkdir(parents=True)
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["score_pd", "joint_pd"])
        writer.writeheader()
        writer.writerow({"score_pd": "0.8", "joint_pd": "0.9"})

    violations = joint_pd_violations(tmp_path)
    assert len(violations) == 1
    assert "joint_pd=0.9 > score_pd=0.8" in violations[0]
