#!/usr/bin/env python3
"""Check cross-file contracts that protect the project's frozen evidence."""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COURSE_PATH = Path("docs/PROJECT_LEARNING_COURSE_ZH.md")
METRIC_DEFINITIONS_PATH = Path("docs/METRIC_DEFINITIONS.md")
CONFIG_PATH = Path("configs/dual_branch_gated_v1.yaml")
TRAINING_PATH = Path("training/train_dual_branch_gated.py")
LOCAL_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def _as_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def joint_pd_violations(root: Path) -> list[str]:
    """Find structured result rows where Joint Pd exceeds Score Pd."""
    violations: list[str] = []
    result_root = root / "results" / "final_evidence"
    pairs = (
        ("score_pd", "joint_pd"),
        ("mean_fixed_score_pd", "mean_fixed_joint_pd"),
    )
    if not result_root.is_dir():
        return ["missing results/final_evidence"]
    for path in sorted(result_root.rglob("*.csv")):
        try:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                fieldnames = set(reader.fieldnames or ())
                active_pairs = [pair for pair in pairs if set(pair) <= fieldnames]
                for row_number, row in enumerate(reader, start=2):
                    for score_name, joint_name in active_pairs:
                        score = _as_float(row.get(score_name, ""))
                        joint = _as_float(row.get(joint_name, ""))
                        if score is not None and joint is not None and joint > score + 1e-9:
                            relative = path.relative_to(root)
                            violations.append(
                                f"{relative}:{row_number}: {joint_name}={joint} > "
                                f"{score_name}={score}"
                            )
        except (OSError, UnicodeError, csv.Error) as exc:
            violations.append(f"{path.relative_to(root)}: cannot read CSV ({exc})")
    return violations


def threshold_contract_violations(root: Path) -> list[str]:
    violations: list[str] = []
    metric_path = root / METRIC_DEFINITIONS_PATH
    config_path = root / CONFIG_PATH
    training_path = root / TRAINING_PATH
    metric_text = metric_path.read_text(encoding="utf-8") if metric_path.is_file() else ""
    config_text = config_path.read_text(encoding="utf-8") if config_path.is_file() else ""
    training_text = training_path.read_text(encoding="utf-8") if training_path.is_file() else ""
    if "`score > threshold`" not in metric_text:
        violations.append("metric definitions do not state the strict score > threshold rule")
    if "不低于该 fold" in metric_text:
        violations.append("metric definitions still contain the ambiguous 不低于阈值 wording")
    if "max_val_false_alarms: 1" not in config_text:
        violations.append("dual_branch_gated_v1.yaml must set max_val_false_alarms: 1")
    if not re.search(r'--max-val-false-alarms"[^\n]*default=1', training_text):
        violations.append("train_dual_branch_gated.py CLI default must be 1")
    return violations


def course_link_violations(root: Path) -> list[str]:
    path = root / COURSE_PATH
    if not path.is_file():
        return [f"missing {COURSE_PATH}"]
    violations: list[str] = []
    text = path.read_text(encoding="utf-8")
    for target in LOCAL_LINK.findall(text):
        target = target.strip().split()[0]
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        target_path = target.split("#", 1)[0]
        if not target_path:
            continue
        resolved = (path.parent / target_path).resolve()
        if not resolved.is_file():
            violations.append(f"{COURSE_PATH}: missing linked file {target}")
    return violations


def governance_violations(root: Path) -> list[str]:
    path = root / COURSE_PATH
    if not path.is_file():
        return [f"missing {COURSE_PATH}"]
    text = path.read_text(encoding="utf-8")
    violations: list[str] = []
    if "TASK_BOARD.md" not in text or not any(
        marker in text for marker in ("正式任务状态唯一以", "正式任务状态应继续以")
    ):
        violations.append("course must identify TASK_BOARD.md as the formal task ledger")
    if "课程内容编写完成；研究方向仍受外部条件阻塞" not in text:
        violations.append("course status must separate course completion from research completion")
    if "IMP-13 | 已实现（结构化主表）" not in text:
        violations.append("IMP-13 must document the implemented structured-table scope")
    return violations


def collect_violations(root: Path = PROJECT_ROOT) -> list[str]:
    root = root.resolve()
    violations: list[str] = []
    for checker in (
        joint_pd_violations,
        threshold_contract_violations,
        course_link_violations,
        governance_violations,
    ):
        violations.extend(checker(root))
    return violations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check project-wide research contracts.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    violations = collect_violations(args.root)
    if violations:
        print("Project contracts: FAIL")
        for violation in violations:
            print(f"- {violation}")
        return 1
    print("Project contracts: PASS")
    print("- Joint Pd <= Score Pd: checked structured final-evidence tables")
    print("- threshold, course links, governance and share prerequisites: checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
