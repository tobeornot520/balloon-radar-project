from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_team_task_claim_template_has_execution_gates() -> None:
    frame = pd.read_csv(PROJECT_ROOT / "configs/team_task_claim_template_v1.csv")
    assert {
        "task_id",
        "owner",
        "reviewer",
        "data_access_level",
        "group_key",
        "test_access_policy",
        "primary_metrics",
        "acceptance_gate",
        "stop_condition",
        "forbidden_claims",
        "review_decision",
    } <= set(frame.columns)


def test_team_onboarding_checklist_covers_evidence_and_environment() -> None:
    frame = pd.read_csv(
        PROJECT_ROOT / "configs/team_onboarding_checklist_template_v1.csv"
    )
    assert {
        "share_package_sha256_passed",
        "three_evidence_objects_explained",
        "claim_boundaries_explained",
        "data_access_level",
        "project_health_passed",
        "pytest_passed",
        "task_id",
        "oral_review_passed",
        "reviewer_decision",
    } <= set(frame.columns)


def test_team_qualification_scorecard_covers_scoring_and_access_decision() -> None:
    frame = pd.read_csv(
        PROJECT_ROOT / "configs/team_qualification_scorecard_template_v1.csv"
    )
    assert {
        "member",
        "reviewer_1",
        "reviewer_2",
        "oral_questions_passed_5",
        "project_understanding_score_25",
        "evidence_boundary_score_20",
        "execution_score_30",
        "reproducibility_score_15",
        "communication_score_10",
        "total_score_100",
        "hard_gate_status",
        "trial_task_result",
        "initial_role",
        "data_access_level",
        "retry_required",
        "reviewer_decision",
    } <= set(frame.columns)


def test_team_qualification_policy_has_fairness_gates_and_scripts() -> None:
    text = (
        PROJECT_ROOT
        / "docs/share/TEAM_QUALIFICATION_AND_ROLE_SCREENING_ZH.md"
    ).read_text(encoding="utf-8")
    for required in (
        "同一标准",
        "最小试做任务",
        "100 分评分表",
        "硬门槛",
        "补验与异议处理",
        "核心候选",
        "暂不分配",
        "群公告：正式版",
        "向指导老师说明",
    ):
        assert required in text


def test_team_manual_names_current_gates_and_red_lines() -> None:
    text = (PROJECT_ROOT / "docs/share/TEAM_START_HERE.md").read_text(
        encoding="utf-8"
    )
    for required in (
        "BLOCKED_EXTERNAL",
        "D04-P0",
        "D17-NX",
        "D17-HX",
        "Tian",
        "LAT-MRICD",
        "禁止随机拆行",
        "不把 UAV/背景结果写成空飘球载荷识别",
        "任务完成定义",
        "成员资格与分工验收办法",
        "4/6",
    ):
        assert required in text
