from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_template(name: str) -> pd.DataFrame:
    return pd.read_csv(
        PROJECT_ROOT / "configs" / name,
        dtype=str,
        keep_default_na=False,
    )


def _assert_template_contract(
    name: str,
    expected_columns: list[str],
    nonempty_defaults: dict[str, str],
) -> pd.DataFrame:
    frame = _read_template(name)
    assert frame.columns.tolist() == expected_columns

    expected_row = dict.fromkeys(expected_columns, "")
    expected_row.update(nonempty_defaults)
    assert frame.to_dict(orient="records") == [expected_row]
    return frame


def test_team_onboarding_checklist_v2_is_onboarding_only() -> None:
    columns = [
        "member",
        "reviewer",
        "date_started",
        "package_name",
        "package_sha256",
        "share_package_sha256_passed",
        "one_page_summary_read",
        "current_status_understood",
        "three_evidence_objects_explained",
        "research_claim_boundaries_explained",
        "current_data_access_level",
        "repository_access",
        "environment_check_applicable",
        "environment_ready",
        "project_health_passed",
        "pytest_passed",
        "oral_review_path",
        "oral_questions_passed_5",
        "onboarding_decision",
        "remaining_gap",
        "next_action",
        "onboarding_completed_at",
    ]
    frame = _assert_template_contract(
        "team_onboarding_checklist_template_v2.csv",
        columns,
        {
            "package_name": (
                "balloon_radar_results_and_team_onboarding_20260807_v15"
            ),
            "share_package_sha256_passed": "pending",
            "one_page_summary_read": "pending",
            "current_status_understood": "pending",
            "three_evidence_objects_explained": "pending",
            "research_claim_boundaries_explained": "pending",
            "current_data_access_level": "A",
            "repository_access": "no",
            "environment_check_applicable": "no",
            "environment_ready": "not_applicable",
            "project_health_passed": "not_applicable",
            "pytest_passed": "not_applicable",
            "oral_questions_passed_5": "0",
            "onboarding_decision": "pending",
        },
    )
    assert {"task_id", "task_claim_path"}.isdisjoint(frame.columns)


def test_team_trial_task_v1_is_a_bounded_qualification_trial() -> None:
    _assert_template_contract(
        "team_trial_task_template_v1.csv",
        [
            "trial_task_id",
            "member",
            "reviewer",
            "trial_option",
            "scope",
            "estimated_hours",
            "input_materials",
            "expected_output",
            "deadline",
            "prohibited_actions",
            "stop_condition",
            "status",
            "result_path",
            "review_decision",
            "followup",
        ],
        {
            "estimated_hours": "3-6",
            "status": "proposed",
            "review_decision": "pending",
        },
    )


def test_team_qualification_scorecard_v2_links_onboarding_and_trial() -> None:
    trial = _read_template("team_trial_task_template_v1.csv")
    qualification = _assert_template_contract(
        "team_qualification_scorecard_template_v2.csv",
        [
            "member",
            "reviewer_1",
            "reviewer_2",
            "assessment_round",
            "date_issued",
            "onboarding_record_path",
            "onboarding_decision",
            "trial_task_deadline",
            "trial_task_id",
            "trial_task_result",
            "project_understanding_score_25",
            "evidence_boundary_score_20",
            "execution_score_30",
            "reproducibility_score_15",
            "communication_score_10",
            "total_score_100",
            "role_gate_status",
            "role_gate_violation",
            "initial_role",
            "granted_data_access_level",
            "retry_required",
            "retry_deadline",
            "qualification_decision",
            "decision_reason",
            "member_acknowledgement",
            "qualification_completed_at",
        ],
        {
            "assessment_round": "1",
            "onboarding_decision": "pending",
            "trial_task_result": "pending",
            "project_understanding_score_25": "0",
            "evidence_boundary_score_20": "0",
            "execution_score_30": "0",
            "reproducibility_score_15": "0",
            "communication_score_10": "0",
            "total_score_100": "0",
            "role_gate_status": "pending",
            "initial_role": "pending",
            "granted_data_access_level": "A",
            "retry_required": "no",
            "qualification_decision": "pending",
            "member_acknowledgement": "pending",
        },
    )
    assert "trial_task_id" in trial.columns
    assert {"onboarding_record_path", "trial_task_id", "trial_task_result"} <= set(
        qualification.columns
    )


def test_team_task_claim_v2_is_a_post_qualification_formal_claim() -> None:
    frame = _assert_template_contract(
        "team_task_claim_template_v2.csv",
        [
            "claim_id",
            "claim_type",
            "task_id",
            "title",
            "owner",
            "reviewer",
            "priority",
            "status",
            "qualification_record_path",
            "required_data_access_level",
            "scope",
            "input_paths",
            "input_hash_or_manifest",
            "git_branch",
            "planned_outputs",
            "group_key",
            "test_access_policy",
            "primary_metrics",
            "acceptance_gate",
            "stop_condition",
            "forbidden_claims",
            "start_date",
            "target_date",
            "last_update",
            "blockers",
            "next_action",
            "claim_decision",
        ],
        {
            "claim_type": "formal",
            "status": "proposed",
            "required_data_access_level": "A",
            "claim_decision": "pending",
        },
    )
    assert "qualification_record_path" in frame.columns
    assert "trial_task_id" not in frame.columns


def test_team_qualification_policy_has_fairness_gates_and_scripts() -> None:
    text = (
        PROJECT_ROOT
        / "docs/share/TEAM_QUALIFICATION_AND_ROLE_SCREENING_ZH.md"
    ).read_text(encoding="utf-8")
    for required in (
        "同一标准",
        "最小试做任务",
        "100 分评分表",
        "角色评估门槛",
        "补验与异议处理",
        "核心候选",
        "暂不分配",
        "群公告：正式版",
        "向指导老师说明",
    ):
        assert required in text


def test_team_documents_define_the_ordered_onboarding_stages() -> None:
    policy = (
        PROJECT_ROOT
        / "docs/share/TEAM_QUALIFICATION_AND_ROLE_SCREENING_ZH.md"
    ).read_text(encoding="utf-8")
    guide = (PROJECT_ROOT / "docs/share/ONB_01_SUBMISSION_GUIDE_ZH.md").read_text(
        encoding="utf-8"
    )
    manual = (PROJECT_ROOT / "docs/share/TEAM_START_HERE.md").read_text(
        encoding="utf-8"
    )
    compact_policy = "".join(policy.split())

    assert (
        "ONB-01（仅基础验收）->Q1-Q6（3-6小时最小试做）->"
        "角色与权限决定->正式任务认领->周交付"
        in compact_policy
    )
    assert "Q1-Q6 不属于 ONB-01，也不属于正式或长期任务" in policy
    assert "角色与权限确定后，成员才能提交" in policy
    assert "ONB-01 不要求" in guide
    assert "正式任务认领表" in guide
    assert "长期任务属于项目分工，不属于 ONB-01 验收" in guide
    assert "先完成 ONB-01，再完成最小试做" in manual
    assert "角色和权限确认后，才认领正式任务" in manual


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
