from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.build_roi_bc_dpg_joint_paper_assets import (
    build_claim_boundaries,
    build_derived_metric_table,
    build_fold_distribution_summary,
    build_paired_mcnemar_table,
    build_scan_group_uncertainty,
    build_simple_combination_diagnostics,
    ensure_output_available,
    markdown_table,
    validate_status,
)


def valid_status() -> dict[str, object]:
    return {
        "status": "ok",
        "bc_decision_source": "base_threshold_test_predictions.csv",
        "roi_decision_source": "refined_fixed_* columns from test_predictions.csv",
        "folds": [1, 2, 3, 4, 5, 6],
        "rows": 1148,
        "exact_alignment": True,
        "test_threshold_retuning": False,
        "output_dir": (
            "results/data_audit/final_roi_bc_dpg_joint_v2_base_threshold"
        ),
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("bc_decision_source", "test_predictions.csv", "BC decision source"),
        ("rows", 1147, "rows must be 1148"),
        ("exact_alignment", False, "exact_alignment must be true"),
        ("test_threshold_retuning", True, "test_threshold_retuning must be false"),
    ],
)
def test_validate_status_rejects_nonfrozen_inputs(
    field: str,
    value: object,
    message: str,
) -> None:
    status = valid_status()
    status[field] = value
    with pytest.raises(ValueError, match=message):
        validate_status(status)


def test_simple_combination_diagnostics_are_derived_and_not_selected() -> None:
    detection = pd.DataFrame(
        [
            {
                "fold": "ALL",
                "model": "bc_dpg_v3",
                "false_alarms": 56,
                "correct_detections": 289,
            },
            {
                "fold": "ALL",
                "model": "roi_ri4",
                "false_alarms": 196,
                "correct_detections": 268,
            },
        ]
    )
    complementarity = pd.DataFrame(
        [
            {
                "fold": "ALL",
                "comparison": "roi_ri4",
                "target_samples": 318,
                "shared_false_alarms": 36,
                "fa_union": 216,
                "shared_correct": 263,
                "correct_union": 294,
            }
        ]
    )

    diagnostics = build_simple_combination_diagnostics(
        detection, complementarity
    ).set_index("rule")

    intersection = diagnostics.loc["AND / intersection"]
    union = diagnostics.loc["OR / union"]
    assert intersection["false_alarms"] == 36
    assert intersection["correct_detections"] == 263
    assert union["false_alarms"] == 216
    assert union["correct_detections"] == 294
    assert intersection["selection_status"] == "diagnostic only, not selected"
    assert union["selection_status"] == "diagnostic only, not selected"


def test_claim_boundaries_do_not_describe_a_trained_joint_model() -> None:
    claims = build_claim_boundaries()
    trained_joint = claims["statement"].str.contains(
        "trained or learned ROI/BC-DPG joint model", case=False
    )
    assert trained_joint.sum() == 1
    assert claims.loc[trained_joint, "boundary"].item() == "not_supported"
    diagnostic = claims["boundary"].eq("diagnostic_only")
    assert claims.loc[diagnostic, "statement"].str.contains("AND and OR").any()


def test_fold_distribution_exposes_worst_fold_and_concentration() -> None:
    rows = []
    pfa = [0.28, 0.0, 0.0, 0.0933333333, 0.0, 0.0]
    joint_pd = [1.0, 0.9245, 0.8654, 0.9038, 0.95, 0.7917]
    false_alarms = [42, 0, 0, 14, 0, 0]
    for model in (
        "bc_dpg_v3",
        "roi_baseline",
        "roi_power_control",
        "roi_ri4",
    ):
        for fold in range(1, 7):
            rows.append(
                {
                    "fold": f"{fold:02d}",
                    "model": model,
                    "pfa": pfa[fold - 1],
                    "joint_pd": joint_pd[fold - 1],
                    "false_alarms": false_alarms[fold - 1],
                }
            )
    summary = build_fold_distribution_summary(pd.DataFrame(rows)).set_index("model")
    bc = summary.loc["bc_dpg_v3"]
    assert bc["worst_pfa_fold"] == "01"
    assert bc["worst_joint_pd_fold"] == "06"
    assert bc["top_two_fold_fa_fraction"] == pytest.approx(1.0)


def test_derived_metrics_use_joint_success_definition() -> None:
    pooled = pd.DataFrame(
        [
            {
                "model": "bc_dpg_v3",
                "display_name": "BC-DPG-FCN v3 (offline)",
                "target_samples": 318,
                "background_samples": 830,
                "correct_detections": 289,
                "false_alarms": 56,
            }
        ]
    )
    metrics = build_derived_metric_table(pooled).iloc[0]
    assert metrics["joint_precision"] == pytest.approx(289 / (289 + 56))
    assert metrics["specificity"] == pytest.approx((830 - 56) / 830)
    assert metrics["joint_f1"] == pytest.approx(0.8717948718)


def test_scan_group_bootstrap_is_deterministic() -> None:
    predictions = pd.DataFrame(
        {
            "fold": [1, 1, 2, 2, 1, 1, 2, 2],
            "scan_group": ["b1", "b1", "b2", "b2", "t1", "t1", "t2", "t2"],
            "sample_id": [f"s{index}" for index in range(8)],
            "target_present": [0, 0, 0, 0, 1, 1, 1, 1],
            "bc_false_alarm": [True, False, False, False, False, False, False, False],
            "bc_correct": [False, False, False, False, True, False, True, True],
            "roi_base_false_alarm": [True, True, False, False, False, False, False, False],
            "roi_base_correct": [False, False, False, False, True, True, True, False],
            "roi_power_false_alarm": [False, False, False, False, False, False, False, False],
            "roi_power_correct": [False, False, False, False, True, True, True, True],
            "roi_ri4_false_alarm": [False, True, False, False, False, False, False, False],
            "roi_ri4_correct": [False, False, False, False, False, True, True, True],
        }
    )
    first = build_scan_group_uncertainty(predictions, iterations=200, seed=7)
    second = build_scan_group_uncertainty(predictions, iterations=200, seed=7)
    pd.testing.assert_frame_equal(first, second)
    assert first["background_scan_groups"].eq(2).all()
    assert first["target_scan_groups"].eq(2).all()
    assert first["bootstrap_unit"].eq(
        "stratified resampling of 2 background and 2 target scan groups"
    ).all()


def test_paired_mcnemar_is_marked_post_test() -> None:
    complementarity = pd.DataFrame(
        [
            {
                "fold": "ALL",
                "comparison": "roi_ri4",
                "bc_only_false_alarms": 20,
                "roi_only_false_alarms": 160,
                "bc_only_correct": 26,
                "roi_only_correct": 5,
            }
        ]
    )
    paired = build_paired_mcnemar_table(complementarity)
    assert paired["two_sided_exact_mcnemar_p"].lt(0.001).all()
    assert paired["status"].str.contains("post-test").all()
    rendered = markdown_table(
        paired,
        ["paired_outcome", "two_sided_exact_mcnemar_p"],
    )
    assert "2.607e-28" in rendered
    assert "1.922e-04" in rendered


def test_nonempty_output_requires_explicit_overwrite(tmp_path: Path) -> None:
    output_dir = tmp_path / "evidence"
    output_dir.mkdir()
    ensure_output_available(output_dir, overwrite=False)

    (output_dir / "existing.txt").write_text("frozen\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="--overwrite"):
        ensure_output_available(output_dir, overwrite=False)
    ensure_output_available(output_dir, overwrite=True)
