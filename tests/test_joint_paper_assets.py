from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.build_roi_bc_dpg_joint_paper_assets import (
    build_claim_boundaries,
    build_simple_combination_diagnostics,
    ensure_output_available,
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


def test_nonempty_output_requires_explicit_overwrite(tmp_path: Path) -> None:
    output_dir = tmp_path / "evidence"
    output_dir.mkdir()
    ensure_output_available(output_dir, overwrite=False)

    (output_dir / "existing.txt").write_text("frozen\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="--overwrite"):
        ensure_output_available(output_dir, overwrite=False)
    ensure_output_available(output_dir, overwrite=True)
