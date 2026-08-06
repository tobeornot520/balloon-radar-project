from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.audit_zero_doppler_target_safety_v1 import (
    LOCAL_ONLY_COLUMNS,
    audit_target_safety,
    load_audit_config,
    transition,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs/zero_doppler_target_safety_audit_v1.json"


def test_target_safety_contract_keeps_claims_closed() -> None:
    config = load_audit_config(CONFIG)

    assert config["expected_counts"]["target_samples"] == 318
    assert config["expected_counts"]["detection_lost"] == 1
    assert config["expected_counts"]["joint_success_lost"] == 0
    assert not any(config["claim_boundaries"].values())


def test_transition_labels_all_paired_states() -> None:
    before = pd.Series([True, True, False, False])
    after = pd.Series([True, False, True, False])

    assert transition(before, after, "detected").tolist() == [
        "retained_detected",
        "lost_detected",
        "gained_detected",
        "retained_not_detected",
    ]


def test_target_safety_matches_frozen_predictions(tmp_path: Path) -> None:
    output_dir = tmp_path / "target-safety"

    summary = audit_target_safety(
        config_path=CONFIG, output_dir=output_dir, overwrite=False
    )

    expected = load_audit_config(CONFIG)["expected_counts"]
    assert summary["counts"] == expected
    assert summary["score_delta"]["maximum"] == pytest.approx(0.0)
    assert summary["detection_loss_context"] == {
        "count": 1,
        "fixed_joint_success_count": 0,
        "residual_joint_success_count": 0,
        "sample_identifiers_published": False,
    }

    local = pd.read_csv(
        output_dir / "target_case_library_local.csv", encoding="utf-8-sig"
    )
    folds = pd.read_csv(
        output_dir / "fold_target_safety_summary.csv", encoding="utf-8-sig"
    )
    shifts = pd.read_csv(output_dir / "peak_shift_histogram.csv", encoding="utf-8-sig")
    quantiles = pd.read_csv(
        output_dir / "score_delta_quantiles.csv", encoding="utf-8-sig"
    )
    saved = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    assert len(local) == 318
    assert folds["fixed_joint_success"].sum() == 290
    assert folds["residual_joint_success"].sum() == 290
    assert folds["detection_lost"].sum() == 1
    assert dict(zip(shifts["shift_band"], shifts["target_count"])) == {
        "unchanged": 312,
        "one_bin": 4,
        "two_to_ten_bins": 0,
        "over_ten_bins": 2,
    }
    assert quantiles.iloc[-1]["score_delta_residual_minus_fixed"] == pytest.approx(0.0)
    assert saved["claim_boundaries"]["deployment_safety_established"] is False
    for frame in (folds, shifts, quantiles):
        assert set(frame.columns).isdisjoint(LOCAL_ONLY_COLUMNS)
