from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.build_bc_dpg_localization_evidence import (
    build_error_distribution,
    build_pooled_summary,
    build_stratified_summary,
    distance_stratum,
    ensure_output_available,
    load_frozen_selection,
    validate_coordinate_alignment,
    validate_predictions,
    velocity_stratum,
)


def prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["t1", "t2", "t3", "b1"],
            "scan_group": ["g1", "g1", "g2", "b"],
            "target_present": [1, 1, 1, 0],
            "pred_range_index": [10, 12, 20, 4],
            "pred_velocity_index": [64, 67, 70, 8],
            "true_range_index": [10, 10, 10, -1],
            "true_velocity_index": [64, 64, 64, -1],
            "range_error_gates": [0.0, 2.0, 10.0, float("nan")],
            "velocity_error_bins": [0.0, 3.0, 6.0, float("nan")],
            "localization_ok": [True, True, False, False],
            "split": ["test"] * 4,
            "score": [0.9, 0.4, 0.8, 0.6],
            "detected": [True, False, True, True],
            "false_alarm": [False, False, False, True],
            "correct_detection": [True, False, False, False],
        }
    )


def target_frame() -> pd.DataFrame:
    frame = validate_predictions(prediction_frame(), threshold=0.5)
    targets = frame.loc[frame["target_present"].eq(1)].copy()
    targets["distance_m"] = [1950.0, 2100.0, 2200.0]
    targets["velocity_mps"] = [-5.0, -2.0, 5.0]
    return targets


def test_prediction_validation_recomputes_frozen_rules() -> None:
    validated = validate_predictions(prediction_frame(), threshold=0.5)
    assert int(validated["detected"].sum()) == 3
    assert int(validated["false_alarm"].sum()) == 1
    assert int(validated["correct_detection"].sum()) == 1

    invalid = prediction_frame()
    invalid.loc[0, "range_error_gates"] = 1.0
    with pytest.raises(ValueError, match="range_error_gates"):
        validate_predictions(invalid, threshold=0.5)


def test_calibrated_coordinates_must_match_raw_predictions() -> None:
    calibrated = validate_predictions(prediction_frame(), threshold=0.5)
    raw = calibrated.copy()
    validate_coordinate_alignment(calibrated, raw)

    raw.loc[0, "pred_range_index"] = 11
    raw.loc[0, "range_error_gates"] = 1.0
    raw.loc[0, "localization_ok"] = True
    with pytest.raises(ValueError, match="pred_range_index"):
        validate_coordinate_alignment(calibrated, raw)


def test_pooled_summary_keeps_detection_and_localization_separate() -> None:
    summary = build_pooled_summary(target_frame()).iloc[0]
    assert summary["target_samples"] == 3
    assert summary["score_detected"] == 2
    assert summary["localization_ok_regardless_score"] == 2
    assert summary["correct_detection"] == 1
    assert summary["detected_but_not_localized"] == 1
    assert summary["localized_but_not_detected"] == 1
    assert summary["joint_pd"] == pytest.approx(1 / 3)


def test_error_distribution_reports_unconditional_and_conditional_scopes() -> None:
    distribution = build_error_distribution(target_frame()).set_index("scope")
    assert set(distribution.index) == {
        "all_targets",
        "score_detected_targets",
        "joint_success_targets",
    }
    assert distribution.loc["all_targets", "range_gates_mae"] == pytest.approx(4.0)
    assert distribution.loc[
        "score_detected_targets", "range_gates_mae"
    ] == pytest.approx(5.0)
    assert distribution.loc[
        "joint_success_targets", "velocity_bins_max"
    ] == pytest.approx(0.0)


def test_fixed_strata_cover_every_target() -> None:
    targets = target_frame()
    distance = build_stratified_summary(
        targets,
        "distance_stratum",
        distance_stratum(targets["distance_m"]),
    )
    velocity = build_stratified_summary(
        targets,
        "velocity_stratum",
        velocity_stratum(targets["velocity_mps"]),
    )
    assert distance["target_samples"].sum() == len(targets)
    assert velocity["target_samples"].sum() == len(targets)
    assert set(distance["distance_stratum"]) == {
        "1950-2040 m",
        "2070-2130 m",
        "2160-2400 m",
    }


def test_selection_requires_one_full_experiment_per_fold(tmp_path: Path) -> None:
    path = tmp_path / "selection.csv"
    pd.DataFrame(
        {
            "fold": list(range(1, 7)),
            "mode": ["full"] * 6,
            "experiment_name": [f"experiment_{fold}" for fold in range(1, 7)],
            "base_threshold": [0.5] * 6,
        }
    ).to_csv(path, index=False)
    selected = load_frozen_selection(path)
    assert selected["fold"].tolist() == list(range(1, 7))

    broken = pd.read_csv(path).iloc[:-1]
    broken.to_csv(path, index=False)
    with pytest.raises(ValueError, match="folds 1-6"):
        load_frozen_selection(path)


def test_nonempty_output_requires_overwrite(tmp_path: Path) -> None:
    output_dir = tmp_path / "localization"
    output_dir.mkdir()
    ensure_output_available(output_dir, overwrite=False)
    (output_dir / "existing.txt").write_text("frozen\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="--overwrite"):
        ensure_output_available(output_dir, overwrite=False)
