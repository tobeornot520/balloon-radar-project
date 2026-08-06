from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from evaluation.tian_fcn_metrics import (
    TianMetricTolerance,
    TianPredictionRecord,
    compute_tian_metrics,
    select_validation_absolute_threshold,
)
from evaluation.tian_fcn_postprocess import tian_pir_mdp, tian_valid_peak_scores
from models.tian_fcn import TianFastUAVFCN
from scripts.run_tian_fcn_sixfold import result_complete
from training.tian_fcn_objective import TianFCNObjective, build_tian_fcn_targets


def test_tian_fcn_geometry_and_local_shape_padding() -> None:
    model = TianFastUAVFCN(in_channels=1)
    assert model.compute_geometry() == ((20, 72), (4, 16))

    output = model(torch.zeros(2, 1, 128, 100))
    assert output.original_shape == (128, 100)
    assert output.padded_shape == (128, 112)
    assert output.classification_logits.shape == (2, 1, 32, 7)
    assert output.normalized_offsets.shape == (2, 2, 32, 7)
    macs, flops = model.estimate_conv_operations(128, 100)
    assert macs > 0
    assert flops == 2 * macs


def test_tian_targets_preserve_point_offset_and_background() -> None:
    targets = build_tian_fcn_targets(
        target_present=torch.tensor([1, 0]),
        velocity_indices=torch.tensor([53, -1]),
        range_indices=torch.tensor([68, -1]),
        padded_shape=(128, 112),
    )
    assert targets.classification.shape == (2, 1, 32, 7)
    assert int(targets.classification[0].sum().item()) > 1
    assert int(targets.classification[1].sum().item()) == 0
    assert targets.regression_mask[0, 0, 13, 4]
    assert int(targets.regression_mask.sum().item()) == 1
    assert torch.isclose(
        targets.normalized_offsets[0, 0, 13, 4], torch.tensor(0.25)
    )
    assert torch.isclose(
        targets.normalized_offsets[0, 1, 13, 4], torch.tensor(0.25)
    )

    point_targets = build_tian_fcn_targets(
        target_present=torch.tensor([1, 0]),
        velocity_indices=torch.tensor([53, -1]),
        range_indices=torch.tensor([68, -1]),
        padded_shape=(128, 112),
        classification_target_mode="responsible_point",
    )
    assert int(point_targets.classification.sum().item()) == 1
    assert point_targets.classification[0, 0, 13, 4]


def test_objective_samples_background_and_backpropagates() -> None:
    torch.manual_seed(42)
    model = TianFastUAVFCN(in_channels=1)
    output = model(torch.randn(2, 1, 128, 100))
    targets = build_tian_fcn_targets(
        torch.tensor([1, 0]),
        torch.tensor([53, -1]),
        torch.tensor([68, -1]),
        output.padded_shape,
    )
    objective = TianFCNObjective(background_negative_units=8)
    loss = objective(
        output.classification_logits,
        output.normalized_offsets,
        targets,
    )
    assert loss.positive_units > 0
    assert loss.sampled_negative_units == loss.positive_units + 8
    assert loss.regression_units == 1
    assert torch.isfinite(loss.total)
    loss.total.backward()
    assert model.shared_conv1.weight.grad is not None


def test_point_gt_can_apply_preregistered_target_negative_floor() -> None:
    logits = torch.zeros((1, 1, 32, 7), requires_grad=True)
    offsets = torch.zeros((1, 2, 32, 7), requires_grad=True)
    targets = build_tian_fcn_targets(
        torch.tensor([1]),
        torch.tensor([53]),
        torch.tensor([68]),
        padded_shape=(128, 112),
        classification_target_mode="responsible_point",
    )
    objective = TianFCNObjective(target_negative_units_floor=16)
    loss = objective(logits, offsets, targets, stage="joint")
    assert loss.positive_units == 1
    assert loss.sampled_negative_units == 16


def test_point_gt_can_supervise_dense_same_range_column_negatives() -> None:
    logits = torch.zeros((2, 1, 32, 7), requires_grad=True)
    offsets = torch.zeros((2, 2, 32, 7), requires_grad=True)
    targets = build_tian_fcn_targets(
        torch.tensor([1, 0]),
        torch.tensor([53, -1]),
        torch.tensor([68, -1]),
        padded_shape=(128, 112),
        classification_target_mode="responsible_point",
    )
    objective = TianFCNObjective(
        background_negative_units=16,
        target_negative_sampling="same_range_column_dense",
    )
    loss = objective(logits, offsets, targets, stage="joint")
    assert loss.positive_units == 1
    assert loss.sampled_negative_units == 31 + 16


def test_training_stages_freeze_expected_modules() -> None:
    model = TianFastUAVFCN()
    model.set_training_stage("classification")
    assert model.shared_conv1.weight.requires_grad
    assert model.classification_head.weight.requires_grad
    assert not model.regression_head.weight.requires_grad

    model.set_training_stage("regression")
    assert not model.shared_conv1.weight.requires_grad
    assert not model.classification_head.weight.requires_grad
    assert model.regression_head.weight.requires_grad

    model.set_training_stage("joint")
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_pir_mdp_recovers_coordinate_and_can_reject_background() -> None:
    logits = torch.full((2, 1, 32, 7), -10.0)
    offsets = torch.zeros((2, 2, 32, 7))
    logits[0, 0, 13, 4] = 10.0
    offsets[0, 0, 13, 4] = 0.25
    offsets[0, 1, 13, 4] = 0.25

    detections = tian_pir_mdp(
        logits,
        offsets,
        original_shape=(128, 100),
        absolute_threshold=0.5,
    )
    assert len(detections[0]) == 1
    assert detections[0][0].range_index == 68
    assert detections[0][0].velocity_index == 53
    assert detections[1] == []

    saturated_background = tian_pir_mdp(
        torch.full((1, 1, 32, 7), 100.0),
        torch.zeros((1, 2, 32, 7)),
        original_shape=(128, 100),
        absolute_threshold=1.0,
    )
    assert saturated_background == [[]]


def test_padding_candidate_is_removed_before_pir_component_selection() -> None:
    logits = torch.full((1, 1, 32, 7), -10.0)
    offsets = torch.zeros((1, 2, 32, 7))
    logits[0, 0, 10, 6] = 10.0
    offsets[0, 0, 10, 6] = 0.9  # Decodes beyond range gate 99.
    logits[0, 0, 13, 4] = 9.0
    offsets[0, 0, 13, 4] = 0.25
    offsets[0, 1, 13, 4] = 0.25

    valid_peaks = tian_valid_peak_scores(
        logits,
        offsets,
        original_shape=(128, 100),
    )
    assert torch.isclose(valid_peaks[0], torch.sigmoid(torch.tensor(9.0)))
    detections = tian_pir_mdp(
        logits,
        offsets,
        original_shape=(128, 100),
    )
    assert detections[0][0].range_index == 68
    assert detections[0][0].velocity_index == 53


def test_mdp_uses_euclidean_deviation_in_highest_mean_component() -> None:
    logits = torch.full((1, 1, 32, 7), -10.0)
    offsets = torch.zeros((1, 2, 32, 7))
    logits[0, 0, 10, 2:4] = 10.0
    offsets[0, :, 10, 2] = torch.tensor([0.6, 0.0])
    offsets[0, :, 10, 3] = torch.tensor([0.4, 0.4])
    logits[0, 0, 20, 5] = 10.0
    offsets[0, :, 20, 5] = torch.tensor([0.25, 0.25])

    detections = tian_pir_mdp(
        logits,
        offsets,
        original_shape=(128, 100),
        absolute_threshold=0.5,
    )[0]
    assert len(detections) == 1
    assert (detections[0].grid_y, detections[0].grid_x) == (10, 3)
    assert detections[0].component_count == 2
    assert detections[0].component_size == 2
    assert detections[0].component_cells == ((10, 2), (10, 3))
    assert detections[0].component_bounds == ((10, 10, 2, 3), (20, 20, 5, 5))
    assert detections[0].component_mean_score == pytest.approx(
        float(torch.sigmoid(torch.tensor(10.0)))
    )
    assert detections[0].component_max_score == pytest.approx(
        float(torch.sigmoid(torch.tensor(10.0)))
    )


def test_validation_threshold_respects_strict_false_alarm_budget() -> None:
    threshold, curve = select_validation_absolute_threshold(
        peak_scores=[0.9, 0.8, 0.8, 0.7, 0.95],
        target_present=[0, 0, 0, 0, 1],
        max_false_alarms=1,
    )
    assert threshold == 0.8
    assert sum(score > threshold for score in (0.9, 0.8, 0.8, 0.7)) == 1
    selected = min(curve, key=lambda row: abs(float(row["threshold"]) - threshold))
    assert selected["false_alarm_count"] == 1


def test_dual_protocol_metrics_keep_false_alarm_denominators_distinct() -> None:
    records = [
        TianPredictionRecord(
            "target_ok",
            1,
            0.9,
            10,
            22,
            0.9,
            11,
            22,
            all_predicted_positions=((10, 22), (21, 22)),
        ),
        TianPredictionRecord("target_miss", 1, 0.8, None, None, None, 30, 40),
        TianPredictionRecord(
            "background_fa",
            0,
            0.7,
            2,
            3,
            0.7,
            -1,
            -1,
            all_predicted_positions=((2, 3), (5, 6)),
        ),
        TianPredictionRecord("background_ok", 0, 0.2, None, None, None, -1, -1),
    ]
    metrics = compute_tian_metrics(
        records,
        TianMetricTolerance(
            range_gates=2,
            velocity_bins=3,
            paper_distance_cells=3,
        ),
    )
    assert metrics["paper_pd"] == 0.5
    assert metrics["paper_pf"] == 0.75
    assert metrics["paper_d_min_euclidean_cells"] == 1.0
    assert metrics["paper_d_5_euclidean_cells"] == 1.0
    assert metrics["paper_d_avg_euclidean_cells"] == 5.5
    assert metrics["joint_pd"] == 0.5
    assert metrics["pfa"] == 0.5
    assert metrics["range_mae_gates"] == 1.0
    assert metrics["velocity_mae_bins"] == 0.0


def test_completed_run_requires_matching_frozen_config(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    (experiment / "tables").mkdir(parents=True)
    (experiment / "checkpoints").mkdir()
    (experiment / "checkpoints" / "best.pt").touch()
    (experiment / "tables" / "summary.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "scope": "smoke",
                "test_split_loaded": False,
                "config": {"joint_epochs": 1, "channel": "H"},
            }
        ),
        encoding="utf-8",
    )
    assert result_complete(
        experiment,
        "smoke",
        {"joint_epochs": 1, "channel": "H"},
    )
    assert not result_complete(
        experiment,
        "smoke",
        {"joint_epochs": 20, "channel": "H"},
    )


@pytest.mark.parametrize(
    ("status", "test_split_loaded"),
    [
        ("FAIL", False),
        (None, False),
        ("PASS", "false"),
        ("PASS", 0),
    ],
)
def test_completed_run_rejects_failed_or_ambiguous_summary_state(
    tmp_path: Path,
    status: str | None,
    test_split_loaded: object,
) -> None:
    experiment = tmp_path / "experiment"
    (experiment / "tables").mkdir(parents=True)
    (experiment / "checkpoints").mkdir()
    (experiment / "checkpoints" / "best.pt").touch()
    (experiment / "tables" / "summary.json").write_text(
        json.dumps(
            {
                "status": status,
                "scope": "smoke",
                "test_split_loaded": test_split_loaded,
                "config": {"joint_epochs": 1, "channel": "H"},
            }
        ),
        encoding="utf-8",
    )
    assert not result_complete(
        experiment,
        "smoke",
        {"joint_epochs": 1, "channel": "H"},
    )
