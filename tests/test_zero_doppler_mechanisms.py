from __future__ import annotations

from argparse import Namespace

import pandas as pd
import pytest
import torch

from models.dual_branch_gated_fcn import DualBranchGatedFCN
from models.zero_doppler_mechanisms import (
    ClutterAwareSuppressionHead,
    FixedNotchResidualSuppressionHead,
    FixedZeroDopplerNotch,
)
from scripts.audit_zero_doppler_candidate_veto_v1 import evaluate_radius
from scripts.summarize_zero_doppler_mechanism_v1 import (
    aggregate_detail,
    output_label,
)
from training.zero_doppler_objectives import (
    DenseZeroDopplerMSE,
    clutter_aware_detection_loss,
    fixed_residual_detection_loss,
)
from training.train_zero_doppler_mechanism import (
    ZeroDopplerMechanismDetector,
    worst_background_group_pfa,
)


def test_fixed_notch_is_symmetric_and_never_increases_logits() -> None:
    logits = torch.ones(2, 1, 128, 5)
    output = FixedZeroDopplerNotch(sigma_bins=4.0, floor=0.05)(logits)

    assert torch.all(output.calibrated_logits <= logits)
    assert torch.all(output.suppression >= 0)
    assert output.suppression[0, 0, 64, 0] > output.suppression[0, 0, 56, 0]
    assert output.suppression[0, 0, 60, 0] == pytest.approx(
        output.suppression[0, 0, 68, 0]
    )


def test_dense_negative_weight_protects_target_region() -> None:
    criterion = DenseZeroDopplerMSE(
        zero_band_radius=2,
        zero_negative_weight=5.0,
        target_guard_level=0.1,
    )
    target = torch.zeros(1, 1, 8, 3)
    target[0, 0, 4, 1] = 1.0
    weights = criterion.pixel_weights(target)

    assert weights[0, 0, 4, 0] == 5.0
    assert weights[0, 0, 4, 1] == 1.0
    assert weights[0, 0, 0, 0] == 1.0


def test_clutter_head_and_loss_preserve_non_increasing_contract() -> None:
    torch.manual_seed(42)
    raw = torch.randn(2, 1, 16, 6, requires_grad=True)
    context = torch.randn(2, 2, 16, 6)
    target = torch.zeros_like(raw)
    target[1, 0, 9, 3] = 1.0
    present = torch.tensor([0, 1])
    head = ClutterAwareSuppressionHead(hidden_channels=4)

    output = head(raw, context)
    total, parts = clutter_aware_detection_loss(
        raw_logits=raw,
        calibrated_logits=output.calibrated_logits,
        suppression=output.suppression,
        target=target,
        target_present=present,
        detection_criterion=DenseZeroDopplerMSE(zero_band_radius=2),
    )
    total.backward()

    assert torch.all(output.calibrated_logits <= raw)
    assert set(parts) == {"detection", "target_keep", "suppression_regularization"}
    assert raw.grad is not None


def test_fixed_residual_is_bounded_and_concentrated_near_zero() -> None:
    logits = torch.ones(2, 1, 128, 5)
    context = torch.randn(2, 2, 128, 5)
    head = FixedNotchResidualSuppressionHead(
        hidden_channels=4,
        maximum_suppression=1.5,
        initial_suppression=0.1,
        zero_sigma_bins=8.0,
    )

    output = head(logits, context)

    assert torch.all(output.calibrated_logits <= logits)
    assert torch.all(output.suppression >= 0)
    assert float(output.suppression.detach().max()) <= 1.5 + 1e-6
    assert output.suppression[0, 0, 64, 0] > output.suppression[0, 0, 32, 0]


def test_fixed_residual_loss_penalizes_background_peaks_and_backpropagates() -> None:
    notched = torch.zeros(2, 1, 8, 3)
    calibrated = notched.clone().requires_grad_(True)
    suppression = torch.zeros_like(calibrated)
    target = torch.zeros_like(calibrated)
    target[1, 0, 4, 1] = 1.0
    present = torch.tensor([0, 1])

    total, parts = fixed_residual_detection_loss(
        notched_logits=notched,
        calibrated_logits=calibrated,
        residual_suppression=suppression,
        target=target,
        target_present=present,
        detection_criterion=DenseZeroDopplerMSE(zero_band_radius=2),
        background_topk=4,
    )
    total.backward()

    assert parts["background_peak"] > 0
    assert calibrated.grad is not None


def test_candidate_veto_audit_counts_tradeoff() -> None:
    frame = pd.DataFrame(
        {
            "target_present": [0, 0, 1, 1],
            "pred_velocity_index": [64, 70, 65, 80],
            "raw_detected": [True, True, True, True],
            "localization_ok": [False, False, True, True],
            "raw_joint_hit": [False, False, True, True],
        }
    )

    result = evaluate_radius(frame, radius=2, center_index=64)

    assert result["false_alarm_count"] == 1
    assert result["joint_hit_count"] == 1
    assert result["removed_false_alarm_count"] == 1
    assert result["lost_joint_hit_count"] == 1


def mechanism_args() -> Namespace:
    return Namespace(
        notch_sigma_bins=4.0,
        notch_floor=0.05,
        maximum_suppression=4.0,
        initial_suppression=0.05,
        residual_hidden_channels=4,
        residual_maximum_suppression=1.5,
        residual_initial_suppression=1e-4,
        residual_zero_sigma_bins=8.0,
    )


@pytest.mark.parametrize(
    ("mode", "expects_trainable"),
    [
        ("baseline", False),
        ("fixed_notch", False),
        ("dense_negative", True),
        ("clutter_aware", True),
        ("fixed_residual", True),
    ],
)
def test_unified_detector_exposes_expected_trainable_scope(
    mode: str, expects_trainable: bool
) -> None:
    model = ZeroDopplerMechanismDetector(
        DualBranchGatedFCN(), mode, mechanism_args()
    )

    assert bool(model.trainable_parameters()) is expects_trainable
    if mode == "dense_negative":
        assert all(
            parameter.requires_grad for parameter in model.base.fusion_head.parameters()
        )
        assert not any(
            parameter.requires_grad for parameter in model.base.h_branch.parameters()
        )
    if mode == "fixed_residual":
        assert not any(parameter.requires_grad for parameter in model.base.parameters())


def test_worst_background_group_pfa_uses_group_maximum() -> None:
    predictions = pd.DataFrame(
        {
            "source_file": ["a", "a", "b", "b", "target"],
            "target_present": [0, 0, 0, 0, 1],
            "false_alarm": [False, True, True, True, False],
        }
    )

    assert worst_background_group_pfa(predictions) == pytest.approx(1.0)


def test_mechanism_summary_uses_pooled_counts_and_worst_fold() -> None:
    detail = pd.DataFrame(
        {
            "mode": ["fixed_notch", "fixed_notch"],
            "split": ["test", "test"],
            "fold": [1, 4],
            "background_count": [100, 50],
            "positive_count": [20, 10],
            "false_alarm_count": [10, 10],
            "joint_hit_count": [19, 8],
            "pfa": [0.1, 0.2],
            "joint_pd": [0.95, 0.8],
            "roc_auc": [0.9, 0.8],
        }
    )

    result = aggregate_detail(detail).iloc[0]

    assert result["pooled_pfa"] == pytest.approx(20 / 150)
    assert result["worst_fold_pfa"] == pytest.approx(0.2)
    assert result["pooled_joint_pd"] == pytest.approx(27 / 30)


def test_mechanism_summary_output_label_is_scope_specific() -> None:
    sixfold = output_label(
        [1, 2, 3, 4, 5, 6], ["baseline", "fixed_notch"], False, None
    )
    learned = output_label(
        [1, 4], ["dense_negative", "clutter_aware"], False, None
    )

    assert sixfold != learned
    assert sixfold.startswith("development_fold01_02_03_04_05_06_")
    assert output_label([1, 4], ["baseline"], True, "smoke_fold01_04_all") == (
        "smoke_fold01_04_all"
    )


@pytest.mark.parametrize("label", ["Development", "bad label", "../escape", ""])
def test_mechanism_summary_rejects_unsafe_output_label(label: str) -> None:
    with pytest.raises(ValueError):
        output_label([1], ["baseline"], False, label)
