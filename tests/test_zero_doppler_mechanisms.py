from __future__ import annotations

from argparse import Namespace

import pandas as pd
import pytest
import torch

from models.dual_branch_gated_fcn import DualBranchGatedFCN
from models.zero_doppler_mechanisms import (
    ClutterAwareSuppressionHead,
    FixedZeroDopplerNotch,
)
from scripts.audit_zero_doppler_candidate_veto_v1 import evaluate_radius
from scripts.summarize_zero_doppler_mechanism_v1 import aggregate_detail
from training.zero_doppler_objectives import (
    DenseZeroDopplerMSE,
    clutter_aware_detection_loss,
)
from training.train_zero_doppler_mechanism import ZeroDopplerMechanismDetector


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
    )


@pytest.mark.parametrize(
    ("mode", "expects_trainable"),
    [
        ("baseline", False),
        ("fixed_notch", False),
        ("dense_negative", True),
        ("clutter_aware", True),
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
