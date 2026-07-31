from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn


class DenseZeroDopplerMSE(nn.Module):
    """Heatmap MSE with explicit zero-Doppler hard-negative weighting.

    Pixels belonging to a target heatmap are protected from the extra negative
    weight. Positive samples retain the existing sample-level weighting used by
    the DPG training pipeline.
    """

    def __init__(
        self,
        *,
        positive_sample_weight: float = 10.0,
        zero_band_radius: int = 7,
        zero_negative_weight: float = 4.0,
        target_guard_level: float = 0.10,
        center_index: int | None = None,
    ) -> None:
        super().__init__()
        if positive_sample_weight <= 0:
            raise ValueError("positive_sample_weight must be positive")
        if zero_band_radius < 0:
            raise ValueError("zero_band_radius must be nonnegative")
        if zero_negative_weight < 1.0:
            raise ValueError("zero_negative_weight must be at least one")
        if not 0.0 <= target_guard_level <= 1.0:
            raise ValueError("target_guard_level must be in [0, 1]")
        self.positive_sample_weight = float(positive_sample_weight)
        self.zero_band_radius = int(zero_band_radius)
        self.zero_negative_weight = float(zero_negative_weight)
        self.target_guard_level = float(target_guard_level)
        self.center_index = center_index

    def pixel_weights(self, target: Tensor) -> Tensor:
        if target.ndim != 4:
            raise ValueError("target must have shape [batch, channels, velocity, range]")
        velocity_bins = target.shape[-2]
        center = velocity_bins // 2 if self.center_index is None else int(self.center_index)
        if not 0 <= center < velocity_bins:
            raise ValueError("center_index lies outside the velocity axis")
        index = torch.arange(velocity_bins, device=target.device)
        zero_band = (index - center).abs().le(self.zero_band_radius)
        zero_band = zero_band.view(1, 1, velocity_bins, 1)
        protected_target = target >= self.target_guard_level
        hard_negative = zero_band & ~protected_target
        return torch.where(
            hard_negative,
            torch.full_like(target, self.zero_negative_weight),
            torch.ones_like(target),
        )

    def forward(self, logits: Tensor, target: Tensor, target_present: Tensor) -> Tensor:
        if logits.shape != target.shape:
            raise ValueError("logits and target must have identical shapes")
        prediction = torch.sigmoid(logits)
        weighted_error = (prediction - target).square() * self.pixel_weights(target)
        per_sample = weighted_error.flatten(1).mean(dim=1)
        present = target_present.reshape(-1).to(device=logits.device).float()
        if len(present) != len(per_sample):
            raise ValueError("target_present batch dimension does not match logits")
        sample_weight = torch.where(
            present > 0.5,
            torch.full_like(per_sample, self.positive_sample_weight),
            torch.ones_like(per_sample),
        )
        return (per_sample * sample_weight).mean()


def clutter_aware_detection_loss(
    *,
    raw_logits: Tensor,
    calibrated_logits: Tensor,
    suppression: Tensor,
    target: Tensor,
    target_present: Tensor,
    detection_criterion: DenseZeroDopplerMSE,
    allowed_target_probability_drop: float = 0.02,
    target_keep_weight: float = 8.0,
    suppression_regularization: float = 0.01,
) -> tuple[Tensor, Mapping[str, Tensor]]:
    """Combine dense detection, target protection, and shift regularization."""
    if not 0.0 <= allowed_target_probability_drop <= 1.0:
        raise ValueError("allowed_target_probability_drop must be in [0, 1]")
    if target_keep_weight < 0 or suppression_regularization < 0:
        raise ValueError("loss weights must be nonnegative")
    if not (raw_logits.shape == calibrated_logits.shape == suppression.shape == target.shape):
        raise ValueError("all heatmap tensors must have identical shapes")
    detection = detection_criterion(calibrated_logits, target, target_present)
    present = target_present.reshape(-1, 1, 1, 1).to(target.device) > 0.5
    target_region = (target >= detection_criterion.target_guard_level) & present
    if bool(target_region.any()):
        probability_drop = torch.sigmoid(raw_logits) - torch.sigmoid(calibrated_logits)
        target_keep = torch.relu(
            probability_drop[target_region] - allowed_target_probability_drop
        ).square().mean()
    else:
        target_keep = calibrated_logits.new_zeros(())
    shift_regularization = suppression.mean()
    total = (
        detection
        + float(target_keep_weight) * target_keep
        + float(suppression_regularization) * shift_regularization
    )
    return total, {
        "detection": detection,
        "target_keep": target_keep,
        "suppression_regularization": shift_regularization,
    }
