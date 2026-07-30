from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


class ThesisTian2024Targets(NamedTuple):
    classification: torch.Tensor
    normalized_offsets: torch.Tensor
    regression_mask: torch.Tensor


class ThesisTian2024Loss(NamedTuple):
    total: torch.Tensor
    classification: torch.Tensor
    regression: torch.Tensor
    positive_units: int
    sampled_negative_units: int
    regression_units: int


def build_thesis_tian2024_targets(
    target_present: torch.Tensor,
    velocity_indices: torch.Tensor,
    range_indices: torch.Tensor,
    padded_shape: tuple[int, int],
    doppler_extent: int = 7,
    range_extent: int = 5,
) -> ThesisTian2024Targets:
    """Pool each 7-Doppler by 5-range positive area to a 4 x 4 grid."""
    if target_present.ndim != 1:
        raise ValueError("target_present must be one-dimensional")
    if velocity_indices.shape != target_present.shape:
        raise ValueError("velocity_indices shape mismatch")
    if range_indices.shape != target_present.shape:
        raise ValueError("range_indices shape mismatch")
    if doppler_extent <= 0 or range_extent <= 0:
        raise ValueError("target extents must be positive")
    height, width = padded_shape
    if height % 4 or width % 4:
        raise ValueError("padded_shape must be divisible by four")

    batch_size = int(target_present.shape[0])
    label_map = torch.zeros(
        (batch_size, 1, height, width),
        dtype=torch.float32,
        device=target_present.device,
    )
    offsets = torch.zeros(
        (batch_size, 2, height // 4, width // 4),
        dtype=torch.float32,
        device=target_present.device,
    )
    regression_mask = torch.zeros(
        (batch_size, 1, height // 4, width // 4),
        dtype=torch.bool,
        device=target_present.device,
    )
    half_doppler = doppler_extent // 2
    half_range = range_extent // 2
    for batch_index in range(batch_size):
        if int(target_present[batch_index].item()) != 1:
            continue
        velocity = int(velocity_indices[batch_index].item())
        range_gate = int(range_indices[batch_index].item())
        if not (0 <= velocity < height and 0 <= range_gate < width):
            raise ValueError("target coordinate lies outside padded_shape")
        top = max(0, velocity - half_doppler)
        bottom = min(height, top + doppler_extent)
        top = max(0, bottom - doppler_extent)
        left = max(0, range_gate - half_range)
        right = min(width, left + range_extent)
        left = max(0, right - range_extent)
        label_map[batch_index, 0, top:bottom, left:right] = 1.0

        grid_y, grid_x = velocity // 4, range_gate // 4
        regression_mask[batch_index, 0, grid_y, grid_x] = True
        offsets[batch_index, 0, grid_y, grid_x] = (range_gate % 4) / 4.0
        offsets[batch_index, 1, grid_y, grid_x] = (velocity % 4) / 4.0

    classification = F.max_pool2d(label_map, kernel_size=4, stride=4)
    return ThesisTian2024Targets(classification, offsets, regression_mask)


class ThesisTian2024Objective(nn.Module):
    """Balanced sampled BCE plus responsible-cell Smooth L1 regression."""

    def __init__(self, regression_weight: float = 10.0) -> None:
        super().__init__()
        if regression_weight < 0:
            raise ValueError("regression_weight must be nonnegative")
        self.regression_weight = float(regression_weight)

    def forward(
        self,
        classification_logits: torch.Tensor,
        normalized_offsets: torch.Tensor,
        targets: ThesisTian2024Targets,
        sample_negatives_randomly: bool = True,
    ) -> ThesisTian2024Loss:
        if classification_logits.shape != targets.classification.shape:
            raise ValueError("classification target shape mismatch")
        if normalized_offsets.shape != targets.normalized_offsets.shape:
            raise ValueError("regression target shape mismatch")

        logits = classification_logits.reshape(-1)
        labels = targets.classification.reshape(-1)
        positive = torch.nonzero(labels > 0.5, as_tuple=False).flatten()
        negative = torch.nonzero(labels <= 0.5, as_tuple=False).flatten()
        negative_count = min(max(int(positive.numel()), 1), int(negative.numel()))
        if sample_negatives_randomly:
            order = torch.randperm(negative.numel(), device=negative.device)
            selected_negative = negative[order[:negative_count]]
        else:
            selected_negative = negative[:negative_count]
        selected = torch.cat((positive, selected_negative))
        classification_loss = F.binary_cross_entropy_with_logits(
            logits[selected], labels[selected]
        )

        mask = targets.regression_mask.expand_as(normalized_offsets)
        if mask.any():
            regression_loss = F.smooth_l1_loss(
                normalized_offsets[mask], targets.normalized_offsets[mask]
            )
        else:
            regression_loss = normalized_offsets.sum() * 0.0
        total = classification_loss + self.regression_weight * regression_loss
        return ThesisTian2024Loss(
            total,
            classification_loss,
            regression_loss,
            int(positive.numel()),
            int(selected_negative.numel()),
            int(targets.regression_mask.sum().item()),
        )
