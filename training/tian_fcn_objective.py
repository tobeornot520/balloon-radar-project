from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


class TianFCNTargets(NamedTuple):
    classification: torch.Tensor
    normalized_offsets: torch.Tensor
    regression_mask: torch.Tensor


class TianFCNLossOutput(NamedTuple):
    total: torch.Tensor
    classification: torch.Tensor
    regression: torch.Tensor
    positive_units: int
    sampled_negative_units: int
    regression_units: int


def _propagate_expanded_labels(label_map: torch.Tensor) -> torch.Tensor:
    """Propagate an expanded point label through the paper FCN geometry."""

    kernels = (
        ((3, 5), (1, 2)),
        ((3, 5), (1, 2)),
    )
    response = label_map
    for kernel_size, padding in kernels:
        kernel = torch.ones(
            (1, 1, *kernel_size),
            dtype=response.dtype,
            device=response.device,
        )
        response = F.conv2d(response, kernel, padding=padding)
    response = F.max_pool2d(response, kernel_size=(2, 4), stride=(2, 4))

    kernel = torch.ones((1, 1, 3, 5), dtype=response.dtype, device=response.device)
    response = F.conv2d(response, kernel, padding=(1, 2))
    response = F.max_pool2d(response, kernel_size=(2, 4), stride=(2, 4))

    kernel = torch.ones((1, 1, 3, 3), dtype=response.dtype, device=response.device)
    response = F.conv2d(response, kernel, padding=1)
    return response.gt(0).to(label_map.dtype)


def build_tian_fcn_targets(
    target_present: torch.Tensor,
    velocity_indices: torch.Tensor,
    range_indices: torch.Tensor,
    padded_shape: tuple[int, int],
    doppler_extent: int = 7,
    range_extent: int = 5,
    output_stride: tuple[int, int] = (4, 16),
    classification_target_mode: str = "expanded",
) -> TianFCNTargets:
    """Create expanded classification GT and point-offset regression GT.

    The paper describes a 5 x 7 range-Doppler target neighborhood but does not
    make the axis order explicit in prose. The primary protocol records the
    physically named extents explicitly: five range cells and seven Doppler
    cells. Both are configurable for the preregistered sensitivity check.
    """

    if target_present.ndim != 1:
        raise ValueError("target_present must be one-dimensional")
    if velocity_indices.shape != target_present.shape:
        raise ValueError("velocity_indices shape mismatch")
    if range_indices.shape != target_present.shape:
        raise ValueError("range_indices shape mismatch")
    if doppler_extent <= 0 or range_extent <= 0:
        raise ValueError("target extents must be positive")
    if classification_target_mode not in {"expanded", "responsible_point"}:
        raise ValueError(
            "classification_target_mode must be expanded or responsible_point"
        )

    padded_h, padded_w = padded_shape
    stride_h, stride_w = output_stride
    if padded_h % stride_h or padded_w % stride_w:
        raise ValueError("padded_shape must be divisible by output_stride")

    batch_size = int(target_present.shape[0])
    device = target_present.device
    label_map = torch.zeros(
        (batch_size, 1, padded_h, padded_w),
        dtype=torch.float32,
        device=device,
    )
    output_h = padded_h // stride_h
    output_w = padded_w // stride_w
    offsets = torch.zeros(
        (batch_size, 2, output_h, output_w),
        dtype=torch.float32,
        device=device,
    )
    regression_mask = torch.zeros(
        (batch_size, 1, output_h, output_w),
        dtype=torch.bool,
        device=device,
    )

    half_doppler = doppler_extent // 2
    half_range = range_extent // 2
    for batch_index in range(batch_size):
        if int(target_present[batch_index].item()) != 1:
            continue
        velocity = int(velocity_indices[batch_index].item())
        range_gate = int(range_indices[batch_index].item())
        if not (0 <= velocity < padded_h and 0 <= range_gate < padded_w):
            raise ValueError("target coordinate lies outside padded_shape")

        top = max(0, velocity - half_doppler)
        bottom = min(padded_h, top + doppler_extent)
        top = max(0, bottom - doppler_extent)
        left = max(0, range_gate - half_range)
        right = min(padded_w, left + range_extent)
        left = max(0, right - range_extent)
        label_map[batch_index, 0, top:bottom, left:right] = 1.0

        grid_y = velocity // stride_h
        grid_x = range_gate // stride_w
        regression_mask[batch_index, 0, grid_y, grid_x] = True
        offsets[batch_index, 0, grid_y, grid_x] = (
            range_gate - grid_x * stride_w
        ) / stride_w
        offsets[batch_index, 1, grid_y, grid_x] = (
            velocity - grid_y * stride_h
        ) / stride_h

    classification = (
        _propagate_expanded_labels(label_map)
        if classification_target_mode == "expanded"
        else regression_mask.to(label_map.dtype)
    )
    return TianFCNTargets(classification, offsets, regression_mask)


class TianFCNObjective(nn.Module):
    """Random-sampled BCE plus Smooth L1 offset regression."""

    def __init__(
        self,
        regression_weight: float = 10.0,
        background_negative_units: int = 16,
        target_negative_units_floor: int = 0,
        target_negative_sampling: str = "balanced_random",
    ) -> None:
        super().__init__()
        if regression_weight < 0:
            raise ValueError("regression_weight must be nonnegative")
        if background_negative_units <= 0:
            raise ValueError("background_negative_units must be positive")
        if target_negative_units_floor < 0:
            raise ValueError("target_negative_units_floor must be nonnegative")
        if target_negative_sampling not in {
            "balanced_random",
            "same_range_column_dense",
        }:
            raise ValueError("invalid target_negative_sampling")
        self.regression_weight = float(regression_weight)
        self.background_negative_units = int(background_negative_units)
        self.target_negative_units_floor = int(target_negative_units_floor)
        self.target_negative_sampling = target_negative_sampling

    def forward(
        self,
        classification_logits: torch.Tensor,
        normalized_offsets: torch.Tensor,
        targets: TianFCNTargets,
        stage: str = "joint",
        sample_negatives_randomly: bool = True,
    ) -> TianFCNLossOutput:
        if stage not in {"classification", "regression", "joint"}:
            raise ValueError("invalid training stage")
        if classification_logits.shape != targets.classification.shape:
            raise ValueError("classification target shape mismatch")
        if normalized_offsets.shape != targets.normalized_offsets.shape:
            raise ValueError("regression target shape mismatch")

        sampled_logits: list[torch.Tensor] = []
        sampled_labels: list[torch.Tensor] = []
        positive_units = 0
        sampled_negative_units = 0
        for batch_index in range(classification_logits.shape[0]):
            logit_map = classification_logits[batch_index, 0]
            label_map = targets.classification[batch_index, 0]
            logits = logit_map.reshape(-1)
            labels = label_map.reshape(-1)
            positive_index = torch.nonzero(labels > 0.5, as_tuple=False).flatten()
            negative_index = torch.nonzero(labels <= 0.5, as_tuple=False).flatten()
            if (
                positive_index.numel()
                and self.target_negative_sampling == "same_range_column_dense"
            ):
                positive_columns = label_map.gt(0.5).any(dim=0)
                same_column_negative = label_map.le(0.5) & positive_columns.view(1, -1)
                selected_negative = torch.nonzero(
                    same_column_negative.reshape(-1), as_tuple=False
                ).flatten()
            else:
                negative_count = (
                    max(
                        int(positive_index.numel()),
                        self.target_negative_units_floor,
                    )
                    if positive_index.numel()
                    else self.background_negative_units
                )
                negative_count = min(negative_count, int(negative_index.numel()))
                if negative_count:
                    if sample_negatives_randomly:
                        order = torch.randperm(
                            negative_index.numel(),
                            device=logits.device,
                        )
                        selected_negative = negative_index[order[:negative_count]]
                    else:
                        selected_negative = negative_index[:negative_count]
                else:
                    selected_negative = negative_index
            selected = torch.cat((positive_index, selected_negative))
            if selected.numel():
                sampled_logits.append(logits[selected])
                sampled_labels.append(labels[selected])
            positive_units += int(positive_index.numel())
            sampled_negative_units += int(selected_negative.numel())

        if sampled_logits:
            classification_loss = F.binary_cross_entropy_with_logits(
                torch.cat(sampled_logits),
                torch.cat(sampled_labels),
            )
        else:
            classification_loss = classification_logits.sum() * 0.0

        mask = targets.regression_mask.expand_as(normalized_offsets)
        if mask.any():
            regression_loss = F.smooth_l1_loss(
                normalized_offsets[mask],
                targets.normalized_offsets[mask],
            )
        else:
            regression_loss = normalized_offsets.sum() * 0.0

        if stage == "classification":
            total = classification_loss
        elif stage == "regression":
            total = regression_loss
        else:
            total = classification_loss + self.regression_weight * regression_loss

        return TianFCNLossOutput(
            total=total,
            classification=classification_loss,
            regression=regression_loss,
            positive_units=positive_units,
            sampled_negative_units=sampled_negative_units,
            regression_units=int(targets.regression_mask.sum().item()),
        )
