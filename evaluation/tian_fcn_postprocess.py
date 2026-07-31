from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TianDetection:
    range_index: int
    velocity_index: int
    score: float
    grid_x: int
    grid_y: int
    component_size: int
    component_count: int
    component_mean_score: float
    component_max_score: float
    component_cells: tuple[tuple[int, int], ...]
    component_bounds: tuple[tuple[int, int, int, int], ...]
    threshold: float


def _connected_components(mask: torch.Tensor) -> list[list[tuple[int, int]]]:
    if mask.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    height, width = mask.shape
    active = mask.bool().cpu()
    visited = torch.zeros_like(active)
    components: list[list[tuple[int, int]]] = []
    neighbors = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    )
    for row in range(height):
        for col in range(width):
            if not active[row, col] or visited[row, col]:
                continue
            stack = [(row, col)]
            visited[row, col] = True
            component: list[tuple[int, int]] = []
            while stack:
                current_row, current_col = stack.pop()
                component.append((current_row, current_col))
                for delta_row, delta_col in neighbors:
                    next_row = current_row + delta_row
                    next_col = current_col + delta_col
                    if not (0 <= next_row < height and 0 <= next_col < width):
                        continue
                    if active[next_row, next_col] and not visited[next_row, next_col]:
                        visited[next_row, next_col] = True
                        stack.append((next_row, next_col))
            components.append(component)
    return components


def _valid_decoded_position_mask(
    normalized_offsets: torch.Tensor,
    original_shape: tuple[int, int],
    output_stride: tuple[int, int],
) -> torch.Tensor:
    if normalized_offsets.ndim != 4 or normalized_offsets.shape[1] != 2:
        raise ValueError("normalized_offsets must have shape [B, 2, H, W]")
    stride_h, stride_w = output_stride
    original_h, original_w = original_shape
    grid_y = torch.arange(
        normalized_offsets.shape[-2],
        dtype=normalized_offsets.dtype,
        device=normalized_offsets.device,
    ).view(1, -1, 1)
    grid_x = torch.arange(
        normalized_offsets.shape[-1],
        dtype=normalized_offsets.dtype,
        device=normalized_offsets.device,
    ).view(1, 1, -1)
    decoded_range = torch.round(
        grid_x * stride_w + normalized_offsets[:, 0] * stride_w
    )
    decoded_velocity = torch.round(
        grid_y * stride_h + normalized_offsets[:, 1] * stride_h
    )
    return (
        (decoded_range >= 0)
        & (decoded_range < original_w)
        & (decoded_velocity >= 0)
        & (decoded_velocity < original_h)
    )


def tian_valid_peak_scores(
    classification_logits: torch.Tensor,
    normalized_offsets: torch.Tensor,
    original_shape: tuple[int, int],
    output_stride: tuple[int, int] = (4, 16),
) -> torch.Tensor:
    """Return the maximum score among candidates decoding inside the RD map."""

    if classification_logits.ndim != 4 or classification_logits.shape[1] != 1:
        raise ValueError("classification_logits must have shape [B, 1, H, W]")
    if classification_logits.shape[0] != normalized_offsets.shape[0]:
        raise ValueError("batch size mismatch")
    if classification_logits.shape[-2:] != normalized_offsets.shape[-2:]:
        raise ValueError("output grid shape mismatch")
    probabilities = torch.sigmoid(classification_logits.detach())[:, 0]
    valid = _valid_decoded_position_mask(
        normalized_offsets.detach(), original_shape, output_stride
    )
    masked = probabilities.masked_fill(~valid, -1.0)
    peaks = masked.flatten(1).max(dim=1).values
    return torch.clamp_min(peaks, 0.0)


def tian_pir_mdp(
    classification_logits: torch.Tensor,
    normalized_offsets: torch.Tensor,
    original_shape: tuple[int, int],
    output_stride: tuple[int, int] = (4, 16),
    probability_margin: float = 0.1,
    absolute_threshold: float | None = None,
) -> list[list[TianDetection]]:
    """Apply probability initial recognition and minimum-deviation positioning.

    ``absolute_threshold`` is the local-data adaptation that permits a
    background map to yield no detection. With ``None``, the paper's dynamic
    threshold ``max_probability - probability_margin`` is used on its own.
    """

    if classification_logits.ndim != 4 or classification_logits.shape[1] != 1:
        raise ValueError("classification_logits must have shape [B, 1, H, W]")
    if normalized_offsets.ndim != 4 or normalized_offsets.shape[1] != 2:
        raise ValueError("normalized_offsets must have shape [B, 2, H, W]")
    if classification_logits.shape[0] != normalized_offsets.shape[0]:
        raise ValueError("batch size mismatch")
    if classification_logits.shape[-2:] != normalized_offsets.shape[-2:]:
        raise ValueError("output grid shape mismatch")
    if not (0 <= probability_margin <= 1):
        raise ValueError("probability_margin must be between zero and one")
    if absolute_threshold is not None and not (0 <= absolute_threshold <= 1):
        raise ValueError("absolute_threshold must be between zero and one")

    probabilities = torch.sigmoid(classification_logits.detach()).cpu()
    offsets = normalized_offsets.detach().cpu()
    valid_positions = _valid_decoded_position_mask(
        offsets, original_shape, output_stride
    )
    stride_h, stride_w = output_stride
    original_h, original_w = original_shape
    batch_detections: list[list[TianDetection]] = []

    for batch_index in range(probabilities.shape[0]):
        probability_map = probabilities[batch_index, 0]
        valid_map = valid_positions[batch_index]
        if not valid_map.any():
            batch_detections.append([])
            continue
        dynamic_threshold = max(
            0.0,
            float(probability_map[valid_map].max().item()) - probability_margin,
        )
        threshold = (
            dynamic_threshold
            if absolute_threshold is None
            else max(dynamic_threshold, float(absolute_threshold))
        )
        components = _connected_components((probability_map > threshold) & valid_map)
        if not components:
            batch_detections.append([])
            continue

        component = max(
            components,
            key=lambda cells: sum(
                float(probability_map[row, col].item()) for row, col in cells
            ) / len(cells),
        )
        component_scores = [
            float(probability_map[row, col].item()) for row, col in component
        ]
        component_bounds = tuple(
            (
                min(row for row, _ in cells),
                max(row for row, _ in cells),
                min(col for _, col in cells),
                max(col for _, col in cells),
            )
            for cells in components
        )
        grid_y, grid_x = min(
            component,
            key=lambda cell: (
                float(offsets[batch_index, 0, cell[0], cell[1]].item()) ** 2
                + float(offsets[batch_index, 1, cell[0], cell[1]].item()) ** 2,
                -float(probability_map[cell[0], cell[1]].item()),
            ),
        )
        range_index = round(
            grid_x * stride_w
            + float(offsets[batch_index, 0, grid_y, grid_x].item()) * stride_w
        )
        velocity_index = round(
            grid_y * stride_h
            + float(offsets[batch_index, 1, grid_y, grid_x].item()) * stride_h
        )
        batch_detections.append(
            [
                TianDetection(
                    range_index=range_index,
                    velocity_index=velocity_index,
                    score=float(probability_map[grid_y, grid_x].item()),
                    grid_x=grid_x,
                    grid_y=grid_y,
                    component_size=len(component),
                    component_count=len(components),
                    component_mean_score=sum(component_scores) / len(component_scores),
                    component_max_score=max(component_scores),
                    component_cells=tuple(component),
                    component_bounds=component_bounds,
                    threshold=threshold,
                )
            ]
        )
    return batch_detections
