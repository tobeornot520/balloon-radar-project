from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ThesisTian2024Detection:
    score: float
    range_index: int
    velocity_index: int
    grid_x: int
    grid_y: int


def direct_max_detections(
    classification_logits: torch.Tensor,
    normalized_offsets: torch.Tensor,
    original_shape: tuple[int, int],
    threshold: float | None = None,
    output_stride: tuple[int, int] = (4, 4),
) -> list[ThesisTian2024Detection | None]:
    """Decode the maximum classification response, without PIR or MDP."""
    if classification_logits.ndim != 4 or classification_logits.shape[1] != 1:
        raise ValueError("classification_logits must have shape [B,1,H,W]")
    if normalized_offsets.shape != (
        classification_logits.shape[0],
        2,
        classification_logits.shape[2],
        classification_logits.shape[3],
    ):
        raise ValueError("normalized_offsets shape mismatch")
    if threshold is not None and not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    probabilities = torch.sigmoid(classification_logits[:, 0])
    flat_scores, flat_indices = probabilities.reshape(probabilities.shape[0], -1).max(1)
    grid_width = int(probabilities.shape[-1])
    stride_y, stride_x = output_stride
    detections: list[ThesisTian2024Detection | None] = []
    for batch_index, (score_tensor, flat_index_tensor) in enumerate(
        zip(flat_scores, flat_indices)
    ):
        score = float(score_tensor.item())
        if threshold is not None and score <= threshold:
            detections.append(None)
            continue
        flat_index = int(flat_index_tensor.item())
        grid_y, grid_x = divmod(flat_index, grid_width)
        offset_x = float(normalized_offsets[batch_index, 0, grid_y, grid_x].item())
        offset_y = float(normalized_offsets[batch_index, 1, grid_y, grid_x].item())
        range_index = min(
            int(original_shape[1]) - 1,
            max(0, int(round((grid_x + offset_x) * stride_x))),
        )
        velocity_index = min(
            int(original_shape[0]) - 1,
            max(0, int(round((grid_y + offset_y) * stride_y))),
        )
        detections.append(
            ThesisTian2024Detection(
                score, range_index, velocity_index, grid_x, grid_y
            )
        )
    return detections
