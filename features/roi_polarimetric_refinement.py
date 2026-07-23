from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch

from features.polarimetric_gated_rd import (
    PolarimetricGateConfig,
    gated_explicit_channels,
)
from features.polarimetric_rd import make_power2, make_ri4


ROI_SOURCE_CHANNELS = (
    "H_power",
    "V_power",
    "Re_H",
    "Im_H",
    "Re_V",
    "Im_V",
    "gated_relative_ZDR_like",
    "gated_local_rho_HV",
    "gated_cos_relative_phase",
    "gated_sin_relative_phase",
)

ROI_MODES = (
    "power2_roi_power_control",
    "power2_roi_ri4",
    "power2_roi_polar6_gated",
    "power2_roi_ri4_polar6_gated",
)


@dataclass(frozen=True)
class ROIConfig:
    velocity_radius: int = 5
    range_radius: int = 4

    def __post_init__(self) -> None:
        if self.velocity_radius < 0 or self.range_radius < 0:
            raise ValueError("ROI radii must be non-negative")

    @property
    def height(self) -> int:
        return 2 * self.velocity_radius + 1

    @property
    def width(self) -> int:
        return 2 * self.range_radius + 1


def build_roi_source(
    features: Mapping[str, np.ndarray],
    *,
    zdr_clip_db: float = 20.0,
    gate_config: PolarimetricGateConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the shared full-frame source used by every Stage-4 ROI mode.

    Returns
    -------
    source:
        Float32 tensor with shape [10, 128, 100].
    confidence:
        Float32 map with shape [1, 128, 100].
    """
    power2 = make_power2(features)
    ri4 = make_ri4(features)
    confidence, gated_zdr, gated_rho, gated_cos, gated_sin = gated_explicit_channels(
        features,
        zdr_clip_db=zdr_clip_db,
        gate_config=gate_config,
    )
    source = np.concatenate(
        (
            power2,
            ri4,
            np.stack((gated_zdr, gated_rho, gated_cos, gated_sin), axis=0),
        ),
        axis=0,
    ).astype(np.float32)
    if source.shape != (10, 128, 100):
        raise RuntimeError(f"Unexpected ROI source shape: {source.shape}")
    if not np.isfinite(source).all() or not np.isfinite(confidence).all():
        raise ValueError("ROI source contains NaN or Inf")
    return source, confidence[None].astype(np.float32)


def canonical_roi_channels(roi_source: torch.Tensor, mode: str) -> torch.Tensor:
    """Select a mode and zero-pad it to a common eight-channel ROI tensor."""
    if roi_source.ndim not in (3, 4) or roi_source.shape[-3] != 10:
        raise ValueError(
            f"roi_source must be [10,H,W] or [B,10,H,W], got {tuple(roi_source.shape)}"
        )
    if mode not in ROI_MODES:
        raise ValueError(f"Unsupported ROI mode: {mode}")
    indices = {
        "power2_roi_power_control": (0, 1),
        "power2_roi_ri4": (2, 3, 4, 5),
        "power2_roi_polar6_gated": (0, 1, 6, 7, 8, 9),
        "power2_roi_ri4_polar6_gated": (2, 3, 4, 5, 6, 7, 8, 9),
    }[mode]
    selected = roi_source[..., list(indices), :, :] if roi_source.ndim == 4 else roi_source[list(indices)]
    output_shape = (*selected.shape[:-3], 8, selected.shape[-2], selected.shape[-1])
    result = torch.zeros(output_shape, dtype=selected.dtype, device=selected.device)
    result[..., : selected.shape[-3], :, :] = selected
    return result


def crop_roi(
    array: torch.Tensor,
    velocity_index: int,
    range_index: int,
    config: ROIConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop a zero-padded ROI and return its valid mask.

    The Doppler axis is not wrapped. This makes the boundary behavior explicit and
    records invalid pixels through ``valid_mask``.
    """
    if array.ndim != 3:
        raise ValueError(f"Expected [C,H,W], got {tuple(array.shape)}")
    channels, height, width = array.shape
    out = torch.zeros(
        (channels, config.height, config.width), dtype=array.dtype, device=array.device
    )
    mask = torch.zeros((1, config.height, config.width), dtype=array.dtype, device=array.device)

    v0 = int(velocity_index) - config.velocity_radius
    r0 = int(range_index) - config.range_radius
    v1 = v0 + config.height
    r1 = r0 + config.width

    src_v0 = max(v0, 0)
    src_r0 = max(r0, 0)
    src_v1 = min(v1, height)
    src_r1 = min(r1, width)
    if src_v1 <= src_v0 or src_r1 <= src_r0:
        return out, mask

    dst_v0 = src_v0 - v0
    dst_r0 = src_r0 - r0
    dst_v1 = dst_v0 + (src_v1 - src_v0)
    dst_r1 = dst_r0 + (src_r1 - src_r0)
    out[:, dst_v0:dst_v1, dst_r0:dst_r1] = array[:, src_v0:src_v1, src_r0:src_r1]
    mask[:, dst_v0:dst_v1, dst_r0:dst_r1] = 1.0
    return out, mask


def logit_from_probability(probability: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probability = probability.clamp(eps, 1.0 - eps)
    return torch.log(probability) - torch.log1p(-probability)
