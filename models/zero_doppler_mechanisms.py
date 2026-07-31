from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import Tensor, nn


class SuppressionOutput(NamedTuple):
    calibrated_logits: Tensor
    suppression: Tensor


def gaussian_notch_profile(
    velocity_bins: int,
    *,
    sigma_bins: float,
    floor: float,
    center_index: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return a symmetric multiplicative odds profile in ``[floor, 1]``."""
    if velocity_bins <= 0:
        raise ValueError("velocity_bins must be positive")
    if sigma_bins <= 0:
        raise ValueError("sigma_bins must be positive")
    if not 0.0 < floor <= 1.0:
        raise ValueError("floor must be in (0, 1]")
    center = velocity_bins // 2 if center_index is None else int(center_index)
    if not 0 <= center < velocity_bins:
        raise ValueError("center_index lies outside the velocity axis")
    indices = torch.arange(velocity_bins, device=device, dtype=dtype)
    distance = indices - float(center)
    center_kernel = torch.exp(-0.5 * (distance / float(sigma_bins)).square())
    return 1.0 - (1.0 - float(floor)) * center_kernel


class FixedZeroDopplerNotch(nn.Module):
    """Apply a fixed, non-increasing zero-Doppler log-odds suppression."""

    def __init__(
        self,
        velocity_bins: int = 128,
        *,
        sigma_bins: float = 4.0,
        floor: float = 0.05,
        center_index: int | None = None,
    ) -> None:
        super().__init__()
        profile = gaussian_notch_profile(
            velocity_bins,
            sigma_bins=sigma_bins,
            floor=floor,
            center_index=center_index,
        )
        self.velocity_bins = int(velocity_bins)
        self.register_buffer("odds_profile", profile.view(1, 1, -1, 1))

    def forward(self, logits: Tensor) -> SuppressionOutput:
        if logits.ndim != 4 or logits.shape[-2] != self.velocity_bins:
            raise ValueError(
                "logits must have shape [batch, channels, velocity_bins, range]"
            )
        suppression = -torch.log(self.odds_profile).to(
            device=logits.device,
            dtype=logits.dtype,
        )
        return SuppressionOutput(
            calibrated_logits=logits - suppression,
            suppression=suppression.expand_as(logits),
        )


class ClutterAwareSuppressionHead(nn.Module):
    """Learn non-negative velocity-wise suppression from RD scene context.

    The head can only lower detector logits. It summarizes H/V context over
    range, predicts one suppression value per velocity row, and preserves the
    base detector's range ordering within each row.
    """

    def __init__(
        self,
        hidden_channels: int = 16,
        maximum_suppression: float = 4.0,
        initial_suppression: float = 0.05,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        if maximum_suppression <= 0:
            raise ValueError("maximum_suppression must be positive")
        if not 0.0 < initial_suppression < maximum_suppression:
            raise ValueError(
                "initial_suppression must be between zero and maximum_suppression"
            )
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.maximum_suppression = float(maximum_suppression)
        self.eps = float(eps)
        self.network = nn.Sequential(
            nn.Conv1d(2, hidden_channels, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, 1, kernel_size=1),
        )
        final = self.network[-1]
        if not isinstance(final, nn.Conv1d):
            raise TypeError("unexpected suppression head output layer")
        nn.init.zeros_(final.weight)
        fraction = initial_suppression / maximum_suppression
        nn.init.constant_(final.bias, math.log(fraction / (1.0 - fraction)))

    def _standardize(self, profile: Tensor) -> Tensor:
        mean = profile.mean(dim=-1, keepdim=True)
        scale = profile.std(dim=-1, keepdim=True, unbiased=False).clamp_min(self.eps)
        return (profile - mean) / scale

    def forward(self, raw_logits: Tensor, rd_context: Tensor) -> SuppressionOutput:
        if raw_logits.ndim != 4 or raw_logits.shape[1] != 1:
            raise ValueError("raw_logits must have shape [batch, 1, velocity, range]")
        if rd_context.ndim != 4:
            raise ValueError("rd_context must have shape [batch, channels, velocity, range]")
        if raw_logits.shape[0] != rd_context.shape[0] or raw_logits.shape[-2:] != rd_context.shape[-2:]:
            raise ValueError("raw_logits and rd_context spatial dimensions must match")
        channel_mean = rd_context.mean(dim=1)
        range_mean = channel_mean.mean(dim=-1)
        range_peak = channel_mean.amax(dim=-1)
        context = torch.stack(
            (self._standardize(range_mean), self._standardize(range_peak)),
            dim=1,
        )
        suppression = self.maximum_suppression * torch.sigmoid(self.network(context))
        suppression_map = suppression.unsqueeze(-1).expand_as(raw_logits)
        return SuppressionOutput(
            calibrated_logits=raw_logits - suppression_map,
            suppression=suppression_map,
        )
