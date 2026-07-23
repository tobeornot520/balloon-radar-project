from __future__ import annotations

import torch
from torch import nn

from models.simple_fcn import ConvBlock


class PolarimetricRepresentationFCN(nn.Module):
    """Fixed-capacity 8-channel FCN for fair representation comparisons."""

    def __init__(self, in_channels: int = 8) -> None:
        super().__init__()
        if in_channels != 8:
            raise ValueError("This benchmark fixes in_channels=8 for fair comparison")
        self.in_channels = 8
        self.network = nn.Sequential(
            ConvBlock(8, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 32),
            ConvBlock(32, 16),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if input_tensor.ndim != 4 or input_tensor.shape[1] != 8:
            raise ValueError(f"Expected [B,8,H,W], got {tuple(input_tensor.shape)}")
        output = self.network(input_tensor)
        if output.shape[-2:] != (128, 100):
            raise RuntimeError(f"Unexpected output shape: {tuple(output.shape)}")
        return output


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
