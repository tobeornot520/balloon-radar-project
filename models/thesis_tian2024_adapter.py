from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


class ThesisTian2024Output(NamedTuple):
    classification_logits: torch.Tensor
    normalized_offsets: torch.Tensor
    original_shape: tuple[int, int]
    padded_shape: tuple[int, int]


def zscore_rd_batch(
    rd_map: torch.Tensor,
    scope: str = "batch_channel",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply the thesis input Z-score with an explicit statistics scope.

    ``batch_channel`` follows the reported batch-dependent preprocessing by
    estimating one mean and standard deviation per channel over B, H and W.
    ``sample_channel`` is the deployment-stable sensitivity alternative.
    """
    if rd_map.ndim != 4:
        raise ValueError("rd_map must have shape [batch, channel, doppler, range]")
    if scope not in {"batch_channel", "sample_channel"}:
        raise ValueError("scope must be batch_channel or sample_channel")
    if eps <= 0:
        raise ValueError("eps must be positive")
    dimensions = (0, 2, 3) if scope == "batch_channel" else (2, 3)
    mean = rd_map.mean(dim=dimensions, keepdim=True)
    variance = rd_map.var(dim=dimensions, keepdim=True, unbiased=False)
    return (rd_map - mean) / torch.sqrt(variance + eps)


class _LocalTaskBranch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv2 = nn.Conv2d(16, 16, kernel_size=(3, 5), padding=(1, 2))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 5), padding=(1, 2))
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.pool1(F.relu(self.conv2(features), inplace=True))
        features = self.pool2(F.relu(self.conv3(features), inplace=True))
        return F.relu(self.conv4(features), inplace=True)


class ThesisTian2024Adapter(nn.Module):
    """Tian2024 adaptation reported for the local 128 x 100 H/V dataset."""

    in_channels = 6
    output_stride = (4, 4)

    def __init__(
        self,
        normalization_scope: str = "batch_channel",
        normalization_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if normalization_scope not in {"batch_channel", "sample_channel"}:
            raise ValueError("invalid normalization_scope")
        self.normalization_scope = normalization_scope
        self.normalization_eps = float(normalization_eps)
        self.shared_conv1 = nn.Conv2d(6, 16, kernel_size=(3, 5), padding=(1, 2))
        self.classification_branch = _LocalTaskBranch()
        self.regression_branch = _LocalTaskBranch()
        self.classification_head = nn.Conv2d(64, 1, kernel_size=1)
        self.regression_head = nn.Conv2d(64, 2, kernel_size=1)

    @classmethod
    def padded_spatial_shape(cls, height: int, width: int) -> tuple[int, int]:
        if height <= 0 or width <= 0:
            raise ValueError("spatial dimensions must be positive")
        return (
            ((height + 3) // 4) * 4,
            ((width + 3) // 4) * 4,
        )

    def forward(self, rd_map: torch.Tensor) -> ThesisTian2024Output:
        if rd_map.ndim != 4 or rd_map.shape[1] != self.in_channels:
            raise ValueError("expected input shape [batch, 6, doppler, range]")
        original_shape = (int(rd_map.shape[-2]), int(rd_map.shape[-1]))
        padded_shape = self.padded_spatial_shape(*original_shape)
        rd_map = zscore_rd_batch(
            rd_map,
            scope=self.normalization_scope,
            eps=self.normalization_eps,
        )
        pad_bottom = padded_shape[0] - original_shape[0]
        pad_right = padded_shape[1] - original_shape[1]
        if pad_bottom or pad_right:
            rd_map = F.pad(rd_map, (0, pad_right, 0, pad_bottom))

        shared = F.relu(self.shared_conv1(rd_map), inplace=True)
        classification = self.classification_branch(shared)
        regression = self.regression_branch(shared)
        return ThesisTian2024Output(
            classification_logits=self.classification_head(classification),
            normalized_offsets=torch.sigmoid(self.regression_head(regression)),
            original_shape=original_shape,
            padded_shape=padded_shape,
        )
