from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from features.roi_polarimetric_refinement import ROI_SOURCE_CHANNELS


class PolarimetricEncoderOutput(NamedTuple):
    embedding: torch.Tensor
    normalized_embedding: torch.Tensor
    feature_map: torch.Tensor


class _BranchEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 24) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, hidden_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class PolarimetricTransferEncoder(nn.Module):
    """Task-independent encoder for calibrated or validity-masked H/V ROIs.

    The ten-channel contract matches ``build_roi_source``:

    - channels 0:2: normalized H/V power;
    - channels 2:6: real/imaginary H/V RD values;
    - channels 6:10: power-gated relative ZDR, rho, phase cosine and sine.

    Relative phase channels may be disabled through ``channel_validity`` when
    coherent H/V acquisition or phase calibration is unavailable.
    """

    input_channels = ROI_SOURCE_CHANNELS

    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.embedding_dim = int(embedding_dim)
        self.power_branch = _BranchEncoder(2)
        self.complex_branch = _BranchEncoder(4)
        self.explicit_branch = _BranchEncoder(4)
        self.fusion = nn.Sequential(
            nn.Conv2d(72, 64, kernel_size=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
        )
        self.embedding_head = nn.Sequential(
            nn.Linear(128, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
        )

    def _validate_inputs(
        self,
        inputs: torch.Tensor,
        channel_validity: torch.Tensor | None,
    ) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != len(self.input_channels):
            raise ValueError(
                f"inputs must be [B,{len(self.input_channels)},H,W], "
                f"got {tuple(inputs.shape)}"
            )
        if inputs.shape[-2] <= 0 or inputs.shape[-1] <= 0:
            raise ValueError("ROI spatial dimensions must be positive")
        if channel_validity is None:
            return inputs
        expected = (inputs.shape[0], len(self.input_channels))
        if tuple(channel_validity.shape) != expected:
            raise ValueError(
                f"channel_validity must be {expected}, got {tuple(channel_validity.shape)}"
            )
        validity = channel_validity.to(device=inputs.device, dtype=inputs.dtype)
        if not torch.logical_and(validity >= 0, validity <= 1).all():
            raise ValueError("channel_validity values must be in [0,1]")
        return inputs * validity[:, :, None, None]

    def forward(
        self,
        inputs: torch.Tensor,
        channel_validity: torch.Tensor | None = None,
    ) -> PolarimetricEncoderOutput:
        inputs = self._validate_inputs(inputs, channel_validity)
        power = self.power_branch(inputs[:, 0:2])
        complex_features = self.complex_branch(inputs[:, 2:6])
        explicit = self.explicit_branch(inputs[:, 6:10])
        feature_map = self.fusion(torch.cat((power, complex_features, explicit), dim=1))
        average = F.adaptive_avg_pool2d(feature_map, 1).flatten(1)
        maximum = F.adaptive_max_pool2d(feature_map, 1).flatten(1)
        embedding = self.embedding_head(torch.cat((average, maximum), dim=1))
        normalized = F.normalize(embedding, p=2, dim=1)
        return PolarimetricEncoderOutput(embedding, normalized, feature_map)


class PolarimetricTransferClassifier(nn.Module):
    """Replaceable task head on top of ``PolarimetricTransferEncoder``."""

    def __init__(
        self,
        num_classes: int,
        *,
        embedding_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least two")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        self.num_classes = int(num_classes)
        self.encoder = PolarimetricTransferEncoder(embedding_dim=embedding_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, self.num_classes),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        channel_validity: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoded = self.encoder(inputs, channel_validity)
        return {
            "logits": self.classifier(encoded.embedding),
            "embedding": encoded.embedding,
            "normalized_embedding": encoded.normalized_embedding,
            "feature_map": encoded.feature_map,
        }


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
