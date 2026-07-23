from __future__ import annotations

import torch
from torch import nn


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = 4 if out_channels % 4 == 0 else 1
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ROIPolarimetricRefiner(nn.Module):
    """Sample-independent, suppression-only candidate refiner.

    The frozen Power2 detector supplies the candidate location and raw score. This
    module only sees the local ROI and can only lower the raw logit. Consequently,
    it cannot create new fixed-threshold false alarms and it never changes the
    Power2 range/velocity estimate.
    """

    def __init__(
        self,
        in_channels: int = 8,
        hidden_dim: int = 32,
        max_suppression_logit: float = 8.0,
    ) -> None:
        super().__init__()
        if in_channels != 8:
            raise ValueError("Stage 4 fixes in_channels=8 across ROI modes")
        if hidden_dim <= 0 or max_suppression_logit <= 0:
            raise ValueError("hidden_dim and max_suppression_logit must be positive")
        self.in_channels = 8
        self.max_suppression_logit = float(max_suppression_logit)
        self.encoder = nn.Sequential(
            ConvNormAct(8, 16),
            ConvNormAct(16, 32),
            nn.AdaptiveAvgPool2d(1),
        )
        # raw_logit, raw_score, valid_fraction, confidence_mean
        self.head = nn.Sequential(
            nn.Linear(32 + 4, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.suppression_head = nn.Linear(hidden_dim, 1)
        self.quality_head = nn.Linear(hidden_dim, 1)
        # Start close to the frozen Power2 identity instead of suppressing every
        # candidate by roughly half of the allowed logit range.
        nn.init.zeros_(self.suppression_head.weight)
        nn.init.constant_(self.suppression_head.bias, -6.0)

    def forward(
        self,
        roi: torch.Tensor,
        raw_logit: torch.Tensor,
        raw_score: torch.Tensor,
        roi_valid_mask: torch.Tensor,
        polarimetric_confidence: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if roi.ndim != 4 or roi.shape[1] != 8:
            raise ValueError(f"Expected ROI [B,8,H,W], got {tuple(roi.shape)}")
        batch = roi.shape[0]
        for name, tensor in (
            ("raw_logit", raw_logit),
            ("raw_score", raw_score),
            ("polarimetric_confidence", polarimetric_confidence),
        ):
            if tensor.numel() != batch:
                raise ValueError(f"{name} must contain one value per sample")
        if roi_valid_mask.shape[0] != batch or roi_valid_mask.ndim != 4:
            raise ValueError("roi_valid_mask must be [B,1,H,W]")

        descriptor = self.encoder(roi).flatten(1)
        valid_fraction = roi_valid_mask.float().mean(dim=(-3, -2, -1), keepdim=False)
        scalar = torch.stack(
            (
                raw_logit.reshape(-1),
                raw_score.reshape(-1),
                valid_fraction.reshape(-1),
                polarimetric_confidence.reshape(-1),
            ),
            dim=1,
        )
        hidden = self.head(torch.cat((descriptor, scalar), dim=1))
        suppression = self.max_suppression_logit * torch.sigmoid(
            self.suppression_head(hidden).reshape(-1)
        )
        refined_logit = raw_logit.reshape(-1) - suppression
        refined_score = torch.sigmoid(refined_logit)
        roi_quality = torch.sigmoid(self.quality_head(hidden).reshape(-1))
        return {
            "raw_power2_score": raw_score.reshape(-1),
            "raw_power2_logit": raw_logit.reshape(-1),
            "refined_logit": refined_logit,
            "refined_score": refined_score,
            "roi_quality": roi_quality,
            "polarimetric_confidence": polarimetric_confidence.reshape(-1),
            "suppression": suppression,
            "score_shift": refined_score - raw_score.reshape(-1),
        }


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
