from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


class TianFCNOutput(NamedTuple):
    classification_logits: torch.Tensor
    normalized_offsets: torch.Tensor
    original_shape: tuple[int, int]
    padded_shape: tuple[int, int]


class _TianFeatureBranch(nn.Module):
    """One task-specific branch after the shared first convolution."""

    def __init__(self) -> None:
        super().__init__()
        self.conv2 = nn.Conv2d(16, 16, kernel_size=(3, 5), padding=(1, 2))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4))
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 5), padding=(1, 2))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4))
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.activation(self.conv2(features))
        features = self.pool1(features)
        features = self.activation(self.conv3(features))
        features = self.pool2(features)
        return self.activation(self.conv4(features))


class TianFastUAVFCN(nn.Module):
    """FCN from Tian et al. (TGRS 2024), adapted to arbitrary RD sizes.

    The paper shares only Conv1 between classification and offset regression.
    The remaining feature layers are task-specific. The layer sequence has a
    theoretical receptive field of 20 Doppler bins by 72 range bins and a
    total stride of 4 by 16.

    Inputs use ``[batch, channel, doppler, range]`` ordering. Inputs that are
    not divisible by the total stride are padded on the bottom and right. The
    padding is a migration requirement for the local 128 x 100 data; it does
    not change coordinates in the original region.
    """

    output_stride = (4, 16)
    receptive_field = (20, 72)

    def __init__(
        self,
        in_channels: int = 1,
        padding_value: float = 0.0,
    ) -> None:
        super().__init__()
        if in_channels not in (1, 2):
            raise ValueError("in_channels must be 1 or 2")

        self.in_channels = int(in_channels)
        self.padding_value = float(padding_value)
        self.shared_conv1 = nn.Conv2d(
            self.in_channels,
            16,
            kernel_size=(3, 5),
            padding=(1, 2),
        )
        self.activation = nn.ReLU(inplace=True)
        self.classification_branch = _TianFeatureBranch()
        self.regression_branch = _TianFeatureBranch()
        self.classification_head = nn.Conv2d(64, 1, kernel_size=1)
        self.regression_head = nn.Conv2d(64, 2, kernel_size=1)
        self.initialize_paper_weights()

    def initialize_paper_weights(self) -> None:
        """Apply the Gaussian-weight and unit-bias initialization in the paper."""

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 1.0)

    @classmethod
    def compute_geometry(cls) -> tuple[tuple[int, int], tuple[int, int]]:
        """Return the theoretical receptive field and total output stride."""

        receptive_field = [1, 1]
        jump = [1, 1]
        layers = (
            ((3, 5), (1, 1)),
            ((3, 5), (1, 1)),
            ((2, 4), (2, 4)),
            ((3, 5), (1, 1)),
            ((2, 4), (2, 4)),
            ((3, 3), (1, 1)),
            ((1, 1), (1, 1)),
        )
        for kernel, stride in layers:
            for axis in range(2):
                receptive_field[axis] += (kernel[axis] - 1) * jump[axis]
                jump[axis] *= stride[axis]
        return tuple(receptive_field), tuple(jump)

    @classmethod
    def padded_spatial_shape(cls, height: int, width: int) -> tuple[int, int]:
        if height <= 0 or width <= 0:
            raise ValueError("spatial dimensions must be positive")
        stride_h, stride_w = cls.output_stride
        padded_h = ((height + stride_h - 1) // stride_h) * stride_h
        padded_w = ((width + stride_w - 1) // stride_w) * stride_w
        return padded_h, padded_w

    def estimate_conv_operations(
        self,
        height: int,
        width: int,
    ) -> tuple[int, int]:
        """Return convolution MACs and FLOPs for one padded input map.

        Pooling, activations, padding, and sigmoid are excluded. FLOPs use the
        common two-operations-per-MAC convention, which is recorded alongside
        the value so comparisons do not mix counting conventions.
        """

        padded_h, padded_w = self.padded_spatial_shape(height, width)
        pooled1_h, pooled1_w = padded_h // 2, padded_w // 4
        pooled2_h, pooled2_w = pooled1_h // 2, pooled1_w // 4

        shared = padded_h * padded_w * 16 * self.in_channels * 3 * 5
        branch_conv2 = padded_h * padded_w * 16 * 16 * 3 * 5
        branch_conv3 = pooled1_h * pooled1_w * 32 * 16 * 3 * 5
        branch_conv4 = pooled2_h * pooled2_w * 64 * 32 * 3 * 3
        classification_head = pooled2_h * pooled2_w * 1 * 64
        regression_head = pooled2_h * pooled2_w * 2 * 64
        macs = (
            shared
            + 2 * (branch_conv2 + branch_conv3 + branch_conv4)
            + classification_head
            + regression_head
        )
        return int(macs), int(2 * macs)

    def set_training_stage(self, stage: str) -> None:
        """Freeze parameters according to the paper's three-stage training."""

        if stage not in {"classification", "regression", "joint"}:
            raise ValueError("stage must be classification, regression, or joint")

        modules = {
            "shared": (self.shared_conv1,),
            "classification": (
                self.classification_branch,
                self.classification_head,
            ),
            "regression": (
                self.regression_branch,
                self.regression_head,
            ),
        }
        enabled = {
            "classification": {"shared", "classification"},
            "regression": {"regression"},
            "joint": set(modules),
        }[stage]
        for group, group_modules in modules.items():
            requires_grad = group in enabled
            for module in group_modules:
                for parameter in module.parameters():
                    parameter.requires_grad = requires_grad

    def forward(self, rd_map: torch.Tensor) -> TianFCNOutput:
        if rd_map.ndim != 4:
            raise ValueError(
                "rd_map must have shape [batch, channel, doppler, range]"
            )
        if rd_map.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} channels, got {rd_map.shape[1]}"
            )

        original_shape = (int(rd_map.shape[-2]), int(rd_map.shape[-1]))
        padded_shape = self.padded_spatial_shape(*original_shape)
        pad_bottom = padded_shape[0] - original_shape[0]
        pad_right = padded_shape[1] - original_shape[1]
        if pad_bottom or pad_right:
            rd_map = F.pad(
                rd_map,
                (0, pad_right, 0, pad_bottom),
                value=self.padding_value,
            )

        shared = self.activation(self.shared_conv1(rd_map))
        classification_features = self.classification_branch(shared)
        regression_features = self.regression_branch(shared)
        classification_logits = self.classification_head(classification_features)
        normalized_offsets = torch.sigmoid(
            self.regression_head(regression_features)
        )

        return TianFCNOutput(
            classification_logits=classification_logits,
            normalized_offsets=normalized_offsets,
            original_shape=original_shape,
            padded_shape=padded_shape,
        )
