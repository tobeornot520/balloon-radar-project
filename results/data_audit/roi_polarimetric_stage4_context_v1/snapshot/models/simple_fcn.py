import torch
from torch import nn


class ConvBlock(nn.Module):
    """
    卷积、分组归一化和激活函数组成的基本模块。

    使用GroupNorm是因为后续小批量训练时，
    它通常比BatchNorm更加稳定。
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
    ) -> None:
        super().__init__()

        if output_channels % 4 != 0:
            raise ValueError(
                "输出通道数必须能被4整除"
            )

        self.block = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=4,
                num_channels=output_channels,
            ),
            nn.SiLU(inplace=True),
        )

    def forward(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        return self.block(
            input_tensor
        )


class SimpleRadarFCN(nn.Module):
    """
    雷达目标热力图定位的简单全卷积网络。

    输入：
        H/V单通道：[B, 1, 128, 100]
        H+V双通道：[B, 2, 128, 100]

    输出：
        [B, 1, 128, 100]

    输出为未经过Sigmoid的logits。
    """

    def __init__(
        self,
        in_channels: int = 2,
    ) -> None:
        super().__init__()

        if in_channels not in {1, 2}:
            raise ValueError(
                "in_channels必须为1或2"
            )

        self.in_channels = int(in_channels)

        self.network = nn.Sequential(
            ConvBlock(
                input_channels=self.in_channels,
                output_channels=16,
            ),
            ConvBlock(
                input_channels=16,
                output_channels=32,
            ),
            ConvBlock(
                input_channels=32,
                output_channels=32,
            ),
            ConvBlock(
                input_channels=32,
                output_channels=16,
            ),
            nn.Conv2d(
                in_channels=16,
                out_channels=1,
                kernel_size=1,
            ),
        )

    def forward(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        output = self.network(
            input_tensor
        )

        if output.shape[-2:] != (
            128,
            100,
        ):
            raise RuntimeError(
                "网络输出尺寸错误："
                f"{tuple(output.shape)}"
            )

        return output


def count_trainable_parameters(
    model: nn.Module,
) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )