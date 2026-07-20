from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from models.simple_fcn import ConvBlock, SimpleRadarFCN


class DualBranchGatedFCN(nn.Module):
    """H/V独立编码、样本级门控融合的热图检测定位网络。

    输入：
        [B, 2, 128, 100]，第0通道为H功率RD，第1通道为V功率RD。

    输出字典：
        fusion_logits: [B, 1, 128, 100]
        h_logits:      [B, 1, 128, 100]
        v_logits:      [B, 1, 128, 100]
        gate_weights:  [B, 2]，依次为w_H、w_V

    H/V分支均保持与现有SimpleRadarFCN完全相同的结构，便于分别加载
    detection_h_baseline_v2与detection_v_baseline_v2的checkpoint。
    """

    def __init__(self, gate_hidden_dim: int = 16) -> None:
        super().__init__()
        if gate_hidden_dim <= 0:
            raise ValueError("gate_hidden_dim必须大于0")

        self.h_branch = SimpleRadarFCN(in_channels=1)
        self.v_branch = SimpleRadarFCN(in_channels=1)

        # 每个分支最后一个卷积块输出16通道；再拼接加权特征、差异和乘积。
        feature_channels = 16
        gate_input_dim = feature_channels * 2 + 2
        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, gate_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(gate_hidden_dim, 2),
        )
        self.fusion_head = nn.Sequential(
            ConvBlock(input_channels=feature_channels * 4, output_channels=32),
            ConvBlock(input_channels=32, output_channels=16),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    @staticmethod
    def _branch_forward(
        branch: SimpleRadarFCN,
        input_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = input_tensor
        for layer in branch.network[:-1]:
            features = layer(features)
        logits = branch.network[-1](features)
        return features, logits

    def forward(self, input_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        if input_tensor.ndim != 4:
            raise ValueError(f"输入必须为4维[B,2,H,W]，实际：{tuple(input_tensor.shape)}")
        if input_tensor.shape[1] != 2:
            raise ValueError(f"输入通道必须为2（H/V），实际：{input_tensor.shape[1]}")
        if input_tensor.shape[-2:] != (128, 100):
            raise ValueError(f"输入空间尺寸必须为(128,100)，实际：{tuple(input_tensor.shape[-2:])}")

        h_input = input_tensor[:, 0:1]
        v_input = input_tensor[:, 1:2]
        h_features, h_logits = self._branch_forward(self.h_branch, h_input)
        v_features, v_logits = self._branch_forward(self.v_branch, v_input)

        h_descriptor = h_features.mean(dim=(-2, -1))
        v_descriptor = v_features.mean(dim=(-2, -1))
        h_score = torch.sigmoid(h_logits).flatten(1).amax(dim=1, keepdim=True)
        v_score = torch.sigmoid(v_logits).flatten(1).amax(dim=1, keepdim=True)
        gate_input = torch.cat((h_descriptor, v_descriptor, h_score, v_score), dim=1)
        gate_weights = torch.softmax(self.gate(gate_input), dim=1)

        w_h = gate_weights[:, 0].view(-1, 1, 1, 1)
        w_v = gate_weights[:, 1].view(-1, 1, 1, 1)
        weighted_h = w_h * h_features
        weighted_v = w_v * v_features
        fusion_features = torch.cat(
            (
                weighted_h,
                weighted_v,
                torch.abs(h_features - v_features),
                h_features * v_features,
            ),
            dim=1,
        )
        fusion_logits = self.fusion_head(fusion_features)

        expected = (input_tensor.shape[0], 1, 128, 100)
        for name, tensor in (
            ("fusion_logits", fusion_logits),
            ("h_logits", h_logits),
            ("v_logits", v_logits),
        ):
            if tuple(tensor.shape) != expected:
                raise RuntimeError(f"{name}输出尺寸错误：{tuple(tensor.shape)}，期望：{expected}")
        return {
            "fusion_logits": fusion_logits,
            "h_logits": h_logits,
            "v_logits": v_logits,
            "gate_weights": gate_weights,
        }

    def set_branch_trainability(self, stage: str) -> None:
        """设置三阶段训练的分支可训练范围。

        warmup: 冻结H/V完整分支，仅训练门控和融合头。
        partial: 解冻每个分支后两个卷积块与辅助输出头。
        full:    解冻两个完整分支。
        """
        if stage not in {"warmup", "partial", "full"}:
            raise ValueError(f"未知训练阶段：{stage}")

        for branch in (self.h_branch, self.v_branch):
            for parameter in branch.parameters():
                parameter.requires_grad = stage == "full"
            if stage == "partial":
                # network[2]、network[3]及network[4]输出头。
                for layer in branch.network[2:]:
                    for parameter in layer.parameters():
                        parameter.requires_grad = True

        # 门控与融合头始终可训练。
        for module in (self.gate, self.fusion_head):
            for parameter in module.parameters():
                parameter.requires_grad = True

    def branch_parameters(self):
        yield from self.h_branch.parameters()
        yield from self.v_branch.parameters()

    def fusion_parameters(self):
        yield from self.gate.parameters()
        yield from self.fusion_head.parameters()


def _extract_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return value
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint
    raise ValueError("checkpoint中找不到可用的模型参数字典")


def _state_variants(state: Mapping[str, torch.Tensor]) -> list[tuple[str, dict[str, torch.Tensor]]]:
    raw = {str(key): value for key, value in state.items()}
    variants: list[tuple[str, dict[str, torch.Tensor]]] = [("raw", raw)]
    prefixes = ("module.", "model.", "network.")
    for prefix in prefixes:
        if raw and all(key.startswith(prefix) for key in raw):
            variants.append((f"strip_{prefix[:-1]}", {key[len(prefix):]: value for key, value in raw.items()}))
    return variants


def load_single_branch_checkpoint(
    branch: SimpleRadarFCN,
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """将现有单通道SimpleRadarFCN checkpoint严格加载到指定分支。"""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"找不到单通道checkpoint：{path}")
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    state = _extract_state_dict(checkpoint)
    expected = branch.state_dict()

    diagnostics: list[str] = []
    for variant_name, candidate in _state_variants(state):
        missing = sorted(set(expected) - set(candidate))
        unexpected = sorted(set(candidate) - set(expected))
        shape_mismatch = [
            key for key in set(expected).intersection(candidate)
            if tuple(expected[key].shape) != tuple(candidate[key].shape)
        ]
        if not missing and not unexpected and not shape_mismatch:
            branch.load_state_dict(candidate, strict=True)
            return {
                "path": str(path),
                "variant": variant_name,
                "epoch": checkpoint.get("epoch") if isinstance(checkpoint, Mapping) else None,
                "threshold": checkpoint.get("threshold") if isinstance(checkpoint, Mapping) else None,
            }
        diagnostics.append(
            f"{variant_name}: missing={missing[:4]}, unexpected={unexpected[:4]}, "
            f"shape_mismatch={shape_mismatch[:4]}"
        )
    raise RuntimeError(
        "checkpoint与SimpleRadarFCN单通道结构不匹配：\n" + "\n".join(diagnostics)
    )


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def count_total_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
