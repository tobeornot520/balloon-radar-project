from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def probability_to_logit(value: float) -> float:
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return float(torch.logit(torch.tensor(value)).item())


class DownwardShiftHead(nn.Module):
    """Predict a non-negative sample-level logit shift.

    calibrated_logits = raw_logits - shift

    The calibrated score can never exceed the raw score, and the spatial
    ordering is preserved exactly.
    """

    def __init__(
        self,
        feature_dim: int = 24,
        hidden_dims: Sequence[int] = (48, 24),
        initial_shift: float = 1e-3,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if initial_shift <= 0:
            raise ValueError("initial_shift must be positive")

        dimensions = [feature_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for input_dim, output_dim in zip(
            dimensions[:-1],
            dimensions[1:],
        ):
            layers.extend(
                (
                    nn.Linear(input_dim, output_dim),
                    nn.ReLU(inplace=True),
                )
            )
        layers.append(nn.Linear(dimensions[-1], 1))
        self.network = nn.Sequential(*layers)
        self._initialize_near_zero(initial_shift)

    def _initialize_near_zero(self, initial_shift: float) -> None:
        final_layer = self.network[-1]
        if not isinstance(final_layer, nn.Linear):
            return

        nn.init.zeros_(final_layer.weight)
        inverse_softplus = torch.log(
            torch.expm1(torch.tensor(float(initial_shift)))
        )
        nn.init.constant_(
            final_layer.bias,
            float(inverse_softplus.item()),
        )

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2:
            raise ValueError(
                f"features must be [B,F], got {tuple(features.shape)}"
            )
        return F.softplus(self.network(features))


class BackgroundTailCalibratedDPGFCN(nn.Module):
    """BC-DPG-FCN v2 with richer morphology and downward-only calibration."""

    def __init__(
        self,
        base_model: nn.Module,
        *,
        topk: int = 16,
        hidden_dims: Sequence[int] = (48, 24),
        initial_shift: float = 1e-3,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        if topk < 2:
            raise ValueError("topk must be at least 2")

        self.base_model = base_model
        self.topk = int(topk)
        self.hidden_dims = tuple(int(v) for v in hidden_dims)
        self.initial_shift = float(initial_shift)
        self.shift_head = DownwardShiftHead(
            feature_dim=24,
            hidden_dims=self.hidden_dims,
            initial_shift=self.initial_shift,
        )

        if freeze_base:
            self.freeze_base()

    def freeze_base(self) -> None:
        self.base_model.eval()
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if not any(
            parameter.requires_grad
            for parameter in self.base_model.parameters()
        ):
            self.base_model.eval()
        return self

    @staticmethod
    def _flatten(tensor: Tensor) -> Tensor:
        return tensor.float().flatten(start_dim=1)

    def _topk_values(self, tensor: Tensor) -> Tensor:
        flattened = self._flatten(tensor)
        count = min(self.topk, flattened.shape[1])
        return flattened.topk(k=count, dim=1).values

    def _four_stats(self, tensor: Tensor) -> Tensor:
        flattened = self._flatten(tensor)
        topk_values = self._topk_values(tensor)
        return torch.cat(
            (
                flattened.mean(dim=1, keepdim=True),
                flattened.std(
                    dim=1,
                    keepdim=True,
                    unbiased=False,
                ),
                flattened.max(dim=1, keepdim=True).values,
                topk_values.mean(dim=1, keepdim=True),
            ),
            dim=1,
        )

    @staticmethod
    def _peak_coordinates(logits: Tensor) -> Tuple[Tensor, Tensor]:
        flattened = logits.float().flatten(start_dim=1)
        index = flattened.argmax(dim=1)
        width = logits.shape[-1]
        row = torch.div(index, width, rounding_mode="floor")
        column = index % width
        return row, column

    @staticmethod
    def _normalized_peak_distance(
        first: Tensor,
        second: Tensor,
    ) -> Tensor:
        first_row, first_column = (
            BackgroundTailCalibratedDPGFCN._peak_coordinates(first)
        )
        second_row, second_column = (
            BackgroundTailCalibratedDPGFCN._peak_coordinates(second)
        )

        row_scale = max(first.shape[-2] - 1, 1)
        column_scale = max(first.shape[-1] - 1, 1)

        distance = torch.sqrt(
            (
                (first_row.float() - second_row.float())
                / float(row_scale)
            ).square()
            + (
                (first_column.float() - second_column.float())
                / float(column_scale)
            ).square()
        )
        return distance.unsqueeze(1)

    @staticmethod
    def _local_peak_contrast(probability: Tensor) -> Tensor:
        probability = probability.float()
        pooled = F.avg_pool2d(
            probability,
            kernel_size=5,
            stride=1,
            padding=2,
        )

        flattened = probability.flatten(start_dim=1)
        peak_index = flattened.argmax(dim=1, keepdim=True)
        peak_value = flattened.gather(1, peak_index)
        local_mean = pooled.flatten(start_dim=1).gather(
            1,
            peak_index,
        )
        return peak_value - local_mean

    def build_features(
        self,
        input_tensor: Tensor,
        base_output: Dict[str, Tensor],
    ) -> Tensor:
        if input_tensor.ndim != 4 or input_tensor.shape[1] != 2:
            raise ValueError(
                "input_tensor must be [B,2,H,W], "
                f"got {tuple(input_tensor.shape)}"
            )

        required = (
            "fusion_logits",
            "h_logits",
            "v_logits",
            "gate_weights",
        )
        missing = [key for key in required if key not in base_output]
        if missing:
            raise KeyError(f"base output missing keys: {missing}")

        h_input = input_tensor[:, 0:1].float()
        v_input = input_tensor[:, 1:2].float()

        fusion_logits = base_output["fusion_logits"].float()
        h_logits = base_output["h_logits"].float()
        v_logits = base_output["v_logits"].float()
        gates = base_output["gate_weights"].float().reshape(
            input_tensor.shape[0],
            -1,
        )[:, :2]

        fusion_probability = torch.sigmoid(fusion_logits)
        h_probability = torch.sigmoid(h_logits)
        v_probability = torch.sigmoid(v_logits)

        fusion_flat = fusion_probability.flatten(start_dim=1)
        fusion_topk = self._topk_values(fusion_probability)
        top1 = fusion_topk[:, :1]
        top2 = fusion_topk[:, 1:2]
        topk_mean = fusion_topk.mean(dim=1, keepdim=True)

        entropy = -(
            fusion_probability
            * torch.log(fusion_probability.clamp_min(1e-6))
            + (1.0 - fusion_probability)
            * torch.log(
                (1.0 - fusion_probability).clamp_min(1e-6)
            )
        ).flatten(start_dim=1).mean(dim=1, keepdim=True)

        h_max = h_probability.flatten(1).max(
            dim=1,
            keepdim=True,
        ).values
        v_max = v_probability.flatten(1).max(
            dim=1,
            keepdim=True,
        ).values

        features = torch.cat(
            (
                # 1-8: normalized H/V RD statistics
                self._four_stats(h_input),
                self._four_stats(v_input),
                # 9-10: polarization gate
                gates,
                # 11-14: fusion distribution
                fusion_flat.mean(dim=1, keepdim=True),
                fusion_flat.std(
                    dim=1,
                    keepdim=True,
                    unbiased=False,
                ),
                top1,
                topk_mean,
                # 15-19: peak/tail morphology
                top1 - top2,
                top1 - topk_mean,
                (fusion_flat > 0.5).float().mean(
                    dim=1,
                    keepdim=True,
                ),
                (fusion_flat > 0.3).float().mean(
                    dim=1,
                    keepdim=True,
                ),
                entropy,
                # 20-22: branch confidence
                h_max,
                v_max,
                torch.abs(h_max - v_max),
                # 23: H/V peak disagreement
                self._normalized_peak_distance(
                    h_logits,
                    v_logits,
                ),
                # 24: local peak contrast
                self._local_peak_contrast(
                    fusion_probability,
                ),
            ),
            dim=1,
        )

        if features.shape[1] != 24:
            raise RuntimeError(
                f"Expected 24 features, got {features.shape[1]}"
            )
        return features

    def forward(self, input_tensor: Tensor) -> Dict[str, Any]:
        base_trainable = any(
            parameter.requires_grad
            for parameter in self.base_model.parameters()
        )
        with torch.set_grad_enabled(base_trainable):
            base_output = self.base_model(input_tensor)

        with torch.autocast(
            device_type=input_tensor.device.type,
            enabled=False,
        ):
            features = self.build_features(
                input_tensor.float(),
                {
                    key: value.float()
                    for key, value in base_output.items()
                },
            )
            shift = self.shift_head(features.float())
            raw_logits = base_output["fusion_logits"].float()
            calibrated_logits = raw_logits - shift.view(
                -1,
                1,
                1,
                1,
            )

        return {
            "base_output": base_output,
            "raw_logits": raw_logits,
            "calibrated_logits": calibrated_logits,
            "raw_heatmap": torch.sigmoid(raw_logits),
            "calibrated_heatmap": torch.sigmoid(
                calibrated_logits
            ),
            "shift": shift,
            "calibration_features": features,
            "gate_weights": base_output["gate_weights"],
        }

    @staticmethod
    def sample_logits(logits: Tensor) -> Tensor:
        return logits.float().flatten(start_dim=1).amax(dim=1)


def background_tail_loss(
    sample_logits: Tensor,
    labels: Tensor,
    shift: Tensor,
    *,
    background_margin_probability: float,
    positive_floor_probability: float = 0.80,
    pairwise_margin: float = 1.0,
    background_weight: float = 1.0,
    positive_weight: float = 1.0,
    pairwise_weight: float = 0.20,
    shift_regularization: float = 0.01,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    labels = labels.float().reshape_as(sample_logits)
    background_mask = labels < 0.5
    positive_mask = ~background_mask

    zero = sample_logits.new_zeros(())
    background_margin = sample_logits.new_tensor(
        probability_to_logit(background_margin_probability)
    )
    positive_floor = sample_logits.new_tensor(
        probability_to_logit(positive_floor_probability)
    )

    if bool(background_mask.any()):
        background_logits = sample_logits[background_mask]
        background_tail = F.relu(
            background_logits - background_margin
        ).square().mean()
    else:
        background_logits = sample_logits.new_empty(0)
        background_tail = zero

    if bool(positive_mask.any()):
        positive_logits = sample_logits[positive_mask]
        positive_floor_loss = F.relu(
            positive_floor - positive_logits
        ).square().mean()
    else:
        positive_logits = sample_logits.new_empty(0)
        positive_floor_loss = zero

    if (
        background_logits.numel() > 0
        and positive_logits.numel() > 0
    ):
        pairwise = F.softplus(
            pairwise_margin
            - positive_logits[:, None]
            + background_logits[None, :]
        ).mean()
    else:
        pairwise = zero

    shift_penalty = shift.square().mean()

    total = (
        background_weight * background_tail
        + positive_weight * positive_floor_loss
        + pairwise_weight * pairwise
        + shift_regularization * shift_penalty
    )

    return total, {
        "loss": total.detach(),
        "background_tail_loss": background_tail.detach(),
        "positive_floor_loss": positive_floor_loss.detach(),
        "pairwise_loss": pairwise.detach(),
        "shift_penalty": shift_penalty.detach(),
    }
