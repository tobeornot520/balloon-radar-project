from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class BackgroundCalibrator(nn.Module):
    """Predict a sample-wise positive temperature and scalar bias."""

    def __init__(
        self,
        feature_dim: int = 12,
        hidden_dims: Sequence[int] = (32, 16),
        min_temperature: float = 0.05,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if min_temperature <= 0:
            raise ValueError("min_temperature must be positive")

        dimensions = [feature_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for input_dim, output_dim in zip(dimensions[:-1], dimensions[1:]):
            layers.extend(
                (
                    nn.Linear(input_dim, output_dim),
                    nn.ReLU(inplace=True),
                )
            )
        layers.append(nn.Linear(dimensions[-1], 2))

        self.network = nn.Sequential(*layers)
        self.min_temperature = float(min_temperature)
        self._initialize_as_identity()

    def _initialize_as_identity(self) -> None:
        final_layer = self.network[-1]
        if not isinstance(final_layer, nn.Linear):
            return

        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

        target = torch.tensor(1.0 - self.min_temperature)
        inverse_softplus = torch.log(torch.expm1(target.clamp_min(1e-6)))
        with torch.no_grad():
            final_layer.bias[0] = inverse_softplus
            final_layer.bias[1] = 0.0

    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        if features.ndim != 2:
            raise ValueError(
                f"features must have shape [B,F], got {tuple(features.shape)}"
            )

        parameters = self.network(features)
        temperature = (
            F.softplus(parameters[:, :1]) + self.min_temperature
        )
        bias = parameters[:, 1:2]
        return temperature, bias


class BackgroundCalibratedDPGFCN(nn.Module):
    """Background-calibrated wrapper for the current DualBranchGatedFCN.

    Input:
        input_tensor: [B, 2, 128, 100]

    Base output:
        fusion_logits: [B, 1, 128, 100]
        gate_weights: [B, 2]

    The calibrator predicts one T > 0 and one b for each sample:

        calibrated_logits = (fusion_logits - b) / T

    Because T and b are spatially uniform within each sample, range-velocity
    ordering and argmax are preserved exactly.
    """

    HEATMAP_KEYS = (
        "fusion_logits",
        "heatmap",
        "fused_heatmap",
        "fusion_heatmap",
        "logits",
        "pred",
        "prediction",
        "output",
    )
    GATE_KEYS = (
        "gate_weights",
        "gate",
        "gates",
        "fusion_weights",
        "polarization_gate",
    )

    def __init__(
        self,
        base_model: nn.Module,
        *,
        topk: int = 16,
        hidden_dims: Sequence[int] = (32, 16),
        min_temperature: float = 0.05,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        if topk <= 0:
            raise ValueError("topk must be positive")

        self.base_model = base_model
        self.topk = int(topk)
        self.hidden_dims = tuple(int(value) for value in hidden_dims)
        self.min_temperature = float(min_temperature)

        self.calibrator = BackgroundCalibrator(
            feature_dim=12,
            hidden_dims=self.hidden_dims,
            min_temperature=self.min_temperature,
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
    def _as_batched_map(tensor: Tensor) -> Tensor:
        if tensor.ndim == 2:
            return tensor.unsqueeze(0).unsqueeze(0)
        if tensor.ndim == 3:
            return tensor.unsqueeze(1)
        if tensor.ndim == 4:
            return tensor
        raise ValueError(
            f"Expected [H,W], [B,H,W], or [B,C,H,W], got {tuple(tensor.shape)}"
        )

    @classmethod
    def _find_tensor(
        cls,
        mapping: Mapping[str, Any],
        keys: Iterable[str],
    ) -> Optional[Tensor]:
        for key in keys:
            value = mapping.get(key)
            if isinstance(value, Tensor):
                return value
        return None

    @classmethod
    def extract_heatmap(cls, output: Any) -> Tensor:
        if isinstance(output, Tensor):
            return output

        if isinstance(output, Mapping):
            tensor = cls._find_tensor(output, cls.HEATMAP_KEYS)
            if tensor is not None:
                return tensor

        raise KeyError(
            "Could not identify DPG-FCN fusion logits. "
            f"Expected one of: {cls.HEATMAP_KEYS}"
        )

    @classmethod
    def extract_gate(cls, output: Any) -> Optional[Tensor]:
        if isinstance(output, Mapping):
            return cls._find_tensor(output, cls.GATE_KEYS)
        return None

    def _topk_mean(self, tensor: Tensor) -> Tensor:
        flattened = tensor.flatten(start_dim=1)
        count = min(self.topk, flattened.shape[1])
        return (
            flattened.topk(k=count, dim=1)
            .values.mean(dim=1, keepdim=True)
        )

    def _four_input_statistics(self, tensor: Tensor) -> Tensor:
        batched = self._as_batched_map(tensor).float()
        flattened = batched.flatten(start_dim=1)
        return torch.cat(
            (
                flattened.mean(dim=1, keepdim=True),
                flattened.std(
                    dim=1,
                    keepdim=True,
                    unbiased=False,
                ),
                flattened.max(dim=1, keepdim=True).values,
                self._topk_mean(batched),
            ),
            dim=1,
        )

    @staticmethod
    def _two_gate_statistics(
        gate: Optional[Tensor],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if gate is None:
            return torch.zeros(
                batch_size,
                2,
                device=device,
                dtype=dtype,
            )

        gate = gate.to(device=device, dtype=dtype)
        if gate.ndim == 1:
            if gate.shape[0] == batch_size:
                return torch.stack(
                    (gate, 1.0 - gate),
                    dim=1,
                )
            gate = gate.unsqueeze(0)

        flattened = gate.reshape(gate.shape[0], -1)
        if flattened.shape[0] != batch_size:
            raise ValueError(
                "gate batch size does not match input batch size: "
                f"{flattened.shape[0]} vs {batch_size}"
            )

        if flattened.shape[1] == 1:
            return torch.cat(
                (flattened, 1.0 - flattened),
                dim=1,
            )
        return flattened[:, :2]

    def build_features(
        self,
        input_tensor: Tensor,
        raw_logits: Tensor,
        gate_weights: Optional[Tensor],
    ) -> Tensor:
        if input_tensor.ndim != 4 or input_tensor.shape[1] != 2:
            raise ValueError(
                "input_tensor must have shape [B,2,H,W], "
                f"got {tuple(input_tensor.shape)}"
            )

        h_input = input_tensor[:, 0:1]
        v_input = input_tensor[:, 1:2]
        raw_logits = self._as_batched_map(raw_logits)
        raw_probability = torch.sigmoid(raw_logits)

        raw_statistics = torch.cat(
            (
                raw_probability.flatten(1)
                .max(dim=1, keepdim=True).values,
                self._topk_mean(raw_probability),
            ),
            dim=1,
        )

        gate_statistics = self._two_gate_statistics(
            gate_weights,
            batch_size=raw_logits.shape[0],
            device=raw_logits.device,
            dtype=raw_logits.dtype,
        )

        features = torch.cat(
            (
                self._four_input_statistics(h_input).to(raw_logits),
                self._four_input_statistics(v_input).to(raw_logits),
                gate_statistics,
                raw_statistics,
            ),
            dim=1,
        )

        if features.shape[1] != 12:
            raise RuntimeError(
                f"Expected 12 calibration features, got {features.shape[1]}"
            )
        return features

    def forward(
        self,
        input_tensor: Tensor,
    ) -> Dict[str, Any]:
        if input_tensor.ndim != 4:
            raise ValueError(
                "input_tensor must be 4D [B,2,H,W], "
                f"got {tuple(input_tensor.shape)}"
            )
        if input_tensor.shape[1] != 2:
            raise ValueError(
                f"input_tensor must have 2 channels, got {input_tensor.shape[1]}"
            )

        base_trainable = any(
            parameter.requires_grad
            for parameter in self.base_model.parameters()
        )
        with torch.set_grad_enabled(base_trainable):
            base_output = self.base_model(input_tensor)

        raw_logits = self._as_batched_map(
            self.extract_heatmap(base_output)
        )
        gate_weights = self.extract_gate(base_output)

        calibration_features = self.build_features(
            input_tensor=input_tensor,
            raw_logits=raw_logits,
            gate_weights=gate_weights,
        )
        temperature, bias = self.calibrator(
            calibration_features
        )

        broadcast_shape = [raw_logits.shape[0]] + [1] * (
            raw_logits.ndim - 1
        )
        calibrated_logits = (
            raw_logits - bias.reshape(broadcast_shape)
        ) / temperature.reshape(broadcast_shape)

        return {
            "base_output": base_output,
            "raw_logits": raw_logits,
            "raw_heatmap": torch.sigmoid(raw_logits),
            "calibrated_logits": calibrated_logits,
            "calibrated_heatmap": torch.sigmoid(
                calibrated_logits
            ),
            "temperature": temperature,
            "bias": bias,
            "calibration_features": calibration_features,
            "gate_weights": gate_weights,
        }

    @staticmethod
    def sample_logits(logits: Tensor) -> Tensor:
        """Match the current detector score: max over the RD heatmap."""
        logits = BackgroundCalibratedDPGFCN._as_batched_map(
            logits
        )
        return logits.flatten(start_dim=1).amax(dim=1)


def calibration_loss(
    sample_logits: Tensor,
    labels: Tensor,
    temperature: Tensor,
    bias: Tensor,
    *,
    temperature_regularization: float = 0.01,
    bias_regularization: float = 0.01,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    labels = labels.float().reshape_as(sample_logits)

    detection_loss = F.binary_cross_entropy_with_logits(
        sample_logits,
        labels,
    )
    temperature_penalty = (
        torch.log(temperature).square().mean()
    )
    bias_penalty = bias.square().mean()

    total_loss = (
        detection_loss
        + temperature_regularization
        * temperature_penalty
        + bias_regularization
        * bias_penalty
    )

    return total_loss, {
        "loss": total_loss.detach(),
        "detection_loss": detection_loss.detach(),
        "temperature_penalty": temperature_penalty.detach(),
        "bias_penalty": bias_penalty.detach(),
    }
