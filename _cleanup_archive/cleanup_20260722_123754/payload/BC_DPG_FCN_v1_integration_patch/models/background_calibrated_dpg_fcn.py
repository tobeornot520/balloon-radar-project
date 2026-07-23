from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class BackgroundCalibrator(nn.Module):
    """Predict sample-wise temperature and bias for DPG-FCN logits."""

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

        dims = [feature_dim, *hidden_dims]
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.ReLU(inplace=True),
                ]
            )
        layers.append(nn.Linear(dims[-1], 2))

        self.network = nn.Sequential(*layers)
        self.min_temperature = float(min_temperature)
        self._initialize_identity()

    def _initialize_identity(self) -> None:
        last = self.network[-1]
        if not isinstance(last, nn.Linear):
            return

        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

        target = torch.tensor(1.0 - self.min_temperature)
        inverse_softplus = torch.log(torch.expm1(target.clamp_min(1e-6)))

        with torch.no_grad():
            last.bias[0] = inverse_softplus
            last.bias[1] = 0.0

    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        if features.ndim != 2:
            raise ValueError(
                f"features must have shape [B,F], got {tuple(features.shape)}"
            )

        params = self.network(features)
        temperature = F.softplus(params[:, :1]) + self.min_temperature
        bias = params[:, 1:2]
        return temperature, bias


class BackgroundCalibratedDPGFCN(nn.Module):
    """Wrap the current DualBranchGatedFCN without modifying its parameters.

    The current DPG-FCN receives a single tensor with shape [B,2,128,100] and
    returns a dictionary containing fusion_logits, h_logits, v_logits, and
    gate_weights. This wrapper matches that exact interface.

    For each sample, one positive temperature T and one bias b are applied to
    all range-velocity cells:

        calibrated_logits = (fusion_logits - b) / T

    Therefore, the spatial ordering and argmax are preserved.
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
        input_is_probability: bool = False,
        topk: int = 16,
        hidden_dims: Sequence[int] = (32, 16),
        min_temperature: float = 0.05,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        if topk <= 0:
            raise ValueError("topk must be positive")

        self.base_model = base_model
        self.input_is_probability = bool(input_is_probability)
        self.topk = int(topk)

        self.calibrator = BackgroundCalibrator(
            feature_dim=12,
            hidden_dims=hidden_dims,
            min_temperature=min_temperature,
        )

        if freeze_base:
            self.freeze_base()

    def freeze_base(self) -> None:
        self.base_model.eval()
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep the frozen DPG-FCN in evaluation mode while calibrator trains.
        if not any(p.requires_grad for p in self.base_model.parameters()):
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

            for value in output.values():
                if isinstance(value, Tensor) and value.ndim >= 3:
                    return value

        if isinstance(output, (tuple, list)):
            for value in output:
                if isinstance(value, Tensor) and value.ndim >= 3:
                    return value
                if isinstance(value, Mapping):
                    try:
                        return cls.extract_heatmap(value)
                    except (KeyError, ValueError):
                        pass

        raise KeyError("Could not identify the DPG-FCN fusion heatmap/logits")

    @classmethod
    def extract_gate(cls, output: Any) -> Optional[Tensor]:
        if isinstance(output, Mapping):
            tensor = cls._find_tensor(output, cls.GATE_KEYS)
            if tensor is not None:
                return tensor

        if isinstance(output, (tuple, list)):
            for value in output:
                if isinstance(value, Mapping):
                    gate = cls.extract_gate(value)
                    if gate is not None:
                        return gate
        return None

    def _topk_mean(self, tensor: Tensor) -> Tensor:
        flat = tensor.flatten(start_dim=1)
        k = min(self.topk, flat.shape[1])
        return flat.topk(k=k, dim=1).values.mean(dim=1, keepdim=True)

    def _four_stats(self, tensor: Tensor) -> Tensor:
        batched = self._as_batched_map(tensor).float()
        flat = batched.flatten(start_dim=1)

        return torch.cat(
            [
                flat.mean(dim=1, keepdim=True),
                flat.std(dim=1, keepdim=True, unbiased=False),
                flat.max(dim=1, keepdim=True).values,
                self._topk_mean(batched),
            ],
            dim=1,
        )

    @staticmethod
    def _two_gate_stats(
        gate: Optional[Tensor],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if gate is None:
            return torch.zeros(batch_size, 2, device=device, dtype=dtype)

        gate = gate.to(device=device, dtype=dtype)

        if gate.ndim == 1:
            if gate.shape[0] == batch_size:
                return torch.stack((gate, 1.0 - gate), dim=1)
            gate = gate.unsqueeze(0)

        flat = gate.reshape(gate.shape[0], -1)
        if flat.shape[0] != batch_size:
            raise ValueError(
                f"Gate batch size {flat.shape[0]} != input batch size {batch_size}"
            )

        if flat.shape[1] == 1:
            return torch.cat((flat, 1.0 - flat), dim=1)

        return flat[:, :2]

    def build_features(
        self,
        input_tensor: Tensor,
        raw_heatmap: Tensor,
        gate: Optional[Tensor],
    ) -> Tensor:
        if input_tensor.ndim != 4 or input_tensor.shape[1] != 2:
            raise ValueError(
                "input_tensor must have shape [B,2,H,W], "
                f"got {tuple(input_tensor.shape)}"
            )

        h_input = input_tensor[:, 0:1]
        v_input = input_tensor[:, 1:2]
        raw_map = self._as_batched_map(raw_heatmap)

        raw_flat = raw_map.flatten(start_dim=1)
        raw_stats = torch.cat(
            [
                raw_flat.max(dim=1, keepdim=True).values,
                self._topk_mean(raw_map),
            ],
            dim=1,
        )

        gate_stats = self._two_gate_stats(
            gate,
            batch_size=raw_map.shape[0],
            device=raw_map.device,
            dtype=raw_map.dtype,
        )

        features = torch.cat(
            [
                self._four_stats(h_input).to(raw_map),
                self._four_stats(v_input).to(raw_map),
                gate_stats,
                raw_stats,
            ],
            dim=1,
        )

        if features.shape[1] != 12:
            raise RuntimeError(
                f"Expected 12 calibration features, got {features.shape[1]}"
            )

        return features

    def _to_logits(self, heatmap: Tensor) -> Tensor:
        if not self.input_is_probability:
            return heatmap
        return torch.logit(heatmap.clamp(1e-6, 1.0 - 1e-6))

    def forward(
        self,
        input_tensor: Tensor,
        **base_kwargs: Any,
    ) -> Dict[str, Any]:
        if input_tensor.ndim != 4:
            raise ValueError(
                f"input_tensor must be 4D [B,2,H,W], got {tuple(input_tensor.shape)}"
            )
        if input_tensor.shape[1] != 2:
            raise ValueError(
                f"input_tensor must have 2 channels, got {input_tensor.shape[1]}"
            )

        with torch.set_grad_enabled(
            any(p.requires_grad for p in self.base_model.parameters())
        ):
            base_output = self.base_model(input_tensor, **base_kwargs)

        raw_heatmap = self._as_batched_map(
            self.extract_heatmap(base_output)
        )
        gate = self.extract_gate(base_output)

        features = self.build_features(
            input_tensor=input_tensor,
            raw_heatmap=raw_heatmap,
            gate=gate,
        )
        temperature, bias = self.calibrator(features)

        broadcast_shape = [raw_heatmap.shape[0]] + [1] * (
            raw_heatmap.ndim - 1
        )
        temperature_map = temperature.reshape(broadcast_shape)
        bias_map = bias.reshape(broadcast_shape)

        raw_logits = self._to_logits(raw_heatmap)
        calibrated_logits = (raw_logits - bias_map) / temperature_map
        calibrated_heatmap = torch.sigmoid(calibrated_logits)

        return {
            "base_output": base_output,
            "raw_heatmap": raw_heatmap,
            "raw_logits": raw_logits,
            "calibrated_logits": calibrated_logits,
            "calibrated_heatmap": calibrated_heatmap,
            "temperature": temperature,
            "bias": bias,
            "calibration_features": features,
            "gate_weights": gate,
        }

    def sample_score(self, calibrated_logits: Tensor) -> Tensor:
        calibrated_logits = self._as_batched_map(calibrated_logits)
        flat = calibrated_logits.flatten(start_dim=1)
        k = min(self.topk, flat.shape[1])
        return flat.topk(k=k, dim=1).values.mean(dim=1)


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
    temperature_penalty = torch.log(temperature).square().mean()
    bias_penalty = bias.square().mean()

    total_loss = (
        detection_loss
        + temperature_regularization * temperature_penalty
        + bias_regularization * bias_penalty
    )

    return total_loss, {
        "loss": total_loss.detach(),
        "detection_loss": detection_loss.detach(),
        "temperature_penalty": temperature_penalty.detach(),
        "bias_penalty": bias_penalty.detach(),
    }
