from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class BackgroundCalibrator(nn.Module):
    def __init__(
        self,
        feature_dim: int = 12,
        hidden_dims: Sequence[int] = (32, 16),
        min_temperature: float = 0.05,
    ) -> None:
        super().__init__()
        dims = [feature_dim, *hidden_dims]
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU(inplace=True)])
        layers.append(nn.Linear(dims[-1], 2))
        self.network = nn.Sequential(*layers)
        self.min_temperature = float(min_temperature)

        last = self.network[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
            target = torch.tensor(1.0 - self.min_temperature)
            inverse_softplus = torch.log(torch.expm1(target.clamp_min(1e-6)))
            with torch.no_grad():
                last.bias[0] = inverse_softplus
                last.bias[1] = 0.0

    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        params = self.network(features)
        temperature = F.softplus(params[:, :1]) + self.min_temperature
        bias = params[:, 1:2]
        return temperature, bias


class BackgroundCalibratedDPGFCN(nn.Module):
    HEATMAP_KEYS = (
        "heatmap", "fused_heatmap", "fusion_heatmap",
        "logits", "pred", "prediction", "output",
    )
    GATE_KEYS = (
        "gate", "gates", "gate_weights",
        "fusion_weights", "polarization_gate",
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

    @staticmethod
    def _as_batched_map(tensor: Tensor) -> Tensor:
        if tensor.ndim == 2:
            return tensor.unsqueeze(0).unsqueeze(0)
        if tensor.ndim == 3:
            return tensor.unsqueeze(1)
        if tensor.ndim == 4:
            return tensor
        raise ValueError(f"Unsupported tensor shape: {tuple(tensor.shape)}")

    @classmethod
    def _find_tensor(
        cls, mapping: Mapping[str, Any], keys: Iterable[str]
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

        raise KeyError("Cannot identify base-model heatmap")

    @classmethod
    def extract_gate(cls, output: Any) -> Optional[Tensor]:
        if isinstance(output, Mapping):
            tensor = cls._find_tensor(output, cls.GATE_KEYS)
            if tensor is not None:
                return tensor

            gate_h = output.get("gate_h")
            gate_v = output.get("gate_v")
            if isinstance(gate_h, Tensor) and isinstance(gate_v, Tensor):
                return torch.stack(
                    [
                        gate_h.reshape(gate_h.shape[0], -1).mean(dim=1),
                        gate_v.reshape(gate_v.shape[0], -1).mean(dim=1),
                    ],
                    dim=1,
                )

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
        if gate.ndim == 0:
            return gate.expand(batch_size, 2)
        if gate.ndim == 1:
            if gate.shape[0] == batch_size:
                return torch.stack([gate, 1.0 - gate], dim=1)
            gate = gate.unsqueeze(0).expand(batch_size, -1)

        flat = gate.reshape(gate.shape[0], -1)
        if flat.shape[1] == 1:
            return torch.cat([flat, 1.0 - flat], dim=1)
        return flat[:, :2]

    def build_features(
        self,
        h_input: Tensor,
        v_input: Tensor,
        raw_heatmap: Tensor,
        gate: Optional[Tensor],
    ) -> Tensor:
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
            raise RuntimeError(f"Expected 12 features, got {features.shape[1]}")
        return features

    def _to_logits(self, heatmap: Tensor) -> Tensor:
        if not self.input_is_probability:
            return heatmap
        return torch.logit(heatmap.clamp(1e-6, 1.0 - 1e-6))

    def forward(
        self,
        h_input: Tensor,
        v_input: Tensor,
        **base_kwargs: Any,
    ) -> Dict[str, Any]:
        base_output = self.base_model(h_input, v_input, **base_kwargs)
        raw_heatmap = self._as_batched_map(self.extract_heatmap(base_output))
        gate = self.extract_gate(base_output)

        features = self.build_features(h_input, v_input, raw_heatmap, gate)
        temperature, bias = self.calibrator(features)

        shape = [raw_heatmap.shape[0]] + [1] * (raw_heatmap.ndim - 1)
        temperature_map = temperature.reshape(shape)
        bias_map = bias.reshape(shape)

        raw_logits = self._to_logits(raw_heatmap)
        calibrated_logits = (raw_logits - bias_map) / temperature_map

        return {
            "base_output": base_output,
            "raw_heatmap": raw_heatmap,
            "calibrated_logits": calibrated_logits,
            "calibrated_heatmap": torch.sigmoid(calibrated_logits),
            "temperature": temperature,
            "bias": bias,
            "calibration_features": features,
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
):
    labels = labels.float().reshape_as(sample_logits)
    detection = F.binary_cross_entropy_with_logits(sample_logits, labels)
    temperature_penalty = torch.log(temperature).square().mean()
    bias_penalty = bias.square().mean()

    total = (
        detection
        + temperature_regularization * temperature_penalty
        + bias_regularization * bias_penalty
    )
    return total, {
        "loss": total.detach(),
        "detection_loss": detection.detach(),
        "temperature_penalty": temperature_penalty.detach(),
        "bias_penalty": bias_penalty.detach(),
    }
