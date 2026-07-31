from __future__ import annotations

from typing import Mapping, NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_DOMAIN_DIMENSIONS = {
    "quality": 3,
    "time": 11,
    "rd": 22,
    "polar": 8,
    "tf": 12,
}


class MultiDomainFusionOutput(NamedTuple):
    embedding: torch.Tensor
    normalized_embedding: torch.Tensor
    domain_weights: torch.Tensor
    domain_embeddings: torch.Tensor


class _DomainEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class MultiDomainFeatureFusion(nn.Module):
    """Validity-masked fusion for current and future radar feature domains.

    Scalar features must be normalized with statistics fitted on training
    acquisition groups before entering this module. Missing domains are omitted
    from ``domain_inputs`` and receive exactly zero fusion weight.
    """

    def __init__(
        self,
        domain_dimensions: Mapping[str, int] | None = None,
        hidden_dim: int = 32,
        embedding_dim: int = 128,
    ) -> None:
        super().__init__()
        dimensions = dict(domain_dimensions or DEFAULT_DOMAIN_DIMENSIONS)
        if not dimensions or any(value <= 0 for value in dimensions.values()):
            raise ValueError("domain dimensions must be positive")
        if hidden_dim <= 0 or embedding_dim <= 0:
            raise ValueError("hidden and embedding dimensions must be positive")
        self.domain_names = tuple(dimensions)
        self.domain_dimensions = dimensions
        self.hidden_dim = int(hidden_dim)
        self.embedding_dim = int(embedding_dim)
        self.encoders = nn.ModuleDict(
            {
                name: _DomainEncoder(input_dim, hidden_dim)
                for name, input_dim in dimensions.items()
            }
        )
        self.gates = nn.ModuleDict(
            {name: nn.Linear(hidden_dim, 1) for name in self.domain_names}
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )

    def _batch_reference(
        self, domain_inputs: Mapping[str, torch.Tensor]
    ) -> tuple[int, torch.device, torch.dtype]:
        if not domain_inputs:
            raise ValueError("at least one domain input is required")
        unknown = set(domain_inputs) - set(self.domain_names)
        if unknown:
            raise ValueError(f"unknown feature domains: {sorted(unknown)}")
        first = next(iter(domain_inputs.values()))
        if first.ndim != 2:
            raise ValueError("domain inputs must have shape [batch, features]")
        return int(first.shape[0]), first.device, first.dtype

    def forward(
        self,
        domain_inputs: Mapping[str, torch.Tensor],
        domain_validity: torch.Tensor | None = None,
    ) -> MultiDomainFusionOutput:
        batch, device, dtype = self._batch_reference(domain_inputs)
        encoded: list[torch.Tensor] = []
        inferred_validity: list[float] = []
        for name in self.domain_names:
            if name in domain_inputs:
                values = domain_inputs[name]
                expected = (batch, self.domain_dimensions[name])
                if tuple(values.shape) != expected:
                    raise ValueError(f"{name} input must be {expected}")
                if values.device != device or values.dtype != dtype:
                    raise ValueError("all domain inputs must share device and dtype")
                if not torch.isfinite(values).all():
                    raise ValueError(f"{name} input contains NaN or Inf")
                inferred_validity.append(1.0)
            else:
                values = torch.zeros(
                    batch,
                    self.domain_dimensions[name],
                    device=device,
                    dtype=dtype,
                )
                inferred_validity.append(0.0)
            encoded.append(self.encoders[name](values))
        embeddings = torch.stack(encoded, dim=1)

        if domain_validity is None:
            validity = torch.tensor(
                inferred_validity, device=device, dtype=dtype
            ).view(1, -1).expand(batch, -1)
        else:
            expected = (batch, len(self.domain_names))
            if tuple(domain_validity.shape) != expected:
                raise ValueError(f"domain_validity must be {expected}")
            validity = domain_validity.to(device=device, dtype=dtype)
            available = torch.tensor(
                inferred_validity, device=device, dtype=dtype
            ).view(1, -1)
            validity = validity * available
        if torch.any(validity.sum(dim=1) <= 0):
            raise ValueError("each sample must have at least one valid domain")

        gate_logits = torch.cat(
            [
                self.gates[name](embeddings[:, index])
                for index, name in enumerate(self.domain_names)
            ],
            dim=1,
        )
        gate_logits = gate_logits.masked_fill(validity <= 0, -torch.inf)
        weights = F.softmax(gate_logits, dim=1)
        fused = torch.sum(embeddings * weights.unsqueeze(-1), dim=1)
        embedding = self.fusion(fused)
        return MultiDomainFusionOutput(
            embedding=embedding,
            normalized_embedding=F.normalize(embedding, p=2, dim=1),
            domain_weights=weights,
            domain_embeddings=embeddings,
        )


class MultiDomainFeatureClassifier(nn.Module):
    """Replaceable classification head for future balloon task labels."""

    def __init__(
        self,
        num_classes: int,
        *,
        domain_dimensions: Mapping[str, int] | None = None,
        hidden_dim: int = 32,
        embedding_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least two")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.encoder = MultiDomainFeatureFusion(
            domain_dimensions=domain_dimensions,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(embedding_dim, num_classes)
        )

    def forward(
        self,
        domain_inputs: Mapping[str, torch.Tensor],
        domain_validity: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoded = self.encoder(domain_inputs, domain_validity)
        return {
            "logits": self.classifier(encoded.embedding),
            "embedding": encoded.embedding,
            "normalized_embedding": encoded.normalized_embedding,
            "domain_weights": encoded.domain_weights,
            "domain_embeddings": encoded.domain_embeddings,
        }
