from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
    ) -> None:
        super().__init__()
        dimensions = [input_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for source, target in zip(
            dimensions[:-1],
            dimensions[1:],
        ):
            layers.extend(
                (
                    nn.Linear(source, target),
                    nn.LayerNorm(target),
                    nn.GELU(),
                )
            )
        layers.append(nn.Linear(dimensions[-1], output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, tensor: Tensor) -> Tensor:
        return self.network(tensor)


class TargetProtectedScanCalibrator(nn.Module):
    """Scan-aware, target-protected sample score calibrator.

    The model predicts:
      1. p_background: probability that the current peak is a background peak.
      2. suppression: maximum logit suppression amount.

    shift = p_background * suppression
    calibrated_logit = raw_logit - shift

    Since shift is non-negative and spatially uniform, scores never increase
    and the original DPG range-velocity argmax is preserved.
    """

    def __init__(
        self,
        sample_feature_dim: int = 24,
        group_feature_dim: int = 12,
        hidden_dims: Sequence[int] = (64, 32),
        maximum_shift: float = 3.0,
        initial_background_probability: float = 0.05,
        initial_suppression: float = 0.10,
    ) -> None:
        super().__init__()
        if maximum_shift <= 0:
            raise ValueError("maximum_shift must be positive")

        self.sample_feature_dim = int(sample_feature_dim)
        self.group_feature_dim = int(group_feature_dim)
        self.maximum_shift = float(maximum_shift)

        input_dim = (
            self.sample_feature_dim
            + self.group_feature_dim
            + 1
        )

        self.background_head = MLP(
            input_dim,
            hidden_dims,
            1,
        )
        self.suppression_head = MLP(
            input_dim,
            hidden_dims,
            1,
        )

        self._initialize_heads(
            initial_background_probability,
            initial_suppression,
        )

    @staticmethod
    def _probability_to_logit(value: float) -> float:
        value = min(max(float(value), 1e-5), 1.0 - 1e-5)
        return float(
            torch.logit(torch.tensor(value)).item()
        )

    def _initialize_heads(
        self,
        initial_background_probability: float,
        initial_suppression: float,
    ) -> None:
        background_final = self.background_head.network[-1]
        suppression_final = self.suppression_head.network[-1]

        if not isinstance(background_final, nn.Linear):
            raise TypeError("Unexpected background head final layer")
        if not isinstance(suppression_final, nn.Linear):
            raise TypeError("Unexpected suppression head final layer")

        nn.init.zeros_(background_final.weight)
        nn.init.constant_(
            background_final.bias,
            self._probability_to_logit(
                initial_background_probability
            ),
        )

        normalized_suppression = (
            float(initial_suppression) / self.maximum_shift
        )
        nn.init.zeros_(suppression_final.weight)
        nn.init.constant_(
            suppression_final.bias,
            self._probability_to_logit(
                normalized_suppression
            ),
        )

    def forward(
        self,
        sample_features: Tensor,
        group_features: Tensor,
        raw_logit: Tensor,
    ) -> Dict[str, Tensor]:
        if sample_features.ndim != 2:
            raise ValueError("sample_features must be [B,F]")
        if group_features.ndim != 2:
            raise ValueError("group_features must be [B,G]")

        raw_logit = raw_logit.float().reshape(-1, 1)
        features = torch.cat(
            (
                sample_features.float(),
                group_features.float(),
                raw_logit,
            ),
            dim=1,
        )

        background_logit = self.background_head(features)
        p_background = torch.sigmoid(background_logit)

        suppression_fraction = torch.sigmoid(
            self.suppression_head(features)
        )
        suppression = (
            self.maximum_shift * suppression_fraction
        )

        shift = p_background * suppression
        calibrated_logit = raw_logit - shift

        return {
            "background_logit": background_logit.reshape(-1),
            "p_background": p_background.reshape(-1),
            "suppression": suppression.reshape(-1),
            "shift": shift.reshape(-1),
            "raw_logit": raw_logit.reshape(-1),
            "calibrated_logit": calibrated_logit.reshape(-1),
            "raw_score": torch.sigmoid(
                raw_logit.reshape(-1)
            ),
            "calibrated_score": torch.sigmoid(
                calibrated_logit.reshape(-1)
            ),
        }


def balanced_background_classification_loss(
    background_logit: Tensor,
    labels: Tensor,
) -> Tensor:
    """Balanced BCE where background=1 and target=0."""
    labels = labels.float().reshape(-1)
    background_target = 1.0 - labels

    background_mask = background_target > 0.5
    target_mask = ~background_mask
    zero = background_logit.new_zeros(())

    if bool(background_mask.any()):
        background_loss = F.binary_cross_entropy_with_logits(
            background_logit[background_mask],
            background_target[background_mask],
        )
    else:
        background_loss = zero

    if bool(target_mask.any()):
        target_loss = F.binary_cross_entropy_with_logits(
            background_logit[target_mask],
            background_target[target_mask],
        )
    else:
        target_loss = zero

    if bool(background_mask.any()) and bool(target_mask.any()):
        return 0.5 * (background_loss + target_loss)
    return background_loss + target_loss


def target_protected_scan_loss(
    outputs: Dict[str, Tensor],
    labels: Tensor,
    *,
    background_margin_logit: float,
    allowed_target_shift: float = 0.10,
    target_keep_weight: float = 8.0,
    background_tail_weight: float = 1.0,
    background_classification_weight: float = 1.0,
    pairwise_weight: float = 0.20,
    shift_selectivity_weight: float = 0.50,
    shift_selectivity_margin: float = 0.20,
    shift_regularization: float = 0.01,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    labels = labels.float().reshape(-1)
    background_mask = labels < 0.5
    target_mask = ~background_mask
    zero = labels.new_zeros(())

    calibrated_logit = outputs["calibrated_logit"]
    shift = outputs["shift"]

    classification = balanced_background_classification_loss(
        outputs["background_logit"],
        labels,
    )

    if bool(background_mask.any()):
        background_tail = F.relu(
            calibrated_logit[background_mask]
            - float(background_margin_logit)
        ).square().mean()
        background_shift_mean = shift[
            background_mask
        ].mean()
    else:
        background_tail = zero
        background_shift_mean = zero

    if bool(target_mask.any()):
        target_excess_shift = F.relu(
            shift[target_mask] - float(allowed_target_shift)
        ).square().mean()
        target_shift_mean = shift[target_mask].mean()
    else:
        target_excess_shift = zero
        target_shift_mean = zero

    if bool(background_mask.any()) and bool(target_mask.any()):
        background_logits = calibrated_logit[
            background_mask
        ]
        target_logits = calibrated_logit[target_mask]
        pairwise = F.softplus(
            1.0
            - target_logits[:, None]
            + background_logits[None, :]
        ).mean()

        selectivity = F.relu(
            target_shift_mean
            + float(shift_selectivity_margin)
            - background_shift_mean
        ).square()
    else:
        pairwise = zero
        selectivity = zero

    regularization = shift.square().mean()

    total = (
        background_classification_weight * classification
        + background_tail_weight * background_tail
        + target_keep_weight * target_excess_shift
        + pairwise_weight * pairwise
        + shift_selectivity_weight * selectivity
        + shift_regularization * regularization
    )

    return total, {
        "loss": total.detach(),
        "background_classification_loss": (
            classification.detach()
        ),
        "background_tail_loss": background_tail.detach(),
        "target_keep_loss": target_excess_shift.detach(),
        "pairwise_loss": pairwise.detach(),
        "shift_selectivity_loss": selectivity.detach(),
        "shift_regularization": regularization.detach(),
        "background_shift_mean": (
            background_shift_mean.detach()
        ),
        "target_shift_mean": target_shift_mean.detach(),
    }
