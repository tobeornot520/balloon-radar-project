from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from features.polarimetric_rd import (
    make_power2,
    make_ri4,
)


@dataclass(frozen=True)
class PolarimetricGateConfig:
    """Power-confidence gate for explicit H/V polarimetric channels.

    The gate is sample-adaptive and only suppresses low-power RD regions. It
    does not convert relative H/V quantities into absolutely calibrated
    polarimetric observables.
    """

    low_percentile: float = 50.0
    high_percentile: float = 99.0
    gamma: float = 1.5
    eps: float = 1e-12

    def __post_init__(self) -> None:
        if not 0.0 <= self.low_percentile < self.high_percentile <= 100.0:
            raise ValueError("Require 0 <= low_percentile < high_percentile <= 100")
        if self.gamma <= 0:
            raise ValueError("gamma must be positive")
        if self.eps <= 0:
            raise ValueError("eps must be positive")


def power_confidence(
    features: Mapping[str, np.ndarray],
    config: PolarimetricGateConfig | None = None,
) -> np.ndarray:
    """Return a soft confidence map in [0,1] from local joint H/V power."""
    config = config or PolarimetricGateConfig()
    total = (
        np.asarray(features["local_power_h"], dtype=np.float64)
        + np.asarray(features["local_power_v"], dtype=np.float64)
    )
    db = 10.0 * np.log10(total + config.eps)
    low = float(np.percentile(db, config.low_percentile))
    high = float(np.percentile(db, config.high_percentile))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(db, dtype=np.float32)
    confidence = np.clip((db - low) / (high - low), 0.0, 1.0)
    confidence = np.power(confidence, config.gamma)
    return confidence.astype(np.float32)


def gated_explicit_channels(
    features: Mapping[str, np.ndarray],
    *,
    zdr_clip_db: float = 20.0,
    gate_config: PolarimetricGateConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return confidence plus four gated explicit polarimetric channels."""
    if zdr_clip_db <= 0:
        raise ValueError("zdr_clip_db must be positive")
    confidence = power_confidence(features, gate_config)
    rho = np.clip(np.asarray(features["rho_hv_local"], dtype=np.float32), 0.0, 1.0)
    zdr = np.clip(
        np.asarray(features["zdr_like_db"], dtype=np.float32) / float(zdr_clip_db),
        -1.0,
        1.0,
    )
    phase_weight = confidence * rho
    gated_zdr = confidence * zdr
    gated_rho = confidence * rho
    gated_cos = phase_weight * np.asarray(features["phi_cos"], dtype=np.float32)
    gated_sin = phase_weight * np.asarray(features["phi_sin"], dtype=np.float32)
    return (
        confidence.astype(np.float32),
        gated_zdr.astype(np.float32),
        gated_rho.astype(np.float32),
        gated_cos.astype(np.float32),
        gated_sin.astype(np.float32),
    )


def make_polar6_gated(
    features: Mapping[str, np.ndarray],
    *,
    zdr_clip_db: float = 20.0,
    gate_config: PolarimetricGateConfig | None = None,
) -> np.ndarray:
    """Power2 plus four power-confidence-gated explicit channels."""
    power2 = make_power2(features)
    _, gated_zdr, gated_rho, gated_cos, gated_sin = gated_explicit_channels(
        features,
        zdr_clip_db=zdr_clip_db,
        gate_config=gate_config,
    )
    return np.concatenate(
        (
            power2,
            np.stack((gated_zdr, gated_rho, gated_cos, gated_sin), axis=0),
        ),
        axis=0,
    ).astype(np.float32)


def make_ri8_gated(
    features: Mapping[str, np.ndarray],
    *,
    zdr_clip_db: float = 20.0,
    gate_config: PolarimetricGateConfig | None = None,
) -> np.ndarray:
    """RI4 plus four power-confidence-gated explicit channels."""
    ri4 = make_ri4(features)
    _, gated_zdr, gated_rho, gated_cos, gated_sin = gated_explicit_channels(
        features,
        zdr_clip_db=zdr_clip_db,
        gate_config=gate_config,
    )
    return np.concatenate(
        (
            ri4,
            np.stack((gated_zdr, gated_rho, gated_cos, gated_sin), axis=0),
        ),
        axis=0,
    ).astype(np.float32)
