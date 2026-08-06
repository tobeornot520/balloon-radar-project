"""Deterministic IQ fixtures for data-free radar interface smoke tests.

The fixtures model only normalized slow-time phase and relative H/V gain/phase.
They are not a physical simulator, contain no labels, and must not be used to
estimate detection performance or to tune a model.  Keeping them in the
feature package gives tests and onboarding examples one shared signal recipe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class SyntheticIQConfig:
    """Configuration for a single target-like H/V IQ scene.

    Frequencies are normalized cycles per slow-time sample.  ``mode="sweep"``
    creates a deliberately time-varying Doppler ridge; it exercises STFT
    descriptors without implying a PRF or a physical micro-Doppler frequency.
    """

    pulses: int = 1024
    range_gates: int = 16
    target_gate: int = 5
    mode: Literal["tone", "sweep"] = "tone"
    carrier_frequency: float = 0.05
    sweep_start: float = -0.18
    sweep_end: float = 0.18
    h_amplitude: float = 1.0
    v_amplitude: float = 0.7
    hv_phase_rad: float = 0.4
    noise_std: float = 0.0
    seed: int = 20260806

    def __post_init__(self) -> None:
        if self.pulses < 8 or self.range_gates < 1:
            raise ValueError("pulses must be >= 8 and range_gates must be positive")
        if not 0 <= self.target_gate < self.range_gates:
            raise ValueError("target_gate must lie inside range_gates")
        if self.mode not in {"tone", "sweep"}:
            raise ValueError("mode must be 'tone' or 'sweep'")
        for name, value in (
            ("carrier_frequency", self.carrier_frequency),
            ("sweep_start", self.sweep_start),
            ("sweep_end", self.sweep_end),
        ):
            if not -0.5 < float(value) < 0.5:
                raise ValueError(f"{name} must lie strictly inside [-0.5, 0.5]")
        if self.h_amplitude <= 0 or self.v_amplitude <= 0:
            raise ValueError("H/V amplitudes must be positive")
        if self.noise_std < 0:
            raise ValueError("noise_std must be nonnegative")


def make_synthetic_hv_iq(
    config: SyntheticIQConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic ``(H, V)`` complex arrays in ``[slow, range]`` order.

    The target occupies one range gate.  H and V share the same normalized
    phase history, while V has a configurable relative amplitude and phase;
    this makes coherence and relative power checks interpretable.  Optional
    noise is independent across channels and range gates, and is seeded by the
    configuration for reproducibility.
    """
    config = config or SyntheticIQConfig()
    slow = np.arange(config.pulses, dtype=np.float64)
    if config.mode == "tone":
        phase_cycles = config.carrier_frequency * slow
    else:
        fraction = slow / float(config.pulses - 1)
        # Integrate a linear normalized-frequency sweep to phase cycles.
        phase_cycles = (
            config.sweep_start * slow
            + 0.5 * (config.sweep_end - config.sweep_start) * slow * fraction
        )
    waveform = np.exp(2j * np.pi * phase_cycles)

    h = np.zeros((config.pulses, config.range_gates), dtype=np.complex64)
    v = np.zeros_like(h)
    h[:, config.target_gate] = config.h_amplitude * waveform
    v[:, config.target_gate] = (
        config.v_amplitude
        * waveform
        * np.exp(1j * config.hv_phase_rad)
    )

    if config.noise_std:
        rng = np.random.default_rng(config.seed)
        scale = float(config.noise_std) / np.sqrt(2.0)
        for array in (h, v):
            noise = scale * (
                rng.normal(size=array.shape) + 1j * rng.normal(size=array.shape)
            )
            array += noise.astype(np.complex64)
    return h, v

