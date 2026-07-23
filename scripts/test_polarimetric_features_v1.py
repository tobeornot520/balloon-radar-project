#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.polarimetric_rd import (
    PolarimetricConfig,
    explicit_polarimetric_rd,
    make_polar6,
    make_ri4,
    make_ri8,
)


def main() -> None:
    rng = np.random.default_rng(42)
    h = (
        rng.normal(size=(128, 100))
        + 1j * rng.normal(size=(128, 100))
    ).astype(np.complex64)
    phase_offset = 0.6
    v = h * np.exp(1j * phase_offset)

    config = PolarimetricConfig(
        velocity_window=5,
        range_window=3,
    )
    features = explicit_polarimetric_rd(h, v, config)
    polar6 = make_polar6(features)
    ri4 = make_ri4(features)
    ri8 = make_ri8(features)

    assert polar6.shape == (6, 128, 100)
    assert ri4.shape == (4, 128, 100)
    assert ri8.shape == (8, 128, 100)
    assert np.isfinite(polar6).all()
    assert np.isfinite(ri4).all()
    assert np.isfinite(ri8).all()
    assert np.min(features["rho_hv_local"]) >= 0.0
    assert np.max(features["rho_hv_local"]) <= 1.00001

    median_zdr = float(
        np.median(features["zdr_like_db"])
    )
    median_rho = float(
        np.median(features["rho_hv_local"])
    )
    phase_cos = float(np.mean(features["phi_cos"]))
    phase_sin = float(np.mean(features["phi_sin"]))

    assert abs(median_zdr) < 1e-3
    assert median_rho > 0.99
    assert abs(phase_cos - np.cos(phase_offset)) < 0.03
    assert abs(phase_sin + np.sin(phase_offset)) < 0.03

    independent = (
        rng.normal(size=(128, 100))
        + 1j * rng.normal(size=(128, 100))
    ).astype(np.complex64)
    independent_features = explicit_polarimetric_rd(
        h,
        independent,
        config,
    )
    independent_rho = float(
        np.median(
            independent_features["rho_hv_local"]
        )
    )
    assert independent_rho < median_rho

    print("=" * 76)
    print("Polarimetric feature smoke test passed")
    print(f"polar6 shape         : {polar6.shape}")
    print(f"ri4 shape            : {ri4.shape}")
    print(f"ri8 shape            : {ri8.shape}")
    print(f"coherent median rho  : {median_rho:.6f}")
    print(f"independent median rho: {independent_rho:.6f}")
    print(f"median ZDR-like dB   : {median_zdr:.6f}")
    print("=" * 76)


if __name__ == "__main__":
    main()
