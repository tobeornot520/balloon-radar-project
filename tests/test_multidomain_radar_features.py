from __future__ import annotations

import numpy as np
import pytest

from features.multidomain_radar_features import (
    MultiDomainFeatureConfig,
    extract_multidomain_features,
)
from models.multidomain_feature_fusion import DEFAULT_DOMAIN_DIMENSIONS


def complex_tone(
    pulses: int,
    gates: int,
    doppler_bin: int,
    gate: int,
) -> np.ndarray:
    slow_time = np.arange(pulses, dtype=np.float64)
    tone = np.exp(2j * np.pi * doppler_bin * slow_time / pulses)
    values = np.zeros((pulses, gates), dtype=np.complex64)
    values[:, gate] = tone.astype(np.complex64)
    return values


def test_tone_anchor_and_coherent_polarization_are_recovered() -> None:
    h = complex_tone(128, 16, doppler_bin=7, gate=5)
    v = h.copy()
    features = extract_multidomain_features(h, v)
    assert features["rd_peak_range_index"] == 5
    assert features["rd_peak_velocity_offset"] == 7
    assert features["time_hv_coherence"] == pytest.approx(1.0, abs=1e-5)
    assert features["time_hv_power_ratio_db"] == pytest.approx(0.0, abs=1e-5)
    assert features["polar_roi_zdr_median_db"] == pytest.approx(0.0, abs=1e-4)
    assert features["tf_long_window_available"] == 0


def test_zero_doppler_and_long_window_gates_are_explicit() -> None:
    rng = np.random.default_rng(42)
    common = np.ones((961, 7), dtype=np.complex64)
    noise = 0.01 * (
        rng.normal(size=common.shape) + 1j * rng.normal(size=common.shape)
    )
    h = common + noise
    v = 0.8 * common + noise
    features = extract_multidomain_features(h, v)
    assert features["rd_peak_at_zero_band"] == 1
    assert features["rd_zero_doppler_energy_fraction"] > 0.9
    assert features["tf_frame_count"] >= 8
    assert features["tf_long_window_available"] == 1


def test_external_candidate_anchors_local_features_without_truth() -> None:
    h = complex_tone(128, 16, doppler_bin=7, gate=5)
    h += complex_tone(128, 16, doppler_bin=-3, gate=9) * 0.5
    v = h.copy()
    features = extract_multidomain_features(
        h,
        v,
        anchor_velocity_index=61,
        anchor_range_index=9,
    )
    assert features["rd_peak_range_index"] == 5
    assert features["rd_anchor_range_index"] == 9
    assert features["rd_anchor_velocity_index"] == 61
    assert features["rd_anchor_is_external_candidate"] == 1
    assert features["rd_anchor_matches_scene_peak"] == 0


def test_extractor_domain_dimensions_match_fusion_contract() -> None:
    h = complex_tone(128, 16, doppler_bin=4, gate=7)
    features = extract_multidomain_features(h, h.copy())

    for domain, expected_dimension in DEFAULT_DOMAIN_DIMENSIONS.items():
        actual = [name for name in features if name.startswith(f"{domain}_")]
        assert len(actual) == expected_dimension


def test_feature_extraction_rejects_invalid_iq() -> None:
    with pytest.raises(TypeError):
        extract_multidomain_features(np.ones((128, 4)), np.ones((128, 4)))
    with pytest.raises(ValueError):
        MultiDomainFeatureConfig(stft_overlap_fraction=1.0)
