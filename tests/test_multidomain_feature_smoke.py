from __future__ import annotations

import numpy as np
import pytest

from features.multidomain_radar_features import (
    MULTIDOMAIN_FEATURE_NAMES,
    extract_multidomain_features,
    split_multidomain_features,
)
from features.synthetic_radar import SyntheticIQConfig, make_synthetic_hv_iq
from models.multidomain_feature_fusion import DEFAULT_DOMAIN_DIMENSIONS
from scripts.run_multidomain_feature_smoke_v1 import run_smoke


def test_synthetic_hv_fixture_is_reproducible_and_aligned() -> None:
    config = SyntheticIQConfig(mode="sweep", noise_std=0.02)
    first = make_synthetic_hv_iq(config)
    second = make_synthetic_hv_iq(config)
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
        assert left.shape == (config.pulses, config.range_gates)
        assert np.iscomplexobj(left)


def test_feature_packer_matches_fusion_contract_and_rejects_bad_scalars() -> None:
    h, v = make_synthetic_hv_iq()
    features = extract_multidomain_features(h, v)
    packed = split_multidomain_features(features)
    assert {name: len(values) for name, values in packed.items()} == (
        DEFAULT_DOMAIN_DIMENSIONS
    )
    assert tuple(packed) == tuple(MULTIDOMAIN_FEATURE_NAMES)
    assert all(values.dtype == np.float32 for values in packed.values())

    missing = dict(features)
    missing.pop("tf_energy_cv")
    with pytest.raises(KeyError, match="tf_energy_cv"):
        split_multidomain_features(missing)

    non_scalar = dict(features)
    non_scalar["time_hv_coherence"] = np.array([1.0])
    with pytest.raises(ValueError, match="scalar"):
        split_multidomain_features(non_scalar)

    nonfinite = dict(features)
    nonfinite["polar_roi_rho_mean"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        split_multidomain_features(nonfinite)


def test_normalized_sweep_changes_stft_ridge_without_physical_units() -> None:
    tone_h, tone_v = make_synthetic_hv_iq(SyntheticIQConfig(mode="tone"))
    sweep_h, sweep_v = make_synthetic_hv_iq(SyntheticIQConfig(mode="sweep"))
    tone = extract_multidomain_features(tone_h, tone_v)
    sweep = extract_multidomain_features(sweep_h, sweep_v)

    assert tone["time_hv_coherence"] == pytest.approx(1.0, abs=1e-5)
    assert sweep["tf_dominant_span_normalized"] > (
        tone["tf_dominant_span_normalized"] + 0.1
    )
    assert sweep["tf_long_window_available"] == 1
    # The extractor deliberately reports normalized frequency only.
    assert "tf_dominant_span_hz" not in sweep


def test_end_to_end_data_free_smoke_is_bounded() -> None:
    summary = run_smoke()
    assert summary["status"] == "PASS"
    assert summary["model_training"] is False
    assert summary["performance_metrics"] is False
    assert summary["fusion"]["masked_polar_weight"] == 0.0

