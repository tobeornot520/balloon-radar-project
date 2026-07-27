from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from features.scan_context import build_scan_context_features
from training.train_target_protected_scan_calibrator import build_group_features


def sample_inputs() -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.DataFrame(
        {
            "sample_id": ["g1_b1", "g1_b2", "g1_b3", "g2_b1"],
            "scan_group": ["g1", "g1", "g1", "g2"],
            "raw_score": [0.2, 0.4, 0.8, 0.6],
            "beam_layer": [1, 1, 2, 1],
            "azimuth_deg": [20.0, 10.0, 5.0, 1.0],
        }
    )
    features = np.zeros((4, 24), dtype=np.float32)
    features[:, 21] = [1.0, 2.0, 3.0, 4.0]
    features[:, 22] = [0.1, 0.2, 0.3, 0.4]
    return frame, features


def test_complete_scan_context_matches_training_wrapper() -> None:
    frame, features = sample_inputs()
    result = build_scan_context_features(
        frame,
        features,
        0.5,
        mode="complete_scan",
    )
    wrapped = build_group_features(frame, features, 0.5)
    np.testing.assert_array_equal(result.values, wrapped)
    assert result.used_history_counts.tolist() == [3, 3, 3, 1]
    assert result.values[0, 0] == pytest.approx(
        math.log1p(3) / math.log1p(256)
    )
    assert result.values[0, 1] == pytest.approx((0.2 + 0.4 + 0.8) / 3)


def test_leave_one_out_excludes_current_sample() -> None:
    frame, features = sample_inputs()
    result = build_scan_context_features(
        frame,
        features,
        0.5,
        mode="leave_one_out",
    )
    assert result.used_history_counts.tolist() == [2, 2, 2, 0]
    assert result.values[0, 1] == pytest.approx((0.4 + 0.8) / 2)
    assert result.values[2, 1] == pytest.approx((0.2 + 0.4) / 2)
    np.testing.assert_array_equal(result.values[3], np.zeros(12, dtype=np.float32))


def test_past_only_respects_order_and_window() -> None:
    frame, features = sample_inputs()
    result = build_scan_context_features(
        frame,
        features,
        0.5,
        mode="past_only",
        window_size=1,
    )
    # g1 order is b2 (az=10), b1 (az=20), then b3 (beam=2).
    assert result.available_history_counts.tolist() == [1, 0, 2, 0]
    assert result.used_history_counts.tolist() == [1, 0, 1, 0]
    assert result.values[0, 1] == pytest.approx(0.4)
    assert result.values[2, 1] == pytest.approx(0.2)


def test_past_only_does_not_change_when_future_samples_change() -> None:
    frame, features = sample_inputs()
    first = build_scan_context_features(
        frame,
        features,
        0.5,
        mode="past_only",
    )
    changed = frame.copy()
    changed.loc[2, "raw_score"] = 0.01
    changed_features = features.copy()
    changed_features[2, 21:23] = 99.0
    second = build_scan_context_features(
        changed,
        changed_features,
        0.5,
        mode="past_only",
    )
    np.testing.assert_array_equal(first.values[:2], second.values[:2])


def test_scan_context_rejects_invalid_window() -> None:
    frame, features = sample_inputs()
    with pytest.raises(ValueError, match="window_size"):
        build_scan_context_features(
            frame,
            features,
            0.5,
            mode="past_only",
            window_size=0,
        )
