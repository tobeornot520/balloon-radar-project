from __future__ import annotations

import pandas as pd

from scripts.build_zero_doppler_human_review_queue_v1 import prepare_review_queue


def test_review_queue_contains_fixed_notch_false_alarms_and_relative_features() -> None:
    fixed_residual_pair = pd.DataFrame(
        {
            "fold": [4, 4, 4],
            "sample_id": ["removed", "retained", "clean"],
            "source_file_fixed": ["scan_a", "scan_a", "scan_a"],
            "target_present_fixed": [0, 0, 0],
            "false_alarm_fixed": [True, True, False],
            "false_alarm_residual": [False, True, False],
            "score_fixed": [0.8, 0.7, 0.1],
            "raw_score_fixed": [0.9, 0.8, 0.2],
            "score_residual": [0.2, 0.6, 0.1],
            "pred_range_index_fixed": [30, 40, 50],
            "pred_velocity_index_fixed": [64, 70, 40],
        }
    )
    features = pd.DataFrame(
        {
            "sample_id": ["removed", "retained", "clean"],
            "feature_rd_anchor_zero_doppler_fraction": [0.8, 0.6, 0.1],
        }
    )

    queue, summary = prepare_review_queue(
        [fixed_residual_pair], features, zero_velocity_index=64
    )

    assert queue["sample_id"].tolist() == ["removed", "retained"]
    assert queue["residual_removed"].tolist() == [True, False]
    assert queue["zero_velocity_distance_bins"].tolist() == [0, 6]
    assert queue.loc[0, "review_priority"] == "P0_removed_by_residual"
    assert queue.loc[1, "review_priority"] == "P1_near_zero_doppler"
    assert queue.loc[0, "physical_class"] == "unknown"
    assert "feature_rd_anchor_zero_doppler_fraction" in queue.columns
    assert summary.loc[0, "fixed_notch_false_alarms"] == 2
    assert summary.loc[0, "residual_false_alarms"] == 1
