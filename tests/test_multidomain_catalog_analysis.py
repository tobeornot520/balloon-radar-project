from __future__ import annotations

import pandas as pd

from scripts.build_multidomain_feature_catalog_v1 import (
    background_group_stress_table,
    feature_columns,
)
from features.multidomain_radar_features import MULTIDOMAIN_FEATURE_NAMES


def test_background_group_stress_preserves_pooled_feature_direction() -> None:
    frame = pd.DataFrame(
        {
            "target_present": [1, 1, 0, 0, 0, 0],
            "source_file": ["target_a", "target_b", "bg_a", "bg_a", "bg_b", "bg_b"],
            "time_h_magnitude_cv": [0.0, 0.1, 0.8, 0.9, 0.6, 0.7],
        }
    )

    result = background_group_stress_table(frame)

    assert result.loc[0, "pooled_direction"] == "target_low"
    assert result.loc[0, "worst_background_group_auc"] == 1.0
    assert result.loc[0, "background_group_count"] == 2


def test_feature_catalog_uses_frozen_names_and_order() -> None:
    ordered = [
        name
        for names in MULTIDOMAIN_FEATURE_NAMES.values()
        for name in names
    ]
    frame = pd.DataFrame(
        {
            ordered[-1]: [1.0],
            "time_future_debug_field": [2.0],
            ordered[0]: [3.0],
        }
    )
    assert feature_columns(frame) == [ordered[0], ordered[-1]]
