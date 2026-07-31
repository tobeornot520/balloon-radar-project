from __future__ import annotations

import pandas as pd

from scripts.build_multidomain_feature_catalog_v1 import (
    background_group_stress_table,
)


def test_background_group_stress_preserves_pooled_feature_direction() -> None:
    frame = pd.DataFrame(
        {
            "target_present": [1, 1, 0, 0, 0, 0],
            "source_file": ["target_a", "target_b", "bg_a", "bg_a", "bg_b", "bg_b"],
            "time_target_low": [0.0, 0.1, 0.8, 0.9, 0.6, 0.7],
        }
    )

    result = background_group_stress_table(frame)

    assert result.loc[0, "pooled_direction"] == "target_low"
    assert result.loc[0, "worst_background_group_auc"] == 1.0
    assert result.loc[0, "background_group_count"] == 2
