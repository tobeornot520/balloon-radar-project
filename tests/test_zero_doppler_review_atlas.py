from __future__ import annotations

import pandas as pd
import pytest

from scripts.build_zero_doppler_review_atlas_v1 import select_cases


def test_select_cases_preserves_queue_order_and_requires_residual_peaks() -> None:
    queue = pd.DataFrame(
        {
            "fold": [4, 4],
            "sample_id": ["p0", "p1"],
            "review_priority": ["P0_removed_by_residual", "P1_near_zero_doppler"],
            "pred_range_index_fixed": [10, 20],
            "pred_velocity_index_fixed": [64, 66],
            "pred_range_index_residual": [10, 21],
            "pred_velocity_index_residual": [63, 67],
        }
    )

    selected = select_cases(queue, ["P0_removed_by_residual"], max_cases=0)

    assert selected["sample_id"].tolist() == ["p0"]


def test_select_cases_rejects_queue_without_residual_peak_columns() -> None:
    queue = pd.DataFrame(
        {
            "fold": [4],
            "sample_id": ["p0"],
            "review_priority": ["P0_removed_by_residual"],
            "pred_range_index_fixed": [10],
            "pred_velocity_index_fixed": [64],
        }
    )

    with pytest.raises(ValueError, match="pred_range_index_residual"):
        select_cases(queue, ["P0_removed_by_residual"], max_cases=0)
