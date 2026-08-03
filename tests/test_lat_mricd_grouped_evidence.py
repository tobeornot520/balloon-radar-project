from __future__ import annotations

import pandas as pd
import pytest

from scripts.build_lat_mricd_grouped_evidence_v1 import (
    EXCLUDED_SOURCE_FILES,
    TABLE_FILES,
    validate_oof_predictions,
)


def small_predictions() -> pd.DataFrame:
    rows = []
    for model in {"dummy", "linear"}:
        for index, (batch, fold, category) in enumerate(
            [(10, 1, 1), (11, 2, 2), (12, 3, 3)]
        ):
            probabilities = [0.1, 0.1, 0.1]
            probabilities[category - 1] = 0.8
            rows.append(
                {
                    "task_id": "task",
                    "source_row_index": index,
                    "model_id": model,
                    "batch_code": batch,
                    "heldout_fold": fold,
                    "category_code": category,
                    "predicted_category_code": category,
                    "probability_uav": probabilities[0],
                    "probability_bird": probabilities[1],
                    "probability_weather": probabilities[2],
                }
            )
    return pd.DataFrame(rows)


def small_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "task_id": ["task", "task", "task"],
            "batch_code": [10, 11, 12],
            "heldout_fold": [1, 2, 3],
        }
    )


def test_oof_validation_requires_exact_rows_probabilities_and_folds() -> None:
    predictions = small_predictions()
    manifest = small_manifest()
    validate_oof_predictions(
        predictions,
        manifest,
        expected_task_records={"task": 3},
        expected_models={"dummy", "linear"},
    )

    broken = predictions.copy()
    broken.loc[0, "heldout_fold"] = 2
    with pytest.raises(ValueError, match="frozen batch manifest"):
        validate_oof_predictions(
            broken,
            manifest,
            expected_task_records={"task": 3},
            expected_models={"dummy", "linear"},
        )


def test_publication_map_excludes_sample_level_predictions() -> None:
    assert "oof_predictions.csv" in EXCLUDED_SOURCE_FILES
    assert not any("oof" in name or "prediction" in name for name in TABLE_FILES)
    assert "split_manifest.csv" in TABLE_FILES
