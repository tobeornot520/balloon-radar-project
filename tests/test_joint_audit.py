from __future__ import annotations

import pandas as pd
import pytest

from scripts.build_final_roi_bc_dpg_joint_audit import build_summaries
from scripts.build_roi_bc_dpg_joint_tables_v1 import (
    align_to_reference,
    boolean_series,
)


def test_boolean_series_parses_supported_values() -> None:
    values = pd.Series([True, False, "yes", "0", 1, None])
    assert boolean_series(values, "flag").tolist() == [
        True,
        False,
        True,
        False,
        True,
        False,
    ]


def test_boolean_series_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="unsupported boolean values"):
        boolean_series(pd.Series(["maybe"]), "flag")


def test_align_to_reference_reorders_and_rejects_mismatch() -> None:
    reference = pd.Series(["sample-b", "sample-a"])
    frame = pd.DataFrame(
        {
            "sample_id": ["sample-a", "sample-b"],
            "value": [1, 2],
        }
    )
    aligned = align_to_reference(reference, frame, "candidate")
    assert aligned["sample_id"].tolist() == ["sample-b", "sample-a"]
    assert aligned["value"].tolist() == [2, 1]

    with pytest.raises(ValueError, match="sample_id mismatch"):
        align_to_reference(
            reference,
            pd.DataFrame({"sample_id": ["sample-a", "sample-c"]}),
            "candidate",
        )


def test_build_summaries_reports_detection_and_complementarity() -> None:
    frame = pd.DataFrame(
        {
            "target_present": [0, 0, 1, 1],
            "bc_false_alarm": [True, False, False, False],
            "bc_correct": [False, False, True, False],
            "roi_base_false_alarm": [False, False, False, False],
            "roi_base_correct": [False, False, True, False],
            "roi_power_false_alarm": [True, True, False, False],
            "roi_power_correct": [False, False, False, True],
            "roi_ri4_false_alarm": [False, True, False, False],
            "roi_ri4_correct": [False, False, True, True],
        }
    )

    detection, complementarity = build_summaries({1: frame})

    bc_all = detection[
        detection["fold"].eq("ALL") & detection["model"].eq("bc_dpg_v3")
    ].iloc[0]
    assert bc_all["false_alarms"] == 1
    assert bc_all["pfa"] == pytest.approx(0.5)
    assert bc_all["correct_detections"] == 1
    assert bc_all["joint_pd"] == pytest.approx(0.5)

    power_all = complementarity[
        complementarity["fold"].eq("ALL")
        & complementarity["comparison"].eq("roi_power_control")
    ].iloc[0]
    assert power_all["shared_false_alarms"] == 1
    assert power_all["roi_only_false_alarms"] == 1
    assert power_all["bc_only_correct"] == 1
    assert power_all["roi_only_correct"] == 1
    assert power_all["correct_union"] == 2
