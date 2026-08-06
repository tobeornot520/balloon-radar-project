from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_zero_doppler_false_alarm_library_v1 import (
    LOCAL_ONLY_COLUMNS,
    assert_sanitized,
    build_library,
    load_config,
    scan_alias,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs/zero_doppler_false_alarm_library_v1.json"


def test_false_alarm_library_contract_is_frozen() -> None:
    config = load_config(CONFIG)

    assert config["folds"] == [1, 2, 3, 4, 5, 6]
    assert config["expected_counts"]["background_samples"] == 830
    assert config["expected_counts"]["fixed_false_alarms"] == 120
    assert config["expected_counts"]["residual_false_alarms"] == 109
    assert not any(config["claim_boundaries"].values())


def test_scan_alias_is_stable_and_does_not_expose_source() -> None:
    config = load_config(CONFIG)

    first = scan_alias("20260204_100739", config)
    second = scan_alias("20260204_100739", config)

    assert first == second
    assert first.startswith("scan_")
    assert "20260204" not in first


def test_sanitized_tables_reject_local_fields() -> None:
    with pytest.raises(ValueError, match="local-only"):
        assert_sanitized(pd.DataFrame({"sample_id": ["case-1"]}), "bad table")


def test_build_false_alarm_library_matches_frozen_evidence(tmp_path: Path) -> None:
    output_dir = tmp_path / "library"

    summary = build_library(
        config_path=CONFIG, output_dir=output_dir, overwrite=False
    )

    assert summary["status"] == "COMPLETE_AS_DEVELOPMENT_AUDIT"
    assert summary["counts"] == load_config(CONFIG)["expected_counts"]
    assert summary["review_completion_fraction"] == pytest.approx(11 / 120)
    assert not any(summary["claim_boundaries"].values())

    local = pd.read_csv(output_dir / "case_library_local.csv", encoding="utf-8-sig")
    fold = pd.read_csv(output_dir / "fold_transition_summary.csv", encoding="utf-8-sig")
    scan = pd.read_csv(output_dir / "scan_transition_summary.csv", encoding="utf-8-sig")
    review = pd.read_csv(output_dir / "review_pattern_summary.csv", encoding="utf-8-sig")
    saved_summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    assert len(local) == 830
    assert local["reviewed"].sum() == 11
    assert set(local.loc[local["reviewed"], "visible_pattern"]) == {
        "near_zero_doppler_peak",
        "broad_structure",
    }
    assert fold["fixed_false_alarms"].tolist() == [53, 0, 0, 67, 0, 0]
    assert fold["residual_false_alarms"].tolist() == [53, 0, 0, 56, 0, 0]
    assert len(scan) == 6
    assert scan["scan_alias"].nunique() == 6
    assert review.groupby("visible_pattern")["case_count"].sum().to_dict() == {
        "broad_structure": 2,
        "near_zero_doppler_peak": 9,
    }
    assert saved_summary["counts"]["added_by_residual"] == 0

    for frame in (fold, scan, review):
        assert set(frame.columns).isdisjoint(LOCAL_ONLY_COLUMNS)
        assert "source_file" not in frame.columns
        assert "review_note" not in frame.columns
