from __future__ import annotations

import pandas as pd
import pytest

from scripts.audit_zero_doppler_human_review_v1 import summarize_reviews, validate_reviews


def review_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold": [4, 4, 4],
            "sample_id": ["reviewed", "waiting", "unavailable"],
            "review_priority": ["P0_removed_by_residual"] * 3,
            "review_status": ["reviewed", "pending", "unavailable"],
            "visible_pattern": ["near_zero_doppler_peak", "unreviewed", "unreviewed"],
            "physical_class": ["unknown", "unknown", "unknown"],
            "evidence_source": ["prediction_and_relative_features_only"] * 3,
            "review_note": ["narrow near-zero ridge", "", "raw context unavailable"],
        }
    )


def test_review_audit_marks_pending_queue_incomplete() -> None:
    normalized = validate_reviews(review_frame())
    summary, patterns, named_labels = summarize_reviews(normalized)

    assert summary["status"] == "INCOMPLETE"
    assert summary["reviewed_count"] == 1
    assert int(patterns["sample_count"].sum()) == 3
    assert named_labels.empty


def test_review_audit_requires_independent_evidence_for_named_physical_class() -> None:
    frame = review_frame()
    frame.loc[0, "physical_class"] = "building"

    with pytest.raises(ValueError, match="independent_scene_record"):
        validate_reviews(frame)


def test_review_audit_rejects_reviewed_row_without_note() -> None:
    frame = review_frame()
    frame.loc[0, "review_note"] = ""

    with pytest.raises(ValueError, match="review_note"):
        validate_reviews(frame)
