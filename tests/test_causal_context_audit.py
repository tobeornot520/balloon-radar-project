from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.audit_bc_dpg_v3_causal_context import (
    PredictionPayload,
    aggregate_metrics,
    align_predictions,
    make_payload,
    paired_record,
)


def test_align_predictions_reorders_and_validates_labels(tmp_path: Path) -> None:
    reference = pd.DataFrame(
        {
            "sample_id": ["b", "a"],
            "target_present": [0, 1],
        }
    )
    candidate = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "target_present": [1, 0],
            "score": [0.9, 0.1],
            "shift": [0.0, 0.2],
            "detected": [True, False],
            "correct_detection": [True, False],
        }
    )
    path = tmp_path / "predictions.csv"
    candidate.to_csv(path, index=False)
    aligned = align_predictions(reference, path)
    assert aligned["sample_id"].tolist() == ["b", "a"]
    assert aligned["score"].tolist() == pytest.approx([0.1, 0.9])

    candidate.loc[0, "target_present"] = 0
    candidate.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Label mismatch"):
        align_predictions(reference, path)


def test_make_payload_uses_strict_frozen_threshold() -> None:
    frame = pd.DataFrame(
        {
            "target_present": [0, 1, 1],
            "localization_ok": [False, True, False],
        }
    )
    payload = make_payload(
        frame,
        np.asarray([0.5, 0.5001, 0.9]),
        np.zeros(3),
        threshold=0.5,
        context=None,
    )
    assert payload.detected.tolist() == [False, True, True]
    assert payload.correct.tolist() == [False, True, False]


def test_aggregate_metrics_uses_pooled_counts_and_exposes_worst_fold() -> None:
    detail = pd.DataFrame(
        [
            {
                "fold": "01",
                "mode": "raw_dpg",
                "model_source": "source",
                "context_role": "role",
                "causality": "sample-independent",
                "training_context_match": True,
                "background_samples": 10,
                "target_samples": 2,
                "false_alarms": 5,
                "pfa": 0.5,
                "score_detections": 2,
                "correct_detections": 2,
                "joint_pd": 1.0,
                "roc_auc": 0.9,
                "background_shift_mean": 0.0,
                "target_shift_mean": 0.0,
            },
            {
                "fold": "02",
                "mode": "raw_dpg",
                "model_source": "source",
                "context_role": "role",
                "causality": "sample-independent",
                "training_context_match": True,
                "background_samples": 30,
                "target_samples": 8,
                "false_alarms": 3,
                "pfa": 0.1,
                "score_detections": 4,
                "correct_detections": 4,
                "joint_pd": 0.5,
                "roc_auc": 0.8,
                "background_shift_mean": 0.0,
                "target_shift_mean": 0.0,
            },
        ]
    )
    row = aggregate_metrics(detail, ["raw_dpg"]).iloc[0]
    assert row["pooled_pfa"] == pytest.approx(8 / 40)
    assert row["macro_pfa"] == pytest.approx(0.3)
    assert row["pooled_joint_pd"] == pytest.approx(6 / 10)
    assert row["worst_pfa_fold"] == "01"
    assert row["worst_joint_pd_fold"] == "02"


def test_paired_record_counts_decision_directions() -> None:
    labels = np.asarray([0, 0, 1, 1])
    baseline = PredictionPayload(
        score=np.zeros(4),
        shift=np.zeros(4),
        detected=np.asarray([True, False, True, True]),
        correct=np.asarray([False, False, True, False]),
        context=None,
    )
    candidate = PredictionPayload(
        score=np.zeros(4),
        shift=np.zeros(4),
        detected=np.asarray([False, True, False, True]),
        correct=np.asarray([False, False, False, True]),
        context=None,
    )
    row = paired_record("01", "candidate", labels, baseline, candidate)
    assert row["background_complete_only_alarm"] == 1
    assert row["background_candidate_only_alarm"] == 1
    assert row["target_complete_only_correct"] == 1
    assert row["target_candidate_only_correct"] == 1
