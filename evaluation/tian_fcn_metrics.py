from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class TianMetricTolerance:
    range_gates: int = 2
    velocity_bins: int = 3
    paper_distance_cells: float = 0.0

    def __post_init__(self) -> None:
        if (
            self.range_gates < 0
            or self.velocity_bins < 0
            or self.paper_distance_cells < 0
        ):
            raise ValueError("metric tolerances must be nonnegative")


@dataclass(frozen=True)
class TianPredictionRecord:
    sample_id: str
    target_present: int
    peak_score: float
    pred_range_index: int | None
    pred_velocity_index: int | None
    detection_score: float | None
    true_range_index: int
    true_velocity_index: int
    all_predicted_positions: tuple[tuple[int, int], ...] | None = None

    def __post_init__(self) -> None:
        if self.target_present not in (0, 1):
            raise ValueError("target_present must be zero or one")
        if not 0.0 <= self.peak_score <= 1.0:
            raise ValueError("peak_score must be between zero and one")
        coordinates = (self.pred_range_index, self.pred_velocity_index)
        if (coordinates[0] is None) != (coordinates[1] is None):
            raise ValueError("predicted coordinates must both be present or absent")
        if self.detection_score is None and coordinates[0] is not None:
            raise ValueError("a positioned detection must have a detection score")
        if self.detection_score is not None and not 0.0 <= self.detection_score <= 1.0:
            raise ValueError("detection_score must be between zero and one")
        if self.all_predicted_positions is not None:
            for position in self.all_predicted_positions:
                if len(position) != 2:
                    raise ValueError("each predicted position must have two coordinates")

    @property
    def predicted_positions(self) -> tuple[tuple[int, int], ...]:
        if self.all_predicted_positions is not None:
            return self.all_predicted_positions
        if self.pred_range_index is None or self.pred_velocity_index is None:
            return ()
        return ((self.pred_range_index, self.pred_velocity_index),)

    @property
    def detected(self) -> bool:
        return bool(self.predicted_positions)


def select_validation_absolute_threshold(
    peak_scores: Sequence[float],
    target_present: Sequence[int],
    max_false_alarms: int,
) -> tuple[float, list[dict[str, float | int]]]:
    """Choose the least restrictive validation threshold within an FA budget."""

    scores = np.asarray(peak_scores, dtype=np.float64)
    labels = np.asarray(target_present, dtype=np.int64)
    if scores.ndim != 1 or labels.ndim != 1 or scores.shape != labels.shape:
        raise ValueError("peak_scores and target_present must be aligned vectors")
    if scores.size == 0:
        raise ValueError("validation predictions are empty")
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("peak_scores must be finite probabilities")
    if np.any((labels != 0) & (labels != 1)):
        raise ValueError("target_present must contain only zero and one")
    if max_false_alarms < 0:
        raise ValueError("max_false_alarms must be nonnegative")

    background_scores = scores[labels == 0]
    if background_scores.size == 0:
        raise ValueError("validation data contain no background samples")

    candidates = np.unique(
        np.concatenate(
            (
                np.array([0.0, 1.0], dtype=np.float64),
                np.unique(background_scores),
            )
        )
    )
    feasible = [
        float(candidate)
        for candidate in candidates
        if int(np.sum(background_scores > candidate)) <= max_false_alarms
    ]
    if not feasible:
        raise RuntimeError("no threshold satisfies the validation false-alarm budget")
    threshold = min(feasible)

    positive_scores = scores[labels == 1]
    curve: list[dict[str, float | int]] = []
    for candidate in candidates:
        false_alarm_count = int(np.sum(background_scores > candidate))
        positive_detected = int(np.sum(positive_scores > candidate))
        curve.append(
            {
                "threshold": float(candidate),
                "false_alarm_count": false_alarm_count,
                "pfa": float(false_alarm_count / background_scores.size),
                "positive_peak_recall": (
                    float(positive_detected / positive_scores.size)
                    if positive_scores.size
                    else math.nan
                ),
            }
        )
    return threshold, curve


def compute_tian_metrics(
    records: Iterable[TianPredictionRecord],
    tolerance: TianMetricTolerance = TianMetricTolerance(),
) -> dict[str, float | int]:
    """Compute paper set metrics and the local map-level metric family.

    The paper uses Euclidean RD-cell distance. Its false-alarm rate is the
    unmatched fraction of output detections, whereas the local ``pfa`` remains
    the fraction of background maps containing at least one detection.
    """

    rows = list(records)
    if not rows:
        raise ValueError("prediction records are empty")

    positives = [row for row in rows if row.target_present == 1]
    backgrounds = [row for row in rows if row.target_present == 0]
    detected_positives = [row for row in positives if row.detected]
    false_alarm_maps = sum(row.detected for row in backgrounds)

    nearest_errors: list[tuple[int, int]] = []
    localization_ok: list[bool] = []
    paper_gt_min_distances: list[float] = []
    paper_detection_distances: list[float] = []
    paper_detection_within_five: list[float] = []
    paper_correct_count = 0
    paper_detection_count = 0

    for row in positives:
        coordinate_errors = [
            (
                abs(pred_range - row.true_range_index),
                abs(pred_velocity - row.true_velocity_index),
            )
            for pred_range, pred_velocity in row.predicted_positions
        ]
        distances = [
            math.hypot(range_error, velocity_error)
            for range_error, velocity_error in coordinate_errors
        ]
        paper_detection_count += len(distances)
        paper_detection_distances.extend(distances)
        paper_detection_within_five.extend(
            distance for distance in distances if distance <= 5.0
        )
        if distances:
            nearest_index = int(np.argmin(distances))
            nearest_errors.append(coordinate_errors[nearest_index])
            minimum_distance = distances[nearest_index]
            paper_gt_min_distances.append(minimum_distance)
            if minimum_distance <= tolerance.paper_distance_cells:
                paper_correct_count += 1
        localization_ok.append(
            any(
                range_error <= tolerance.range_gates
                and velocity_error <= tolerance.velocity_bins
                for range_error, velocity_error in coordinate_errors
            )
        )

    background_detection_count = sum(
        len(row.predicted_positions) for row in backgrounds
    )
    paper_detection_count += background_detection_count
    positive_count = len(positives)
    background_count = len(backgrounds)
    detected_positive_count = len(detected_positives)
    correct_detection_count = int(sum(localization_ok))
    range_errors = [item[0] for item in nearest_errors]
    velocity_errors = [item[1] for item in nearest_errors]
    paper_unmatched_detections = max(
        0, paper_detection_count - paper_correct_count
    )

    return {
        "sample_count": len(rows),
        "positive_count": positive_count,
        "background_count": background_count,
        "detected_positive_count": detected_positive_count,
        "correct_detection_count": correct_detection_count,
        "false_alarm_count": int(false_alarm_maps),
        "paper_detection_count": paper_detection_count,
        "paper_correct_detection_count": paper_correct_count,
        "paper_distance_threshold_cells": tolerance.paper_distance_cells,
        "paper_pd": (
            paper_correct_count / positive_count if positive_count else math.nan
        ),
        "paper_pf": (
            paper_unmatched_detections / paper_detection_count
            if paper_detection_count
            else 0.0
        ),
        "paper_d_min_euclidean_cells": (
            float(np.mean(paper_gt_min_distances))
            if paper_gt_min_distances
            else math.nan
        ),
        "paper_d_5_euclidean_cells": (
            float(np.mean(paper_detection_within_five))
            if paper_detection_within_five
            else math.nan
        ),
        "paper_d_avg_euclidean_cells": (
            float(np.mean(paper_detection_distances))
            if paper_detection_distances
            else math.nan
        ),
        "joint_pd": (
            correct_detection_count / positive_count if positive_count else math.nan
        ),
        "pfa": (
            false_alarm_maps / background_count if background_count else math.nan
        ),
        "range_mae_gates": (
            float(np.mean(range_errors)) if range_errors else math.nan
        ),
        "velocity_mae_bins": (
            float(np.mean(velocity_errors)) if velocity_errors else math.nan
        ),
        "range_tolerance_gates": tolerance.range_gates,
        "velocity_tolerance_bins": tolerance.velocity_bins,
    }
