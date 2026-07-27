from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd


GROUP_FEATURE_DIM = 12
DEFAULT_ORDER_COLUMNS = ("beam_layer", "azimuth_deg", "sample_id")
ContextMode = Literal["complete_scan", "leave_one_out", "past_only"]


@dataclass(frozen=True)
class ScanContextResult:
    values: np.ndarray
    used_history_counts: np.ndarray
    available_history_counts: np.ndarray


def _validate_inputs(
    frame: pd.DataFrame,
    sample_features: np.ndarray,
    mode: ContextMode,
    window_size: int | None,
    order_columns: Sequence[str],
) -> None:
    if len(frame) != sample_features.shape[0]:
        raise ValueError("frame/features length mismatch")
    if sample_features.ndim != 2 or sample_features.shape[1] <= 22:
        raise ValueError("sample_features must be [N,F] with F >= 23")
    required = {"scan_group", "raw_score"}
    if mode == "past_only":
        required.update(order_columns)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing scan-context columns: {missing}")
    if frame["scan_group"].isna().any():
        raise ValueError("scan_group contains missing values")
    if not np.isfinite(frame["raw_score"].to_numpy(dtype=np.float64)).all():
        raise ValueError("raw_score contains non-finite values")
    if not np.isfinite(sample_features[:, [21, 22]]).all():
        raise ValueError("scan-context sample features contain non-finite values")
    if mode != "past_only" and window_size is not None:
        raise ValueError("window_size is only valid for past_only context")
    if window_size is not None and window_size <= 0:
        raise ValueError("window_size must be positive")
    if mode == "past_only":
        if frame[list(order_columns)].isna().any().any():
            raise ValueError("past-only order columns contain missing values")
        if frame["sample_id"].astype(str).duplicated().any():
            raise ValueError("sample_id must be unique for past-only ordering")


def _summarize_indices(
    indices: np.ndarray,
    raw_score: np.ndarray,
    branch_diff: np.ndarray,
    peak_distance: np.ndarray,
    base_threshold: float,
) -> np.ndarray:
    if len(indices) == 0:
        return np.zeros(GROUP_FEATURE_DIM, dtype=np.float32)

    scores = raw_score[indices]
    count = len(indices)
    return np.asarray(
        [
            min(math.log1p(count) / math.log1p(256.0), 1.0),
            float(np.mean(scores)),
            float(np.std(scores)),
            float(np.median(scores)),
            float(np.quantile(scores, 0.75)),
            float(np.quantile(scores, 0.90)),
            float(np.max(scores)),
            float(np.mean(scores > 0.30)),
            float(np.mean(scores > 0.50)),
            float(np.mean(scores > base_threshold)),
            float(np.mean(branch_diff[indices])),
            float(np.mean(peak_distance[indices])),
        ],
        dtype=np.float32,
    )


def build_scan_context_features(
    frame: pd.DataFrame,
    sample_features: np.ndarray,
    base_threshold: float,
    *,
    mode: ContextMode,
    window_size: int | None = None,
    order_columns: Sequence[str] = DEFAULT_ORDER_COLUMNS,
) -> ScanContextResult:
    """Build the 12 scan statistics without consulting labels.

    ``past_only`` excludes the current sample. Its first sample in each scan
    receives an all-zero context, matching the no-context representation.
    """

    if mode not in {"complete_scan", "leave_one_out", "past_only"}:
        raise ValueError(f"Unsupported scan-context mode: {mode}")
    _validate_inputs(frame, sample_features, mode, window_size, order_columns)

    local_frame = frame.reset_index(drop=True).copy()
    raw_score = local_frame["raw_score"].to_numpy(dtype=np.float64)
    branch_diff = sample_features[:, 21]
    peak_distance = sample_features[:, 22]
    output = np.zeros((len(local_frame), GROUP_FEATURE_DIM), dtype=np.float32)
    used_counts = np.zeros(len(local_frame), dtype=np.int64)
    available_counts = np.zeros(len(local_frame), dtype=np.int64)

    group_indices = local_frame.groupby("scan_group", sort=False).indices
    for indices_value in group_indices.values():
        indices = np.asarray(indices_value, dtype=np.int64)
        if mode == "complete_scan":
            values = _summarize_indices(
                indices,
                raw_score,
                branch_diff,
                peak_distance,
                base_threshold,
            )
            output[indices] = values
            used_counts[indices] = len(indices)
            available_counts[indices] = len(indices)
            continue

        if mode == "leave_one_out":
            for current in indices:
                history = indices[indices != current]
                output[current] = _summarize_indices(
                    history,
                    raw_score,
                    branch_diff,
                    peak_distance,
                    base_threshold,
                )
                used_counts[current] = len(history)
                available_counts[current] = len(history)
            continue

        group_frame = local_frame.iloc[indices].copy()
        group_frame["_row_position"] = indices
        ordered = group_frame.sort_values(
            list(order_columns),
            kind="mergesort",
        )["_row_position"].to_numpy(dtype=np.int64)
        for position, current in enumerate(ordered):
            available = ordered[:position]
            history = (
                available[-window_size:]
                if window_size is not None
                else available
            )
            output[current] = _summarize_indices(
                history,
                raw_score,
                branch_diff,
                peak_distance,
                base_threshold,
            )
            used_counts[current] = len(history)
            available_counts[current] = len(available)

    return ScanContextResult(
        values=output,
        used_history_counts=used_counts,
        available_history_counts=available_counts,
    )
