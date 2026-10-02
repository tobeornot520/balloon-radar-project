#!/usr/bin/env python3
"""Train an exploratory file-level classifier for the new RSTM collection.

This is an intake/development baseline.  It deliberately uses one feature
vector per source IQ file and holds out the last temporal quintile inside each
source category.  It must not be reported as a date/site-independent blind
test: all target categories were collected at one site on one date.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import struct
from datetime import datetime
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGET_CATEGORIES = (
    "target_six_wing",
    "target_reflector",
    "target_small_satellite",
    "target_uav",
    "target_balloon",
)
PAYLOAD_OFFSET = 0x860
BLOCK_FLOATS = 8192
NUM_BLOCKS = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def sample_payload(path: Path) -> np.ndarray:
    """Sample fixed-size float32 blocks across the payload without full loading."""
    payload_bytes = max(0, path.stat().st_size - PAYLOAD_OFFSET)
    available = payload_bytes // 4
    if available < 16:
        return np.empty(0, dtype=np.float32)
    starts = np.linspace(0, max(0, available - BLOCK_FLOATS), NUM_BLOCKS, dtype=np.int64)
    parts: list[np.ndarray] = []
    with path.open("rb") as handle:
        for start in np.unique(starts):
            handle.seek(PAYLOAD_OFFSET + int(start) * 4)
            raw = handle.read(BLOCK_FLOATS * 4)
            values = np.frombuffer(raw, dtype="<f4")
            if values.size:
                parts.append(values[np.isfinite(values)])
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else math.nan


def feature_names() -> list[str]:
    names: list[str] = []
    for prefix in ("component_a", "component_b", "magnitude"):
        names.extend(
            [f"{prefix}_{suffix}" for suffix in ("mean", "std", "p01", "p05", "p50", "p95", "p99")]
        )
    names.extend(["ab_correlation", "magnitude_max", "payload_finite_fraction"])
    names.extend([f"block_rms_{suffix}" for suffix in ("mean", "std", "p10", "p50", "p90")])
    return names


def make_features(values: np.ndarray) -> np.ndarray:
    total = max(1, values.size)
    # RSTM stores per-ray metadata alongside the payload.  Interpreting those
    # integer words as float32 creates very large outliers; normalized IQ
    # samples in this collection are well inside this conservative bound.
    values = values[np.isfinite(values) & (np.abs(values) <= 16.0)]
    finite_fraction = float(values.size / total)
    if values.size < 16:
        return np.full(len(feature_names()), np.nan, dtype=np.float64)
    a = values[0::2].astype(np.float64, copy=False)
    b = values[1::2].astype(np.float64, copy=False)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    magnitude = np.sqrt(a * a + b * b)
    def summary(x: np.ndarray) -> list[float]:
        return [
            float(np.mean(x)),
            float(np.std(x)),
            percentile(x, 1),
            percentile(x, 5),
            percentile(x, 50),
            percentile(x, 95),
            percentile(x, 99),
        ]
    if n > 1 and float(np.std(a)) > 0 and float(np.std(b)) > 0:
        correlation = float(np.corrcoef(a, b)[0, 1])
    else:
        correlation = 0.0
    block_size = max(1, n // NUM_BLOCKS)
    rms = np.asarray(
        [float(np.sqrt(np.mean(a[i : i + block_size] ** 2 + b[i : i + block_size] ** 2)))
         for i in range(0, n - block_size + 1, block_size)],
        dtype=np.float64,
    )
    features = summary(a) + summary(b) + summary(magnitude)
    features += [correlation, float(np.max(magnitude)), finite_fraction]
    features += [float(np.mean(rms)), float(np.std(rms)), percentile(rms, 10), percentile(rms, 50), percentile(rms, 90)]
    return np.asarray(features, dtype=np.float64)


def make_rows(manifest: Path, data_root: Path, output_features: Path) -> tuple[list[dict[str, object]], np.ndarray]:
    with manifest.open(encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    rows: list[dict[str, object]] = []
    vectors: list[np.ndarray] = []
    for source in source_rows:
        category = source["category_dir"]
        if category not in TARGET_CATEGORIES or source["extension"] != ".iq":
            continue
        path = (data_root / source["relative_path"]).resolve()
        values = sample_payload(path)
        vector = make_features(values)
        record: dict[str, object] = {
            "relative_path": source["relative_path"],
            "category_dir": category,
            "filename_timestamp_utc": source["filename_timestamp_utc"],
            "n_sampled_float32": int(values.size),
            "payload_offset": PAYLOAD_OFFSET,
        }
        for name, value in zip(feature_names(), vector):
            record[name] = float(value)
        rows.append(record)
        vectors.append(vector)
    order = np.argsort([parse_time(str(row["filename_timestamp_utc"])) for row in rows])
    rows = [rows[int(i)] for i in order]
    matrix = np.vstack([vectors[int(i)] for i in order])
    output_features.parent.mkdir(parents=True, exist_ok=True)
    with output_features.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows, matrix


def add_temporal_split(rows: list[dict[str, object]]) -> None:
    by_category: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        by_category.setdefault(str(row["category_dir"]), []).append(index)
    for category, indices in by_category.items():
        indices.sort(key=lambda i: parse_time(str(rows[i]["filename_timestamp_utc"])))
        for rank, index in enumerate(indices):
            block = min(4, (rank * 5) // max(1, len(indices)))
            rows[index]["temporal_block"] = block
            rows[index]["split"] = "test" if block == 4 else "train"


def fit_and_report(
    rows: list[dict[str, object]],
    matrix: np.ndarray,
    target_mode: str,
    output_dir: Path,
) -> dict[str, object]:
    if target_mode == "target_type":
        selected = np.asarray([str(row["category_dir"]) in TARGET_CATEGORIES for row in rows])
        labels = [str(row["category_dir"]) for row in rows]
    elif target_mode == "balloon_vs_other_target":
        selected = np.asarray([str(row["category_dir"]) in TARGET_CATEGORIES for row in rows])
        labels = ["balloon" if row["category_dir"] == "target_balloon" else "other_target" for row in rows]
    else:
        raise ValueError(target_mode)
    x = matrix[selected]
    selected_rows = [row for row, keep in zip(rows, selected) if keep]
    train = np.asarray([row["split"] == "train" for row in selected_rows])
    test = ~train
    y = np.asarray(labels)[selected]
    if np.isnan(x).any():
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)),
        ]
    )
    model.fit(x[train], y[train])
    prediction = model.predict(x[test])
    classes = list(model.named_steps["classifier"].classes_)
    report = classification_report(y[test], prediction, labels=classes, output_dict=True, zero_division=0)
    payload = {
        "task": target_mode,
        "feature_contract": {
            "payload_offset_bytes": PAYLOAD_OFFSET,
            "sampling_blocks": NUM_BLOCKS,
            "block_float32_count": BLOCK_FLOATS,
            "features": feature_names(),
            "timestamp_and_header_features_excluded": True,
        },
        "split_contract": {
            "unit": "one source IQ file",
            "test_rule": "last temporal quintile within each source category",
            "train_files": int(train.sum()),
            "test_files": int(test.sum()),
            "train_class_counts": {c: int((y[train] == c).sum()) for c in classes},
            "test_class_counts": {c: int((y[test] == c).sum()) for c in classes},
        },
        "metrics": {
            "accuracy": float(accuracy_score(y[test], prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(y[test], prediction)),
            "macro_f1": float(f1_score(y[test], prediction, average="macro", zero_division=0)),
            "classification_report": report,
            "confusion_matrix": confusion_matrix(y[test], prediction, labels=classes).tolist(),
        },
        "claim_boundary": "exploratory same-site same-date development result; not an independent blind test",
    }
    task_dir = output_dir / target_mode
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    joblib.dump(model, task_dir / "model.joblib")
    return payload


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    rows, matrix = make_rows(
        args.manifest.resolve(),
        args.data_root.resolve(),
        output_dir / "file_features.csv",
    )
    add_temporal_split(rows)
    with (output_dir / "file_features_with_split.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    results = [
        fit_and_report(rows, matrix, "target_type", output_dir),
        fit_and_report(rows, matrix, "balloon_vs_other_target", output_dir),
    ]
    summary = {
        "collection": "field_collection_20250425_st001",
        "n_target_files": len(rows),
        "tasks": [result["task"] for result in results],
        "status": "development_only",
        "limitations": [
            "all categories share one site and one date",
            "source folder labels are not independently verified physical labels",
            "header physical axes and H/V semantics are unresolved",
            "temporal blocks are not independent sessions",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
