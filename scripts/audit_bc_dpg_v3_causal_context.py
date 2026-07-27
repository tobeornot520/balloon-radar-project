#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.scan_context import (
    GROUP_FEATURE_DIM,
    ScanContextResult,
    build_scan_context_features,
)
from models.target_protected_scan_calibrator import TargetProtectedScanCalibrator


DEFAULT_EXPERIMENT_ROOT = PROJECT_ROOT / "results" / "experiments"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "data_audit"
    / "bc_dpg_v3_causal_context_audit"
)
EXPECTED_FOLDS = (1, 2, 3, 4, 5, 6)
EXPECTED_ROWS = 1148
EXPECTED_COMPLETE_FALSE_ALARMS = 56
EXPECTED_COMPLETE_CORRECT = 289
EXPECTED_INDEPENDENT_FALSE_ALARMS = 122
SAMPLE_FEATURE_COLUMNS = tuple(
    f"sample_feature_{index:02d}" for index in range(24)
)
STORED_GROUP_FEATURE_COLUMNS = tuple(
    f"group_feature_{index:02d}" for index in range(GROUP_FEATURE_DIM)
)


@dataclass(frozen=True)
class FoldSources:
    fold: int
    full_checkpoint: Path
    precomputed_test: Path
    full_predictions: Path
    full_summary: Path
    independent_predictions: Path
    independent_summary: Path


@dataclass(frozen=True)
class PredictionPayload:
    score: np.ndarray
    shift: np.ndarray
    detected: np.ndarray
    correct: np.ndarray
    context: ScanContextResult | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay frozen BC-DPG v3 checkpoints under complete-scan, "
            "leave-one-out, and assumed-order past-only contexts. No training "
            "or threshold selection is performed."
        )
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=list(EXPECTED_FOLDS),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--windows",
        nargs="+",
        type=int,
        default=[4, 16, 64],
        help="Past-only history windows; an unbounded past-only mode is always added.",
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def fold_sources(experiment_root: Path, fold: int, seed: int) -> FoldSources:
    fold_tag = f"fold{fold:02d}"
    full = experiment_root / f"bc_dpg_v3_scan_target_v4_{fold_tag}_seed{seed}"
    independent = (
        experiment_root
        / f"bc_dpg_v3_ablation_no_scan_context_v4_{fold_tag}_seed{seed}"
    )
    return FoldSources(
        fold=fold,
        full_checkpoint=full / "checkpoints" / "best.pt",
        precomputed_test=full / "tables" / "precomputed_test.csv",
        full_predictions=full / "tables" / "base_threshold_test_predictions.csv",
        full_summary=full / "tables" / "summary.json",
        independent_predictions=(
            independent / "tables" / "base_threshold_test_predictions.csv"
        ),
        independent_summary=independent / "tables" / "summary.json",
    )


def validate_args(args: argparse.Namespace) -> None:
    if not args.folds or len(args.folds) != len(set(args.folds)):
        raise ValueError("folds must be nonempty and unique")
    if any(fold not in EXPECTED_FOLDS for fold in args.folds):
        raise ValueError("folds must be in 1-6")
    if not args.windows or len(args.windows) != len(set(args.windows)):
        raise ValueError("windows must be nonempty and unique")
    if any(window <= 0 for window in args.windows):
        raise ValueError("windows must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")


def validate_sources(sources: Iterable[FoldSources]) -> None:
    missing: list[str] = []
    for source in sources:
        for path in (
            source.full_checkpoint,
            source.precomputed_test,
            source.full_predictions,
            source.full_summary,
            source.independent_predictions,
            source.independent_summary,
        ):
            if not path.is_file():
                missing.append(display_path(path))
    if missing:
        raise FileNotFoundError(f"Missing causal-context audit inputs: {missing}")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {display_path(path)}")
    return payload


def boolean_array(series: pd.Series, name: str) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=bool)
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    parsed = normalized.map(mapping)
    if parsed.isna().any():
        values = sorted(normalized.loc[parsed.isna()].unique().tolist())
        raise ValueError(f"Unsupported boolean values in {name}: {values}")
    return parsed.to_numpy(dtype=bool)


def load_precomputed(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "sample_id",
        "scan_group",
        "target_present",
        "raw_logit",
        "raw_score",
        "localization_ok",
        "beam_layer",
        "azimuth_deg",
        *SAMPLE_FEATURE_COLUMNS,
        *STORED_GROUP_FEATURE_COLUMNS,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing precomputed columns in {display_path(path)}: {missing}")
    if frame["sample_id"].astype(str).duplicated().any():
        raise ValueError(f"Duplicate sample IDs in {display_path(path)}")
    labels = frame["target_present"].to_numpy(dtype=np.int64)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"Invalid labels in {display_path(path)}")
    label_counts = frame.groupby("scan_group")["target_present"].nunique()
    if label_counts.gt(1).any():
        raise ValueError(f"Mixed-label scan group in {display_path(path)}")
    return frame.reset_index(drop=True)


def align_predictions(reference: pd.DataFrame, path: Path) -> pd.DataFrame:
    candidate = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "sample_id",
        "target_present",
        "score",
        "shift",
        "detected",
        "correct_detection",
    }
    missing = sorted(required.difference(candidate.columns))
    if missing:
        raise ValueError(f"Missing prediction columns in {display_path(path)}: {missing}")
    if candidate["sample_id"].astype(str).duplicated().any():
        raise ValueError(f"Duplicate sample IDs in {display_path(path)}")
    reference_ids = reference["sample_id"].astype(str)
    candidate_ids = candidate["sample_id"].astype(str)
    if set(reference_ids) != set(candidate_ids):
        raise ValueError(f"Sample ID mismatch in {display_path(path)}")
    aligned = candidate.assign(sample_id=candidate_ids).set_index("sample_id").loc[
        reference_ids
    ].reset_index()
    if not np.array_equal(
        aligned["target_present"].to_numpy(dtype=np.int64),
        reference["target_present"].to_numpy(dtype=np.int64),
    ):
        raise ValueError(f"Label mismatch in {display_path(path)}")
    return aligned


def load_calibrator(path: Path) -> tuple[TargetProtectedScanCalibrator, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    model = TargetProtectedScanCalibrator(
        sample_feature_dim=24,
        group_feature_dim=GROUP_FEATURE_DIM,
        hidden_dims=tuple(config.get("hidden_dims", (64, 32))),
        maximum_shift=float(config.get("maximum_shift", 3.0)),
        initial_background_probability=float(
            config.get("initial_background_probability", 0.05)
        ),
        initial_suppression=float(config.get("initial_suppression", 0.10)),
    )
    result = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Calibrator checkpoint mismatch: {result}")
    model.eval()
    return model, checkpoint


def infer_calibrator(
    model: TargetProtectedScanCalibrator,
    sample_features: np.ndarray,
    group_features: np.ndarray,
    raw_logit: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    scores: list[np.ndarray] = []
    shifts: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(raw_logit), batch_size):
            stop = min(start + batch_size, len(raw_logit))
            outputs = model(
                torch.from_numpy(sample_features[start:stop].astype(np.float32)),
                torch.from_numpy(group_features[start:stop].astype(np.float32)),
                torch.from_numpy(raw_logit[start:stop].astype(np.float32)),
            )
            scores.append(outputs["calibrated_score"].numpy())
            shifts.append(outputs["shift"].numpy())
    return np.concatenate(scores), np.concatenate(shifts)


def mode_metadata(mode: str) -> dict[str, Any]:
    if mode == "raw_dpg":
        return {
            "model_source": "frozen DPG output",
            "context_role": "sample-level baseline",
            "causality": "sample-independent",
            "training_context_match": True,
        }
    if mode == "sample_independent_bc":
        return {
            "model_source": "separately trained no-scan-context BC",
            "context_role": "online-oriented reference",
            "causality": "sample-independent",
            "training_context_match": True,
        }
    if mode == "full_complete_scan":
        return {
            "model_source": "frozen complete-scan BC-DPG v3",
            "context_role": "matched frozen offline result",
            "causality": "non-causal complete scan",
            "training_context_match": True,
        }
    if mode == "full_leave_one_out":
        return {
            "model_source": "frozen complete-scan BC-DPG v3",
            "context_role": "post-hoc self-inclusion sensitivity",
            "causality": "non-causal; excludes self but uses future samples",
            "training_context_match": False,
        }
    if mode.startswith("full_past_only"):
        return {
            "model_source": "frozen complete-scan BC-DPG v3",
            "context_role": "post-hoc assumed-order sensitivity",
            "causality": "past-only under inferred beam/azimuth order",
            "training_context_match": False,
        }
    raise ValueError(f"Unknown mode: {mode}")


def make_payload(
    frame: pd.DataFrame,
    score: np.ndarray,
    shift: np.ndarray,
    threshold: float,
    context: ScanContextResult | None,
) -> PredictionPayload:
    labels = frame["target_present"].to_numpy(dtype=np.int64)
    localization_ok = boolean_array(frame["localization_ok"], "localization_ok")
    detected = score > threshold
    correct = labels.astype(bool) & detected & localization_ok
    return PredictionPayload(
        score=score.astype(np.float64),
        shift=shift.astype(np.float64),
        detected=detected,
        correct=correct,
        context=context,
    )


def metric_record(
    fold: int,
    mode: str,
    frame: pd.DataFrame,
    payload: PredictionPayload,
    threshold: float,
) -> dict[str, Any]:
    labels = frame["target_present"].to_numpy(dtype=np.int64)
    background = labels == 0
    target = labels == 1
    false_alarms = background & payload.detected
    correct = target & payload.correct
    score_detected = target & payload.detected
    precision_denominator = int(correct.sum() + false_alarms.sum())
    joint_precision = (
        float(correct.sum() / precision_denominator)
        if precision_denominator
        else float("nan")
    )
    joint_pd = float(correct.sum() / target.sum())
    joint_f1 = (
        2.0 * joint_precision * joint_pd / (joint_precision + joint_pd)
        if joint_precision + joint_pd > 0
        else 0.0
    )
    metadata = mode_metadata(mode)
    return {
        "fold": f"{fold:02d}",
        "mode": mode,
        **metadata,
        "threshold": threshold,
        "samples": len(frame),
        "background_samples": int(background.sum()),
        "target_samples": int(target.sum()),
        "false_alarms": int(false_alarms.sum()),
        "pfa": float(false_alarms.sum() / background.sum()),
        "score_detections": int(score_detected.sum()),
        "score_pd": float(score_detected.sum() / target.sum()),
        "correct_detections": int(correct.sum()),
        "joint_pd": float(correct.sum() / target.sum()),
        "joint_precision": joint_precision,
        "joint_f1": joint_f1,
        "roc_auc": float(roc_auc_score(labels, payload.score)),
        "background_shift_mean": float(payload.shift[background].mean()),
        "target_shift_mean": float(payload.shift[target].mean()),
    }


def aggregate_metrics(detail: pd.DataFrame, mode_order: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for mode in mode_order:
        group = detail.loc[detail["mode"].eq(mode)].copy()
        background_samples = int(group["background_samples"].sum())
        target_samples = int(group["target_samples"].sum())
        false_alarms = int(group["false_alarms"].sum())
        correct = int(group["correct_detections"].sum())
        score_detections = int(group["score_detections"].sum())
        precision = correct / (correct + false_alarms)
        joint_pd = correct / target_samples
        worst_pfa_index = group["pfa"].idxmax()
        worst_pd_index = group["joint_pd"].idxmin()
        rows.append(
            {
                "mode": mode,
                "model_source": group["model_source"].iloc[0],
                "context_role": group["context_role"].iloc[0],
                "causality": group["causality"].iloc[0],
                "training_context_match": bool(
                    group["training_context_match"].iloc[0]
                ),
                "folds": len(group),
                "background_samples": background_samples,
                "target_samples": target_samples,
                "false_alarms": false_alarms,
                "pooled_pfa": false_alarms / background_samples,
                "macro_pfa": float(group["pfa"].mean()),
                "median_pfa": float(group["pfa"].median()),
                "worst_fold_pfa": float(group.loc[worst_pfa_index, "pfa"]),
                "worst_pfa_fold": str(group.loc[worst_pfa_index, "fold"]),
                "score_detections": score_detections,
                "pooled_score_pd": score_detections / target_samples,
                "correct_detections": correct,
                "pooled_joint_pd": joint_pd,
                "macro_joint_pd": float(group["joint_pd"].mean()),
                "worst_fold_joint_pd": float(
                    group.loc[worst_pd_index, "joint_pd"]
                ),
                "worst_joint_pd_fold": str(group.loc[worst_pd_index, "fold"]),
                "joint_precision": precision,
                "joint_f1": (
                    2.0 * precision * joint_pd / (precision + joint_pd)
                ),
                "macro_roc_auc": float(group["roc_auc"].mean()),
                "background_shift_mean": float(
                    np.average(
                        group["background_shift_mean"],
                        weights=group["background_samples"],
                    )
                ),
                "target_shift_mean": float(
                    np.average(
                        group["target_shift_mean"],
                        weights=group["target_samples"],
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def paired_record(
    fold: str,
    mode: str,
    labels: np.ndarray,
    baseline: PredictionPayload,
    candidate: PredictionPayload,
) -> dict[str, Any]:
    background = labels == 0
    target = labels == 1
    return {
        "fold": fold,
        "mode": mode,
        "background_shared_alarm": int(
            (background & baseline.detected & candidate.detected).sum()
        ),
        "background_complete_only_alarm": int(
            (background & baseline.detected & ~candidate.detected).sum()
        ),
        "background_candidate_only_alarm": int(
            (background & ~baseline.detected & candidate.detected).sum()
        ),
        "target_shared_correct": int(
            (target & baseline.correct & candidate.correct).sum()
        ),
        "target_complete_only_correct": int(
            (target & baseline.correct & ~candidate.correct).sum()
        ),
        "target_candidate_only_correct": int(
            (target & ~baseline.correct & candidate.correct).sum()
        ),
        "status": "post-hoc paired sensitivity; not a selection rule",
    }


def history_records(
    fold: int,
    mode: str,
    labels: np.ndarray,
    context: ScanContextResult,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, class_name in ((0, "background"), (1, "target")):
        mask = labels == label
        used = context.used_history_counts[mask]
        available = context.available_history_counts[mask]
        rows.append(
            {
                "fold": f"{fold:02d}",
                "mode": mode,
                "class_name": class_name,
                "samples": int(mask.sum()),
                "zero_context_samples": int((used == 0).sum()),
                "used_context_min": int(used.min()),
                "used_context_median": float(np.median(used)),
                "used_context_mean": float(used.mean()),
                "used_context_max": int(used.max()),
                "available_history_max": int(available.max()),
            }
        )
    return rows


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    view = frame[columns].copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: f"{value:.4f}")
    headers = [str(column) for column in view.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for _, row in view.iterrows():
        lines.append(
            "| "
            + " | ".join(
                str(row[column]).replace("|", "\\|") for column in view.columns
            )
            + " |"
        )
    return "\n".join(lines)


def build_report(
    aggregate: pd.DataFrame,
    paired: pd.DataFrame,
    replay: pd.DataFrame,
    windows: list[int],
) -> str:
    complete = aggregate.loc[aggregate["mode"].eq("full_complete_scan")].iloc[0]
    independent = aggregate.loc[
        aggregate["mode"].eq("sample_independent_bc")
    ].iloc[0]
    leave_one_out = aggregate.loc[
        aggregate["mode"].eq("full_leave_one_out")
    ].iloc[0]
    past_all = aggregate.loc[
        aggregate["mode"].eq("full_past_only_all")
    ].iloc[0]
    past_pair = paired.loc[
        paired["fold"].eq("ALL")
        & paired["mode"].eq("full_past_only_all")
    ].iloc[0]
    return "\n".join(
        [
            "# BC-DPG v3 causal-context sensitivity audit",
            "",
            "## Audit status",
            "",
            "This is a deterministic replay of frozen checkpoints and frozen test "
            "features. It performs no training, checkpoint selection, threshold "
            "selection, or test-threshold retuning. The complete-scan replay must "
            "reproduce the authoritative frozen decisions before any sensitivity "
            "result is written.",
            "",
            "The full model was trained with complete-scan context. Leave-one-out and "
            "past-only rows substitute a different context at inference time and are "
            "therefore post-hoc out-of-distribution sensitivity diagnostics, not newly "
            "trained causal models.",
            "",
            "## Six-fold aggregate",
            "",
            markdown_table(
                aggregate,
                [
                    "mode",
                    "false_alarms",
                    "pooled_pfa",
                    "correct_detections",
                    "pooled_joint_pd",
                    "worst_fold_pfa",
                    "worst_fold_joint_pd",
                    "training_context_match",
                ],
            ),
            "",
            f"The matched complete-scan replay has {int(complete['false_alarms'])} "
            f"false alarms and {int(complete['correct_detections'])}/318 joint "
            "successes. The separately trained sample-independent BC reference has "
            f"{int(independent['false_alarms'])} false alarms and "
            f"{int(independent['correct_detections'])}/318 joint successes.",
            "",
            f"Removing only the current sample from each complete scan gives "
            f"{int(leave_one_out['false_alarms'])} false alarms and "
            f"{int(leave_one_out['correct_detections'])}/318 joint successes. "
            "This isolates self-inclusion sensitivity but remains non-causal because "
            "later scan samples are still available.",
            "",
            f"Using all assumed-order prior samples with the frozen complete-scan model "
            f"gives {int(past_all['false_alarms'])} false alarms and "
            f"{int(past_all['correct_detections'])}/318 joint successes. Relative to "
            "the complete-scan replay, the paired changes are: "
            "complete-only alarms removed="
            f"{int(past_pair['background_complete_only_alarm'])}, candidate-only alarms "
            f"added={int(past_pair['background_candidate_only_alarm'])}, complete-only "
            "joint successes lost="
            f"{int(past_pair['target_complete_only_correct'])}, and candidate-only joint "
            f"successes gained={int(past_pair['target_candidate_only_correct'])}.",
            "",
            "## Context definitions",
            "",
            "- `full_complete_scan`: all samples in the scan, including the current "
            "sample and possible future samples; matched to training but non-causal.",
            "- `full_leave_one_out`: all other samples in the scan; excludes self but "
            "still uses possible future samples.",
            f"- `full_past_only_w*`: the latest {', '.join(map(str, windows))} prior "
            "samples under the assumed order.",
            "- `full_past_only_all`: all prior samples under the assumed order.",
            "- `sample_independent_bc`: separately trained with all 12 group features "
            "fixed to zero; it is not the full checkpoint with context removed.",
            "",
            "Past-only order is inferred from `(beam_layer, azimuth_deg, sample_id)`. "
            "The source data do not provide a verified per-sample acquisition timestamp, "
            "so these rows are causal only under that ordering assumption.",
            "",
            "## Replay validation",
            "",
            markdown_table(
                replay,
                [
                    "fold",
                    "rows",
                    "stored_context_max_abs_delta",
                    "frozen_score_max_abs_delta",
                    "frozen_decision_mismatches",
                ],
            ),
            "",
            "## Claim boundary",
            "",
            "This audit measures inference sensitivity of an already inspected model. "
            "It does not establish a deployable causal BC-DPG, does not remove the "
            "class/date confounding in the current data, and must not be used to select "
            "a preferred history window from test outcomes. A deployable next model "
            "must be trained and selected with causal context on training/validation "
            "data, then evaluated once on a locked external test set with recorded "
            "sample timestamps.",
            "",
        ]
    )


def source_records(sources: Iterable[FoldSources]) -> list[dict[str, Any]]:
    paths: list[tuple[str, Path]] = []
    for source in sources:
        paths.extend(
            [
                ("full_checkpoint", source.full_checkpoint),
                ("precomputed_test", source.precomputed_test),
                ("full_frozen_predictions", source.full_predictions),
                ("full_summary", source.full_summary),
                ("independent_frozen_predictions", source.independent_predictions),
                ("independent_summary", source.independent_summary),
            ]
        )
    paths.extend(
        [
            ("build_script", Path(__file__).resolve()),
            (
                "context_implementation",
                PROJECT_ROOT / "features" / "scan_context.py",
            ),
            (
                "model_implementation",
                PROJECT_ROOT / "models" / "target_protected_scan_calibrator.py",
            ),
        ]
    )
    return [
        {
            "category": category,
            "path": display_path(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for category, path in paths
    ]


def generated_records(output_dir: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(output_dir)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "audit_manifest.json"
    ]


def run_audit(
    sources: list[FoldSources],
    windows: list[int],
    batch_size: int,
    output_dir: Path,
) -> None:
    mode_order = [
        "raw_dpg",
        "sample_independent_bc",
        "full_complete_scan",
        "full_leave_one_out",
        *[f"full_past_only_w{window:02d}" for window in windows],
        "full_past_only_all",
    ]
    metric_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []

    for source in sources:
        frame = load_precomputed(source.precomputed_test)
        full_frozen = align_predictions(frame, source.full_predictions)
        independent = align_predictions(frame, source.independent_predictions)
        model, checkpoint = load_calibrator(source.full_checkpoint)
        full_summary = read_json(source.full_summary)
        independent_summary = read_json(source.independent_summary)
        threshold = float(checkpoint["base_threshold"])
        if not np.isclose(threshold, float(full_summary["base_threshold"])):
            raise ValueError(f"Full threshold mismatch in Fold {source.fold}")
        if not np.isclose(
            threshold,
            float(independent_summary["base_threshold"]),
        ):
            raise ValueError(f"Independent threshold mismatch in Fold {source.fold}")

        sample_features = frame[list(SAMPLE_FEATURE_COLUMNS)].to_numpy(np.float32)
        stored_group_features = frame[
            list(STORED_GROUP_FEATURE_COLUMNS)
        ].to_numpy(np.float32)
        raw_logit = frame["raw_logit"].to_numpy(np.float32)
        raw_score = frame["raw_score"].to_numpy(np.float64)
        labels = frame["target_present"].to_numpy(np.int64)

        contexts: dict[str, ScanContextResult] = {
            "full_complete_scan": build_scan_context_features(
                frame,
                sample_features,
                threshold,
                mode="complete_scan",
            ),
            "full_leave_one_out": build_scan_context_features(
                frame,
                sample_features,
                threshold,
                mode="leave_one_out",
            ),
            "full_past_only_all": build_scan_context_features(
                frame,
                sample_features,
                threshold,
                mode="past_only",
            ),
        }
        for window in windows:
            contexts[f"full_past_only_w{window:02d}"] = (
                build_scan_context_features(
                    frame,
                    sample_features,
                    threshold,
                    mode="past_only",
                    window_size=window,
                )
            )

        payloads: dict[str, PredictionPayload] = {
            "raw_dpg": make_payload(
                frame,
                raw_score,
                np.zeros(len(frame), dtype=np.float64),
                threshold,
                None,
            ),
            "sample_independent_bc": make_payload(
                frame,
                independent["score"].to_numpy(np.float64),
                independent["shift"].to_numpy(np.float64),
                threshold,
                None,
            ),
        }
        for mode, context in contexts.items():
            score, shift = infer_calibrator(
                model,
                sample_features,
                context.values,
                raw_logit,
                batch_size,
            )
            payloads[mode] = make_payload(
                frame,
                score,
                shift,
                threshold,
                context,
            )

        stored_context_delta = float(
            np.max(
                np.abs(
                    contexts["full_complete_scan"].values
                    - stored_group_features
                )
            )
        )
        frozen_score = full_frozen["score"].to_numpy(np.float64)
        frozen_detected = boolean_array(full_frozen["detected"], "detected")
        replay_score_delta = float(
            np.max(
                np.abs(
                    payloads["full_complete_scan"].score - frozen_score
                )
            )
        )
        replay_mismatches = int(
            np.count_nonzero(
                payloads["full_complete_scan"].detected != frozen_detected
            )
        )
        if stored_context_delta > 1e-7:
            raise ValueError(f"Stored context replay mismatch in Fold {source.fold}")
        if replay_score_delta > 1e-6 or replay_mismatches:
            raise ValueError(f"Frozen prediction replay mismatch in Fold {source.fold}")
        replay_rows.append(
            {
                "fold": f"{source.fold:02d}",
                "rows": len(frame),
                "threshold": threshold,
                "stored_context_max_abs_delta": stored_context_delta,
                "frozen_score_max_abs_delta": replay_score_delta,
                "frozen_decision_mismatches": replay_mismatches,
            }
        )

        baseline = payloads["full_complete_scan"]
        for mode in mode_order:
            payload = payloads[mode]
            metric_rows.append(
                metric_record(source.fold, mode, frame, payload, threshold)
            )
            if mode != "full_complete_scan":
                paired_rows.append(
                    paired_record(
                        f"{source.fold:02d}",
                        mode,
                        labels,
                        baseline,
                        payload,
                    )
                )
            if payload.context is not None:
                history_rows.extend(
                    history_records(
                        source.fold,
                        mode,
                        labels,
                        payload.context,
                    )
                )

    detail = pd.DataFrame(metric_rows)
    aggregate = aggregate_metrics(detail, mode_order)
    paired = pd.DataFrame(paired_rows)
    paired_all = (
        paired.drop(columns=["fold", "status"])
        .groupby("mode", sort=False, as_index=False)
        .sum(numeric_only=True)
    )
    paired_all.insert(0, "fold", "ALL")
    paired_all["status"] = "post-hoc paired sensitivity; not a selection rule"
    paired = pd.concat([paired, paired_all], ignore_index=True)
    history = pd.DataFrame(history_rows)
    replay = pd.DataFrame(replay_rows)

    if {source.fold for source in sources} == set(EXPECTED_FOLDS):
        complete = aggregate.loc[
            aggregate["mode"].eq("full_complete_scan")
        ].iloc[0]
        independent = aggregate.loc[
            aggregate["mode"].eq("sample_independent_bc")
        ].iloc[0]
        if (
            int(complete["false_alarms"]) != EXPECTED_COMPLETE_FALSE_ALARMS
            or int(complete["correct_detections"]) != EXPECTED_COMPLETE_CORRECT
        ):
            raise ValueError("Complete-scan pooled result does not match frozen evidence")
        if int(independent["false_alarms"]) != EXPECTED_INDEPENDENT_FALSE_ALARMS:
            raise ValueError("Sample-independent pooled result does not match frozen evidence")
        complete_rows = int(
            detail.loc[
                detail["mode"].eq("full_complete_scan"),
                "samples",
            ].sum()
        )
        if complete_rows != EXPECTED_ROWS:
            raise ValueError("Six-fold row count does not match frozen evidence")

    tables = {
        "context_metrics_by_fold.csv": detail,
        "context_metrics_aggregate.csv": aggregate,
        "paired_deltas_vs_complete_scan.csv": paired,
        "history_coverage_by_fold.csv": history,
        "complete_replay_validation.csv": replay,
    }
    for name, table in tables.items():
        table.to_csv(output_dir / name, index=False, lineterminator="\n")
    (output_dir / "CAUSAL_CONTEXT_AUDIT.md").write_text(
        build_report(aggregate, paired, replay, windows),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "audit_role": "post-hoc frozen-checkpoint context sensitivity",
        "deterministic_build": True,
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "threshold_selection_performed": False,
        "test_threshold_retuning": False,
        "complete_scan_replay_required": True,
        "past_only_order_columns": ["beam_layer", "azimuth_deg", "sample_id"],
        "past_only_order_verified_by_timestamp": False,
        "past_only_windows": windows,
        "folds": [source.fold for source in sources],
        "sources": source_records(sources),
        "generated_files": generated_records(output_dir),
    }
    (output_dir / "audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_output(
    sources: list[FoldSources],
    windows: list[int],
    batch_size: int,
    output_dir: Path,
    overwrite: bool,
) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"Output path is not a directory: {output_dir}")
    if output_dir.is_dir() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is nonempty: {output_dir}. Use --overwrite to replace it."
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_parent = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.build-", dir=output_dir.parent)
    )
    staging_dir = staging_parent / output_dir.name
    staging_dir.mkdir()
    try:
        run_audit(sources, windows, batch_size, staging_dir)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        staging_dir.replace(output_dir)
    except Exception:
        shutil.rmtree(staging_parent, ignore_errors=True)
        raise
    shutil.rmtree(staging_parent, ignore_errors=True)


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        experiment_root = resolve_path(args.experiment_root)
        output_dir = resolve_path(args.output_dir)
        sources = [
            fold_sources(experiment_root, fold, args.seed)
            for fold in args.folds
        ]
        validate_sources(sources)
        build_output(
            sources,
            sorted(args.windows),
            args.batch_size,
            output_dir,
            args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError, OSError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    print("BC-DPG causal-context sensitivity audit: PASS")
    print(f"output_dir={display_path(output_dir)}")
    print(f"folds={','.join(map(str, args.folds))}")
    print(f"past_only_windows={','.join(map(str, sorted(args.windows)))}")
    print("training_performed=False")
    print("test_threshold_retuning=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
