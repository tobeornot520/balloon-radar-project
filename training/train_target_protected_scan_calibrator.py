#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.detection_dataset_v3 import DetectionRadarDatasetV3
from features.scan_context import (
    DEFAULT_ORDER_COLUMNS,
    GROUP_FEATURE_DIM,
    build_scan_context_features,
)
from models.background_tail_calibrated_dpg_fcn import (
    BackgroundTailCalibratedDPGFCN,
)
from models.dual_branch_gated_fcn import DualBranchGatedFCN
from models.target_protected_scan_calibrator import (
    TargetProtectedScanCalibrator,
    target_protected_scan_loss,
)
from scripts.train_detection_baseline_v2 import (
    DetectionTolerance,
    MemoryCachedDataset,
    apply_threshold_and_metrics,
    json_safe,
    make_debug_subset,
    select_threshold_at_false_alarm_budget,
    set_seed,
)


SCAN_PATTERN = re.compile(r"^(\d{8}_\d{6})")
SAMPLE_FEATURE_DIM = 24


def parse_hidden_dims(text: str) -> tuple[int, ...]:
    values = tuple(
        int(part.strip())
        for part in text.split(",")
        if part.strip()
    )
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(
            "hidden dims must be positive comma-separated integers"
        )
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train BC-DPG-FCN v3 target-protected, scan-aware "
            "calibrator on frozen DPG features."
        )
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument(
        "--output-root",
        default="results/experiments",
    )

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--scheduler-patience",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=5.0,
    )

    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument(
        "--scan-context-mode",
        choices=("complete_scan", "past_only"),
        default="complete_scan",
        help="Context used consistently for train, validation, and test splits.",
    )
    parser.add_argument(
        "--history-window",
        type=int,
        default=None,
        help="Maximum past samples for past_only; omit to use all prior samples.",
    )
    parser.add_argument(
        "--allow-inferred-order",
        action="store_true",
        help=(
            "Acknowledge that past_only currently orders samples by beam layer, "
            "azimuth, and sample ID rather than verified acquisition timestamps."
        ),
    )
    parser.add_argument(
        "--hidden-dims",
        type=parse_hidden_dims,
        default=(64, 32),
    )
    parser.add_argument("--maximum-shift", type=float, default=3.0)
    parser.add_argument(
        "--initial-background-probability",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--initial-suppression",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--allowed-target-shift",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--target-keep-weight",
        type=float,
        default=8.0,
    )
    parser.add_argument(
        "--background-tail-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--background-classification-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--pairwise-weight",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--shift-selectivity-weight",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--shift-selectivity-margin",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--shift-regularization",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--pd-tolerance",
        type=float,
        default=0.01,
        help=(
            "Best checkpoints must preserve validation Pd within this "
            "absolute tolerance whenever possible."
        ),
    )

    parser.add_argument("--range-sigma", type=float, default=None)
    parser.add_argument("--velocity-sigma", type=float, default=None)
    parser.add_argument(
        "--max-val-false-alarms",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--range-tolerance-gates",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--velocity-tolerance-bins",
        type=int,
        default=None,
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug-per-class", type=int, default=0)
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Train and evaluate train/validation splits without loading test data.",
    )
    parser.add_argument("--no-memory-cache", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_scan_context_args(args: argparse.Namespace) -> None:
    if args.history_window is not None and args.history_window <= 0:
        raise ValueError("history-window must be positive")
    if args.scan_context_mode == "complete_scan":
        if args.history_window is not None:
            raise ValueError("history-window is only valid for past_only context")
        if args.allow_inferred_order:
            raise ValueError("allow-inferred-order is only valid for past_only context")
        return
    if not args.allow_inferred_order:
        columns = ", ".join(DEFAULT_ORDER_COLUMNS)
        raise ValueError(
            "past_only requires verified acquisition order, which is unavailable. "
            f"For development-only smoke runs using ({columns}), explicitly pass "
            "--allow-inferred-order."
        )


def scan_context_metadata(args: argparse.Namespace) -> dict[str, Any]:
    inferred = args.scan_context_mode == "past_only"
    return {
        "mode": args.scan_context_mode,
        "history_window": args.history_window,
        "order_columns": list(DEFAULT_ORDER_COLUMNS) if inferred else [],
        "order_verified_by_timestamp": False,
        "evidence_role": (
            "development_only_inferred_order"
            if inferred
            else "offline_complete_scan"
        ),
    }


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def scan_group(sample_id: str, mat_path: str = "") -> str:
    for candidate in (str(sample_id), Path(str(mat_path)).stem):
        match = SCAN_PATTERN.match(candidate)
        if match:
            return match.group(1)
    return str(sample_id).split("_beam", 1)[0]


def probability_to_logit(value: float) -> float:
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return float(torch.logit(torch.tensor(value)).item())


class CachedFeatureDataset(Dataset):
    def __init__(
        self,
        sample_features: np.ndarray,
        group_features: np.ndarray,
        raw_logit: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self.sample_features = torch.from_numpy(
            sample_features.astype(np.float32)
        )
        self.group_features = torch.from_numpy(
            group_features.astype(np.float32)
        )
        self.raw_logit = torch.from_numpy(
            raw_logit.astype(np.float32)
        )
        self.labels = torch.from_numpy(
            labels.astype(np.float32)
        )

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "sample_features": self.sample_features[index],
            "group_features": self.group_features[index],
            "raw_logit": self.raw_logit[index],
            "label": self.labels[index],
            "index": torch.tensor(index, dtype=torch.long),
        }


def load_base(
    checkpoint_path: Path,
    device: torch.device,
    topk: int,
) -> tuple[
    DualBranchGatedFCN,
    BackgroundTailCalibratedDPGFCN,
    dict[str, Any],
]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint.get("config", {})
    base = DualBranchGatedFCN(
        gate_hidden_dim=int(config.get("gate_hidden_dim", 16))
    )
    result = base.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"base checkpoint mismatch: {result}"
        )

    base.to(device)
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)

    feature_extractor = BackgroundTailCalibratedDPGFCN(
        base,
        topk=topk,
        hidden_dims=(8,),
        initial_shift=1e-3,
        freeze_base=True,
    ).to(device)
    feature_extractor.eval()

    return base, feature_extractor, checkpoint


def resolve_defaults(
    args: argparse.Namespace,
    checkpoint: dict[str, Any],
) -> None:
    config = checkpoint.get("config", {})
    defaults = {
        "range_sigma": config.get("range_sigma", 5.0),
        "velocity_sigma": config.get("velocity_sigma", 5.0),
        "max_val_false_alarms": config.get(
            "max_val_false_alarms",
            2,
        ),
        "range_tolerance_gates": config.get(
            "range_tolerance_gates",
            2,
        ),
        "velocity_tolerance_bins": config.get(
            "velocity_tolerance_bins",
            3,
        ),
    }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)


def build_source_dataset(
    manifest_path: Path,
    split: str,
    args: argparse.Namespace,
) -> Dataset:
    dataset: Dataset = DetectionRadarDatasetV3(
        manifest_path=manifest_path,
        split=split,
        channel_mode="HV",
        range_sigma=float(args.range_sigma),
        velocity_sigma=float(args.velocity_sigma),
    )
    dataset = make_debug_subset(
        dataset,
        args.debug_per_class,
    )
    if not args.no_memory_cache:
        dataset = MemoryCachedDataset(dataset)
    return dataset


def extract_raw_peak(
    logits: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flattened = logits.float().flatten(start_dim=1)
    max_logit, flat_index = flattened.max(dim=1)
    score = torch.sigmoid(max_logit)
    width = logits.shape[-1]
    velocity_index = torch.div(
        flat_index,
        width,
        rounding_mode="floor",
    )
    range_index = flat_index % width
    return (
        max_logit.cpu().numpy(),
        score.cpu().numpy(),
        range_index.cpu().numpy(),
        velocity_index.cpu().numpy(),
    )


def precompute_split(
    split: str,
    dataset: Dataset,
    base: DualBranchGatedFCN,
    feature_extractor: BackgroundTailCalibratedDPGFCN,
    device: torch.device,
    args: argparse.Namespace,
    tolerance: DetectionTolerance,
    amp_enabled: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    feature_batches: list[np.ndarray] = []
    raw_logit_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(
                device,
                non_blocking=True,
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                base_output = base(inputs)

            with torch.autocast(
                device_type=device.type,
                enabled=False,
            ):
                features = feature_extractor.build_features(
                    inputs.float(),
                    {
                        key: value.float()
                        for key, value in base_output.items()
                    },
                )
                raw_logits = base_output[
                    "fusion_logits"
                ].float()

            raw_logit, raw_score, raw_r, raw_v = extract_raw_peak(
                raw_logits
            )
            labels = (
                batch["target_present"]
                .cpu().numpy().astype(np.int64)
            )
            true_r = np.asarray(
                batch["range_index"]
            ).astype(np.int64)
            true_v = np.asarray(
                batch["velocity_index"]
            ).astype(np.int64)
            gates = (
                base_output["gate_weights"]
                .float().cpu().numpy()
            )

            feature_batches.append(
                features.cpu().numpy().astype(np.float32)
            )
            raw_logit_batches.append(
                raw_logit.astype(np.float32)
            )
            label_batches.append(
                labels.astype(np.float32)
            )

            for index in range(inputs.shape[0]):
                if labels[index]:
                    range_error = abs(
                        int(raw_r[index]) - int(true_r[index])
                    )
                    velocity_error = abs(
                        int(raw_v[index]) - int(true_v[index])
                    )
                    localization_ok = (
                        range_error <= tolerance.range_gates
                        and velocity_error <= tolerance.velocity_bins
                    )
                else:
                    range_error = math.nan
                    velocity_error = math.nan
                    localization_ok = False

                sample_id = batch["sample_id"][index]
                mat_path = batch["mat_path"][index]

                rows.append(
                    {
                        "sample_id": sample_id,
                        "scan_group": scan_group(
                            sample_id,
                            mat_path,
                        ),
                        "target_present": int(labels[index]),
                        "raw_logit": float(raw_logit[index]),
                        "raw_score": float(raw_score[index]),
                        "pred_range_index": int(raw_r[index]),
                        "pred_velocity_index": int(raw_v[index]),
                        "true_range_index": int(true_r[index]),
                        "true_velocity_index": int(true_v[index]),
                        "range_error_gates": range_error,
                        "velocity_error_bins": velocity_error,
                        "localization_ok": bool(localization_ok),
                        "gate_h": float(gates[index, 0]),
                        "gate_v": float(gates[index, 1]),
                        "beam_layer": int(
                            np.asarray(batch["beam_layer"])[index]
                        ),
                        "azimuth_deg": float(
                            np.asarray(batch["azimuth_deg"])[index]
                        ),
                        "mat_path": mat_path,
                        "split": split,
                    }
                )

    sample_features = np.concatenate(
        feature_batches,
        axis=0,
    )
    raw_logit = np.concatenate(
        raw_logit_batches,
        axis=0,
    )
    labels = np.concatenate(
        label_batches,
        axis=0,
    )
    frame = pd.DataFrame(rows)

    if sample_features.shape[1] != SAMPLE_FEATURE_DIM:
        raise RuntimeError(
            f"Expected {SAMPLE_FEATURE_DIM} sample features, "
            f"got {sample_features.shape[1]}"
        )

    context = build_scan_context_features(
        frame,
        sample_features,
        float(args.base_threshold),
        mode=args.scan_context_mode,
        window_size=args.history_window,
    )
    frame["context_used_samples"] = context.used_history_counts
    frame["context_available_samples"] = context.available_history_counts
    return (
        sample_features,
        context.values,
        raw_logit,
        labels,
        frame,
    )


def build_group_features(
    frame: pd.DataFrame,
    sample_features: np.ndarray,
    base_threshold: float,
    *,
    mode: str = "complete_scan",
    window_size: int | None = None,
) -> np.ndarray:
    return build_scan_context_features(
        frame,
        sample_features,
        base_threshold,
        mode=mode,
        window_size=window_size,
    ).values


def context_coverage(
    precomputed: dict[
        str,
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame],
    ],
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for split, values in precomputed.items():
        frame = values[4]
        used = frame["context_used_samples"].to_numpy(dtype=np.int64)
        available = frame["context_available_samples"].to_numpy(dtype=np.int64)
        result[split] = {
            "samples": int(len(frame)),
            "zero_context_samples": int((used == 0).sum()),
            "used_min": int(used.min()),
            "used_median": float(np.median(used)),
            "used_mean": float(used.mean()),
            "used_max": int(used.max()),
            "available_max": int(available.max()),
        }
    return result


def write_validation_only_outputs(
    *,
    args: argparse.Namespace,
    model: TargetProtectedScanCalibrator,
    best: dict[str, Any],
    precomputed: dict[
        str,
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame],
    ],
    val_frame: pd.DataFrame,
    val_losses: dict[str, float],
    tolerance: DetectionTolerance,
    table_dir: Path,
    start_time: float,
) -> None:
    selected_threshold, selected_curve = select_threshold_at_false_alarm_budget(
        val_frame,
        int(args.max_val_false_alarms),
    )
    val_results, val_metrics = apply_metrics(
        val_frame,
        selected_threshold,
        tolerance,
    )
    raw_val = val_frame.copy()
    raw_val["score"] = raw_val["raw_score"]
    raw_threshold, raw_curve = select_threshold_at_false_alarm_budget(
        raw_val,
        int(args.max_val_false_alarms),
    )
    _, raw_val_metrics = apply_metrics(raw_val, raw_threshold, tolerance)
    base_val_results, base_val_metrics = apply_metrics(
        val_frame,
        args.base_threshold,
        tolerance,
    )
    _, raw_base_val_metrics = apply_metrics(
        raw_val,
        args.base_threshold,
        tolerance,
    )

    val_results.to_csv(
        table_dir / "val_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    base_val_results.to_csv(
        table_dir / "base_threshold_val_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selected_curve.to_csv(
        table_dir / "val_threshold_curve.csv",
        index=False,
        encoding="utf-8-sig",
    )
    raw_curve.to_csv(
        table_dir / "raw_val_threshold_curve.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "experiment_name": args.name,
        "evaluation_scope": "validation_only",
        "test_evaluation_performed": False,
        "performance_evidence_role": (
            "interface_smoke_only"
            if args.allow_inferred_order
            else "validation_selection"
        ),
        "best_epoch": int(best["epoch"]),
        "base_threshold": args.base_threshold,
        "selected_threshold": float(selected_threshold),
        "raw_selected_threshold": float(raw_threshold),
        "dataset_sizes": {
            split: int(precomputed[split][3].shape[0])
            for split in ("train", "val")
        },
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "scan_context": scan_context_metadata(args),
        "context_coverage": context_coverage(precomputed),
        "validation_metrics": val_metrics,
        "raw_validation_metrics": raw_val_metrics,
        "base_threshold_validation_metrics": base_val_metrics,
        "raw_base_threshold_validation_metrics": raw_base_val_metrics,
        "validation_losses": val_losses,
        "pd_floor_validation": {
            "raw_pd": float(raw_base_val_metrics["joint_pd"]),
            "required_minimum": float(
                raw_base_val_metrics["joint_pd"] - args.pd_tolerance
            ),
            "bc_pd": float(base_val_metrics["joint_pd"]),
            "satisfied": bool(
                base_val_metrics["joint_pd"]
                >= raw_base_val_metrics["joint_pd"] - args.pd_tolerance
            ),
        },
        "score_never_increased_validation": bool(
            val_frame["score_never_increased"].all()
        ),
        "elapsed_seconds": time.time() - start_time,
        "config": vars(args),
    }
    (table_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 82)
    print(f"Best epoch: {best['epoch']}")
    print(
        "Selected-threshold validation: "
        f"Pd={val_metrics['joint_pd']:.4f}, "
        f"Pfa={val_metrics['pfa']:.4f}, "
        f"AUC={val_metrics['roc_auc']:.4f}"
    )
    print("Test split loaded/evaluated: False")
    print(f"Summary: {table_dir / 'summary.json'}")
    print("=" * 82)


def save_precomputed(
    table_dir: Path,
    split: str,
    sample_features: np.ndarray,
    group_features: np.ndarray,
    frame: pd.DataFrame,
) -> None:
    output = frame.copy()
    for index in range(sample_features.shape[1]):
        output[f"sample_feature_{index:02d}"] = (
            sample_features[:, index]
        )
    for index in range(group_features.shape[1]):
        output[f"group_feature_{index:02d}"] = (
            group_features[:, index]
        )
    output.to_csv(
        table_dir / f"precomputed_{split}.csv",
        index=False,
        encoding="utf-8-sig",
    )


def create_feature_loader(
    data: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        pd.DataFrame,
    ],
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = CachedFeatureDataset(
        data[0],
        data[1],
        data[2],
        data[3],
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def train_epoch(
    model: TargetProtectedScanCalibrator,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    background_margin_logit: float,
) -> dict[str, float]:
    model.train()
    keys = (
        "loss",
        "background_classification_loss",
        "background_tail_loss",
        "target_keep_loss",
        "pairwise_loss",
        "shift_selectivity_loss",
        "shift_regularization",
        "background_shift_mean",
        "target_shift_mean",
    )
    sums = {key: 0.0 for key in keys}
    count = 0

    for batch in loader:
        sample_features = batch["sample_features"].to(device)
        group_features = batch["group_features"].to(device)
        raw_logit = batch["raw_logit"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            sample_features,
            group_features,
            raw_logit,
        )
        loss, parts = target_protected_scan_loss(
            outputs,
            labels,
            background_margin_logit=background_margin_logit,
            allowed_target_shift=args.allowed_target_shift,
            target_keep_weight=args.target_keep_weight,
            background_tail_weight=args.background_tail_weight,
            background_classification_weight=(
                args.background_classification_weight
            ),
            pairwise_weight=args.pairwise_weight,
            shift_selectivity_weight=(
                args.shift_selectivity_weight
            ),
            shift_selectivity_margin=(
                args.shift_selectivity_margin
            ),
            shift_regularization=args.shift_regularization,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            args.gradient_clip_norm,
        )
        optimizer.step()

        batch_size = labels.shape[0]
        for key in keys:
            sums[key] += float(parts[key].item()) * batch_size
        count += batch_size

    return {
        key: value / max(count, 1)
        for key, value in sums.items()
    }


def collect_predictions(
    model: TargetProtectedScanCalibrator,
    loader: DataLoader,
    metadata: pd.DataFrame,
    device: torch.device,
    args: argparse.Namespace,
    background_margin_logit: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    model.eval()
    outputs_by_index: dict[int, dict[str, float]] = {}
    keys = (
        "loss",
        "background_classification_loss",
        "background_tail_loss",
        "target_keep_loss",
        "pairwise_loss",
        "shift_selectivity_loss",
        "shift_regularization",
        "background_shift_mean",
        "target_shift_mean",
    )
    sums = {key: 0.0 for key in keys}
    count = 0

    with torch.no_grad():
        for batch in loader:
            sample_features = batch["sample_features"].to(device)
            group_features = batch["group_features"].to(device)
            raw_logit = batch["raw_logit"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                sample_features,
                group_features,
                raw_logit,
            )
            _, parts = target_protected_scan_loss(
                outputs,
                labels,
                background_margin_logit=background_margin_logit,
                allowed_target_shift=args.allowed_target_shift,
                target_keep_weight=args.target_keep_weight,
                background_tail_weight=args.background_tail_weight,
                background_classification_weight=(
                    args.background_classification_weight
                ),
                pairwise_weight=args.pairwise_weight,
                shift_selectivity_weight=(
                    args.shift_selectivity_weight
                ),
                shift_selectivity_margin=(
                    args.shift_selectivity_margin
                ),
                shift_regularization=args.shift_regularization,
            )

            indices = batch["index"].cpu().numpy()
            values = {
                key: tensor.cpu().numpy()
                for key, tensor in outputs.items()
            }
            for position, row_index in enumerate(indices):
                outputs_by_index[int(row_index)] = {
                    "score": float(
                        values["calibrated_score"][position]
                    ),
                    "raw_score": float(
                        values["raw_score"][position]
                    ),
                    "p_background": float(
                        values["p_background"][position]
                    ),
                    "suppression": float(
                        values["suppression"][position]
                    ),
                    "shift": float(
                        values["shift"][position]
                    ),
                    "calibrated_logit": float(
                        values["calibrated_logit"][position]
                    ),
                }

            batch_size = labels.shape[0]
            for key in keys:
                sums[key] += float(parts[key].item()) * batch_size
            count += batch_size

    frame = metadata.copy().reset_index(drop=True)
    ordered = [
        outputs_by_index[index]
        for index in range(len(frame))
    ]
    prediction_frame = pd.DataFrame(ordered)

    frame["score"] = prediction_frame["score"]
    frame["raw_score"] = prediction_frame["raw_score"]
    frame["p_background"] = prediction_frame[
        "p_background"
    ]
    frame["suppression"] = prediction_frame["suppression"]
    frame["shift"] = prediction_frame["shift"]
    frame["calibrated_logit"] = prediction_frame[
        "calibrated_logit"
    ]
    frame["score_never_increased"] = (
        frame["score"] <= frame["raw_score"] + 1e-7
    )

    return frame, {
        key: value / max(count, 1)
        for key, value in sums.items()
    }


def checkpoint_key(
    bc_metrics: dict[str, Any],
    raw_validation_pd: float,
    val_auc: float,
    val_loss: float,
    pd_tolerance: float,
) -> tuple[float, ...]:
    bc_pd = float(bc_metrics["joint_pd"])
    bc_pfa = float(bc_metrics["pfa"])
    eligible = bc_pd >= raw_validation_pd - pd_tolerance

    if eligible:
        return (
            1.0,
            -bc_pfa,
            bc_pd,
            val_auc,
            -val_loss,
        )
    return (
        0.0,
        bc_pd,
        -bc_pfa,
        val_auc,
        -val_loss,
    )


def apply_metrics(
    frame: pd.DataFrame,
    threshold: float,
    tolerance: DetectionTolerance,
):
    return apply_threshold_and_metrics(
        frame,
        float(threshold),
        tolerance,
    )


def main() -> None:
    args = parse_args()
    validate_scan_context_args(args)
    set_seed(args.seed)

    manifest_path = resolve_path(args.manifest_path)
    base_checkpoint_path = resolve_path(
        args.base_checkpoint
    )
    output_root = resolve_path(args.output_root)
    experiment_dir = output_root / args.name

    if experiment_dir.exists() and args.overwrite:
        shutil.rmtree(experiment_dir)
    if experiment_dir.exists():
        raise FileExistsError(
            f"Experiment already exists: {experiment_dir}"
        )

    checkpoint_dir = experiment_dir / "checkpoints"
    table_dir = experiment_dir / "tables"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    amp_enabled = device.type == "cuda" and not args.no_amp

    base, feature_extractor, checkpoint = load_base(
        base_checkpoint_path,
        device,
        args.topk,
    )
    resolve_defaults(args, checkpoint)

    args.base_threshold = float(checkpoint["threshold"])
    background_margin_logit = probability_to_logit(
        args.base_threshold
    )

    tolerance = DetectionTolerance(
        range_gates=int(args.range_tolerance_gates),
        velocity_bins=int(args.velocity_tolerance_bins),
    )

    print("=" * 82)
    print(f"Experiment             : {args.name}")
    print(f"Device / AMP           : {device} / {amp_enabled}")
    print(f"Base checkpoint        : {base_checkpoint_path}")
    print(f"Base threshold         : {args.base_threshold:.6f}")
    print(f"Pd tolerance           : {args.pd_tolerance:.4f}")
    print(f"Scan context           : {args.scan_context_mode}")
    if args.scan_context_mode == "past_only":
        window = "all" if args.history_window is None else args.history_window
        print(f"History window         : {window}")
        print("Order timestamp-verified: False (development-only)")
    print("=" * 82)

    precomputed = {}
    splits = ("train", "val") if args.validation_only else ("train", "val", "test")
    for split in splits:
        source_dataset = build_source_dataset(
            manifest_path,
            split,
            args,
        )
        print(
            f"Precomputing frozen DPG features: "
            f"{split} ({len(source_dataset)} samples)"
        )
        precomputed[split] = precompute_split(
            split,
            source_dataset,
            base,
            feature_extractor,
            device,
            args,
            tolerance,
            amp_enabled,
        )
        save_precomputed(
            table_dir,
            split,
            precomputed[split][0],
            precomputed[split][1],
            precomputed[split][4],
        )

    train_loader = create_feature_loader(
        precomputed["train"],
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = create_feature_loader(
        precomputed["val"],
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = (
        None
        if args.validation_only
        else create_feature_loader(
            precomputed["test"],
            batch_size=args.batch_size,
            shuffle=False,
        )
    )

    model = TargetProtectedScanCalibrator(
        sample_feature_dim=SAMPLE_FEATURE_DIM,
        group_feature_dim=GROUP_FEATURE_DIM,
        hidden_dims=args.hidden_dims,
        maximum_shift=args.maximum_shift,
        initial_background_probability=(
            args.initial_background_probability
        ),
        initial_suppression=args.initial_suppression,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.scheduler_patience,
        min_lr=args.min_learning_rate,
    )

    raw_val = precomputed["val"][4].copy()
    raw_val["score"] = raw_val["raw_score"]
    _, raw_base_val_metrics = apply_metrics(
        raw_val,
        args.base_threshold,
        tolerance,
    )
    raw_validation_pd = float(
        raw_base_val_metrics["joint_pd"]
    )

    print(
        "Raw validation at base threshold: "
        f"Pd={raw_base_val_metrics['joint_pd']:.4f}, "
        f"Pfa={raw_base_val_metrics['pfa']:.4f}"
    )
    print(
        "Trainable parameters  : "
        f"{sum(p.numel() for p in model.parameters())}"
    )
    print("=" * 82)

    best_key = (-math.inf,)
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    stale = 0
    history: list[dict[str, Any]] = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_losses = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args,
            background_margin_logit,
        )
        val_frame, val_losses = collect_predictions(
            model,
            val_loader,
            precomputed["val"][4],
            device,
            args,
            background_margin_logit,
        )

        selected_threshold, _ = (
            select_threshold_at_false_alarm_budget(
                val_frame,
                int(args.max_val_false_alarms),
            )
        )
        _, selected_metrics = apply_metrics(
            val_frame,
            selected_threshold,
            tolerance,
        )
        _, base_metrics = apply_metrics(
            val_frame,
            args.base_threshold,
            tolerance,
        )

        current_key = checkpoint_key(
            base_metrics,
            raw_validation_pd,
            float(selected_metrics["roc_auc"]),
            val_losses["loss"],
            args.pd_tolerance,
        )
        improved = current_key > best_key

        checkpoint_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "selection_key": current_key,
            "selected_threshold": float(selected_threshold),
            "selected_validation_metrics": selected_metrics,
            "base_threshold_validation_metrics": base_metrics,
            "raw_base_threshold_validation_metrics": (
                raw_base_val_metrics
            ),
            "validation_losses": val_losses,
            "base_checkpoint": str(base_checkpoint_path),
            "base_threshold": args.base_threshold,
            "config": vars(args),
        }
        torch.save(checkpoint_payload, last_path)

        if improved:
            best_key = current_key
            stale = 0
            shutil.copy2(last_path, best_path)
        else:
            stale += 1

        scheduler.step(val_losses["loss"])

        background_part = val_frame.loc[
            val_frame["target_present"] == 0
        ]
        target_part = val_frame.loc[
            val_frame["target_present"] == 1
        ]

        history.append(
            {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **{
                    f"train_{key}": value
                    for key, value in train_losses.items()
                },
                **{
                    f"val_{key}": value
                    for key, value in val_losses.items()
                },
                "selected_threshold": selected_threshold,
                "selected_pd": selected_metrics["joint_pd"],
                "selected_pfa": selected_metrics["pfa"],
                "selected_auc": selected_metrics["roc_auc"],
                "base_pd": base_metrics["joint_pd"],
                "base_pfa": base_metrics["pfa"],
                "raw_base_pd": raw_validation_pd,
                "pd_floor_satisfied": (
                    base_metrics["joint_pd"]
                    >= raw_validation_pd - args.pd_tolerance
                ),
                "background_shift_mean": float(
                    background_part["shift"].mean()
                ),
                "target_shift_mean": float(
                    target_part["shift"].mean()
                ),
                "background_probability_mean": float(
                    background_part["p_background"].mean()
                ),
                "target_probability_mean": float(
                    target_part["p_background"].mean()
                ),
            }
        )
        pd.DataFrame(history).to_csv(
            table_dir / "training_history.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"epoch={epoch:03d} "
            f"train={train_losses['loss']:.5f} "
            f"val={val_losses['loss']:.5f} "
            f"basePd={base_metrics['joint_pd']:.4f} "
            f"basePfa={base_metrics['pfa']:.4f} "
            f"rawPd={raw_validation_pd:.4f} "
            f"bgShift={background_part['shift'].mean():.3f} "
            f"targetShift={target_part['shift'].mean():.3f} "
            f"{'*' if improved else ''}"
        )

        if stale >= args.early_stopping_patience:
            print(
                f"Early stopping after {stale} stale epochs"
            )
            break

    best = torch.load(
        best_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(
        best["model_state_dict"],
        strict=True,
    )

    val_frame, val_losses = collect_predictions(
        model,
        val_loader,
        precomputed["val"][4],
        device,
        args,
        background_margin_logit,
    )
    if args.validation_only:
        write_validation_only_outputs(
            args=args,
            model=model,
            best=best,
            precomputed=precomputed,
            val_frame=val_frame,
            val_losses=val_losses,
            tolerance=tolerance,
            table_dir=table_dir,
            start_time=start_time,
        )
        return

    if test_loader is None:
        raise RuntimeError("test loader is required outside validation-only mode")
    test_frame, test_losses = collect_predictions(
        model,
        test_loader,
        precomputed["test"][4],
        device,
        args,
        background_margin_logit,
    )

    selected_threshold, selected_curve = (
        select_threshold_at_false_alarm_budget(
            val_frame,
            int(args.max_val_false_alarms),
        )
    )
    val_results, val_metrics = apply_metrics(
        val_frame,
        selected_threshold,
        tolerance,
    )
    test_results, test_metrics = apply_metrics(
        test_frame,
        selected_threshold,
        tolerance,
    )

    raw_val = val_frame.copy()
    raw_val["score"] = raw_val["raw_score"]
    raw_test = test_frame.copy()
    raw_test["score"] = raw_test["raw_score"]

    raw_threshold, raw_curve = (
        select_threshold_at_false_alarm_budget(
            raw_val,
            int(args.max_val_false_alarms),
        )
    )
    _, raw_val_metrics = apply_metrics(
        raw_val,
        raw_threshold,
        tolerance,
    )
    _, raw_test_metrics = apply_metrics(
        raw_test,
        raw_threshold,
        tolerance,
    )

    base_val_results, base_val_metrics = apply_metrics(
        val_frame,
        args.base_threshold,
        tolerance,
    )
    base_test_results, base_test_metrics = apply_metrics(
        test_frame,
        args.base_threshold,
        tolerance,
    )
    _, raw_base_val_metrics = apply_metrics(
        raw_val,
        args.base_threshold,
        tolerance,
    )
    raw_base_test_results, raw_base_test_metrics = apply_metrics(
        raw_test,
        args.base_threshold,
        tolerance,
    )

    for name, frame in (
        ("val_predictions", val_results),
        ("test_predictions", test_results),
        ("base_threshold_val_predictions", base_val_results),
        ("base_threshold_test_predictions", base_test_results),
        (
            "raw_base_threshold_test_predictions",
            raw_base_test_results,
        ),
    ):
        frame.to_csv(
            table_dir / f"{name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    selected_curve.to_csv(
        table_dir / "val_threshold_curve.csv",
        index=False,
        encoding="utf-8-sig",
    )
    raw_curve.to_csv(
        table_dir / "raw_val_threshold_curve.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "experiment_name": args.name,
        "evaluation_scope": "validation_and_test",
        "test_evaluation_performed": True,
        "best_epoch": int(best["epoch"]),
        "base_threshold": args.base_threshold,
        "selected_threshold": float(selected_threshold),
        "raw_selected_threshold": float(raw_threshold),
        "dataset_sizes": {
            split: int(precomputed[split][3].shape[0])
            for split in ("train", "val", "test")
        },
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
        ),
        "scan_context": scan_context_metadata(args),
        "context_coverage": context_coverage(precomputed),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "raw_validation_metrics": raw_val_metrics,
        "raw_test_metrics": raw_test_metrics,
        "base_threshold_validation_metrics": base_val_metrics,
        "base_threshold_test_metrics": base_test_metrics,
        "raw_base_threshold_validation_metrics": (
            raw_base_val_metrics
        ),
        "raw_base_threshold_test_metrics": (
            raw_base_test_metrics
        ),
        "pd_floor_validation": {
            "raw_pd": float(
                raw_base_val_metrics["joint_pd"]
            ),
            "required_minimum": float(
                raw_base_val_metrics["joint_pd"]
                - args.pd_tolerance
            ),
            "bc_pd": float(base_val_metrics["joint_pd"]),
            "satisfied": bool(
                base_val_metrics["joint_pd"]
                >= raw_base_val_metrics["joint_pd"]
                - args.pd_tolerance
            ),
        },
        "shift_statistics_validation": {
            "background_mean": float(
                val_frame.loc[
                    val_frame["target_present"] == 0,
                    "shift",
                ].mean()
            ),
            "target_mean": float(
                val_frame.loc[
                    val_frame["target_present"] == 1,
                    "shift",
                ].mean()
            ),
        },
        "shift_statistics_test": {
            "background_mean": float(
                test_frame.loc[
                    test_frame["target_present"] == 0,
                    "shift",
                ].mean()
            ),
            "target_mean": float(
                test_frame.loc[
                    test_frame["target_present"] == 1,
                    "shift",
                ].mean()
            ),
        },
        "background_probability_statistics_test": {
            "background_mean": float(
                test_frame.loc[
                    test_frame["target_present"] == 0,
                    "p_background",
                ].mean()
            ),
            "target_mean": float(
                test_frame.loc[
                    test_frame["target_present"] == 1,
                    "p_background",
                ].mean()
            ),
        },
        "score_never_increased_validation": bool(
            val_frame["score_never_increased"].all()
        ),
        "score_never_increased_test": bool(
            test_frame["score_never_increased"].all()
        ),
        "elapsed_seconds": time.time() - start_time,
        "config": vars(args),
    }

    (table_dir / "summary.json").write_text(
        json.dumps(
            json_safe(summary),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 82)
    print(f"Best epoch: {best['epoch']}")
    print(
        "Selected-threshold v3 test : "
        f"Pd={test_metrics['joint_pd']:.4f}, "
        f"Pfa={test_metrics['pfa']:.4f}, "
        f"AUC={test_metrics['roc_auc']:.4f}"
    )
    print(
        "Selected-threshold raw test: "
        f"Pd={raw_test_metrics['joint_pd']:.4f}, "
        f"Pfa={raw_test_metrics['pfa']:.4f}, "
        f"AUC={raw_test_metrics['roc_auc']:.4f}"
    )
    print(
        f"Base threshold {args.base_threshold:.6f} v3 test : "
        f"Pd={base_test_metrics['joint_pd']:.4f}, "
        f"Pfa={base_test_metrics['pfa']:.4f}"
    )
    print(
        f"Base threshold {args.base_threshold:.6f} raw test: "
        f"Pd={raw_base_test_metrics['joint_pd']:.4f}, "
        f"Pfa={raw_base_test_metrics['pfa']:.4f}"
    )
    print(
        "Test mean shift: "
        f"background={summary['shift_statistics_test']['background_mean']:.4f}, "
        f"target={summary['shift_statistics_test']['target_mean']:.4f}"
    )
    print(
        "Test p(background): "
        f"background={summary['background_probability_statistics_test']['background_mean']:.4f}, "
        f"target={summary['background_probability_statistics_test']['target_mean']:.4f}"
    )
    print(
        "Validation Pd floor satisfied: "
        f"{summary['pd_floor_validation']['satisfied']}"
    )
    print(
        "Score never increased: "
        f"val={summary['score_never_increased_validation']}, "
        f"test={summary['score_never_increased_test']}"
    )
    print(f"Summary: {table_dir / 'summary.json'}")
    print("=" * 82)


if __name__ == "__main__":
    main()
