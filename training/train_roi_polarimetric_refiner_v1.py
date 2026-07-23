#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.roi_polarimetric_refinement_dataset import ROICandidateCacheDataset
from features.roi_polarimetric_refinement import ROI_MODES, logit_from_probability
from models.roi_polarimetric_refiner import ROIPolarimetricRefiner, count_parameters

BASELINE_MODE = "power2_baseline"
ALL_MODES = (BASELINE_MODE, *ROI_MODES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage-4 candidate-guided ROI refiner v1.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--mode", choices=ALL_MODES, required=True)
    parser.add_argument("--fold-id", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--max-suppression-logit", type=float, default=8.0)
    parser.add_argument("--background-tail-weight", type=float, default=1.0)
    parser.add_argument("--target-protection-weight", type=float, default=2.0)
    parser.add_argument("--shift-regularization-weight", type=float, default=0.01)
    parser.add_argument("--quality-loss-weight", type=float, default=0.2)
    parser.add_argument("--tail-margin-logit", type=float, default=0.25)
    parser.add_argument("--max-val-pd-drop", type=float, default=0.01)
    parser.add_argument("--max-val-false-alarms", type=int, default=2)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug-per-class", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: str) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = PROJECT_ROOT / value
    return value.resolve()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def debug_subset(dataset: ROICandidateCacheDataset, per_class: int) -> Dataset:
    if per_class <= 0:
        return dataset
    target = dataset.payload["target_present"].numpy().astype(int)
    indices: list[int] = []
    for label in (0, 1):
        indices.extend(np.flatnonzero(target == label)[:per_class].tolist())
    return Subset(dataset, indices)


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def safe_auc(labels: np.ndarray, scores: np.ndarray, max_fpr: float | None = None) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, scores, max_fpr=max_fpr))
    except Exception:
        if max_fpr is not None:
            return float("nan")
        order = np.argsort(scores)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(scores) + 1)
        pos = labels == 1
        n_pos = int(pos.sum())
        n_neg = int((~pos).sum())
        return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, budget: float) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    try:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(labels, scores)
        eligible = tpr[fpr <= budget + 1e-12]
        return float(eligible.max()) if eligible.size else 0.0
    except Exception:
        negatives = scores[labels == 0]
        positives = scores[labels == 1]
        allowed = int(math.floor(budget * len(negatives) + 1e-12))
        thresholds = np.r_[np.inf, np.unique(scores)[::-1]]
        best = 0.0
        for threshold in thresholds:
            if int((negatives >= threshold).sum()) <= allowed:
                best = max(best, float((positives >= threshold).mean()))
        return best


def prediction_metrics(frame: pd.DataFrame, score_column: str, threshold: float) -> dict[str, float | int]:
    labels = frame["target_present"].to_numpy(dtype=int)
    scores = frame[score_column].to_numpy(dtype=float)
    detected = scores >= float(threshold)
    positive = labels == 1
    background = labels == 0
    localized = frame["localization_ok"].to_numpy(dtype=bool)
    correct = detected & positive & localized
    false_alarm = detected & background
    range_errors = np.abs(
        frame.loc[positive, "pred_range_index"].to_numpy(dtype=float)
        - frame.loc[positive, "true_range_index"].to_numpy(dtype=float)
    )
    velocity_errors = np.abs(
        frame.loc[positive, "pred_velocity_index"].to_numpy(dtype=float)
        - frame.loc[positive, "true_velocity_index"].to_numpy(dtype=float)
    )
    return {
        "positive_count": int(positive.sum()),
        "background_count": int(background.sum()),
        "score_detected_positive_count": int((detected & positive).sum()),
        "correct_detection_count": int(correct.sum()),
        "false_alarm_count": int(false_alarm.sum()),
        "score_detection_pd": float((detected & positive).sum() / max(positive.sum(), 1)),
        "joint_pd": float(correct.sum() / max(positive.sum(), 1)),
        "pfa": float(false_alarm.sum() / max(background.sum(), 1)),
        "roc_auc": safe_auc(labels, scores),
        "partial_auc_5pct_fpr": safe_auc(labels, scores, max_fpr=0.05),
        "tpr_at_1pct_fpr": tpr_at_fpr(labels, scores, 0.01),
        "tpr_at_5pct_fpr": tpr_at_fpr(labels, scores, 0.05),
        "all_positive_range_mae_gates": float(range_errors.mean()) if range_errors.size else float("nan"),
        "all_positive_velocity_mae_bins": float(velocity_errors.mean()) if velocity_errors.size else float("nan"),
    }


def apply_threshold(frame: pd.DataFrame, score_column: str, threshold: float, prefix: str) -> pd.DataFrame:
    result = frame.copy()
    detected = result[score_column].to_numpy(dtype=float) >= float(threshold)
    positive = result["target_present"].to_numpy(dtype=int) == 1
    background = ~positive
    localized = result["localization_ok"].to_numpy(dtype=bool)
    result[f"{prefix}_threshold"] = float(threshold)
    result[f"{prefix}_detected"] = detected
    result[f"{prefix}_false_alarm"] = detected & background
    result[f"{prefix}_correct_detection"] = detected & positive & localized
    return result


def select_threshold(frame: pd.DataFrame, score_column: str, max_false_alarms: int) -> float:
    scores = frame[score_column].to_numpy(dtype=float)
    thresholds = np.r_[np.nextafter(scores.max(), np.inf), np.unique(scores)[::-1], np.nextafter(scores.min(), -np.inf)]
    best_threshold = float(thresholds[0])
    best_key: tuple[float, float, float] | None = None
    for threshold in thresholds:
        metrics = prediction_metrics(frame, score_column, float(threshold))
        if int(metrics["false_alarm_count"]) > max_false_alarms:
            continue
        key = (
            float(metrics["joint_pd"]),
            float(metrics["score_detection_pd"]),
            -float(threshold),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def batch_to_frame(batch: dict[str, Any], output: dict[str, torch.Tensor] | None) -> pd.DataFrame:
    size = len(batch["sample_id"])
    def cpu_list(name: str) -> list[Any]:
        value = batch[name]
        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
        return list(value)

    raw_score = batch["raw_score"].detach().cpu().numpy().reshape(-1)
    raw_logit = batch["raw_logit"].detach().cpu().numpy().reshape(-1)
    if output is None:
        refined_score = raw_score.copy()
        refined_logit = raw_logit.copy()
        suppression = np.zeros(size, dtype=float)
        score_shift = np.zeros(size, dtype=float)
        predicted_quality = batch["roi_quality"].detach().cpu().numpy().reshape(-1)
    else:
        refined_score = output["refined_score"].detach().cpu().numpy().reshape(-1)
        refined_logit = output["refined_logit"].detach().cpu().numpy().reshape(-1)
        suppression = output["suppression"].detach().cpu().numpy().reshape(-1)
        score_shift = output["score_shift"].detach().cpu().numpy().reshape(-1)
        predicted_quality = output["roi_quality"].detach().cpu().numpy().reshape(-1)
    return pd.DataFrame({
        "sample_id": list(batch["sample_id"]),
        "source_file": list(batch["source_file"]),
        "target_present": cpu_list("target_present"),
        "raw_score": raw_score,
        "raw_logit": raw_logit,
        "refined_score": refined_score,
        "refined_logit": refined_logit,
        "suppression": suppression,
        "score_shift": score_shift,
        "predicted_roi_quality": predicted_quality,
        "roi_quality": cpu_list("roi_quality"),
        "polarimetric_confidence": cpu_list("polarimetric_confidence"),
        "pred_range_index": cpu_list("pred_range_index"),
        "pred_velocity_index": cpu_list("pred_velocity_index"),
        "true_range_index": cpu_list("true_range_index"),
        "true_velocity_index": cpu_list("true_velocity_index"),
        "localization_ok": cpu_list("localization_ok"),
        "beam_layer": cpu_list("beam_layer"),
        "azimuth_deg": cpu_list("azimuth_deg"),
        "distance_m": cpu_list("distance_m"),
        "velocity_mps": cpu_list("velocity_mps"),
        "mat_path": list(batch["mat_path"]),
    })


def loss_terms(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    threshold_logit: torch.Tensor,
    pos_weight: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    refined_logit = output["refined_logit"]
    suppression = output["suppression"]
    detection = F.binary_cross_entropy_with_logits(
        refined_logit, target, pos_weight=pos_weight
    )
    quality = F.binary_cross_entropy(output["roi_quality"].clamp(1e-6, 1 - 1e-6), target)
    background = target < 0.5
    positive = target >= 0.5
    if background.any():
        tail = F.softplus(
            refined_logit[background] - threshold_logit + args.tail_margin_logit
        ).mean()
    else:
        tail = refined_logit.new_zeros(())
    target_protection = suppression[positive].mean() if positive.any() else suppression.new_zeros(())
    shift_regularization = suppression.square().mean()
    total = (
        detection
        + args.quality_loss_weight * quality
        + args.background_tail_weight * tail
        + args.target_protection_weight * target_protection
        + args.shift_regularization_weight * shift_regularization
    )
    return {
        "loss_total": total,
        "loss_detection": detection,
        "loss_quality": quality,
        "loss_background_tail": tail,
        "loss_target_protection": target_protection,
        "loss_shift_regularization": shift_regularization,
    }


def run_epoch(
    model: ROIPolarimetricRefiner,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    base_threshold: float,
    pos_weight: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, float]]:
    training = optimizer is not None
    model.train(training)
    frames: list[pd.DataFrame] = []
    totals: dict[str, float] = {}
    samples = 0
    threshold_logit = logit_from_probability(
        torch.tensor(base_threshold, dtype=torch.float32, device=device)
    )
    for batch in loader:
        roi = batch["roi"].to(device, non_blocking=True)
        raw_logit = batch["raw_logit"].to(device, non_blocking=True).float().reshape(-1)
        raw_score = batch["raw_score"].to(device, non_blocking=True).float().reshape(-1)
        mask = batch["roi_valid_mask"].to(device, non_blocking=True).float()
        confidence = batch["polarimetric_confidence"].to(device, non_blocking=True).float().reshape(-1)
        target = batch["target_present"].to(device, non_blocking=True).float().reshape(-1)
        with torch.set_grad_enabled(training):
            output = model(roi, raw_logit, raw_score, mask, confidence)
            losses = loss_terms(output, target, threshold_logit, pos_weight, args)
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses["loss_total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
                optimizer.step()
        batch_size = roi.shape[0]
        samples += batch_size
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu()) * batch_size
        frames.append(batch_to_frame(batch, output))
    frame = pd.concat(frames, ignore_index=True)
    return frame, {key: value / max(samples, 1) for key, value in totals.items()}


def evaluate_baseline(loader: DataLoader) -> pd.DataFrame:
    return pd.concat([batch_to_frame(batch, None) for batch in loader], ignore_index=True)


def save_checkpoint(
    path: Path,
    model: ROIPolarimetricRefiner,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    base_threshold: float,
    metrics: dict[str, Any],
) -> None:
    torch.save({
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "mode": args.mode,
        "fold_id": int(args.fold_id),
        "base_threshold": float(base_threshold),
        "model_config": {
            "in_channels": 8,
            "hidden_dim": args.hidden_dim,
            "max_suppression_logit": args.max_suppression_logit,
        },
        "validation_fixed_threshold_metrics": metrics,
        "config": vars(args),
        "sample_independent": True,
        "scan_context": False,
        "suppression_only": True,
        "power2_location_frozen": True,
    }, path)


def write_results(
    args: argparse.Namespace,
    experiment_dir: Path,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    base_threshold: float,
    best_epoch: int,
    parameter_count: int,
    history: pd.DataFrame,
    elapsed: float,
) -> None:
    table_dir = experiment_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    raw_val_metrics = prediction_metrics(val_frame, "raw_score", base_threshold)
    raw_test_metrics = prediction_metrics(test_frame, "raw_score", base_threshold)
    refined_val_metrics = prediction_metrics(val_frame, "refined_score", base_threshold)
    refined_test_metrics = prediction_metrics(test_frame, "refined_score", base_threshold)
    selected_threshold = select_threshold(
        val_frame, "refined_score", args.max_val_false_alarms
    )
    selected_val_metrics = prediction_metrics(val_frame, "refined_score", selected_threshold)
    selected_test_metrics = prediction_metrics(test_frame, "refined_score", selected_threshold)

    val_results = apply_threshold(val_frame, "raw_score", base_threshold, "raw_fixed")
    val_results = apply_threshold(val_results, "refined_score", base_threshold, "refined_fixed")
    val_results = apply_threshold(val_results, "refined_score", selected_threshold, "refined_selected")
    test_results = apply_threshold(test_frame, "raw_score", base_threshold, "raw_fixed")
    test_results = apply_threshold(test_results, "refined_score", base_threshold, "refined_fixed")
    test_results = apply_threshold(test_results, "refined_score", selected_threshold, "refined_selected")
    val_results.to_csv(table_dir / "val_predictions.csv", index=False, encoding="utf-8-sig")
    test_results.to_csv(table_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    history.to_csv(table_dir / "training_history.csv", index=False, encoding="utf-8-sig")

    summary = {
        "experiment_name": args.name,
        "stage": "candidate_guided_roi_polarimetric_refinement_v1",
        "mode": args.mode,
        "fold_id": int(args.fold_id),
        "seed": int(args.seed),
        "parameter_count": int(parameter_count),
        "best_epoch": int(best_epoch),
        "base_power2_threshold": float(base_threshold),
        "selected_refined_validation_threshold": float(selected_threshold),
        "primary_evaluation": "refined score at the unchanged Power2 deployment threshold",
        "raw_validation_fixed_threshold_metrics": raw_val_metrics,
        "raw_test_fixed_threshold_metrics": raw_test_metrics,
        "refined_validation_fixed_threshold_metrics": refined_val_metrics,
        "refined_test_fixed_threshold_metrics": refined_test_metrics,
        "refined_validation_selected_threshold_metrics": selected_val_metrics,
        "refined_test_selected_threshold_metrics": selected_test_metrics,
        "fixed_threshold_delta": {
            "false_alarms": int(refined_test_metrics["false_alarm_count"]) - int(raw_test_metrics["false_alarm_count"]),
            "joint_pd": float(refined_test_metrics["joint_pd"]) - float(raw_test_metrics["joint_pd"]),
            "roc_auc": float(refined_test_metrics["roc_auc"]) - float(raw_test_metrics["roc_auc"]),
        },
        "sample_independent": True,
        "scan_context": False,
        "suppression_only": args.mode != BASELINE_MODE,
        "power2_location_frozen": True,
        "new_fixed_threshold_false_alarms_mathematically_possible": False,
        "elapsed_seconds": float(elapsed),
        "config": vars(args),
    }
    (table_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if args.fold_id <= 0 or args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("fold, epochs and batch size must be positive")
    if not 0.0 <= args.max_val_pd_drop <= 1.0:
        raise ValueError("max_val_pd_drop must be in [0,1]")
    set_seed(args.seed)
    cache_dir = resolve(args.cache_dir)
    datasets = {
        split: ROICandidateCacheDataset(cache_dir / f"{split}.pt", ROI_MODES[0] if args.mode == BASELINE_MODE else args.mode)
        for split in ("train", "val", "test")
    }
    base_thresholds = {round(dataset.base_threshold, 12) for dataset in datasets.values()}
    if len(base_thresholds) != 1:
        raise ValueError(f"Cache threshold mismatch: {base_thresholds}")
    base_threshold = float(next(iter(base_thresholds)))
    active_datasets: dict[str, Dataset] = {
        split: debug_subset(dataset, args.debug_per_class)
        for split, dataset in datasets.items()
    }
    loaders = {
        split: make_loader(active_datasets[split], args.batch_size, split == "train", args.num_workers)
        for split in ("train", "val", "test")
    }

    experiment_dir = PROJECT_ROOT / "results/experiments" / args.name
    if experiment_dir.exists() and not args.overwrite:
        if (experiment_dir / "tables/summary.json").is_file() and (experiment_dir / "checkpoints/best.pt").is_file():
            print(f"[complete] {experiment_dir}")
            return
        raise FileExistsError(f"Incomplete result exists: {experiment_dir}; use --overwrite")
    if experiment_dir.exists():
        shutil.rmtree(experiment_dir)
    (experiment_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (experiment_dir / "tables").mkdir(parents=True, exist_ok=True)

    print("=" * 94)
    print(f"experiment / fold / mode : {args.name} / {args.fold_id} / {args.mode}")
    print(f"base threshold           : {base_threshold:.6f}")
    print("dataset sizes            : " + ", ".join(f"{k}={len(v)}" for k, v in active_datasets.items()))
    print("sample independent       : true")
    print("Power2 location frozen   : true")
    print("=" * 94)
    start = time.time()

    if args.mode == BASELINE_MODE:
        val_frame = evaluate_baseline(loaders["val"])
        test_frame = evaluate_baseline(loaders["test"])
        checkpoint = {
            "epoch": 0,
            "mode": BASELINE_MODE,
            "fold_id": int(args.fold_id),
            "base_threshold": base_threshold,
            "model_state_dict": {},
            "sample_independent": True,
            "power2_location_frozen": True,
            "config": vars(args),
        }
        torch.save(checkpoint, experiment_dir / "checkpoints/best.pt")
        history = pd.DataFrame([{"epoch": 0, "role": "frozen_power2_baseline"}])
        write_results(
            args, experiment_dir, val_frame, test_frame, base_threshold,
            best_epoch=0, parameter_count=0, history=history, elapsed=time.time() - start,
        )
        print(f"baseline result: {experiment_dir}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ROIPolarimetricRefiner(
        hidden_dim=args.hidden_dim,
        max_suppression_logit=args.max_suppression_logit,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.scheduler_patience,
        min_lr=args.min_learning_rate,
    )
    train_target = datasets["train"].payload["target_present"].float()
    n_pos = float(train_target.sum().item())
    n_neg = float(len(train_target) - n_pos)
    pos_weight = torch.tensor(max(n_neg / max(n_pos, 1.0), 1.0), device=device)

    raw_val_frame = evaluate_baseline(loaders["val"])
    raw_val_metrics = prediction_metrics(raw_val_frame, "raw_score", base_threshold)
    minimum_pd = max(0.0, float(raw_val_metrics["joint_pd"]) - args.max_val_pd_drop)
    history_rows: list[dict[str, Any]] = []
    best_key: tuple[Any, ...] | None = None
    stale = 0

    for epoch in range(1, args.epochs + 1):
        _, train_losses = run_epoch(
            model, loaders["train"], optimizer, device, base_threshold, pos_weight, args
        )
        val_frame, val_losses = run_epoch(
            model, loaders["val"], None, device, base_threshold, pos_weight, args
        )
        metrics = prediction_metrics(val_frame, "refined_score", base_threshold)
        scheduler.step(val_losses["loss_total"])
        eligible = float(metrics["joint_pd"]) + 1e-12 >= minimum_pd
        if eligible:
            key = (
                1,
                -float(metrics["pfa"]),
                float(metrics["joint_pd"]),
                float(metrics["partial_auc_5pct_fpr"]),
                -float(val_losses["loss_total"]),
            )
        else:
            key = (
                0,
                float(metrics["joint_pd"]),
                -float(metrics["pfa"]),
                float(metrics["partial_auc_5pct_fpr"]),
                -float(val_losses["loss_total"]),
            )
        improved = best_key is None or key > best_key
        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_losses.items()},
            **{f"val_{k}": v for k, v in val_losses.items()},
            "val_fixed_joint_pd": metrics["joint_pd"],
            "val_fixed_pfa": metrics["pfa"],
            "val_auc": metrics["roc_auc"],
            "val_pauc_5pct": metrics["partial_auc_5pct_fpr"],
            "eligible_pd_constraint": eligible,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history_rows.append(row)
        if improved:
            best_key = key
            stale = 0
            save_checkpoint(
                experiment_dir / "checkpoints/best.pt", model, optimizer, epoch,
                args, base_threshold, metrics,
            )
        else:
            stale += 1
        save_checkpoint(
            experiment_dir / "checkpoints/last.pt", model, optimizer, epoch,
            args, base_threshold, metrics,
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs} | loss={val_losses['loss_total']:.5f} "
            f"Pd={float(metrics['joint_pd']):.4f} Pfa={float(metrics['pfa']):.4f} "
            f"pAUC5={float(metrics['partial_auc_5pct_fpr']):.4f} "
            f"eligible={eligible}" + (" | best" if improved else "")
        )
        if stale >= args.early_stopping_patience:
            print(f"Early stopping after {stale} stale epochs")
            break

    best = torch.load(experiment_dir / "checkpoints/best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"], strict=True)
    val_frame, _ = run_epoch(
        model, loaders["val"], None, device, base_threshold, pos_weight, args
    )
    test_frame, _ = run_epoch(
        model, loaders["test"], None, device, base_threshold, pos_weight, args
    )
    history = pd.DataFrame(history_rows)
    write_results(
        args, experiment_dir, val_frame, test_frame, base_threshold,
        best_epoch=int(best["epoch"]), parameter_count=count_parameters(model),
        history=history, elapsed=time.time() - start,
    )
    summary = json.loads((experiment_dir / "tables/summary.json").read_text(encoding="utf-8"))
    metrics = summary["refined_test_fixed_threshold_metrics"]
    print("=" * 94)
    print(f"best epoch : {best['epoch']}")
    print(
        f"test fixed: Pd={float(metrics['joint_pd']):.4f} "
        f"Pfa={float(metrics['pfa']):.4f} FA={int(metrics['false_alarm_count'])} "
        f"AUC={float(metrics['roc_auc']):.4f}"
    )
    print(f"result     : {experiment_dir}")
    print("=" * 94)


if __name__ == "__main__":
    main()
