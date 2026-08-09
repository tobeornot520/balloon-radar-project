#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.detection_dataset_v3 import DetectionRadarDatasetV3
from models.background_tail_calibrated_dpg_fcn import (
    BackgroundTailCalibratedDPGFCN,
    background_tail_loss,
)
from models.dual_branch_gated_fcn import DualBranchGatedFCN
from scripts.train_detection_baseline_v2 import (
    DetectionTolerance,
    MemoryCachedDataset,
    apply_threshold_and_metrics,
    json_safe,
    make_debug_subset,
    select_threshold_at_false_alarm_budget,
    set_seed,
)


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
            "Train BC-DPG-FCN v2 with downward-only background "
            "tail suppression"
        )
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument(
        "--output-root",
        default="results/experiments",
    )

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler-patience", type=int, default=3)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=5.0,
    )

    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument(
        "--hidden-dims",
        type=parse_hidden_dims,
        default=(48, 24),
    )
    parser.add_argument("--initial-shift", type=float, default=1e-3)
    parser.add_argument(
        "--background-margin-probability",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--positive-floor-probability",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--pairwise-margin",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--background-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--positive-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--pairwise-weight",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--shift-regularization",
        type=float,
        default=0.01,
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
    parser.add_argument("--no-memory-cache", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def extract_scores_and_peaks(
    logits: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        score.detach().cpu().numpy(),
        range_index.detach().cpu().numpy(),
        velocity_index.detach().cpu().numpy(),
    )


def load_model(
    checkpoint_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[BackgroundTailCalibratedDPGFCN, dict[str, Any]]:
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
            f"checkpoint mismatch: {result}"
        )

    model = BackgroundTailCalibratedDPGFCN(
        base,
        topk=args.topk,
        hidden_dims=args.hidden_dims,
        initial_shift=args.initial_shift,
        freeze_base=True,
    ).to(device)
    return model, checkpoint


def resolve_defaults(
    args: argparse.Namespace,
    base_checkpoint: dict[str, Any],
) -> None:
    config = base_checkpoint.get("config", {})
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
        "background_margin_probability": float(
            base_checkpoint.get("threshold", 0.5)
        ),
    }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)


def build_datasets(
    manifest_path: Path,
    args: argparse.Namespace,
) -> dict[str, Dataset]:
    datasets: dict[str, Dataset] = {}
    for split in ("train", "val", "test"):
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
        datasets[split] = dataset
    return datasets


def build_loaders(
    datasets: dict[str, Dataset],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, DataLoader]:
    return {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
        )
        for split, dataset in datasets.items()
    }


def loss_from_outputs(
    model: BackgroundTailCalibratedDPGFCN,
    outputs: dict[str, Any],
    labels: torch.Tensor,
    args: argparse.Namespace,
):
    sample_logits = model.sample_logits(
        outputs["calibrated_logits"]
    )
    return background_tail_loss(
        sample_logits,
        labels,
        outputs["shift"],
        background_margin_probability=(
            args.background_margin_probability
        ),
        positive_floor_probability=(
            args.positive_floor_probability
        ),
        pairwise_margin=args.pairwise_margin,
        background_weight=args.background_weight,
        positive_weight=args.positive_weight,
        pairwise_weight=args.pairwise_weight,
        shift_regularization=args.shift_regularization,
    )


def train_epoch(
    model: BackgroundTailCalibratedDPGFCN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
    amp_enabled: bool,
) -> dict[str, float]:
    model.train()
    keys = (
        "loss",
        "background_tail_loss",
        "positive_floor_loss",
        "pairwise_loss",
        "shift_penalty",
    )
    sums = {key: 0.0 for key in keys}
    count = 0

    for batch in loader:
        inputs = batch["input"].to(
            device,
            non_blocking=True,
        )
        labels = batch["target_present"].to(
            device,
            non_blocking=True,
        ).float()

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            outputs = model(inputs)
            loss, parts = loss_from_outputs(
                model,
                outputs,
                labels,
                args,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.shift_head.parameters(),
            args.gradient_clip_norm,
        )
        scaler.step(optimizer)
        scaler.update()

        batch_size = inputs.shape[0]
        for key in keys:
            sums[key] += float(parts[key].item()) * batch_size
        count += batch_size

    if count == 0:
        raise RuntimeError("training loader is empty")
    return {key: value / count for key, value in sums.items()}


def collect(
    model: BackgroundTailCalibratedDPGFCN,
    loader: DataLoader,
    device: torch.device,
    tolerance: DetectionTolerance,
    args: argparse.Namespace,
    amp_enabled: bool,
) -> tuple[pd.DataFrame, dict[str, float]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    keys = (
        "loss",
        "background_tail_loss",
        "positive_floor_loss",
        "pairwise_loss",
        "shift_penalty",
    )
    sums = {key: 0.0 for key in keys}
    count = 0

    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(
                device,
                non_blocking=True,
            )
            labels = batch["target_present"].to(
                device,
                non_blocking=True,
            ).float()

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                outputs = model(inputs)
                _, parts = loss_from_outputs(
                    model,
                    outputs,
                    labels,
                    args,
                )

            calibrated_score, raw_r, raw_v = (
                extract_scores_and_peaks(
                    outputs["calibrated_logits"]
                )
            )
            raw_score, _, _ = extract_scores_and_peaks(
                outputs["raw_logits"]
            )

            present = (
                batch["target_present"]
                .cpu().numpy().astype(int)
            )
            true_r = np.asarray(batch["range_index"]).astype(int)
            true_v = np.asarray(batch["velocity_index"]).astype(int)
            shifts = (
                outputs["shift"].cpu().numpy().reshape(-1)
            )
            gates = (
                outputs["gate_weights"]
                .float().cpu().numpy()
            )

            batch_size = inputs.shape[0]
            for key in keys:
                sums[key] += float(parts[key].item()) * batch_size
            count += batch_size

            for index in range(batch_size):
                if present[index]:
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

                rows.append(
                    {
                        "sample_id": batch["sample_id"][index],
                        "target_present": int(present[index]),
                        "score": float(calibrated_score[index]),
                        "raw_score": float(raw_score[index]),
                        "pred_range_index": int(raw_r[index]),
                        "pred_velocity_index": int(raw_v[index]),
                        "true_range_index": int(true_r[index]),
                        "true_velocity_index": int(true_v[index]),
                        "range_error_gates": range_error,
                        "velocity_error_bins": velocity_error,
                        "localization_ok": bool(localization_ok),
                        "shift": float(shifts[index]),
                        "gate_h": float(gates[index, 0]),
                        "gate_v": float(gates[index, 1]),
                        "beam_layer": int(
                            np.asarray(batch["beam_layer"])[index]
                        ),
                        "azimuth_deg": float(
                            np.asarray(batch["azimuth_deg"])[index]
                        ),
                        "mat_path": batch["mat_path"][index],
                    }
                )

    if count == 0:
        raise RuntimeError("evaluation loader is empty")

    return pd.DataFrame(rows), {
        key: value / count for key, value in sums.items()
    }


def fixed_metrics(
    frame: pd.DataFrame,
    threshold: float,
    tolerance: DetectionTolerance,
):
    return apply_threshold_and_metrics(
        frame,
        float(threshold),
        tolerance,
    )


def selection_key(
    metrics: dict[str, Any],
    auc: float,
    loss: float,
) -> tuple[float, float, float, float]:
    return (
        float(metrics.get("joint_pd", -math.inf)),
        -float(metrics.get("pfa", math.inf)),
        float(auc),
        -float(loss),
    )


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: BackgroundTailCalibratedDPGFCN,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    selected_threshold: float,
    selected_metrics: dict[str, Any],
    base_threshold_metrics: dict[str, Any],
    losses: dict[str, float],
    args: argparse.Namespace,
    base_checkpoint_path: Path,
    base_checkpoint: dict[str, Any],
) -> None:
    torch.save(
        {
            "epoch": int(epoch),
            "shift_head_state_dict": (
                model.shift_head.state_dict()
            ),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "selected_threshold": float(selected_threshold),
            "selected_validation_metrics": selected_metrics,
            "base_threshold_validation_metrics": (
                base_threshold_metrics
            ),
            "validation_losses": losses,
            "base_checkpoint": str(base_checkpoint_path),
            "base_checkpoint_threshold": float(
                base_checkpoint.get("threshold", 0.5)
            ),
            "model_config": {
                "topk": args.topk,
                "hidden_dims": list(args.hidden_dims),
                "initial_shift": args.initial_shift,
            },
            "config": vars(args),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    manifest_path = resolve_project_path(args.manifest_path)
    base_checkpoint_path = resolve_project_path(
        args.base_checkpoint
    )
    output_root = resolve_project_path(args.output_root)
    experiment_dir = output_root / args.name

    if experiment_dir.exists() and args.overwrite:
        shutil.rmtree(experiment_dir)
    if experiment_dir.exists() and not args.resume:
        raise FileExistsError(
            f"experiment exists: {experiment_dir}"
        )

    checkpoint_dir = experiment_dir / "checkpoints"
    table_dir = experiment_dir / "tables"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    amp_enabled = device.type == "cuda" and not args.no_amp

    model, base_checkpoint = load_model(
        base_checkpoint_path,
        args,
        device,
    )
    resolve_defaults(args, base_checkpoint)

    base_threshold = float(base_checkpoint["threshold"])
    tolerance = DetectionTolerance(
        int(args.range_tolerance_gates),
        int(args.velocity_tolerance_bins),
    )

    datasets = build_datasets(manifest_path, args)
    loaders = build_loaders(datasets, args, device)

    optimizer = torch.optim.AdamW(
        model.shift_head.parameters(),
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
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=amp_enabled,
    )

    history: list[dict[str, Any]] = []
    best_key = (-math.inf, -math.inf, -math.inf, -math.inf)
    stale = 0
    start_epoch = 1
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"

    if args.resume:
        resume = torch.load(
            last_path,
            map_location=device,
            weights_only=False,
        )
        model.shift_head.load_state_dict(
            resume["shift_head_state_dict"],
            strict=True,
        )
        optimizer.load_state_dict(
            resume["optimizer_state_dict"]
        )
        scheduler.load_state_dict(
            resume["scheduler_state_dict"]
        )
        start_epoch = int(resume["epoch"]) + 1

    print("=" * 78)
    print(f"Experiment               : {args.name}")
    print(f"Device / AMP             : {device} / {amp_enabled}")
    print(f"Base threshold           : {base_threshold:.6f}")
    print(
        "Background margin       : "
        f"{args.background_margin_probability:.6f}"
    )
    print(
        "Dataset sizes           : "
        f"{ {k: len(v) for k, v in datasets.items()} }"
    )
    print(
        "Trainable parameters    : "
        f"{sum(p.numel() for p in model.shift_head.parameters())}"
    )
    print("=" * 78)

    start_time = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        train_losses = train_epoch(
            model,
            loaders["train"],
            optimizer,
            scaler,
            device,
            args,
            amp_enabled,
        )
        val_frame, val_losses = collect(
            model,
            loaders["val"],
            device,
            tolerance,
            args,
            amp_enabled,
        )

        selected_threshold, _ = (
            select_threshold_at_false_alarm_budget(
                val_frame,
                int(args.max_val_false_alarms),
            )
        )
        _, selected_metrics = fixed_metrics(
            val_frame,
            selected_threshold,
            tolerance,
        )
        _, base_metrics = fixed_metrics(
            val_frame,
            base_threshold,
            tolerance,
        )

        scheduler.step(val_losses["loss"])
        current_key = selection_key(
            base_metrics,
            float(selected_metrics.get("roc_auc", math.nan)),
            val_losses["loss"],
        )
        improved = current_key > best_key

        save_checkpoint(
            last_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            selected_threshold=selected_threshold,
            selected_metrics=selected_metrics,
            base_threshold_metrics=base_metrics,
            losses=val_losses,
            args=args,
            base_checkpoint_path=base_checkpoint_path,
            base_checkpoint=base_checkpoint,
        )

        if improved:
            best_key = current_key
            stale = 0
            shutil.copy2(last_path, best_path)
        else:
            stale += 1

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
                "base_threshold_pd": base_metrics["joint_pd"],
                "base_threshold_pfa": base_metrics["pfa"],
                "shift_mean": float(val_frame["shift"].mean()),
                "shift_background_mean": float(
                    val_frame.loc[
                        val_frame["target_present"] == 0,
                        "shift",
                    ].mean()
                ),
                "shift_positive_mean": float(
                    val_frame.loc[
                        val_frame["target_present"] == 1,
                        "shift",
                    ].mean()
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
            f"train={train_losses['loss']:.6f} "
            f"val={val_losses['loss']:.6f} "
            f"basePd={base_metrics['joint_pd']:.4f} "
            f"basePfa={base_metrics['pfa']:.4f} "
            f"AUC={selected_metrics['roc_auc']:.4f} "
            f"shift={val_frame['shift'].mean():.4f} "
            f"{'*' if improved else ''}"
        )

        if stale >= args.early_stopping_patience:
            print(f"Early stopping after {stale} stale epochs")
            break

    best = torch.load(
        best_path,
        map_location=device,
        weights_only=False,
    )
    model.shift_head.load_state_dict(
        best["shift_head_state_dict"],
        strict=True,
    )

    frames = {}
    losses = {}
    for split in ("val", "test"):
        frame, split_losses = collect(
            model,
            loaders[split],
            device,
            tolerance,
            args,
            amp_enabled,
        )
        frames[split] = frame
        losses[split] = split_losses

    selected_threshold, selected_curve = (
        select_threshold_at_false_alarm_budget(
            frames["val"],
            int(args.max_val_false_alarms),
        )
    )
    val_results, val_metrics = fixed_metrics(
        frames["val"],
        selected_threshold,
        tolerance,
    )
    test_results, test_metrics = fixed_metrics(
        frames["test"],
        selected_threshold,
        tolerance,
    )

    raw_val = frames["val"].copy()
    raw_val["score"] = raw_val["raw_score"]
    raw_test = frames["test"].copy()
    raw_test["score"] = raw_test["raw_score"]

    raw_selected_threshold, raw_curve = (
        select_threshold_at_false_alarm_budget(
            raw_val,
            int(args.max_val_false_alarms),
        )
    )
    _, raw_val_metrics = fixed_metrics(
        raw_val,
        raw_selected_threshold,
        tolerance,
    )
    _, raw_test_metrics = fixed_metrics(
        raw_test,
        raw_selected_threshold,
        tolerance,
    )

    base_val_results, base_val_metrics = fixed_metrics(
        frames["val"],
        base_threshold,
        tolerance,
    )
    base_test_results, base_test_metrics = fixed_metrics(
        frames["test"],
        base_threshold,
        tolerance,
    )
    _, raw_base_val_metrics = fixed_metrics(
        raw_val,
        base_threshold,
        tolerance,
    )
    raw_base_test_results, raw_base_test_metrics = (
        fixed_metrics(
            raw_test,
            base_threshold,
            tolerance,
        )
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
        "best_epoch": int(best["epoch"]),
        "base_checkpoint": str(base_checkpoint_path),
        "base_threshold": base_threshold,
        "background_margin_probability": (
            args.background_margin_probability
        ),
        "selected_threshold": float(selected_threshold),
        "raw_selected_threshold": float(raw_selected_threshold),
        "dataset_sizes": {
            split: len(dataset)
            for split, dataset in datasets.items()
        },
        "trainable_parameter_count": sum(
            p.numel() for p in model.shift_head.parameters()
        ),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "raw_validation_metrics": raw_val_metrics,
        "raw_test_metrics": raw_test_metrics,
        "base_threshold_validation_metrics": base_val_metrics,
        "base_threshold_test_metrics": base_test_metrics,
        "raw_base_threshold_validation_metrics": (
            raw_base_val_metrics
        ),
        "raw_base_threshold_test_metrics": raw_base_test_metrics,
        "validation_losses": losses["val"],
        "test_losses": losses["test"],
        "shift_statistics_validation": {
            "all_mean": float(frames["val"]["shift"].mean()),
            "background_mean": float(
                frames["val"].loc[
                    frames["val"]["target_present"] == 0,
                    "shift",
                ].mean()
            ),
            "positive_mean": float(
                frames["val"].loc[
                    frames["val"]["target_present"] == 1,
                    "shift",
                ].mean()
            ),
        },
        "shift_statistics_test": {
            "all_mean": float(frames["test"]["shift"].mean()),
            "background_mean": float(
                frames["test"].loc[
                    frames["test"]["target_present"] == 0,
                    "shift",
                ].mean()
            ),
            "positive_mean": float(
                frames["test"].loc[
                    frames["test"]["target_present"] == 1,
                    "shift",
                ].mean()
            ),
        },
        "score_never_increased_validation": bool(
            (
                frames["val"]["score"]
                <= frames["val"]["raw_score"] + 1e-7
            ).all()
        ),
        "score_never_increased_test": bool(
            (
                frames["test"]["score"]
                <= frames["test"]["raw_score"] + 1e-7
            ).all()
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

    print("\n" + "=" * 78)
    print(f"Best epoch: {best['epoch']}")
    print(
        "Selected-threshold BC test : "
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
        f"Base threshold {base_threshold:.6f} BC test : "
        f"Pd={base_test_metrics['joint_pd']:.4f}, "
        f"Pfa={base_test_metrics['pfa']:.4f}"
    )
    print(
        f"Base threshold {base_threshold:.6f} raw test: "
        f"Pd={raw_base_test_metrics['joint_pd']:.4f}, "
        f"Pfa={raw_base_test_metrics['pfa']:.4f}"
    )
    print(
        "Score never increased: "
        f"val={summary['score_never_increased_validation']}, "
        f"test={summary['score_never_increased_test']}"
    )
    print(f"Summary: {table_dir / 'summary.json'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
