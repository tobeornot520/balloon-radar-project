#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.polarimetric_detection_dataset_v2 import (
    PolarimetricDetectionDatasetV2,
    representation_channels,
)
from models.polarimetric_representation_fcn import (
    PolarimetricRepresentationFCN,
    count_parameters,
)
from scripts.train_detection_baseline_v2 import (
    DetectionTolerance,
    MemoryCachedDataset,
    SampleWeightedHeatmapMSE,
    apply_threshold_and_metrics,
    collect_predictions,
    json_safe,
    make_debug_subset,
    make_loader,
    plot_history,
    plot_score_histogram,
    resolve_project_path,
    select_threshold_at_false_alarm_budget,
    set_seed,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a fixed-capacity sample-independent gated polarimetric representation FCN v2."
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument(
        "--input-mode", choices=["power2", "ri4", "polar6_gated", "ri8_gated"], required=True
    )
    parser.add_argument("--dataset-version", default="V4")
    parser.add_argument(
        "--split-strategy", default="grouped_source_file_and_scan_temporal_holdout"
    )
    parser.add_argument("--fold-id", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--positive-sample-weight", type=float, default=10.0)
    parser.add_argument("--range-sigma", type=float, default=5.0)
    parser.add_argument("--velocity-sigma", type=float, default=5.0)
    parser.add_argument("--velocity-window", type=int, default=5)
    parser.add_argument("--range-window", type=int, default=3)
    parser.add_argument("--zdr-clip-db", type=float, default=20.0)
    parser.add_argument("--gate-low-percentile", type=float, default=50.0)
    parser.add_argument("--gate-high-percentile", type=float, default=99.0)
    parser.add_argument("--gate-gamma", type=float, default=1.5)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--max-val-false-alarms", type=int, default=2)
    parser.add_argument("--range-tolerance-gates", type=int, default=2)
    parser.add_argument("--velocity-tolerance-bins", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug-per-class", type=int, default=0)
    parser.add_argument("--no-memory-cache", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "epochs", "batch_size", "learning_rate", "min_learning_rate",
        "positive_sample_weight", "range_sigma", "velocity_sigma",
        "velocity_window", "range_window", "zdr_clip_db", "gate_gamma",
        "early_stopping_patience", "scheduler_patience", "gradient_clip_norm",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not 0.0 <= args.gate_low_percentile < args.gate_high_percentile <= 100.0:
        raise ValueError("Require gate low < gate high within [0,100]")
    if args.fold_id <= 0:
        raise ValueError("--fold-id must be positive")
    if args.num_workers < 0 or args.debug_per_class < 0:
        raise ValueError("count arguments cannot be negative")


def save_checkpoint(
    path: Path,
    model: PolarimetricRepresentationFCN,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    threshold: float,
    val_metrics: dict[str, float | int],
) -> None:
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "input_mode": args.input_mode,
            "active_channels": list(representation_channels(args.input_mode)),
            "canonical_channels": 8,
            "threshold": float(threshold),
            "validation_metrics": val_metrics,
            "dataset_version": args.dataset_version,
            "manifest_path": args.manifest_path,
            "split_strategy": args.split_strategy,
            "fold_id": int(args.fold_id),
            "config": vars(args),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    manifest_path = resolve_project_path(args.manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    args.manifest_path = str(manifest_path)

    experiment_dir = PROJECT_ROOT / "results" / "experiments" / args.name
    if experiment_dir.exists() and not args.overwrite:
        complete = (
            (experiment_dir / "checkpoints/best.pt").is_file()
            and (experiment_dir / "tables/summary.json").is_file()
        )
        if complete:
            print(f"[complete] {experiment_dir}")
            return
        raise FileExistsError(
            f"Incomplete experiment exists: {experiment_dir}. Use --overwrite to replace it."
        )
    if experiment_dir.exists() and args.overwrite:
        import shutil
        shutil.rmtree(experiment_dir)
    checkpoint_dir = experiment_dir / "checkpoints"
    table_dir = experiment_dir / "tables"
    figure_dir = experiment_dir / "figures"
    for directory in (checkpoint_dir, table_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda" and not args.no_amp
    tolerance = DetectionTolerance(
        range_gates=args.range_tolerance_gates,
        velocity_bins=args.velocity_tolerance_bins,
    )

    base_datasets = {
        split: PolarimetricDetectionDatasetV2(
            manifest_path=manifest_path,
            split=split,
            input_mode=args.input_mode,
            range_sigma=args.range_sigma,
            velocity_sigma=args.velocity_sigma,
            velocity_window=args.velocity_window,
            range_window=args.range_window,
            zdr_clip_db=args.zdr_clip_db,
            gate_low_percentile=args.gate_low_percentile,
            gate_high_percentile=args.gate_high_percentile,
            gate_gamma=args.gate_gamma,
        )
        for split in ("train", "val", "test")
    }
    datasets: dict[str, Dataset] = {
        split: make_debug_subset(dataset, args.debug_per_class)
        for split, dataset in base_datasets.items()
    }
    if not args.no_memory_cache:
        datasets = {
            split: MemoryCachedDataset(dataset, label=f"{args.input_mode}-{split}")
            for split, dataset in datasets.items()
        }
    loaders = {
        split: make_loader(
            datasets[split], args.batch_size, split == "train", args.num_workers, device
        )
        for split in ("train", "val", "test")
    }

    model = PolarimetricRepresentationFCN().to(device)
    criterion = SampleWeightedHeatmapMSE(args.positive_sample_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.scheduler_patience,
        min_lr=args.min_learning_rate,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    print("=" * 82)
    print(f"experiment       : {args.name}")
    print(f"fold / mode      : {args.fold_id} / {args.input_mode}")
    print(f"active channels  : {representation_channels(args.input_mode)}")
    print(f"canonical input  : 8 channels")
    print(f"parameters       : {count_parameters(model):,}")
    print(f"device / AMP     : {device} / {amp_enabled}")
    print("dataset sizes    : " + ", ".join(f"{k}={len(v)}" for k, v in datasets.items()))
    print("=" * 82)

    history_rows: list[dict[str, float | int]] = []
    best_key: tuple[float, float, float] | None = None
    stale_epochs = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, loaders["train"], criterion, optimizer, scaler, device,
            args.gradient_clip_norm, amp_enabled,
        )
        val_frame, val_loss = collect_predictions(
            model, loaders["val"], criterion, device, tolerance, amp_enabled
        )
        threshold, _ = select_threshold_at_false_alarm_budget(
            val_frame, args.max_val_false_alarms
        )
        _, val_metrics = apply_threshold_and_metrics(val_frame, threshold, tolerance)
        scheduler.step(val_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": current_lr,
            "val_threshold": threshold,
            "val_joint_pd": float(val_metrics["joint_pd"]),
            "val_pfa": float(val_metrics["pfa"]),
            "val_auc": float(val_metrics["roc_auc"]),
        }
        history_rows.append(row)
        key = (
            float(val_metrics["joint_pd"]),
            -float(val_metrics["pfa"]),
            -float(val_loss),
        )
        improved = best_key is None or key > best_key
        if improved:
            best_key = key
            stale_epochs = 0
            save_checkpoint(
                checkpoint_dir / "best.pt", model, optimizer, epoch, args,
                threshold, val_metrics,
            )
        else:
            stale_epochs += 1
        save_checkpoint(
            checkpoint_dir / "last.pt", model, optimizer, epoch, args,
            threshold, val_metrics,
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs} | train={train_loss:.6f} "
            f"val={val_loss:.6f} | thr={threshold:.5f} "
            f"Pd={float(val_metrics['joint_pd']):.4f} "
            f"Pfa={float(val_metrics['pfa']):.4f} | lr={current_lr:.2e}"
            + (" | best" if improved else "")
        )
        if stale_epochs >= args.early_stopping_patience:
            print(f"Early stopping after {stale_epochs} stale epochs")
            break

    history = pd.DataFrame(history_rows)
    history.to_csv(table_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    plot_history(history, figure_dir / "training_loss.png")

    best_checkpoint = torch.load(
        checkpoint_dir / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])
    val_frame, best_val_loss = collect_predictions(
        model, loaders["val"], criterion, device, tolerance, amp_enabled
    )
    threshold, threshold_curve = select_threshold_at_false_alarm_budget(
        val_frame, args.max_val_false_alarms
    )
    val_results, val_metrics = apply_threshold_and_metrics(
        val_frame, threshold, tolerance
    )
    test_frame, test_loss = collect_predictions(
        model, loaders["test"], criterion, device, tolerance, amp_enabled
    )
    test_results, test_metrics = apply_threshold_and_metrics(
        test_frame, threshold, tolerance
    )

    val_results.to_csv(table_dir / "val_predictions.csv", index=False, encoding="utf-8-sig")
    test_results.to_csv(table_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    threshold_curve.to_csv(
        table_dir / "val_threshold_curve.csv", index=False, encoding="utf-8-sig"
    )
    plot_score_histogram(
        val_results, threshold, figure_dir / "val_score_histogram.png", "Validation scores"
    )
    plot_score_histogram(
        test_results, threshold, figure_dir / "test_score_histogram.png", "Test scores"
    )

    summary = {
        "experiment_name": args.name,
        "benchmark": "sample_independent_polarimetric_representation_v2_gated",
        "input_mode": args.input_mode,
        "active_channels": list(representation_channels(args.input_mode)),
        "canonical_channels": 8,
        "same_architecture_across_modes": True,
        "parameter_count": count_parameters(model),
        "dataset_version": args.dataset_version,
        "manifest_path": args.manifest_path,
        "split_strategy": args.split_strategy,
        "fold_id": int(args.fold_id),
        "seed": int(args.seed),
        "device": str(device),
        "amp_enabled": amp_enabled,
        "dataset_sizes": {k: len(v) for k, v in datasets.items()},
        "best_epoch": int(best_checkpoint["epoch"]),
        "validation_threshold": float(threshold),
        "max_val_false_alarms": int(args.max_val_false_alarms),
        "tolerance": asdict(tolerance),
        "best_val_loss": float(best_val_loss),
        "test_loss": float(test_loss),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "elapsed_seconds": float(time.time() - start_time),
        "gate_config": {
            "low_percentile": float(args.gate_low_percentile),
            "high_percentile": float(args.gate_high_percentile),
            "gamma": float(args.gate_gamma),
        },
        "calibration_warning": (
            "relative_ZDR_like and relative phase are not absolute calibrated polarimetric quantities"
        ),
        "config": vars(args),
    }
    (table_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n" + "=" * 82)
    print(f"best epoch : {best_checkpoint['epoch']}")
    print(f"threshold  : {threshold:.6f}")
    print(
        f"test       : Pd={float(test_metrics['joint_pd']):.4f}, "
        f"Pfa={float(test_metrics['pfa']):.4f}, "
        f"FA={int(test_metrics['false_alarm_count'])}, "
        f"AUC={float(test_metrics['roc_auc']):.4f}"
    )
    print(f"result dir : {experiment_dir}")
    print("=" * 82)


if __name__ == "__main__":
    main()
