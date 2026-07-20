#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.detection_dataset_v3 import DetectionRadarDatasetV3
from models.simple_fcn import SimpleRadarFCN
from scripts.train_detection_baseline_v2 import (
    DetectionTolerance,
    SampleWeightedHeatmapMSE,
    apply_threshold_and_metrics,
    collect_predictions,
    json_safe,
    MemoryCachedDataset,
    make_debug_subset,
    make_loader,
    plot_score_histogram,
    resolve_project_path,
    select_threshold_at_false_alarm_budget,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="重新评价完整检测基线 checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data/raw/detection_dataset")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-val-false-alarms", type=int, default=1)
    parser.add_argument("--reselect-threshold", action="store_true")
    parser.add_argument("--debug-per-class", type=int, default=0)
    parser.add_argument("--no-memory-cache", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_project_path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    channel = checkpoint.get("channel", config.get("channel", "HV"))
    in_channels = int(checkpoint.get("in_channels", 2 if channel == "HV" else 1))
    tolerance = DetectionTolerance(
        range_gates=int(config.get("range_tolerance_gates", 2)),
        velocity_bins=int(config.get("velocity_tolerance_bins", 3)),
    )
    range_sigma = float(config.get("range_sigma", 5.0))
    velocity_sigma = float(config.get("velocity_sigma", 5.0))
    positive_sample_weight = float(config.get("positive_sample_weight", 10.0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda" and not args.no_amp
    data_root = resolve_project_path(args.data_root)
    output_dir = (
        resolve_project_path(args.output_dir)
        if args.output_dir
        else checkpoint_path.parent.parent / "reevaluation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    model = SimpleRadarFCN(in_channels=in_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = SampleWeightedHeatmapMSE(positive_sample_weight)

    base_datasets = {
        split: DetectionRadarDatasetV3(
            split=split,
            channel_mode=channel,
            range_sigma=range_sigma,
            velocity_sigma=velocity_sigma,
        )
        for split in ("val", "test")
    }
    datasets = {
        split: make_debug_subset(dataset, args.debug_per_class)
        for split, dataset in base_datasets.items()
    }
    if not args.no_memory_cache:
        datasets = {
            split: MemoryCachedDataset(dataset, label=split)
            for split, dataset in datasets.items()
        }
    loaders = {
        split: make_loader(ds, args.batch_size, False, args.num_workers, device)
        for split, ds in datasets.items()
    }

    val_frame, val_loss = collect_predictions(
        model, loaders["val"], criterion, device, tolerance, amp_enabled
    )
    if args.reselect_threshold or "threshold" not in checkpoint:
        threshold, curve = select_threshold_at_false_alarm_budget(
            val_frame, args.max_val_false_alarms
        )
    else:
        threshold = float(checkpoint["threshold"])
        _, curve = select_threshold_at_false_alarm_budget(
            val_frame, args.max_val_false_alarms
        )
    val_results, val_metrics = apply_threshold_and_metrics(val_frame, threshold, tolerance)

    test_frame, test_loss = collect_predictions(
        model, loaders["test"], criterion, device, tolerance, amp_enabled
    )
    test_results, test_metrics = apply_threshold_and_metrics(test_frame, threshold, tolerance)

    val_results.to_csv(output_dir / "val_predictions.csv", index=False, encoding="utf-8-sig")
    test_results.to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    curve.to_csv(output_dir / "val_threshold_curve.csv", index=False, encoding="utf-8-sig")
    plot_score_histogram(val_results, threshold, output_dir / "val_score_histogram.png", "Validation scores")
    plot_score_histogram(test_results, threshold, output_dir / "test_score_histogram.png", "Test scores")
    summary = {
        "checkpoint": str(checkpoint_path),
        "channel": channel,
        "threshold": threshold,
        "val_loss": val_loss,
        "test_loss": test_loss,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))
    print(f"保存目录: {output_dir}")


if __name__ == "__main__":
    main()
