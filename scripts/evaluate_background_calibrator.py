#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.detection_dataset_v3 import DetectionRadarDatasetV3
from scripts.train_detection_baseline_v2 import (
    DetectionTolerance,
    MemoryCachedDataset,
    apply_threshold_and_metrics,
    json_safe,
    make_debug_subset,
    select_threshold_at_false_alarm_budget,
)
from training.train_background_calibrator import (
    collect_predictions,
    load_frozen_model,
    resolve_project_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained BC-DPG calibration head"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--base-checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--debug-per-class", type=int, default=0)
    parser.add_argument("--no-memory-cache", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--reselect-threshold", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration_checkpoint_path = resolve_project_path(
        args.checkpoint
    )
    checkpoint = torch.load(
        calibration_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    stored_config = checkpoint.get("config", {})
    model_config = checkpoint.get("model_config", {})

    manifest_path = resolve_project_path(
        args.manifest_path
        or stored_config["manifest_path"]
    )
    base_checkpoint_path = resolve_project_path(
        args.base_checkpoint
        or checkpoint["base_checkpoint"]
    )

    output_dir = (
        resolve_project_path(args.output_dir)
        if args.output_dir
        else calibration_checkpoint_path.parent.parent
        / "evaluation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    amp_enabled = (
        device.type == "cuda" and not args.no_amp
    )

    model, base_checkpoint = load_frozen_model(
        base_checkpoint_path,
        topk=int(model_config.get("topk", 16)),
        hidden_dims=tuple(
            model_config.get("hidden_dims", [32, 16])
        ),
        min_temperature=float(
            model_config.get("min_temperature", 0.05)
        ),
        device=device,
    )
    model.calibrator.load_state_dict(
        checkpoint["calibrator_state_dict"],
        strict=True,
    )

    range_sigma = float(
        stored_config.get(
            "range_sigma",
            base_checkpoint.get("config", {}).get(
                "range_sigma",
                5.0,
            ),
        )
    )
    velocity_sigma = float(
        stored_config.get(
            "velocity_sigma",
            base_checkpoint.get("config", {}).get(
                "velocity_sigma",
                5.0,
            ),
        )
    )

    tolerance = DetectionTolerance(
        int(
            stored_config.get(
                "range_tolerance_gates",
                2,
            )
        ),
        int(
            stored_config.get(
                "velocity_tolerance_bins",
                3,
            )
        ),
    )

    frames = {}
    losses = {}
    for split in ("val", "test"):
        dataset = DetectionRadarDatasetV3(
            manifest_path=manifest_path,
            split=split,
            channel_mode="HV",
            range_sigma=range_sigma,
            velocity_sigma=velocity_sigma,
        )
        dataset = make_debug_subset(
            dataset,
            args.debug_per_class,
        )
        if not args.no_memory_cache:
            dataset = MemoryCachedDataset(dataset)

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
        )

        frame, split_losses = collect_predictions(
            model,
            loader,
            device,
            tolerance,
            argparse.Namespace(
                temperature_regularization=float(
                    stored_config.get(
                        "temperature_regularization",
                        0.01,
                    )
                ),
                bias_regularization=float(
                    stored_config.get(
                        "bias_regularization",
                        0.01,
                    )
                ),
            ),
            amp_enabled,
        )
        frames[split] = frame
        losses[split] = split_losses

    if args.reselect_threshold:
        threshold, curve = (
            select_threshold_at_false_alarm_budget(
                frames["val"],
                int(
                    stored_config.get(
                        "max_val_false_alarms",
                        2,
                    )
                ),
            )
        )
    else:
        threshold = float(checkpoint["threshold"])
        _, curve = select_threshold_at_false_alarm_budget(
            frames["val"],
            int(
                stored_config.get(
                    "max_val_false_alarms",
                    2,
                )
            ),
        )

    val_results, val_metrics = apply_threshold_and_metrics(
        frames["val"],
        threshold,
        tolerance,
    )
    test_results, test_metrics = apply_threshold_and_metrics(
        frames["test"],
        threshold,
        tolerance,
    )

    val_results.to_csv(
        output_dir / "val_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    test_results.to_csv(
        output_dir / "test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    curve.to_csv(
        output_dir / "val_threshold_curve.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "calibration_checkpoint": str(
            calibration_checkpoint_path
        ),
        "base_checkpoint": str(base_checkpoint_path),
        "manifest_path": str(manifest_path),
        "threshold": float(threshold),
        "threshold_reselected": bool(args.reselect_threshold),
        "tolerance": asdict(tolerance),
        "validation_losses": losses["val"],
        "test_losses": losses["test"],
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "argmax_preserved_validation": bool(
            frames["val"]["argmax_preserved"].all()
        ),
        "argmax_preserved_test": bool(
            frames["test"]["argmax_preserved"].all()
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(
            json_safe(summary),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"threshold={threshold:.6f}")
    print(
        "validation: "
        f"Pd={float(val_metrics['joint_pd']):.4f}, "
        f"Pfa={float(val_metrics['pfa']):.4f}, "
        f"AUC={float(val_metrics['roc_auc']):.4f}"
    )
    print(
        "test      : "
        f"Pd={float(test_metrics['joint_pd']):.4f}, "
        f"Pfa={float(test_metrics['pfa']):.4f}, "
        f"AUC={float(test_metrics['roc_auc']):.4f}"
    )
    print(f"summary={output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
