#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.detection_dataset_v3 import DEFAULT_MANIFEST, DetectionRadarDatasetV3
from models.tian_fcn import TianFastUAVFCN
from training.tian_fcn_objective import TianFCNObjective, build_tian_fcn_targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a bounded train/validation-only Tian FCN interface smoke"
    )
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--channel", choices=("H", "V", "HV"), default="H")
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def bounded_subset(dataset: DetectionRadarDatasetV3, per_class: int) -> Dataset:
    if not 1 <= per_class <= 4:
        raise ValueError("--per-class must be between 1 and 4")
    background: list[int] = []
    target: list[int] = []
    for index, record in enumerate(dataset.records):
        destination = target if int(record["target_present"]) else background
        if len(destination) < per_class:
            destination.append(index)
        if len(background) == per_class and len(target) == per_class:
            break
    if not background or not target:
        raise ValueError("smoke requires target and background records")
    return Subset(dataset, background + target)


def make_loader(
    manifest_path: str,
    split: str,
    channel: str,
    per_class: int,
    batch_size: int,
) -> DataLoader:
    dataset = DetectionRadarDatasetV3(
        manifest_path=manifest_path,
        split=split,
        channel_mode=channel,
    )
    return DataLoader(
        bounded_subset(dataset, per_class),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


def optimize_one_batch(
    model: TianFastUAVFCN,
    objective: TianFCNObjective,
    batch: dict[str, Any],
    stage: str,
    learning_rate: float,
) -> dict[str, float | int]:
    model.set_training_stage(stage)
    optimizer = torch.optim.SGD(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        momentum=0.9,
    )
    output = model(batch["input"])
    targets = build_tian_fcn_targets(
        target_present=batch["target_present"],
        velocity_indices=batch["velocity_index"],
        range_indices=batch["range_index"],
        padded_shape=output.padded_shape,
    )
    loss = objective(
        output.classification_logits,
        output.normalized_offsets,
        targets,
        stage=stage,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.total.backward()
    optimizer.step()
    return {
        "total_loss": float(loss.total.detach().item()),
        "classification_loss": float(loss.classification.detach().item()),
        "regression_loss": float(loss.regression.detach().item()),
        "positive_units": loss.positive_units,
        "sampled_negative_units": loss.sampled_negative_units,
        "regression_units": loss.regression_units,
    }


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    set_seed(args.seed)

    # Test is intentionally never constructed by this smoke.
    train_loader = make_loader(
        args.manifest_path, "train", args.channel, args.per_class, args.batch_size
    )
    validation_loader = make_loader(
        args.manifest_path, "val", args.channel, args.per_class, args.batch_size
    )
    train_batch = next(iter(train_loader))
    validation_batch = next(iter(validation_loader))

    model = TianFastUAVFCN(in_channels=2 if args.channel == "HV" else 1)
    objective = TianFCNObjective(regression_weight=10.0)
    stages = (
        ("classification", 0.1),
        ("regression", 0.1),
        ("joint", 0.05),
    )
    stage_results = {
        stage: optimize_one_batch(
            model, objective, train_batch, stage, learning_rate
        )
        for stage, learning_rate in stages
    }

    model.eval()
    with torch.no_grad():
        validation_output = model(validation_batch["input"])
        validation_targets = build_tian_fcn_targets(
            validation_batch["target_present"],
            validation_batch["velocity_index"],
            validation_batch["range_index"],
            validation_output.padded_shape,
        )
        validation_loss = objective(
            validation_output.classification_logits,
            validation_output.normalized_offsets,
            validation_targets,
            stage="joint",
        )

    summary = {
        "status": "PASS",
        "evidence_role": "interface_only_validation_smoke",
        "test_split_loaded": False,
        "manifest_path": str(Path(args.manifest_path).expanduser()),
        "channel": args.channel,
        "seed": args.seed,
        "per_class_per_split": args.per_class,
        "input_shape": list(train_batch["input"].shape),
        "padded_spatial_shape": list(validation_output.padded_shape),
        "output_grid_shape": list(validation_output.classification_logits.shape[-2:]),
        "receptive_field": list(model.compute_geometry()[0]),
        "output_stride": list(model.compute_geometry()[1]),
        "stage_results": stage_results,
        "validation_interface_loss": float(validation_loss.total.item()),
    }
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
