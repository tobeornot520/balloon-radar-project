#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.detection_dataset_v3 import DEFAULT_MANIFEST, DetectionRadarDatasetV3
from evaluation.tian_fcn_metrics import (
    TianMetricTolerance,
    TianPredictionRecord,
    compute_tian_metrics,
    select_validation_absolute_threshold,
)
from evaluation.tian_fcn_postprocess import tian_pir_mdp, tian_valid_peak_scores
from models.tian_fcn import TianFastUAVFCN
from training.tian_fcn_objective import TianFCNObjective, build_tian_fcn_targets


@dataclass(frozen=True)
class LossSummary:
    total: float
    classification: float
    regression: float


@dataclass(frozen=True)
class RawPrediction:
    sample_id: str
    target_present: int
    true_range_index: int
    true_velocity_index: int
    source_file: str
    beam_layer: int
    azimuth_deg: float
    classification_logits: torch.Tensor
    normalized_offsets: torch.Tensor
    peak_score: float


class MemoryCachedDataset(Dataset):
    def __init__(self, dataset: Dataset, label: str) -> None:
        self.samples: list[dict[str, Any]] = []
        print(f"Caching {label}: {len(dataset)} samples", flush=True)
        for index in range(len(dataset)):
            self.samples.append(dataset[index])
            if (index + 1) % 200 == 0 or index + 1 == len(dataset):
                print(f"  {label}: {index + 1}/{len(dataset)}", flush=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one leakage-controlled fold of the Tian 2024 FCN"
    )
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--scope",
        choices=("smoke", "diagnostic", "formal"),
        required=True,
    )
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-root", default="results/experiments")
    parser.add_argument("--fold-id", type=int, default=0)
    parser.add_argument("--channel", choices=("H", "V", "HV"), default="H")
    parser.add_argument("--classification-epochs", type=int, default=20)
    parser.add_argument("--regression-epochs", type=int, default=20)
    parser.add_argument("--joint-epochs", type=int, default=20)
    parser.add_argument("--classification-learning-rate", type=float, default=0.1)
    parser.add_argument("--regression-learning-rate", type=float, default=0.1)
    parser.add_argument("--joint-learning-rate", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--lr-decay-every", type=int, default=10)
    parser.add_argument("--lr-decay-factor", type=float, default=0.1)
    parser.add_argument("--regression-weight", type=float, default=10.0)
    parser.add_argument("--background-negative-units", type=int, default=16)
    parser.add_argument("--target-negative-units-floor", type=int, default=0)
    parser.add_argument(
        "--target-negative-sampling",
        choices=("balanced_random", "same_range_column_dense"),
        default="balanced_random",
    )
    parser.add_argument("--range-extent", type=int, default=5)
    parser.add_argument("--doppler-extent", type=int, default=7)
    parser.add_argument(
        "--classification-target-mode",
        choices=("expanded", "responsible_point"),
        default="expanded",
    )
    parser.add_argument("--probability-margin", type=float, default=0.1)
    parser.add_argument("--max-val-false-alarms", type=int, default=2)
    parser.add_argument("--range-tolerance-gates", type=int, default=2)
    parser.add_argument("--velocity-tolerance-bins", type=int, default=3)
    parser.add_argument("--paper-distance-cells", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--debug-per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-memory-cache", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "classification_epochs",
        "regression_epochs",
        "joint_epochs",
        "classification_learning_rate",
        "regression_learning_rate",
        "joint_learning_rate",
        "lr_decay_every",
        "lr_decay_factor",
        "background_negative_units",
        "range_extent",
        "doppler_extent",
        "batch_size",
        "gradient_clip_norm",
        "early_stopping_patience",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.fold_id < 0 or args.num_workers < 0 or args.max_val_false_alarms < 0:
        raise ValueError("fold, worker, and false-alarm counts must be nonnegative")
    if args.target_negative_units_floor < 0:
        raise ValueError("--target-negative-units-floor must be nonnegative")
    if args.regression_weight < 0:
        raise ValueError("--regression-weight must be nonnegative")
    if not 0.0 <= args.momentum < 1.0:
        raise ValueError("--momentum must be in [0, 1)")
    if not 0.0 < args.lr_decay_factor <= 1.0:
        raise ValueError("--lr-decay-factor must be in (0, 1]")
    if not 0.0 <= args.probability_margin <= 1.0:
        raise ValueError("--probability-margin must be in [0, 1]")
    if args.debug_per_class is None:
        args.debug_per_class = 4 if args.scope == "smoke" else 0
    if args.debug_per_class < 0:
        raise ValueError("--debug-per-class must be nonnegative")
    if args.scope in {"diagnostic", "formal"} and args.debug_per_class:
        raise ValueError(f"{args.scope} scope cannot use --debug-per-class")
    if args.scope == "smoke" and not args.debug_per_class:
        raise ValueError("smoke scope requires a bounded --debug-per-class value")
    TianMetricTolerance(
        range_gates=args.range_tolerance_gates,
        velocity_bins=args.velocity_tolerance_bins,
        paper_distance_cells=args.paper_distance_cells,
    )


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_bounded_subset(dataset: DetectionRadarDatasetV3, per_class: int) -> Dataset:
    if per_class <= 0:
        return dataset
    background: list[int] = []
    positive: list[int] = []
    for index, record in enumerate(dataset.records):
        destination = positive if int(record["target_present"]) else background
        if len(destination) < per_class:
            destination.append(index)
    if len(background) != per_class or len(positive) != per_class:
        raise ValueError("bounded dataset requires enough samples from both classes")
    return Subset(dataset, background + positive)


def make_dataset(
    manifest_path: Path,
    split: str,
    channel: str,
    per_class: int,
    memory_cache: bool,
) -> Dataset:
    base = DetectionRadarDatasetV3(
        manifest_path=manifest_path,
        split=split,
        channel_mode=channel,
    )
    dataset = make_bounded_subset(base, per_class)
    if memory_cache:
        return MemoryCachedDataset(dataset, f"{channel}-{split}")
    return dataset


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
        "generator": generator,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(**kwargs)


def make_targets(
    batch: dict[str, Any],
    padded_shape: tuple[int, int],
    device: torch.device,
    args: argparse.Namespace,
):
    return build_tian_fcn_targets(
        batch["target_present"].to(device, non_blocking=True),
        batch["velocity_index"].to(device, non_blocking=True),
        batch["range_index"].to(device, non_blocking=True),
        padded_shape=padded_shape,
        doppler_extent=args.doppler_extent,
        range_extent=args.range_extent,
        classification_target_mode=args.classification_target_mode,
    )


def train_one_epoch(
    model: TianFastUAVFCN,
    loader: DataLoader,
    objective: TianFCNObjective,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    stage: str,
    args: argparse.Namespace,
) -> LossSummary:
    model.train()
    sums = np.zeros(3, dtype=np.float64)
    sample_count = 0
    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        padded_shape = model.padded_spatial_shape(*inputs.shape[-2:])
        targets = make_targets(batch, padded_shape, device, args)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            output = model(inputs)
            loss = objective(
                output.classification_logits,
                output.normalized_offsets,
                targets,
                stage=stage,
                sample_negatives_randomly=True,
            )
        scaler.scale(loss.total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            args.gradient_clip_norm,
        )
        scaler.step(optimizer)
        scaler.update()
        batch_size = int(inputs.shape[0])
        sums += batch_size * np.asarray(
            [loss.total.item(), loss.classification.item(), loss.regression.item()]
        )
        sample_count += batch_size
    if not sample_count:
        raise RuntimeError("training loader is empty")
    return LossSummary(*(float(value / sample_count) for value in sums))


def collect_predictions(
    model: TianFastUAVFCN,
    loader: DataLoader,
    objective: TianFCNObjective,
    device: torch.device,
    amp_enabled: bool,
    stage: str,
    args: argparse.Namespace,
) -> tuple[list[RawPrediction], LossSummary, float]:
    model.eval()
    predictions: list[RawPrediction] = []
    sums = np.zeros(3, dtype=np.float64)
    sample_count = 0
    forward_seconds = 0.0
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device, non_blocking=True)
            padded_shape = model.padded_spatial_shape(*inputs.shape[-2:])
            targets = make_targets(batch, padded_shape, device, args)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                output = model(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_seconds += time.perf_counter() - started
            loss = objective(
                output.classification_logits,
                output.normalized_offsets,
                targets,
                stage=stage,
                sample_negatives_randomly=False,
            )
            batch_size = int(inputs.shape[0])
            sums += batch_size * np.asarray(
                [loss.total.item(), loss.classification.item(), loss.regression.item()]
            )
            sample_count += batch_size
            logits = output.classification_logits.detach().float().cpu()
            offsets = output.normalized_offsets.detach().float().cpu()
            probabilities = tian_valid_peak_scores(
                logits,
                offsets,
                original_shape=output.original_shape,
                output_stride=model.output_stride,
            )
            for index in range(batch_size):
                predictions.append(
                    RawPrediction(
                        sample_id=str(batch["sample_id"][index]),
                        target_present=int(batch["target_present"][index]),
                        true_range_index=int(batch["range_index"][index]),
                        true_velocity_index=int(batch["velocity_index"][index]),
                        source_file=str(batch["source_file"][index]),
                        beam_layer=int(batch["beam_layer"][index]),
                        azimuth_deg=float(batch["azimuth_deg"][index]),
                        classification_logits=logits[index],
                        normalized_offsets=offsets[index],
                        peak_score=float(probabilities[index]),
                    )
                )
    if not sample_count:
        raise RuntimeError("evaluation loader is empty")
    losses = LossSummary(*(float(value / sample_count) for value in sums))
    return predictions, losses, forward_seconds / sample_count


def evaluate_predictions(
    raw: list[RawPrediction],
    tolerance: TianMetricTolerance,
    probability_margin: float,
    absolute_threshold: float | None,
) -> tuple[list[TianPredictionRecord], dict[str, float | int], pd.DataFrame]:
    logits = torch.stack([row.classification_logits for row in raw])
    offsets = torch.stack([row.normalized_offsets for row in raw])
    detections = tian_pir_mdp(
        logits,
        offsets,
        original_shape=(128, 100),
        probability_margin=probability_margin,
        absolute_threshold=absolute_threshold,
    )
    records: list[TianPredictionRecord] = []
    table_rows: list[dict[str, Any]] = []
    for source, sample_detections in zip(raw, detections):
        detection = sample_detections[0] if sample_detections else None
        predicted_positions = tuple(
            (item.range_index, item.velocity_index) for item in sample_detections
        )
        record = TianPredictionRecord(
            sample_id=source.sample_id,
            target_present=source.target_present,
            peak_score=source.peak_score,
            pred_range_index=None if detection is None else detection.range_index,
            pred_velocity_index=None if detection is None else detection.velocity_index,
            detection_score=None if detection is None else detection.score,
            true_range_index=source.true_range_index,
            true_velocity_index=source.true_velocity_index,
            all_predicted_positions=predicted_positions,
        )
        records.append(record)
        if source.target_present and sample_detections:
            errors = [
                (
                    abs(item.range_index - source.true_range_index),
                    abs(item.velocity_index - source.true_velocity_index),
                )
                for item in sample_detections
            ]
            range_error, velocity_error = min(
                errors, key=lambda item: math.hypot(item[0], item[1])
            )
            localization_ok = any(
                item[0] <= tolerance.range_gates
                and item[1] <= tolerance.velocity_bins
                for item in errors
            )
        else:
            range_error = math.nan
            velocity_error = math.nan
            localization_ok = False
        table_rows.append(
            {
                "sample_id": source.sample_id,
                "source_file": source.source_file,
                "beam_layer": source.beam_layer,
                "azimuth_deg": source.azimuth_deg,
                "target_present": source.target_present,
                "peak_score": source.peak_score,
                "detected": record.detected,
                "detection_count": len(sample_detections),
                "all_predicted_positions": json.dumps(predicted_positions),
                "detection_score": math.nan if detection is None else detection.score,
                "pred_range_index": -1 if detection is None else detection.range_index,
                "pred_velocity_index": -1 if detection is None else detection.velocity_index,
                "true_range_index": source.true_range_index,
                "true_velocity_index": source.true_velocity_index,
                "range_error_gates": range_error,
                "velocity_error_bins": velocity_error,
                "localization_ok": localization_ok,
                "absolute_threshold": (
                    math.nan if absolute_threshold is None else absolute_threshold
                ),
            }
        )
    metrics = compute_tian_metrics(records, tolerance)
    return records, metrics, pd.DataFrame(table_rows)


def validation_evaluation(
    raw: list[RawPrediction],
    args: argparse.Namespace,
    tolerance: TianMetricTolerance,
) -> tuple[float, list[dict[str, float | int]], dict[str, float | int]]:
    threshold, curve = select_validation_absolute_threshold(
        [row.peak_score for row in raw],
        [row.target_present for row in raw],
        args.max_val_false_alarms,
    )
    _, metrics, _ = evaluate_predictions(
        raw,
        tolerance,
        args.probability_margin,
        threshold,
    )
    return threshold, curve, metrics


def stage_selection_key(
    stage: str,
    losses: LossSummary,
    metrics: dict[str, float | int],
) -> tuple[float, ...]:
    if stage == "classification":
        return (-losses.classification,)
    if stage == "regression":
        return (-losses.regression,)
    return (
        float(metrics["joint_pd"]),
        -float(metrics["pfa"]),
        -losses.total,
    )


def save_checkpoint(
    path: Path,
    model: TianFastUAVFCN,
    optimizer: torch.optim.Optimizer,
    stage: str,
    stage_epoch: int,
    threshold: float,
    metrics: dict[str, float | int],
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "stage": stage,
            "stage_epoch": stage_epoch,
            "channel": args.channel,
            "in_channels": model.in_channels,
            "fold_id": args.fold_id,
            "manifest_path": args.manifest_path,
            "validation_threshold": threshold,
            "validation_metrics": metrics,
            "config": vars(args),
        },
        path,
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def paper_metric_subset(metrics: dict[str, float | int]) -> dict[str, float | int]:
    return {key: value for key, value in metrics.items() if key.startswith("paper_")}


def main() -> int:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    manifest_path = resolve_path(args.manifest_path)
    output_root = resolve_path(args.output_root)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    args.manifest_path = str(manifest_path)
    args.output_root = str(output_root)

    experiment_dir = output_root / args.name
    if experiment_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"experiment output exists: {experiment_dir}; use --overwrite explicitly"
        )
    if experiment_dir.exists():
        shutil.rmtree(experiment_dir)
    checkpoint_dir = experiment_dir / "checkpoints"
    table_dir = experiment_dir / "tables"
    checkpoint_dir.mkdir(parents=True)
    table_dir.mkdir(parents=True)

    device = resolve_device(args.device)
    amp_enabled = device.type == "cuda" and not args.no_amp
    memory_cache = not args.no_memory_cache
    tolerance = TianMetricTolerance(
        range_gates=args.range_tolerance_gates,
        velocity_bins=args.velocity_tolerance_bins,
        paper_distance_cells=args.paper_distance_cells,
    )
    train_dataset = make_dataset(
        manifest_path, "train", args.channel, args.debug_per_class, memory_cache
    )
    val_dataset = make_dataset(
        manifest_path, "val", args.channel, args.debug_per_class, memory_cache
    )
    train_loader = make_loader(
        train_dataset, args.batch_size, True, args.num_workers, device, args.seed
    )
    val_loader = make_loader(
        val_dataset, args.batch_size, False, args.num_workers, device, args.seed
    )

    model = TianFastUAVFCN(in_channels=2 if args.channel == "HV" else 1).to(device)
    objective = TianFCNObjective(
        regression_weight=args.regression_weight,
        background_negative_units=args.background_negative_units,
        target_negative_units_floor=args.target_negative_units_floor,
        target_negative_sampling=args.target_negative_sampling,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    stage_specs = (
        ("classification", args.classification_epochs, args.classification_learning_rate),
        ("regression", args.regression_epochs, args.regression_learning_rate),
        ("joint", args.joint_epochs, args.joint_learning_rate),
    )
    history_rows: list[dict[str, Any]] = []
    stage_best: dict[str, dict[str, Any]] = {}
    global_epoch = 0
    started = time.time()

    print("=" * 88)
    print(f"experiment       : {args.name}")
    print(f"scope / fold     : {args.scope} / {args.fold_id}")
    print(f"channel          : {args.channel}")
    print(f"device / AMP     : {device} / {amp_enabled}")
    print(f"train / val      : {len(train_dataset)} / {len(val_dataset)}")
    print("test split       : deferred until final model selection")
    print("=" * 88, flush=True)

    for stage, epochs, learning_rate in stage_specs:
        model.set_training_stage(stage)
        optimizer = torch.optim.SGD(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=learning_rate,
            momentum=args.momentum,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.lr_decay_every,
            gamma=args.lr_decay_factor,
        )
        best_key: tuple[float, ...] | None = None
        best_state: dict[str, torch.Tensor] | None = None
        stale_epochs = 0
        for stage_epoch in range(1, epochs + 1):
            global_epoch += 1
            train_losses = train_one_epoch(
                model,
                train_loader,
                objective,
                optimizer,
                scaler,
                device,
                amp_enabled,
                stage,
                args,
            )
            raw_val, val_losses, val_seconds_per_map = collect_predictions(
                model, val_loader, objective, device, amp_enabled, stage, args
            )
            threshold, _, val_metrics = validation_evaluation(
                raw_val, args, tolerance
            )
            current_lr = float(optimizer.param_groups[0]["lr"])
            key = stage_selection_key(stage, val_losses, val_metrics)
            improved = best_key is None or key > best_key
            if improved:
                best_key = key
                stale_epochs = 0
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
                save_checkpoint(
                    checkpoint_dir / f"{stage}_best.pt",
                    model,
                    optimizer,
                    stage,
                    stage_epoch,
                    threshold,
                    val_metrics,
                    args,
                )
                if stage == "joint":
                    save_checkpoint(
                        checkpoint_dir / "best.pt",
                        model,
                        optimizer,
                        stage,
                        stage_epoch,
                        threshold,
                        val_metrics,
                        args,
                    )
            else:
                stale_epochs += 1
            history_rows.append(
                {
                    "global_epoch": global_epoch,
                    "stage": stage,
                    "stage_epoch": stage_epoch,
                    "learning_rate": current_lr,
                    "train_total_loss": train_losses.total,
                    "train_classification_loss": train_losses.classification,
                    "train_regression_loss": train_losses.regression,
                    "val_total_loss": val_losses.total,
                    "val_classification_loss": val_losses.classification,
                    "val_regression_loss": val_losses.regression,
                    "val_threshold": threshold,
                    "val_joint_pd": val_metrics["joint_pd"],
                    "val_pfa": val_metrics["pfa"],
                    "val_seconds_per_map": val_seconds_per_map,
                    "stage_best": improved,
                }
            )
            print(
                f"{stage:14s} {stage_epoch:03d}/{epochs:03d} "
                f"train={train_losses.total:.5f} val={val_losses.total:.5f} "
                f"Pd={float(val_metrics['joint_pd']):.4f} "
                f"Pfa={float(val_metrics['pfa']):.4f} lr={current_lr:.2e}"
                + (" best" if improved else ""),
                flush=True,
            )
            scheduler.step()
            if stale_epochs >= args.early_stopping_patience:
                print(f"early stop: {stage} after {stale_epochs} stale epochs")
                break
        if best_state is None:
            raise RuntimeError(f"stage {stage} did not produce a checkpoint")
        model.load_state_dict(best_state)
        stage_best[stage] = {
            "stage_epoch": int(
                max(
                    row["stage_epoch"]
                    for row in history_rows
                    if row["stage"] == stage and row["stage_best"]
                )
            ),
            "selection_key": list(best_key or ()),
        }

    pd.DataFrame(history_rows).to_csv(
        table_dir / "training_history.csv", index=False, encoding="utf-8-sig"
    )
    raw_val, final_val_losses, val_seconds_per_map = collect_predictions(
        model, val_loader, objective, device, amp_enabled, "joint", args
    )
    threshold, threshold_curve, val_project_metrics = validation_evaluation(
        raw_val, args, tolerance
    )
    _, val_paper_all, _ = evaluate_predictions(
        raw_val, tolerance, args.probability_margin, None
    )
    _, _, val_table = evaluate_predictions(
        raw_val, tolerance, args.probability_margin, threshold
    )
    val_table.to_csv(
        table_dir / "val_predictions.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(threshold_curve).to_csv(
        table_dir / "val_threshold_curve.csv", index=False, encoding="utf-8-sig"
    )

    test_split_loaded = False
    test_summary: dict[str, Any] | None = None
    if args.scope == "formal":
        test_dataset = make_dataset(
            manifest_path, "test", args.channel, 0, memory_cache
        )
        test_split_loaded = True
        test_loader = make_loader(
            test_dataset, args.batch_size, False, args.num_workers, device, args.seed
        )
        raw_test, test_losses, test_seconds_per_map = collect_predictions(
            model, test_loader, objective, device, amp_enabled, "joint", args
        )
        _, test_project_metrics, test_table = evaluate_predictions(
            raw_test, tolerance, args.probability_margin, threshold
        )
        _, test_paper_all, _ = evaluate_predictions(
            raw_test, tolerance, args.probability_margin, None
        )
        test_table.to_csv(
            table_dir / "test_predictions.csv", index=False, encoding="utf-8-sig"
        )
        test_summary = {
            "losses": asdict(test_losses),
            "paper_alignment_dynamic_pir": paper_metric_subset(test_paper_all),
            "project_protocol_locked_threshold": test_project_metrics,
            "seconds_per_map": test_seconds_per_map,
            "sample_count": len(test_dataset),
        }

    macs, flops = model.estimate_conv_operations(128, 100)
    summary = {
        "status": "PASS",
        "paper_metric_definition": "tian_2024_set_euclidean_v2",
        "experiment_name": args.name,
        "evidence_role": {
            "formal": "method_reproduction_and_local_data_transfer",
            "diagnostic": "train_validation_only_model_rescue_diagnostic",
            "smoke": "interface_only_validation_smoke",
        }[args.scope],
        "scope": args.scope,
        "fold_id": args.fold_id,
        "channel": args.channel,
        "seed": args.seed,
        "manifest_path": str(manifest_path),
        "split_strategy": "V4 grouped scan and contiguous UAV block holdout",
        "test_split_loaded": test_split_loaded,
        "test_constructed_after_model_and_threshold_selection": test_split_loaded,
        "dataset_sizes": {
            "train": len(train_dataset),
            "validation": len(val_dataset),
            "test": None if test_summary is None else test_summary["sample_count"],
        },
        "stage_best": stage_best,
        "validation_threshold": threshold,
        "threshold_comparison": "probability > threshold",
        "max_validation_false_alarms": args.max_val_false_alarms,
        "validation": {
            "losses": asdict(final_val_losses),
            "paper_alignment_dynamic_pir": paper_metric_subset(val_paper_all),
            "project_protocol_selected_threshold": val_project_metrics,
            "seconds_per_map": val_seconds_per_map,
        },
        "test": test_summary,
        "model": {
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "conv_macs_per_map": macs,
            "conv_flops_per_map": flops,
            "flop_convention": "two FLOPs per convolution MAC; non-convolution ops excluded",
            "receptive_field": list(model.receptive_field),
            "output_stride": list(model.output_stride),
        },
        "elapsed_seconds": time.time() - started,
        "config": vars(args),
    }
    (table_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
