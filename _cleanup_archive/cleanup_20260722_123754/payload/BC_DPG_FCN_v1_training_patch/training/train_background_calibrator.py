#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.detection_dataset_v3 import DetectionRadarDatasetV3
from models.background_calibrated_dpg_fcn import (
    BackgroundCalibratedDPGFCN,
    calibration_loss,
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
            "Train a background calibration head on a frozen "
            "DPG-FCN checkpoint"
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
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--scheduler-patience",
        type=int,
        default=3,
    )
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
        default=(32, 16),
    )
    parser.add_argument(
        "--min-temperature",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--temperature-regularization",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--bias-regularization",
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
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--debug-per-class",
        type=int,
        default=0,
    )
    parser.add_argument("--no-memory-cache", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def resolve_from_base_config(
    args: argparse.Namespace,
    base_config: dict[str, Any],
) -> None:
    defaults = {
        "range_sigma": 5.0,
        "velocity_sigma": 5.0,
        "max_val_false_alarms": 2,
        "range_tolerance_gates": 2,
        "velocity_tolerance_bins": 3,
    }
    for name, fallback in defaults.items():
        if getattr(args, name) is None:
            setattr(
                args,
                name,
                base_config.get(name, fallback),
            )


def load_frozen_model(
    checkpoint_path: Path,
    *,
    topk: int,
    hidden_dims: Sequence[int],
    min_temperature: float,
    device: torch.device,
) -> tuple[
    BackgroundCalibratedDPGFCN,
    dict[str, Any],
]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"model_state_dict missing in {checkpoint_path}"
        )

    base_config = checkpoint.get("config", {})
    gate_hidden_dim = int(
        base_config.get("gate_hidden_dim", 16)
    )
    base_model = DualBranchGatedFCN(
        gate_hidden_dim=gate_hidden_dim
    )
    load_result = base_model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "Strict base checkpoint load unexpectedly returned "
            f"missing={load_result.missing_keys}, "
            f"unexpected={load_result.unexpected_keys}"
        )

    model = BackgroundCalibratedDPGFCN(
        base_model=base_model,
        topk=topk,
        hidden_dims=hidden_dims,
        min_temperature=min_temperature,
        freeze_base=True,
    ).to(device)
    return model, checkpoint


def make_datasets(
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


def make_loaders(
    datasets: dict[str, Dataset],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, DataLoader]:
    loaders: dict[str, DataLoader] = {}
    for split, dataset in datasets.items():
        loaders[split] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(
                args.num_workers > 0
            ),
            drop_last=False,
        )
    return loaders


def extract_scores_and_peaks(
    logits: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probability = torch.sigmoid(logits)
    flattened = probability.flatten(start_dim=1)
    score, flat_index = flattened.max(dim=1)

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


def train_one_epoch(
    model: BackgroundCalibratedDPGFCN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
    amp_enabled: bool,
) -> dict[str, float]:
    model.train()
    sums = {
        "loss": 0.0,
        "detection_loss": 0.0,
        "temperature_penalty": 0.0,
        "bias_penalty": 0.0,
    }
    sample_count = 0

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
            sample_logits = model.sample_logits(
                outputs["calibrated_logits"]
            )
            loss, parts = calibration_loss(
                sample_logits,
                labels,
                outputs["temperature"],
                outputs["bias"],
                temperature_regularization=(
                    args.temperature_regularization
                ),
                bias_regularization=args.bias_regularization,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.calibrator.parameters(),
            args.gradient_clip_norm,
        )
        scaler.step(optimizer)
        scaler.update()

        count = inputs.shape[0]
        for key in sums:
            sums[key] += float(parts[key].item()) * count
        sample_count += count

    if sample_count == 0:
        raise RuntimeError("Training DataLoader is empty")

    return {
        key: value / sample_count
        for key, value in sums.items()
    }


def collect_predictions(
    model: BackgroundCalibratedDPGFCN,
    loader: DataLoader,
    device: torch.device,
    tolerance: DetectionTolerance,
    args: argparse.Namespace,
    amp_enabled: bool,
) -> tuple[pd.DataFrame, dict[str, float]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    sums = {
        "loss": 0.0,
        "detection_loss": 0.0,
        "temperature_penalty": 0.0,
        "bias_penalty": 0.0,
    }
    sample_count = 0

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
                sample_logits = model.sample_logits(
                    outputs["calibrated_logits"]
                )
                _, parts = calibration_loss(
                    sample_logits,
                    labels,
                    outputs["temperature"],
                    outputs["bias"],
                    temperature_regularization=(
                        args.temperature_regularization
                    ),
                    bias_regularization=args.bias_regularization,
                )

            calibrated_score, calibrated_r, calibrated_v = (
                extract_scores_and_peaks(
                    outputs["calibrated_logits"]
                )
            )
            raw_score, raw_r, raw_v = extract_scores_and_peaks(
                outputs["raw_logits"]
            )

            base_output = outputs["base_output"]
            h_score, h_r, h_v = extract_scores_and_peaks(
                base_output["h_logits"]
            )
            v_score, v_r, v_v = extract_scores_and_peaks(
                base_output["v_logits"]
            )

            gates = outputs["gate_weights"].detach().cpu().numpy()
            temperatures = (
                outputs["temperature"]
                .detach().cpu().numpy().reshape(-1)
            )
            biases = (
                outputs["bias"]
                .detach().cpu().numpy().reshape(-1)
            )

            present = (
                batch["target_present"]
                .detach().cpu().numpy().astype(int)
            )
            true_r = np.asarray(batch["range_index"]).astype(int)
            true_v = np.asarray(batch["velocity_index"]).astype(int)

            count = inputs.shape[0]
            for key in sums:
                sums[key] += float(parts[key].item()) * count
            sample_count += count

            for index in range(count):
                if present[index] == 1:
                    range_error = abs(
                        int(calibrated_r[index])
                        - int(true_r[index])
                    )
                    velocity_error = abs(
                        int(calibrated_v[index])
                        - int(true_v[index])
                    )
                    localization_ok = (
                        range_error <= tolerance.range_gates
                        and velocity_error <= tolerance.velocity_bins
                    )
                else:
                    range_error = math.nan
                    velocity_error = math.nan
                    localization_ok = False

                argmax_preserved = (
                    int(calibrated_r[index]) == int(raw_r[index])
                    and int(calibrated_v[index]) == int(raw_v[index])
                )

                rows.append(
                    {
                        "sample_id": batch["sample_id"][index],
                        "target_present": int(present[index]),
                        "score": float(calibrated_score[index]),
                        "raw_score": float(raw_score[index]),
                        "pred_range_index": int(
                            calibrated_r[index]
                        ),
                        "pred_velocity_index": int(
                            calibrated_v[index]
                        ),
                        "raw_pred_range_index": int(raw_r[index]),
                        "raw_pred_velocity_index": int(raw_v[index]),
                        "true_range_index": int(true_r[index]),
                        "true_velocity_index": int(true_v[index]),
                        "range_error_gates": range_error,
                        "velocity_error_bins": velocity_error,
                        "localization_ok": bool(localization_ok),
                        "argmax_preserved": bool(argmax_preserved),
                        "temperature": float(temperatures[index]),
                        "bias": float(biases[index]),
                        "gate_h": float(gates[index, 0]),
                        "gate_v": float(gates[index, 1]),
                        "h_aux_score": float(h_score[index]),
                        "v_aux_score": float(v_score[index]),
                        "h_pred_range_index": int(h_r[index]),
                        "h_pred_velocity_index": int(h_v[index]),
                        "v_pred_range_index": int(v_r[index]),
                        "v_pred_velocity_index": int(v_v[index]),
                        "beam_layer": int(
                            np.asarray(batch["beam_layer"])[index]
                        ),
                        "azimuth_deg": float(
                            np.asarray(batch["azimuth_deg"])[index]
                        ),
                        "distance_m": float(
                            np.asarray(batch["distance_m"])[index]
                        ),
                        "velocity_mps": float(
                            np.asarray(batch["velocity_mps"])[index]
                        ),
                        "mat_path": batch["mat_path"][index],
                    }
                )

    if sample_count == 0:
        raise RuntimeError("Evaluation DataLoader is empty")

    frame = pd.DataFrame(rows)
    if not bool(frame["argmax_preserved"].all()):
        failed = frame.loc[
            ~frame["argmax_preserved"],
            ["sample_id"],
        ]
        raise RuntimeError(
            "Spatial argmax changed for samples: "
            + ", ".join(failed["sample_id"].head(10))
        )

    return frame, {
        key: value / sample_count
        for key, value in sums.items()
    }


def make_raw_frame(
    calibrated_frame: pd.DataFrame,
) -> pd.DataFrame:
    raw = calibrated_frame.copy()
    raw["score"] = raw["raw_score"]
    raw["pred_range_index"] = raw["raw_pred_range_index"]
    raw["pred_velocity_index"] = (
        raw["raw_pred_velocity_index"]
    )

    positive = raw["target_present"].to_numpy(int) == 1
    raw.loc[positive, "range_error_gates"] = np.abs(
        raw.loc[positive, "pred_range_index"].to_numpy(int)
        - raw.loc[positive, "true_range_index"].to_numpy(int)
    )
    raw.loc[positive, "velocity_error_bins"] = np.abs(
        raw.loc[positive, "pred_velocity_index"].to_numpy(int)
        - raw.loc[positive, "true_velocity_index"].to_numpy(int)
    )
    raw["localization_ok"] = (
        positive
        & (
            raw["range_error_gates"]
            <= raw.attrs.get("range_tolerance_gates", math.inf)
        )
        & (
            raw["velocity_error_bins"]
            <= raw.attrs.get("velocity_tolerance_bins", math.inf)
        )
    )
    return raw


def apply_metrics(
    frame: pd.DataFrame,
    threshold: float,
    tolerance: DetectionTolerance,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    return apply_threshold_and_metrics(
        frame,
        float(threshold),
        tolerance,
    )


def calibration_statistics(
    frame: pd.DataFrame,
) -> dict[str, float]:
    statistics: dict[str, float] = {}
    for prefix, mask in (
        ("all", np.ones(len(frame), dtype=bool)),
        (
            "background",
            frame["target_present"].to_numpy(int) == 0,
        ),
        (
            "positive",
            frame["target_present"].to_numpy(int) == 1,
        ),
    ):
        part = frame.loc[mask]
        statistics[f"{prefix}_count"] = int(len(part))
        for column in ("temperature", "bias", "score", "raw_score"):
            statistics[f"{prefix}_{column}_mean"] = (
                float(part[column].mean())
                if len(part)
                else math.nan
            )
            statistics[f"{prefix}_{column}_std"] = (
                float(part[column].std(ddof=0))
                if len(part)
                else math.nan
            )
    return statistics


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: BackgroundCalibratedDPGFCN,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    threshold: float,
    validation_metrics: dict[str, Any],
    validation_losses: dict[str, float],
    args: argparse.Namespace,
    base_checkpoint: dict[str, Any],
) -> None:
    torch.save(
        {
            "epoch": int(epoch),
            "calibrator_state_dict": (
                model.calibrator.state_dict()
            ),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "threshold": float(threshold),
            "validation_metrics": validation_metrics,
            "validation_losses": validation_losses,
            "base_checkpoint": str(
                resolve_project_path(args.base_checkpoint)
            ),
            "base_checkpoint_epoch": base_checkpoint.get("epoch"),
            "base_checkpoint_stage": base_checkpoint.get("stage"),
            "base_checkpoint_threshold": base_checkpoint.get(
                "threshold"
            ),
            "base_model_config": base_checkpoint.get(
                "config",
                {},
            ),
            "model_config": {
                "topk": args.topk,
                "hidden_dims": list(args.hidden_dims),
                "min_temperature": args.min_temperature,
            },
            "config": vars(args),
        },
        path,
    )


def metric_key(
    metrics: dict[str, Any],
    val_loss: float,
) -> tuple[float, float, float, float]:
    return (
        float(metrics.get("joint_pd", -math.inf)),
        -float(metrics.get("pfa", math.inf)),
        float(metrics.get("roc_auc", -math.inf)),
        -float(val_loss),
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

    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not base_checkpoint_path.is_file():
        raise FileNotFoundError(base_checkpoint_path)

    if experiment_dir.exists() and args.overwrite:
        shutil.rmtree(experiment_dir)
    if experiment_dir.exists() and not args.resume:
        raise FileExistsError(
            f"Experiment already exists: {experiment_dir}. "
            "Use --resume or --overwrite."
        )

    checkpoint_dir = experiment_dir / "checkpoints"
    table_dir = experiment_dir / "tables"
    for directory in (
        checkpoint_dir,
        table_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    amp_enabled = (
        device.type == "cuda" and not args.no_amp
    )

    model, base_checkpoint = load_frozen_model(
        base_checkpoint_path,
        topk=args.topk,
        hidden_dims=args.hidden_dims,
        min_temperature=args.min_temperature,
        device=device,
    )
    base_config = base_checkpoint.get("config", {})
    resolve_from_base_config(args, base_config)

    tolerance = DetectionTolerance(
        int(args.range_tolerance_gates),
        int(args.velocity_tolerance_bins),
    )

    datasets = make_datasets(
        manifest_path,
        args,
    )
    loaders = make_loaders(
        datasets,
        args,
        device,
    )

    optimizer = torch.optim.AdamW(
        model.calibrator.parameters(),
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

    start_epoch = 1
    best_epoch = 0
    best_key = (
        -math.inf,
        -math.inf,
        -math.inf,
        -math.inf,
    )
    stale_epochs = 0
    history_rows: list[dict[str, Any]] = []

    last_path = checkpoint_dir / "last.pt"
    best_path = checkpoint_dir / "best.pt"

    if args.resume:
        if not last_path.is_file():
            raise FileNotFoundError(
                f"Resume checkpoint missing: {last_path}"
            )
        resume = torch.load(
            last_path,
            map_location=device,
            weights_only=False,
        )
        model.calibrator.load_state_dict(
            resume["calibrator_state_dict"],
            strict=True,
        )
        optimizer.load_state_dict(
            resume["optimizer_state_dict"]
        )
        scheduler.load_state_dict(
            resume["scheduler_state_dict"]
        )
        start_epoch = int(resume["epoch"]) + 1

        history_path = table_dir / "training_history.csv"
        if history_path.is_file():
            history_rows = (
                pd.read_csv(
                    history_path,
                    encoding="utf-8-sig",
                ).to_dict("records")
            )

        if best_path.is_file():
            previous_best = torch.load(
                best_path,
                map_location="cpu",
                weights_only=False,
            )
            previous_metrics = previous_best.get(
                "validation_metrics",
                {},
            )
            previous_losses = previous_best.get(
                "validation_losses",
                {},
            )
            best_key = metric_key(
                previous_metrics,
                float(
                    previous_losses.get(
                        "loss",
                        math.inf,
                    )
                ),
            )
            best_epoch = int(previous_best.get("epoch", 0))

    start_time = time.time()

    print("=" * 78)
    print(f"Experiment          : {args.name}")
    print(f"Device              : {device}")
    print(f"AMP                 : {amp_enabled}")
    print(f"Manifest            : {manifest_path}")
    print(f"Base checkpoint     : {base_checkpoint_path}")
    print(f"Base epoch/stage    : {base_checkpoint.get('epoch')}/{base_checkpoint.get('stage')}")
    print(f"Base threshold      : {base_checkpoint.get('threshold')}")
    print(f"Dataset sizes       : { {k: len(v) for k, v in datasets.items()} }")
    print(f"Trainable parameters: {sum(p.numel() for p in model.calibrator.parameters())}")
    print("=" * 78)

    for epoch in range(start_epoch, args.epochs + 1):
        train_losses = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            scaler,
            device,
            args,
            amp_enabled,
        )

        val_frame, val_losses = collect_predictions(
            model,
            loaders["val"],
            device,
            tolerance,
            args,
            amp_enabled,
        )
        threshold, _ = select_threshold_at_false_alarm_budget(
            val_frame,
            args.max_val_false_alarms,
        )
        _, val_metrics = apply_metrics(
            val_frame,
            threshold,
            tolerance,
        )

        scheduler.step(val_losses["loss"])

        row = {
            "epoch": epoch,
            "learning_rate": float(
                optimizer.param_groups[0]["lr"]
            ),
            **{
                f"train_{key}": value
                for key, value in train_losses.items()
            },
            **{
                f"val_{key}": value
                for key, value in val_losses.items()
            },
            "val_threshold": float(threshold),
            "val_joint_pd": float(
                val_metrics.get("joint_pd", math.nan)
            ),
            "val_pfa": float(
                val_metrics.get("pfa", math.nan)
            ),
            "val_roc_auc": float(
                val_metrics.get("roc_auc", math.nan)
            ),
            "val_temperature_mean": float(
                val_frame["temperature"].mean()
            ),
            "val_bias_mean": float(
                val_frame["bias"].mean()
            ),
        }
        history_rows.append(row)

        current_key = metric_key(
            val_metrics,
            val_losses["loss"],
        )
        improved = current_key > best_key

        save_checkpoint(
            last_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            threshold=threshold,
            validation_metrics=val_metrics,
            validation_losses=val_losses,
            args=args,
            base_checkpoint=base_checkpoint,
        )

        if improved:
            best_key = current_key
            best_epoch = epoch
            stale_epochs = 0
            shutil.copy2(last_path, best_path)
        else:
            stale_epochs += 1

        pd.DataFrame(history_rows).to_csv(
            table_dir / "training_history.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"epoch={epoch:03d} "
            f"train={train_losses['loss']:.6f} "
            f"val={val_losses['loss']:.6f} "
            f"Pd={float(val_metrics.get('joint_pd', math.nan)):.4f} "
            f"Pfa={float(val_metrics.get('pfa', math.nan)):.4f} "
            f"AUC={float(val_metrics.get('roc_auc', math.nan)):.4f} "
            f"T={val_frame['temperature'].mean():.4f} "
            f"b={val_frame['bias'].mean():.4f} "
            f"{'*' if improved else ''}"
        )

        if stale_epochs >= args.early_stopping_patience:
            print(
                "Early stopping: "
                f"{stale_epochs} epochs without improvement"
            )
            break

    if not best_path.is_file():
        raise RuntimeError("best.pt was not created")

    best = torch.load(
        best_path,
        map_location=device,
        weights_only=False,
    )
    model.calibrator.load_state_dict(
        best["calibrator_state_dict"],
        strict=True,
    )

    val_frame, val_losses = collect_predictions(
        model,
        loaders["val"],
        device,
        tolerance,
        args,
        amp_enabled,
    )
    threshold, threshold_curve = (
        select_threshold_at_false_alarm_budget(
            val_frame,
            args.max_val_false_alarms,
        )
    )
    val_results, val_metrics = apply_metrics(
        val_frame,
        threshold,
        tolerance,
    )

    test_frame, test_losses = collect_predictions(
        model,
        loaders["test"],
        device,
        tolerance,
        args,
        amp_enabled,
    )
    test_results, test_metrics = apply_metrics(
        test_frame,
        threshold,
        tolerance,
    )

    raw_val = val_frame.copy()
    raw_val["score"] = raw_val["raw_score"]
    raw_test = test_frame.copy()
    raw_test["score"] = raw_test["raw_score"]

    raw_threshold, raw_threshold_curve = (
        select_threshold_at_false_alarm_budget(
            raw_val,
            args.max_val_false_alarms,
        )
    )
    raw_val_results, raw_val_metrics = apply_metrics(
        raw_val,
        raw_threshold,
        tolerance,
    )
    raw_test_results, raw_test_metrics = apply_metrics(
        raw_test,
        raw_threshold,
        tolerance,
    )

    _, fixed_val_metrics = apply_metrics(
        val_frame,
        args.fixed_threshold,
        tolerance,
    )
    fixed_test_results, fixed_test_metrics = apply_metrics(
        test_frame,
        args.fixed_threshold,
        tolerance,
    )
    _, raw_fixed_val_metrics = apply_metrics(
        raw_val,
        args.fixed_threshold,
        tolerance,
    )
    raw_fixed_test_results, raw_fixed_test_metrics = (
        apply_metrics(
            raw_test,
            args.fixed_threshold,
            tolerance,
        )
    )

    val_results.to_csv(
        table_dir / "val_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    test_results.to_csv(
        table_dir / "test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    raw_val_results.to_csv(
        table_dir / "raw_val_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    raw_test_results.to_csv(
        table_dir / "raw_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fixed_test_results.to_csv(
        table_dir / "fixed_threshold_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    raw_fixed_test_results.to_csv(
        table_dir / "raw_fixed_threshold_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    threshold_curve.to_csv(
        table_dir / "val_threshold_curve.csv",
        index=False,
        encoding="utf-8-sig",
    )
    raw_threshold_curve.to_csv(
        table_dir / "raw_val_threshold_curve.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "experiment_name": args.name,
        "device": str(device),
        "amp_enabled": amp_enabled,
        "manifest_path": str(manifest_path),
        "base_checkpoint": str(base_checkpoint_path),
        "base_checkpoint_epoch": base_checkpoint.get("epoch"),
        "base_checkpoint_stage": base_checkpoint.get("stage"),
        "base_checkpoint_threshold": base_checkpoint.get(
            "threshold"
        ),
        "dataset_sizes": {
            split: len(dataset)
            for split, dataset in datasets.items()
        },
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.calibrator.parameters()
        ),
        "best_epoch": int(best["epoch"]),
        "validation_threshold": float(threshold),
        "raw_validation_threshold": float(raw_threshold),
        "fixed_threshold": float(args.fixed_threshold),
        "max_val_false_alarms": int(
            args.max_val_false_alarms
        ),
        "tolerance": asdict(tolerance),
        "validation_losses": val_losses,
        "test_losses": test_losses,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "raw_validation_metrics": raw_val_metrics,
        "raw_test_metrics": raw_test_metrics,
        "fixed_validation_metrics": fixed_val_metrics,
        "fixed_test_metrics": fixed_test_metrics,
        "raw_fixed_validation_metrics": (
            raw_fixed_val_metrics
        ),
        "raw_fixed_test_metrics": raw_fixed_test_metrics,
        "calibration_statistics_validation": (
            calibration_statistics(val_frame)
        ),
        "calibration_statistics_test": (
            calibration_statistics(test_frame)
        ),
        "argmax_preserved_validation": bool(
            val_frame["argmax_preserved"].all()
        ),
        "argmax_preserved_test": bool(
            test_frame["argmax_preserved"].all()
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
        "BC validation: "
        f"threshold={threshold:.6f}, "
        f"Pd={float(val_metrics['joint_pd']):.4f}, "
        f"Pfa={float(val_metrics['pfa']):.4f}, "
        f"AUC={float(val_metrics['roc_auc']):.4f}"
    )
    print(
        "BC test      : "
        f"Pd={float(test_metrics['joint_pd']):.4f}, "
        f"Pfa={float(test_metrics['pfa']):.4f}, "
        f"AUC={float(test_metrics['roc_auc']):.4f}"
    )
    print(
        "Raw test     : "
        f"Pd={float(raw_test_metrics['joint_pd']):.4f}, "
        f"Pfa={float(raw_test_metrics['pfa']):.4f}, "
        f"AUC={float(raw_test_metrics['roc_auc']):.4f}"
    )
    print(
        f"Fixed {args.fixed_threshold:.3f} BC test: "
        f"Pd={float(fixed_test_metrics['joint_pd']):.4f}, "
        f"Pfa={float(fixed_test_metrics['pfa']):.4f}"
    )
    print(
        f"Fixed {args.fixed_threshold:.3f} raw test: "
        f"Pd={float(raw_fixed_test_metrics['joint_pd']):.4f}, "
        f"Pfa={float(raw_fixed_test_metrics['pfa']):.4f}"
    )
    print(
        "Spatial argmax preserved: "
        f"val={summary['argmax_preserved_validation']}, "
        f"test={summary['argmax_preserved_test']}"
    )
    print(f"Summary: {table_dir / 'summary.json'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
