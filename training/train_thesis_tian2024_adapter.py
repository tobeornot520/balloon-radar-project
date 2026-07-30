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

from datasets.thesis_tian2024_dataset import (  # noqa: E402
    DEFAULT_MANIFEST,
    ThesisTian2024Dataset,
)
from evaluation.tian_fcn_metrics import select_validation_absolute_threshold  # noqa: E402
from evaluation.thesis_tian2024_postprocess import direct_max_detections  # noqa: E402
from models.thesis_tian2024_adapter import ThesisTian2024Adapter  # noqa: E402
from training.thesis_tian2024_objective import (  # noqa: E402
    ThesisTian2024Objective,
    build_thesis_tian2024_targets,
)


@dataclass(frozen=True)
class LossSummary:
    total: float
    classification: float
    regression: float


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
        description="Train the thesis Tian2024 local six-channel adaptation"
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--scope", choices=("smoke", "validation"), required=True)
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-root", default="results/experiments")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--regression-weight", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-val-false-alarms", type=int, default=1)
    parser.add_argument("--range-tolerance-gates", type=int, default=2)
    parser.add_argument("--velocity-tolerance-bins", type=int, default=3)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--debug-per-class", type=int, default=None)
    parser.add_argument("--train-background-limit", type=int, default=208)
    parser.add_argument("--train-target-limit", type=int, default=80)
    parser.add_argument(
        "--normalization-scope",
        choices=("batch_channel", "sample_channel"),
        default="batch_channel",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-memory-cache", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "epochs",
        "learning_rate",
        "batch_size",
        "gradient_clip_norm",
        "early_stopping_patience",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.regression_weight < 0 or args.max_val_false_alarms < 0:
        raise ValueError("loss weight and false-alarm budget must be nonnegative")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be nonnegative")
    if args.train_background_limit <= 0 or args.train_target_limit <= 0:
        raise ValueError("training class limits must be positive")
    if args.debug_per_class is None:
        args.debug_per_class = 4 if args.scope == "smoke" else 0
    if args.scope == "smoke" and args.debug_per_class <= 0:
        raise ValueError("smoke scope requires --debug-per-class")
    if args.scope == "validation" and args.debug_per_class:
        raise ValueError("validation scope cannot use --debug-per-class")


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


def make_subset(dataset: ThesisTian2024Dataset, per_class: int) -> Dataset:
    if per_class <= 0:
        return dataset
    selected: dict[int, list[int]] = {0: [], 1: []}
    for index, record in enumerate(dataset.records):
        label = int(record["target_present"])
        if len(selected[label]) < per_class:
            selected[label].append(index)
    if any(len(indices) != per_class for indices in selected.values()):
        raise ValueError("not enough target/background samples for bounded smoke")
    return Subset(dataset, selected[0] + selected[1])


def make_seeded_training_subset(
    dataset: ThesisTian2024Dataset,
    background_limit: int,
    target_limit: int,
    seed: int,
) -> Dataset:
    """Select the unavailable thesis subset IDs deterministically from train."""
    by_class: dict[int, list[int]] = {0: [], 1: []}
    for index, record in enumerate(dataset.records):
        by_class[int(record["target_present"])].append(index)
    limits = {0: background_limit, 1: target_limit}
    generator = np.random.default_rng(seed)
    selected: list[int] = []
    for label in (0, 1):
        candidates = np.asarray(by_class[label], dtype=np.int64)
        if len(candidates) < limits[label]:
            raise ValueError(
                f"class {label} has {len(candidates)} samples, needs {limits[label]}"
            )
        choice = generator.choice(candidates, size=limits[label], replace=False)
        selected.extend(sorted(int(index) for index in choice))
    return Subset(dataset, selected)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
        drop_last=False,
    )


def make_targets(
    batch: dict[str, Any],
    device: torch.device,
    padded_shape: tuple[int, int],
):
    return build_thesis_tian2024_targets(
        batch["target_present"].to(device),
        batch["velocity_index"].to(device),
        batch["range_index"].to(device),
        padded_shape,
    )


def train_epoch(
    model: ThesisTian2024Adapter,
    loader: DataLoader,
    objective: ThesisTian2024Objective,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    clip_norm: float,
) -> LossSummary:
    model.train()
    totals = np.zeros(3, dtype=np.float64)
    count = 0
    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        targets = make_targets(
            batch, device, model.padded_spatial_shape(*inputs.shape[-2:])
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            output = model(inputs)
            loss = objective(
                output.classification_logits,
                output.normalized_offsets,
                targets,
            )
        scaler.scale(loss.total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        scaler.step(optimizer)
        scaler.update()
        batch_size = int(inputs.shape[0])
        totals += batch_size * np.array(
            [loss.total.item(), loss.classification.item(), loss.regression.item()]
        )
        count += batch_size
    return LossSummary(*(float(value / count) for value in totals))


def _mean_pairwise_template_correlation(maps: list[np.ndarray]) -> float:
    if len(maps) < 2:
        return math.nan
    correlations: list[float] = []
    for left_index in range(len(maps)):
        left = maps[left_index].reshape(-1).astype(np.float64)
        for right in maps[left_index + 1 :]:
            right_flat = right.reshape(-1).astype(np.float64)
            left_std = float(left.std())
            right_std = float(right_flat.std())
            if left_std <= 1e-12 or right_std <= 1e-12:
                correlations.append(1.0)
            else:
                correlations.append(float(np.corrcoef(left, right_flat)[0, 1]))
    return float(np.mean(correlations))


def evaluate(
    model: ThesisTian2024Adapter,
    loader: DataLoader,
    objective: ThesisTian2024Objective,
    device: torch.device,
    amp_enabled: bool,
    max_false_alarms: int,
    range_tolerance: int,
    velocity_tolerance: int,
) -> tuple[LossSummary, dict[str, Any], pd.DataFrame]:
    model.eval()
    totals = np.zeros(3, dtype=np.float64)
    count = 0
    rows: list[dict[str, Any]] = []
    target_maps: list[np.ndarray] = []
    started = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device, non_blocking=True)
            targets = make_targets(
                batch, device, model.padded_spatial_shape(*inputs.shape[-2:])
            )
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                output = model(inputs)
            loss = objective(
                output.classification_logits,
                output.normalized_offsets,
                targets,
                sample_negatives_randomly=False,
            )
            logits = output.classification_logits.detach().float().cpu()
            offsets = output.normalized_offsets.detach().float().cpu()
            detections = direct_max_detections(
                logits, offsets, output.original_shape, threshold=None
            )
            probabilities = torch.sigmoid(logits[:, 0]).numpy()
            batch_size = int(inputs.shape[0])
            totals += batch_size * np.array(
                [loss.total.item(), loss.classification.item(), loss.regression.item()]
            )
            count += batch_size
            for index, detection in enumerate(detections):
                if detection is None:
                    raise RuntimeError("unthresholded direct-max decoding failed")
                present = int(batch["target_present"][index])
                if present:
                    target_maps.append(probabilities[index])
                rows.append(
                    {
                        "sample_id": str(batch["sample_id"][index]),
                        "target_present": present,
                        "peak_score": detection.score,
                        "peak_grid_x": detection.grid_x,
                        "peak_grid_y": detection.grid_y,
                        "pred_range_index": detection.range_index,
                        "pred_velocity_index": detection.velocity_index,
                        "true_range_index": int(batch["range_index"][index]),
                        "true_velocity_index": int(batch["velocity_index"][index]),
                    }
                )
    table = pd.DataFrame(rows)
    threshold, _ = select_validation_absolute_threshold(
        table["peak_score"].tolist(),
        table["target_present"].tolist(),
        max_false_alarms,
    )
    table["detected"] = table["peak_score"] > threshold
    positive = table["target_present"] == 1
    background = ~positive
    table["range_error_gates"] = np.where(
        positive,
        (table["pred_range_index"] - table["true_range_index"]).abs(),
        np.nan,
    )
    table["velocity_error_bins"] = np.where(
        positive,
        (table["pred_velocity_index"] - table["true_velocity_index"]).abs(),
        np.nan,
    )
    table["responsible_grid_selected"] = positive & (
        table["peak_grid_x"] == table["true_range_index"] // 4
    ) & (table["peak_grid_y"] == table["true_velocity_index"] // 4)
    table["localization_ok"] = positive & (
        table["range_error_gates"] <= range_tolerance
    ) & (table["velocity_error_bins"] <= velocity_tolerance)
    table["joint_ok"] = table["detected"] & table["localization_ok"]

    target_count = int(positive.sum())
    background_count = int(background.sum())
    unique_peaks = int(
        table.loc[positive, ["peak_grid_y", "peak_grid_x"]].drop_duplicates().shape[0]
    )
    correlation = _mean_pairwise_template_correlation(target_maps)
    unique_ratio = float(unique_peaks / target_count) if target_count else math.nan
    audit_pass = bool(
        target_count >= 2
        and unique_peaks >= 2
        and unique_ratio >= 0.25
        and math.isfinite(correlation)
        and correlation < 0.98
    )
    metrics = {
        "validation_threshold": float(threshold),
        "target_count": target_count,
        "background_count": background_count,
        "joint_pd": float(table.loc[positive, "joint_ok"].mean()),
        "pfa": float(table.loc[background, "detected"].mean()),
        "responsible_grid_selection_rate": float(
            table.loc[positive, "responsible_grid_selected"].mean()
        ),
        "range_mae_gates": float(table.loc[positive, "range_error_gates"].mean()),
        "velocity_mae_bins": float(
            table.loc[positive, "velocity_error_bins"].mean()
        ),
        "unique_target_peak_grids": unique_peaks,
        "unique_target_peak_ratio": unique_ratio,
        "mean_target_template_correlation": correlation,
        "non_degenerate_validation_output": audit_pass,
        "seconds_per_map": float((time.perf_counter() - started) / count),
    }
    losses = LossSummary(*(float(value / count) for value in totals))
    table["validation_threshold"] = threshold
    return losses, metrics, table


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> int:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    manifest_path = resolve_path(args.manifest_path)
    output_root = resolve_path(args.output_root)
    experiment_dir = output_root / args.name
    if experiment_dir.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {experiment_dir}; use --overwrite")
    if experiment_dir.exists():
        shutil.rmtree(experiment_dir)
    checkpoint_dir = experiment_dir / "checkpoints"
    table_dir = experiment_dir / "tables"
    checkpoint_dir.mkdir(parents=True)
    table_dir.mkdir(parents=True)

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    amp_enabled = device.type == "cuda" and not args.no_amp

    train_base = ThesisTian2024Dataset(manifest_path, "train")
    val_base = ThesisTian2024Dataset(manifest_path, "val")
    if args.scope == "smoke":
        train_dataset = make_subset(train_base, args.debug_per_class)
        val_dataset = make_subset(val_base, args.debug_per_class)
    else:
        train_dataset = make_seeded_training_subset(
            train_base,
            args.train_background_limit,
            args.train_target_limit,
            args.seed,
        )
        val_dataset = val_base
    if not args.no_memory_cache:
        train_dataset = MemoryCachedDataset(train_dataset, "thesis-tian6-train")
        val_dataset = MemoryCachedDataset(val_dataset, "thesis-tian6-val")
    train_loader = make_loader(
        train_dataset, args.batch_size, True, args.num_workers, device, args.seed
    )
    val_loader = make_loader(
        val_dataset, args.batch_size, False, args.num_workers, device, args.seed
    )

    model = ThesisTian2024Adapter(args.normalization_scope).to(device)
    objective = ThesisTian2024Objective(args.regression_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    history: list[dict[str, Any]] = []
    best_key: tuple[float, ...] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    print(
        f"{args.name}: train={len(train_dataset)} val={len(val_dataset)} "
        f"device={device} test=DEFERRED",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        train_losses = train_epoch(
            model,
            train_loader,
            objective,
            optimizer,
            scaler,
            device,
            amp_enabled,
            args.gradient_clip_norm,
        )
        val_losses, metrics, _ = evaluate(
            model,
            val_loader,
            objective,
            device,
            amp_enabled,
            args.max_val_false_alarms,
            args.range_tolerance_gates,
            args.velocity_tolerance_bins,
        )
        key = (
            float(metrics["joint_pd"]),
            float(metrics["responsible_grid_selection_rate"]),
            float(metrics["unique_target_peak_ratio"]),
            -float(metrics["mean_target_template_correlation"]),
            -val_losses.total,
        )
        improved = best_key is None or key > best_key
        if improved:
            best_key = key
            stale = 0
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            torch.save(
                {
                    "model_state_dict": best_state,
                    "epoch": epoch,
                    "validation_metrics": metrics,
                    "config": vars(args),
                },
                checkpoint_dir / "best.pt",
            )
        else:
            stale += 1
        history.append(
            {
                "epoch": epoch,
                **{f"train_{key}": value for key, value in asdict(train_losses).items()},
                **{f"val_{key}": value for key, value in asdict(val_losses).items()},
                **metrics,
                "best": improved,
            }
        )
        print(
            f"epoch {epoch:03d}/{args.epochs:03d} loss={val_losses.total:.4f} "
            f"Pd={metrics['joint_pd']:.3f} Pfa={metrics['pfa']:.3f} "
            f"peaks={metrics['unique_target_peak_grids']} "
            f"corr={metrics['mean_target_template_correlation']:.4f}"
            + (" best" if improved else ""),
            flush=True,
        )
        if stale >= args.early_stopping_patience:
            break
    if best_state is None:
        raise RuntimeError("training did not produce a best state")
    model.load_state_dict(best_state)
    final_losses, final_metrics, final_table = evaluate(
        model,
        val_loader,
        objective,
        device,
        amp_enabled,
        args.max_val_false_alarms,
        args.range_tolerance_gates,
        args.velocity_tolerance_bins,
    )
    pd.DataFrame(history).to_csv(table_dir / "training_history.csv", index=False)
    final_table.to_csv(table_dir / "val_predictions.csv", index=False)
    summary = {
        "status": (
            "VALIDATION_NON_DEGENERATE"
            if final_metrics["non_degenerate_validation_output"]
            else "VALIDATION_DEGENERATE_STOP"
        ),
        "experiment_name": args.name,
        "evidence_role": "train_validation_only_method_level_local_adaptation",
        "method": "thesis_tian2024_local_adaptation_v1",
        "not_claimed": [
            "exact_numeric_thesis_reproduction",
            "original_tian_2024_reproduction",
            "balloon_payload_recognition",
        ],
        "input_channels": ["Re_H", "Im_H", "Re_V", "Im_V", "sZdr", "sRhoCo"],
        "input_shape": [6, 128, 100],
        "output_shape": [1, 32, 25],
        "normalization_scope": args.normalization_scope,
        "normalization_warning": (
            "batch_channel statistics depend on batch composition and require a "
            "deployment-stable sensitivity run before freezing"
            if args.normalization_scope == "batch_channel"
            else None
        ),
        "train_sample_count": len(train_dataset),
        "validation_sample_count": len(val_dataset),
        "training_subset_contract": {
            "background_limit": args.train_background_limit,
            "target_limit": args.train_target_limit,
            "selection": "seeded_without_replacement_within_train_partition",
            "exact_thesis_sample_ids_available": False,
        },
        "test_split_loaded": False,
        "best_selection_key": best_key,
        "validation_losses": asdict(final_losses),
        "validation_metrics": final_metrics,
        "config": vars(args),
    }
    with (experiment_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(summary), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
