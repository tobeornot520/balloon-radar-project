#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.detection_dataset_v3 import (
    DEFAULT_MANIFEST,
    DetectionRadarDatasetV3,
)
from models.simple_fcn import SimpleRadarFCN, count_trainable_parameters


@dataclass(frozen=True)
class DetectionTolerance:
    range_gates: int = 2
    velocity_bins: int = 3


class SampleWeightedHeatmapMSE(nn.Module):
    """与学长论文设置一致的样本级加权热图 MSE。

    正样本整幅热图损失乘 positive_sample_weight，背景样本权重为 1。
    输入为 logits，内部先执行 sigmoid。
    """

    def __init__(self, positive_sample_weight: float = 10.0) -> None:
        super().__init__()
        if positive_sample_weight <= 0:
            raise ValueError("positive_sample_weight 必须大于 0")
        self.positive_sample_weight = float(positive_sample_weight)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        target_present: torch.Tensor,
    ) -> torch.Tensor:
        prediction = torch.sigmoid(logits)
        per_sample_mse = (prediction - target).pow(2).flatten(1).mean(dim=1)
        sample_weights = torch.where(
            target_present.reshape(-1).float() > 0.5,
            torch.full_like(per_sample_mse, self.positive_sample_weight),
            torch.ones_like(per_sample_mse),
        )
        return (per_sample_mse * sample_weights).mean()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在 Background+UAV 完整数据上训练热图回归检测定位基线"
    )
    parser.add_argument("--name", default="detection_hv_baseline_v2")
    parser.add_argument(
        "--data-root",
        default="data/raw/detection_dataset",
        help="原始MAT/TXT数据根目录，仅用于实验溯源；V3划分由manifest决定",
    )
    parser.add_argument(
        "--manifest-path",
        default=str(DEFAULT_MANIFEST),
        help="分组数据清单路径",
    )
    parser.add_argument(
        "--dataset-version",
        default="V3",
        help="写入checkpoint和summary的数据版本，例如V3或V4",
    )
    parser.add_argument(
        "--split-strategy",
        default="grouped_source_file_and_scan_temporal_holdout",
        help="写入checkpoint和summary的划分策略标识",
    )
    parser.add_argument(
        "--fold-id",
        type=int,
        default=0,
        help="多折实验编号；非多折实验使用0",
    )
    parser.add_argument("--channel", choices=["H", "V", "HV"], default="HV")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--positive-sample-weight", type=float, default=10.0)
    parser.add_argument("--range-sigma", type=float, default=5.0)
    parser.add_argument("--velocity-sigma", type=float, default=5.0)
    parser.add_argument("--early-stopping-patience", type=int, default=25)
    parser.add_argument("--scheduler-patience", type=int, default=6)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument(
        "--max-val-false-alarms",
        type=int,
        default=1,
        help="验证集选阈值时允许的背景虚警绝对数量",
    )
    parser.add_argument("--range-tolerance-gates", type=int, default=2)
    parser.add_argument("--velocity-tolerance-bins", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--debug-per-class",
        type=int,
        default=0,
        help="大于 0 时每个 split 每类只取前 N 个样本，用于烟雾测试",
    )
    parser.add_argument(
        "--no-memory-cache",
        action="store_true",
        help="禁用内存缓存；默认预加载 RD 与热图以避免每轮重复 MAT/FFT",
    )
    parser.add_argument("--no-amp", action="store_true", help="禁用 CUDA AMP")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_names = (
        "epochs",
        "batch_size",
        "learning_rate",
        "min_learning_rate",
        "positive_sample_weight",
        "range_sigma",
        "velocity_sigma",
        "early_stopping_patience",
        "scheduler_patience",
        "gradient_clip_norm",
    )
    for name in positive_names:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} 必须大于 0")
    if args.num_workers < 0:
        raise ValueError("--num-workers 不能小于 0")
    if args.max_val_false_alarms < 0:
        raise ValueError("--max-val-false-alarms 不能小于 0")
    if args.range_tolerance_gates < 0 or args.velocity_tolerance_bins < 0:
        raise ValueError("定位容差不能小于 0")
    if args.debug_per_class < 0:
        raise ValueError("--debug-per-class 不能小于 0")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def make_debug_subset(dataset: DetectionRadarDatasetV3, per_class: int) -> Dataset:
    if per_class <= 0:
        return dataset
    background_indices: list[int] = []
    positive_indices: list[int] = []
    for index, record in enumerate(dataset.records):
        if int(record["target_present"]) == 1:
            positive_indices.append(index)
        else:
            background_indices.append(index)
    indices = background_indices[:per_class] + positive_indices[:per_class]
    if not indices:
        raise ValueError("debug 子集为空")
    return Subset(dataset, indices)


class MemoryCachedDataset(Dataset):
    """将预处理后的样本缓存到内存，避免每个 epoch 重复 loadmat 和 FFT。"""

    def __init__(self, dataset: Dataset, label: str = "dataset") -> None:
        self.samples: list[dict[str, Any]] = []
        total = len(dataset)
        print(f"预加载 {label}: {total} 个样本...")
        for index in range(total):
            self.samples.append(dataset[index])
            if (index + 1) % 200 == 0 or index + 1 == total:
                print(f"  {label}: {index + 1}/{total}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(42)
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


def extract_scores_and_peaks(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prediction = torch.sigmoid(logits)
    batch_size = prediction.shape[0]
    range_size = prediction.shape[-1]
    flattened = prediction.reshape(batch_size, -1)
    scores, flat_indices = flattened.max(dim=1)
    pred_velocity = flat_indices // range_size
    pred_range = flat_indices % range_size
    return scores, pred_range, pred_velocity


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    combined = np.concatenate([positive, negative])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, combined.size + 1, dtype=np.float64)
    # 对相同分数使用平均秩。
    sorted_scores = combined[order]
    start = 0
    while start < sorted_scores.size:
        end = start + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    positive_ranks = ranks[: positive.size].sum()
    return float(
        (positive_ranks - positive.size * (positive.size + 1) / 2.0)
        / (positive.size * negative.size)
    )


def select_threshold_at_false_alarm_budget(
    frame: pd.DataFrame,
    max_false_alarms: int,
) -> tuple[float, pd.DataFrame]:
    background_scores = frame.loc[frame["target_present"] == 0, "score"].to_numpy(float)
    if background_scores.size == 0:
        raise ValueError("验证集没有背景样本，无法选择检测阈值")
    if max_false_alarms >= background_scores.size:
        threshold = float(np.nextafter(background_scores.min(), -np.inf))
    else:
        unique_scores = np.unique(background_scores)
        candidates = np.concatenate(
            [
                np.array([np.nextafter(unique_scores.max(), np.inf)]),
                unique_scores,
            ]
        )
        candidates = np.sort(np.unique(candidates))
        feasible: list[tuple[float, int]] = []
        for candidate in candidates:
            false_count = int(np.sum(background_scores > candidate))
            if false_count <= max_false_alarms:
                feasible.append((float(candidate), false_count))
        if not feasible:
            threshold = float(np.nextafter(background_scores.max(), np.inf))
        else:
            # 最低的可行阈值在给定虚警预算下保留最多正样本。
            threshold = min(item[0] for item in feasible)

    positive = frame[frame["target_present"] == 1]
    curve_rows: list[dict[str, float | int]] = []
    candidates = np.sort(
        np.unique(
            np.concatenate(
                [
                    background_scores,
                    np.array([threshold, np.nextafter(background_scores.max(), np.inf)]),
                ]
            )
        )
    )
    for candidate in candidates:
        bg_detected = background_scores > candidate
        pos_detected = positive["score"].to_numpy(float) > candidate
        pos_correct = pos_detected & positive["localization_ok"].to_numpy(bool)
        curve_rows.append(
            {
                "threshold": float(candidate),
                "false_alarm_count": int(bg_detected.sum()),
                "pfa": float(bg_detected.mean()),
                "score_recall": float(pos_detected.mean()) if len(positive) else float("nan"),
                "joint_pd": float(pos_correct.mean()) if len(positive) else float("nan"),
            }
        )
    return threshold, pd.DataFrame(curve_rows)


def apply_threshold_and_metrics(
    frame: pd.DataFrame,
    threshold: float,
    tolerance: DetectionTolerance,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    result = frame.copy()
    result["detected"] = result["score"] > float(threshold)
    result["false_alarm"] = (result["target_present"] == 0) & result["detected"]
    result["correct_detection"] = (
        (result["target_present"] == 1)
        & result["detected"]
        & result["localization_ok"]
    )

    positives = result[result["target_present"] == 1]
    backgrounds = result[result["target_present"] == 0]

    tp_binary = int(((result["target_present"] == 1) & result["detected"]).sum())
    fn_binary = int(((result["target_present"] == 1) & ~result["detected"]).sum())
    fp_binary = int(((result["target_present"] == 0) & result["detected"]).sum())
    tn_binary = int(((result["target_present"] == 0) & ~result["detected"]).sum())
    precision = tp_binary / (tp_binary + fp_binary) if (tp_binary + fp_binary) else 0.0
    recall = tp_binary / (tp_binary + fn_binary) if (tp_binary + fn_binary) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    detected_positives = positives[positives["detected"]]
    correct_positives = positives[positives["correct_detection"]]

    metrics: dict[str, float | int] = {
        "threshold": float(threshold),
        "sample_count": int(len(result)),
        "positive_count": int(len(positives)),
        "background_count": int(len(backgrounds)),
        "false_alarm_count": int(backgrounds["detected"].sum()),
        "pfa": float(backgrounds["detected"].mean()) if len(backgrounds) else float("nan"),
        "score_detection_count": int(positives["detected"].sum()),
        "score_recall": float(positives["detected"].mean()) if len(positives) else float("nan"),
        "correct_detection_count": int(positives["correct_detection"].sum()),
        "joint_pd": float(positives["correct_detection"].mean()) if len(positives) else float("nan"),
        "binary_tp": tp_binary,
        "binary_fn": fn_binary,
        "binary_fp": fp_binary,
        "binary_tn": tn_binary,
        "binary_accuracy": float((tp_binary + tn_binary) / len(result)) if len(result) else float("nan"),
        "binary_precision": float(precision),
        "binary_recall": float(recall),
        "binary_f1": float(f1),
        "roc_auc": binary_auc(
            result["target_present"].to_numpy(int),
            result["score"].to_numpy(float),
        ),
        "all_positive_range_mae_gates": float(positives["range_error_gates"].mean()) if len(positives) else float("nan"),
        "all_positive_velocity_mae_bins": float(positives["velocity_error_bins"].mean()) if len(positives) else float("nan"),
        "detected_positive_range_mae_gates": float(detected_positives["range_error_gates"].mean()) if len(detected_positives) else float("nan"),
        "detected_positive_velocity_mae_bins": float(detected_positives["velocity_error_bins"].mean()) if len(detected_positives) else float("nan"),
        "correct_positive_range_mae_gates": float(correct_positives["range_error_gates"].mean()) if len(correct_positives) else float("nan"),
        "correct_positive_velocity_mae_bins": float(correct_positives["velocity_error_bins"].mean()) if len(correct_positives) else float("nan"),
        "range_tolerance_gates": int(tolerance.range_gates),
        "velocity_tolerance_bins": int(tolerance.velocity_bins),
    }
    return result, metrics


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    criterion: SampleWeightedHeatmapMSE,
    device: torch.device,
    tolerance: DetectionTolerance,
    amp_enabled: bool,
) -> tuple[pd.DataFrame, float]:
    model.eval()
    rows: list[dict[str, Any]] = []
    loss_sum = 0.0
    sample_count = 0
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            present = batch["target_present"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(inputs)
                loss = criterion(logits, targets, present)
            scores, pred_range, pred_velocity = extract_scores_and_peaks(logits)
            batch_size = inputs.shape[0]
            loss_sum += float(loss.item()) * batch_size
            sample_count += batch_size

            present_np = present.detach().cpu().numpy().astype(int)
            true_range = np.asarray(batch["range_index"], dtype=np.int64)
            true_velocity = np.asarray(batch["velocity_index"], dtype=np.int64)
            pred_range_np = pred_range.detach().cpu().numpy().astype(int)
            pred_velocity_np = pred_velocity.detach().cpu().numpy().astype(int)
            scores_np = scores.detach().cpu().numpy().astype(float)

            for index in range(batch_size):
                is_positive = int(present_np[index]) == 1
                if is_positive:
                    range_error = abs(int(pred_range_np[index]) - int(true_range[index]))
                    velocity_error = abs(int(pred_velocity_np[index]) - int(true_velocity[index]))
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
                        "target_present": int(present_np[index]),
                        "score": float(scores_np[index]),
                        "pred_range_index": int(pred_range_np[index]),
                        "pred_velocity_index": int(pred_velocity_np[index]),
                        "true_range_index": int(true_range[index]),
                        "true_velocity_index": int(true_velocity[index]),
                        "range_error_gates": range_error,
                        "velocity_error_bins": velocity_error,
                        "localization_ok": bool(localization_ok),
                        "beam_layer": int(np.asarray(batch["beam_layer"])[index]),
                        "azimuth_deg": float(np.asarray(batch["azimuth_deg"])[index]),
                        "distance_m": float(np.asarray(batch["distance_m"])[index]),
                        "velocity_mps": float(np.asarray(batch["velocity_mps"])[index]),
                        "mat_path": batch["mat_path"][index],
                    }
                )
    if sample_count == 0:
        raise RuntimeError("评价 DataLoader 为空")
    return pd.DataFrame(rows), loss_sum / sample_count


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: SampleWeightedHeatmapMSE,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    gradient_clip_norm: float,
    amp_enabled: bool,
) -> float:
    model.train()
    loss_sum = 0.0
    sample_count = 0
    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        present = batch["target_present"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(inputs)
            loss = criterion(logits, targets, present)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        batch_size = inputs.shape[0]
        loss_sum += float(loss.item()) * batch_size
        sample_count += batch_size
    if sample_count == 0:
        raise RuntimeError("训练 DataLoader 为空")
    return loss_sum / sample_count


def save_checkpoint(
    path: Path,
    model: nn.Module,
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
            "channel": args.channel,
            "in_channels": 2 if args.channel == "HV" else 1,
            "threshold": float(threshold),
            "validation_metrics": val_metrics,
            "dataset_version": getattr(
                args,
                "dataset_version",
                "unknown",
            ),
            "manifest_path": getattr(
                args,
                "manifest_path",
                "",
            ),
            "split_strategy": getattr(
                args,
                "split_strategy",
                "unknown",
            ),
            "fold_id": int(
                getattr(
                    args,
                    "fold_id",
                    0,
                )
            ),
            "raw_data_root": getattr(
                args,
                "data_root",
                "",
            ),
            "config": vars(args),
        },
        path,
    )


def plot_history(history: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history["epoch"], history["train_loss"], label="train loss")
    ax.plot(history["epoch"], history["val_loss"], label="val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history["epoch"], history["val_joint_pd"], label="val joint Pd")
    ax.plot(history["epoch"], history["val_pfa"], label="val Pfa")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Rate")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path.with_name("validation_detection_curve.png"), dpi=180)
    plt.close(fig)


def plot_score_histogram(frame: pd.DataFrame, threshold: float, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    background = frame.loc[frame["target_present"] == 0, "score"]
    positive = frame.loc[frame["target_present"] == 1, "score"]
    ax.hist(background, bins=30, alpha=0.65, label="background")
    ax.hist(positive, bins=30, alpha=0.65, label="UAV")
    ax.axvline(threshold, linestyle="--", label=f"threshold={threshold:.4f}")
    ax.set_xlabel("Maximum heatmap score")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    raw_data_root = resolve_project_path(args.data_root)
    manifest_path = resolve_project_path(args.manifest_path)

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"数据manifest不存在: {manifest_path}"
        )

    dataset_version = str(
        args.dataset_version
    ).strip()
    split_strategy = str(
        args.split_strategy
    ).strip()
    fold_id = int(args.fold_id)

    if not dataset_version:
        raise ValueError(
            "--dataset-version不能为空"
        )

    if not split_strategy:
        raise ValueError(
            "--split-strategy不能为空"
        )

    if fold_id < 0:
        raise ValueError(
            "--fold-id不能小于0"
        )

    args.data_root = str(raw_data_root)
    args.manifest_path = str(manifest_path)
    args.dataset_version = dataset_version
    args.split_strategy = split_strategy
    args.fold_id = fold_id

    experiment_dir = PROJECT_ROOT / "results" / "experiments" / args.name
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
        split: DetectionRadarDatasetV3(
            manifest_path=manifest_path,
            split=split,
            channel_mode=args.channel,
            range_sigma=args.range_sigma,
            velocity_sigma=args.velocity_sigma,
        )
        for split in ("train", "val", "test")
    }
    datasets: dict[str, Dataset] = {
        split: make_debug_subset(dataset, args.debug_per_class)
        for split, dataset in base_datasets.items()
    }
    if not args.no_memory_cache:
        datasets = {
            split: MemoryCachedDataset(dataset, label=split)
            for split, dataset in datasets.items()
        }
    loaders = {
        "train": make_loader(
            datasets["train"], args.batch_size, True, args.num_workers, device
        ),
        "val": make_loader(
            datasets["val"], args.batch_size, False, args.num_workers, device
        ),
        "test": make_loader(
            datasets["test"], args.batch_size, False, args.num_workers, device
        ),
    }

    in_channels = 2 if args.channel == "HV" else 1
    model = SimpleRadarFCN(in_channels=in_channels).to(device)
    criterion = SampleWeightedHeatmapMSE(args.positive_sample_weight)
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
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    print("=" * 78)
    print(f"实验: {args.name}")
    print(f"设备: {device}; AMP: {amp_enabled}")
    print(f"通道: {args.channel}; 参数量: {count_trainable_parameters(model):,}")
    print(
        "数据: "
        + ", ".join(f"{split}={len(dataset)}" for split, dataset in datasets.items())
    )
    print(
        f"验证阈值预算: 最多 {args.max_val_false_alarms} 个背景虚警; "
        f"定位容差: ±{tolerance.range_gates} 距离门, ±{tolerance.velocity_bins} 速度单元"
    )
    print("=" * 78)

    history_rows: list[dict[str, float | int]] = []
    best_key: tuple[float, float, float] | None = None
    best_epoch = 0
    stale_epochs = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            scaler,
            device,
            args.gradient_clip_norm,
            amp_enabled,
        )
        val_frame, val_loss = collect_predictions(
            model,
            loaders["val"],
            criterion,
            device,
            tolerance,
            amp_enabled,
        )
        val_threshold, _ = select_threshold_at_false_alarm_budget(
            val_frame, args.max_val_false_alarms
        )
        _, val_metrics = apply_threshold_and_metrics(
            val_frame, val_threshold, tolerance
        )
        scheduler.step(val_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": current_lr,
            "val_threshold": val_threshold,
            "val_joint_pd": float(val_metrics["joint_pd"]),
            "val_pfa": float(val_metrics["pfa"]),
            "val_score_recall": float(val_metrics["score_recall"]),
            "val_auc": float(val_metrics["roc_auc"]),
            "val_range_mae_gates": float(val_metrics["all_positive_range_mae_gates"]),
            "val_velocity_mae_bins": float(val_metrics["all_positive_velocity_mae_bins"]),
        }
        history_rows.append(row)

        # Pd 优先；同 Pd 时优先更低 Pfa，再优先更低 val loss。
        key = (
            float(val_metrics["joint_pd"]),
            -float(val_metrics["pfa"]),
            -float(val_loss),
        )
        improved = best_key is None or key > best_key
        if improved:
            best_key = key
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                optimizer,
                epoch,
                args,
                val_threshold,
                val_metrics,
            )
        else:
            stale_epochs += 1

        save_checkpoint(
            checkpoint_dir / "last.pt",
            model,
            optimizer,
            epoch,
            args,
            val_threshold,
            val_metrics,
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train={train_loss:.6f} val={val_loss:.6f} | "
            f"thr={val_threshold:.5f} Pd={float(val_metrics['joint_pd']):.4f} "
            f"Pfa={float(val_metrics['pfa']):.4f} | lr={current_lr:.2e}"
            + (" | best" if improved else "")
        )
        if stale_epochs >= args.early_stopping_patience:
            print(f"连续 {stale_epochs} 轮未提升，提前停止。")
            break

    history = pd.DataFrame(history_rows)
    history.to_csv(table_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    plot_history(history, figure_dir / "training_loss.png")

    best_checkpoint = torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    val_frame, best_val_loss = collect_predictions(
        model, loaders["val"], criterion, device, tolerance, amp_enabled
    )
    threshold, threshold_curve = select_threshold_at_false_alarm_budget(
        val_frame, args.max_val_false_alarms
    )
    val_results, val_metrics = apply_threshold_and_metrics(val_frame, threshold, tolerance)
    test_frame, test_loss = collect_predictions(
        model, loaders["test"], criterion, device, tolerance, amp_enabled
    )
    test_results, test_metrics = apply_threshold_and_metrics(test_frame, threshold, tolerance)

    val_results.to_csv(table_dir / "val_predictions.csv", index=False, encoding="utf-8-sig")
    test_results.to_csv(table_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    threshold_curve.to_csv(table_dir / "val_threshold_curve.csv", index=False, encoding="utf-8-sig")
    plot_score_histogram(
        val_results, threshold, figure_dir / "val_score_histogram.png", "Validation scores"
    )
    plot_score_histogram(
        test_results, threshold, figure_dir / "test_score_histogram.png", "Test scores"
    )

    elapsed_seconds = time.time() - start_time
    summary = {
        "experiment_name": args.name,
        "dataset_version": args.dataset_version,
        "manifest_path": args.manifest_path,
        "split_strategy": args.split_strategy,
        "fold_id": int(args.fold_id),
        "raw_data_root": args.data_root,
        "channel": args.channel,
        "device": str(device),
        "amp_enabled": amp_enabled,
        "parameter_count": count_trainable_parameters(model),
        "dataset_sizes": {split: len(dataset) for split, dataset in datasets.items()},
        "best_epoch": int(best_checkpoint["epoch"]),
        "validation_threshold": float(threshold),
        "max_val_false_alarms": int(args.max_val_false_alarms),
        "tolerance": asdict(tolerance),
        "best_val_loss": float(best_val_loss),
        "test_loss": float(test_loss),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "elapsed_seconds": float(elapsed_seconds),
        "config": vars(args),
    }
    (table_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print(f"最佳 epoch: {int(best_checkpoint['epoch'])}")
    print(f"验证集选定阈值: {threshold:.6f}")
    print(
        "验证集: "
        f"Pd={float(val_metrics['joint_pd']):.4f}, "
        f"Pfa={float(val_metrics['pfa']):.4f}, "
        f"AUC={float(val_metrics['roc_auc']):.4f}"
    )
    print(
        "测试集: "
        f"Pd={float(test_metrics['joint_pd']):.4f}, "
        f"Pfa={float(test_metrics['pfa']):.4f}, "
        f"AUC={float(test_metrics['roc_auc']):.4f}, "
        f"距离MAE={float(test_metrics['all_positive_range_mae_gates']):.3f}门, "
        f"速度MAE={float(test_metrics['all_positive_velocity_mae_bins']):.3f}单元"
    )
    print(f"结果目录: {experiment_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
