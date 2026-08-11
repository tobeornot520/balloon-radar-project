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
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.detection_dataset_v3 import (
    DEFAULT_MANIFEST,
    DetectionRadarDatasetV3,
)
from models.dual_branch_gated_fcn import (
    DualBranchGatedFCN,
    count_total_parameters,
    count_trainable_parameters,
    load_single_branch_checkpoint,
)
from scripts.train_detection_baseline_v2 import (
    DetectionTolerance,
    MemoryCachedDataset,
    SampleWeightedHeatmapMSE,
    apply_threshold_and_metrics,
    json_safe,
    make_debug_subset,
    make_loader,
    plot_history,
    plot_score_histogram,
    resolve_project_path,
    select_threshold_at_false_alarm_budget,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练H/V独立编码的门控融合FCN")
    parser.add_argument("--name", default="dpg_fcn_v1_seed42")
    parser.add_argument(
        "--data-root",
        default="data/raw/detection_dataset",
        help="原始MAT/TXT根目录，仅用于实验溯源",
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
    parser.add_argument(
        "--h-checkpoint",
        default="results/experiments/detection_h_v3_grouped_seed42/checkpoints/best.pt",
    )
    parser.add_argument(
        "--v-checkpoint",
        default="results/experiments/detection_v_v3_grouped_seed42/checkpoints/best.pt",
    )
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--warmup-epochs", type=int, default=15)
    parser.add_argument("--partial-unfreeze-epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--fusion-learning-rate", type=float, default=3e-4)
    parser.add_argument("--branch-learning-rate", type=float, default=3e-5)
    parser.add_argument("--min-fusion-learning-rate", type=float, default=3e-6)
    parser.add_argument("--min-branch-learning-rate", type=float, default=3e-7)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--positive-sample-weight", type=float, default=10.0)
    parser.add_argument("--h-aux-weight", type=float, default=0.2)
    parser.add_argument("--v-aux-weight", type=float, default=0.2)
    parser.add_argument("--range-sigma", type=float, default=5.0)
    parser.add_argument("--velocity-sigma", type=float, default=5.0)
    parser.add_argument("--gate-hidden-dim", type=int, default=16)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    # Keep the CLI default aligned with configs/dual_branch_gated_v1.yaml.
    parser.add_argument("--max-val-false-alarms", type=int, default=1)
    parser.add_argument("--range-tolerance-gates", type=int, default=2)
    parser.add_argument("--velocity-tolerance-bins", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug-per-class", type=int, default=0)
    parser.add_argument("--no-memory-cache", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "epochs", "batch_size", "fusion_learning_rate", "branch_learning_rate",
        "min_fusion_learning_rate", "min_branch_learning_rate", "positive_sample_weight",
        "range_sigma", "velocity_sigma", "gate_hidden_dim", "early_stopping_patience",
        "scheduler_patience", "gradient_clip_norm",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')}必须大于0")
    if args.warmup_epochs < 0 or args.partial_unfreeze_epochs < 0:
        raise ValueError("训练阶段轮数不能小于0")
    if args.warmup_epochs + args.partial_unfreeze_epochs > args.epochs:
        raise ValueError("warmup_epochs + partial_unfreeze_epochs不能超过epochs")
    if args.h_aux_weight < 0 or args.v_aux_weight < 0:
        raise ValueError("辅助损失权重不能小于0")
    if args.max_val_false_alarms < 0 or args.num_workers < 0 or args.debug_per_class < 0:
        raise ValueError("计数参数不能小于0")


def stage_for_epoch(epoch: int, args: argparse.Namespace) -> str:
    if epoch <= args.warmup_epochs:
        return "warmup"
    if epoch <= args.warmup_epochs + args.partial_unfreeze_epochs:
        return "partial"
    return "full"


def extract_scores_and_peaks(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prediction = torch.sigmoid(logits)
    flattened = prediction.flatten(1)
    scores, indices = flattened.max(dim=1)
    range_size = prediction.shape[-1]
    return scores, indices % range_size, indices // range_size


def compute_losses(
    outputs: dict[str, torch.Tensor],
    target: torch.Tensor,
    present: torch.Tensor,
    criterion: SampleWeightedHeatmapMSE,
    h_aux_weight: float,
    v_aux_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    fusion_loss = criterion(outputs["fusion_logits"], target, present)
    h_loss = criterion(outputs["h_logits"], target, present)
    v_loss = criterion(outputs["v_logits"], target, present)
    total = fusion_loss + h_aux_weight * h_loss + v_aux_weight * v_loss
    return total, {"fusion": fusion_loss, "h_aux": h_loss, "v_aux": v_loss}


def train_one_epoch(
    model: DualBranchGatedFCN,
    loader: DataLoader,
    criterion: SampleWeightedHeatmapMSE,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
    amp_enabled: bool,
) -> dict[str, float]:
    model.train()
    sums = {"total": 0.0, "fusion": 0.0, "h_aux": 0.0, "v_aux": 0.0}
    sample_count = 0
    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        present = batch["target_present"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            outputs = model(inputs)
            total, parts = compute_losses(
                outputs, target, present, criterion, args.h_aux_weight, args.v_aux_weight
            )
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        n = inputs.shape[0]
        sums["total"] += float(total.item()) * n
        for key in ("fusion", "h_aux", "v_aux"):
            sums[key] += float(parts[key].item()) * n
        sample_count += n
    if sample_count == 0:
        raise RuntimeError("训练DataLoader为空")
    return {key: value / sample_count for key, value in sums.items()}


def collect_predictions(
    model: DualBranchGatedFCN,
    loader: DataLoader,
    criterion: SampleWeightedHeatmapMSE,
    device: torch.device,
    tolerance: DetectionTolerance,
    args: argparse.Namespace,
    amp_enabled: bool,
) -> tuple[pd.DataFrame, dict[str, float]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    sums = {"total": 0.0, "fusion": 0.0, "h_aux": 0.0, "v_aux": 0.0}
    sample_count = 0
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            present = batch["target_present"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                outputs = model(inputs)
                total, parts = compute_losses(
                    outputs, target, present, criterion, args.h_aux_weight, args.v_aux_weight
                )
            fusion_score, fusion_r, fusion_v = extract_scores_and_peaks(outputs["fusion_logits"])
            h_score, h_r, h_v = extract_scores_and_peaks(outputs["h_logits"])
            v_score, v_r, v_v = extract_scores_and_peaks(outputs["v_logits"])
            gates = outputs["gate_weights"]
            n = inputs.shape[0]
            sums["total"] += float(total.item()) * n
            for key in ("fusion", "h_aux", "v_aux"):
                sums[key] += float(parts[key].item()) * n
            sample_count += n

            present_np = present.cpu().numpy().astype(int)
            true_r = np.asarray(batch["range_index"], dtype=np.int64)
            true_v = np.asarray(batch["velocity_index"], dtype=np.int64)
            arrays = {
                "score": fusion_score.cpu().numpy(),
                "pred_r": fusion_r.cpu().numpy(),
                "pred_v": fusion_v.cpu().numpy(),
                "h_score": h_score.cpu().numpy(), "h_r": h_r.cpu().numpy(), "h_v": h_v.cpu().numpy(),
                "v_score": v_score.cpu().numpy(), "v_r": v_r.cpu().numpy(), "v_v": v_v.cpu().numpy(),
                "gates": gates.cpu().numpy(),
            }
            for i in range(n):
                positive = int(present_np[i]) == 1
                if positive:
                    range_error = abs(int(arrays["pred_r"][i]) - int(true_r[i]))
                    velocity_error = abs(int(arrays["pred_v"][i]) - int(true_v[i]))
                    localization_ok = (
                        range_error <= tolerance.range_gates
                        and velocity_error <= tolerance.velocity_bins
                    )
                else:
                    range_error = math.nan
                    velocity_error = math.nan
                    localization_ok = False
                rows.append({
                    "sample_id": batch["sample_id"][i],
                    "target_present": int(present_np[i]),
                    "score": float(arrays["score"][i]),
                    "pred_range_index": int(arrays["pred_r"][i]),
                    "pred_velocity_index": int(arrays["pred_v"][i]),
                    "true_range_index": int(true_r[i]),
                    "true_velocity_index": int(true_v[i]),
                    "range_error_gates": range_error,
                    "velocity_error_bins": velocity_error,
                    "localization_ok": bool(localization_ok),
                    "h_aux_score": float(arrays["h_score"][i]),
                    "v_aux_score": float(arrays["v_score"][i]),
                    "h_pred_range_index": int(arrays["h_r"][i]),
                    "h_pred_velocity_index": int(arrays["h_v"][i]),
                    "v_pred_range_index": int(arrays["v_r"][i]),
                    "v_pred_velocity_index": int(arrays["v_v"][i]),
                    "gate_h": float(arrays["gates"][i, 0]),
                    "gate_v": float(arrays["gates"][i, 1]),
                    "beam_layer": int(np.asarray(batch["beam_layer"])[i]),
                    "azimuth_deg": float(np.asarray(batch["azimuth_deg"])[i]),
                    "distance_m": float(np.asarray(batch["distance_m"])[i]),
                    "velocity_mps": float(np.asarray(batch["velocity_mps"])[i]),
                    "mat_path": batch["mat_path"][i],
                })
    if sample_count == 0:
        raise RuntimeError("评价DataLoader为空")
    return pd.DataFrame(rows), {key: value / sample_count for key, value in sums.items()}


def add_gate_diagnostics(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    result = frame.copy()
    result["detected"] = result["score"] > float(threshold)
    result["false_alarm"] = (result["target_present"] == 0) & result["detected"]
    result["correct_detection"] = (
        (result["target_present"] == 1) & result["detected"] & result["localization_ok"]
    )
    result["dominant_branch"] = np.where(result["gate_h"] >= result["gate_v"], "H", "V")
    result["gate_margin"] = np.abs(result["gate_h"] - result["gate_v"])
    return result


def gate_statistics(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    masks = {
        "all": np.ones(len(frame), dtype=bool),
        "positive": frame["target_present"].to_numpy(int) == 1,
        "background": frame["target_present"].to_numpy(int) == 0,
        "correct_positive": frame["correct_detection"].to_numpy(bool),
        "failed_positive": (frame["target_present"].to_numpy(int) == 1) & ~frame["correct_detection"].to_numpy(bool),
        "false_alarm": frame["false_alarm"].to_numpy(bool),
    }
    rows = []
    for group, mask in masks.items():
        part = frame.loc[mask]
        rows.append({
            "split": split,
            "group": group,
            "count": int(len(part)),
            "mean_gate_h": float(part["gate_h"].mean()) if len(part) else math.nan,
            "mean_gate_v": float(part["gate_v"].mean()) if len(part) else math.nan,
            "median_gate_h": float(part["gate_h"].median()) if len(part) else math.nan,
            "h_dominant_count": int((part["dominant_branch"] == "H").sum()) if len(part) else 0,
            "v_dominant_count": int((part["dominant_branch"] == "V").sum()) if len(part) else 0,
            "mean_gate_margin": float(part["gate_margin"].mean()) if len(part) else math.nan,
        })
    return pd.DataFrame(rows)


def save_checkpoint(
    path: Path,
    model: DualBranchGatedFCN,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    epoch: int,
    stage: str,
    args: argparse.Namespace,
    threshold: float,
    metrics: dict[str, float | int],
    init_info: dict[str, Any],
) -> None:
    torch.save({
        "epoch": int(epoch),
        "stage": stage,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "threshold": float(threshold),
        "validation_metrics": metrics,
        "initialization": init_info,
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
        "h_checkpoint": getattr(
            args,
            "h_checkpoint",
            "",
        ),
        "v_checkpoint": getattr(
            args,
            "v_checkpoint",
            "",
        ),
        "config": vars(args),
    }, path)


def validate_branch_checkpoint(
    checkpoint_path: Path,
    expected_channel: str,
    manifest_path: Path,
    expected_dataset_version: str,
    expected_split_strategy: str,
    expected_fold_id: int,
) -> dict[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"{expected_channel}初始化checkpoint不存在: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    actual_channel = str(
        checkpoint.get("channel", "")
    )
    if actual_channel != expected_channel:
        raise RuntimeError(
            f"{expected_channel}分支checkpoint通道错误: "
            f"期望{expected_channel}, 实际{actual_channel}"
        )

    dataset_version = str(
        checkpoint.get("dataset_version", "")
    )

    if dataset_version != expected_dataset_version:
        raise RuntimeError(
            f"{expected_channel}分支数据版本错误: "
            f"期望{expected_dataset_version!r}, "
            f"实际{dataset_version!r}"
        )

    checkpoint_manifest_text = str(
        checkpoint.get("manifest_path", "")
    )
    if not checkpoint_manifest_text:
        raise RuntimeError(
            f"{expected_channel}分支checkpoint缺少manifest_path"
        )

    checkpoint_manifest = Path(
        checkpoint_manifest_text
    ).expanduser().resolve()

    if checkpoint_manifest != manifest_path.resolve():
        raise RuntimeError(
            f"{expected_channel}分支manifest不一致:\n"
            f"checkpoint: {checkpoint_manifest}\n"
            f"current:    {manifest_path.resolve()}"
        )

    checkpoint_split_strategy = str(
        checkpoint.get("split_strategy", "")
    )

    if checkpoint_split_strategy != expected_split_strategy:
        raise RuntimeError(
            f"{expected_channel}分支划分策略错误:\n"
            f"期望: {expected_split_strategy}\n"
            f"实际: {checkpoint_split_strategy}"
        )

    config = checkpoint.get("config", {})

    checkpoint_fold_id = int(
        checkpoint.get(
            "fold_id",
            config.get("fold_id", 0),
        )
    )

    if checkpoint_fold_id != expected_fold_id:
        raise RuntimeError(
            f"{expected_channel}分支fold错误: "
            f"期望{expected_fold_id}, "
            f"实际{checkpoint_fold_id}"
        )

    return {
        "checkpoint_path": str(checkpoint_path),
        "experiment_name": config.get("name"),
        "channel": actual_channel,
        "dataset_version": dataset_version,
        "manifest_path": str(checkpoint_manifest),
        "split_strategy": checkpoint_split_strategy,
        "fold_id": checkpoint_fold_id,
        "seed": config.get("seed"),
        "best_epoch": checkpoint.get("epoch"),
        "threshold": checkpoint.get("threshold"),
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    raw_data_root = resolve_project_path(args.data_root)
    manifest_path = resolve_project_path(args.manifest_path)
    h_checkpoint = resolve_project_path(args.h_checkpoint)
    v_checkpoint = resolve_project_path(args.v_checkpoint)

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
    args.h_checkpoint = str(h_checkpoint)
    args.v_checkpoint = str(v_checkpoint)
    args.dataset_version = dataset_version
    args.split_strategy = split_strategy
    args.fold_id = fold_id

    h_checkpoint_metadata = validate_branch_checkpoint(
        h_checkpoint,
        expected_channel="H",
        manifest_path=manifest_path,
        expected_dataset_version=args.dataset_version,
        expected_split_strategy=args.split_strategy,
        expected_fold_id=args.fold_id,
    )

    v_checkpoint_metadata = validate_branch_checkpoint(
        v_checkpoint,
        expected_channel="V",
        manifest_path=manifest_path,
        expected_dataset_version=args.dataset_version,
        expected_split_strategy=args.split_strategy,
        expected_fold_id=args.fold_id,
    )

    experiment_dir = PROJECT_ROOT / "results" / "experiments" / args.name
    checkpoint_dir = experiment_dir / "checkpoints"
    table_dir = experiment_dir / "tables"
    figure_dir = experiment_dir / "figures"
    for directory in (checkpoint_dir, table_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda" and not args.no_amp
    tolerance = DetectionTolerance(args.range_tolerance_gates, args.velocity_tolerance_bins)

    base = {
        split: DetectionRadarDatasetV3(
            manifest_path=manifest_path,
            split=split,
            channel_mode="HV",
            range_sigma=args.range_sigma,
            velocity_sigma=args.velocity_sigma,
        ) for split in ("train", "val", "test")
    }
    datasets: dict[str, Dataset] = {
        split: make_debug_subset(dataset, args.debug_per_class)
        for split, dataset in base.items()
    }
    if not args.no_memory_cache:
        datasets = {split: MemoryCachedDataset(ds, label=split) for split, ds in datasets.items()}
    loaders = {
        "train": make_loader(datasets["train"], args.batch_size, True, args.num_workers, device),
        "val": make_loader(datasets["val"], args.batch_size, False, args.num_workers, device),
        "test": make_loader(datasets["test"], args.batch_size, False, args.num_workers, device),
    }

    model = DualBranchGatedFCN(args.gate_hidden_dim).to(device)
    init_info = {
        "H": {
            "source": h_checkpoint_metadata,
            "load_result": load_single_branch_checkpoint(
                model.h_branch,
                h_checkpoint,
                device,
            ),
        },
        "V": {
            "source": v_checkpoint_metadata,
            "load_result": load_single_branch_checkpoint(
                model.v_branch,
                v_checkpoint,
                device,
            ),
        },
    }
    criterion = SampleWeightedHeatmapMSE(args.positive_sample_weight)
    optimizer = torch.optim.AdamW([
        {"params": list(model.fusion_parameters()), "lr": args.fusion_learning_rate, "name": "fusion"},
        {"params": list(model.branch_parameters()), "lr": args.branch_learning_rate, "name": "branches"},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.scheduler_patience,
        min_lr=[args.min_fusion_learning_rate, args.min_branch_learning_rate],
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    start_epoch = 1
    best_key = None
    best_epoch = 0
    stale_epochs = 0
    history_rows: list[dict[str, Any]] = []
    last_path = checkpoint_dir / "last.pt"
    if args.resume and last_path.exists():
        resume = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(resume["model_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        if "scheduler_state_dict" in resume:
            scheduler.load_state_dict(resume["scheduler_state_dict"])
        start_epoch = int(resume["epoch"]) + 1
        history_path = table_dir / "training_history.csv"
        if history_path.exists():
            history_rows = pd.read_csv(history_path, encoding="utf-8-sig").to_dict("records")
        best_path = checkpoint_dir / "best.pt"
        if best_path.exists():
            previous_best = torch.load(best_path, map_location="cpu", weights_only=False)
            previous_metrics = previous_best.get("validation_metrics", {})
            if previous_metrics:
                best_key = (
                    float(previous_metrics.get("joint_pd", -math.inf)),
                    -float(previous_metrics.get("pfa", math.inf)),
                    float(previous_metrics.get("roc_auc", -math.inf)),
                    -math.inf,
                )
                best_epoch = int(previous_best.get("epoch", 0))
        print(f"从epoch {start_epoch}继续训练：{last_path}")

    print("=" * 78)
    print(f"实验：{args.name}; 设备：{device}; AMP：{amp_enabled}; seed：{args.seed}")
    print(f"H初始化：{init_info['H']}")
    print(f"V初始化：{init_info['V']}")
    print(f"总参数量：{count_total_parameters(model):,}")
    print("数据：" + ", ".join(f"{s}={len(d)}" for s, d in datasets.items()))
    print(f"阶段：warmup={args.warmup_epochs}, partial={args.partial_unfreeze_epochs}, full={args.epochs-args.warmup_epochs-args.partial_unfreeze_epochs}")
    print("=" * 78)

    start_time = time.time()
    current_stage = None
    for epoch in range(start_epoch, args.epochs + 1):
        stage = stage_for_epoch(epoch, args)
        if stage != current_stage:
            model.set_branch_trainability(stage)
            current_stage = stage
            stale_epochs = 0
            print(f"进入训练阶段：{stage}; 可训练参数：{count_trainable_parameters(model):,}")

        train_losses = train_one_epoch(
            model, loaders["train"], criterion, optimizer, scaler, device, args, amp_enabled
        )
        val_frame, val_losses = collect_predictions(
            model, loaders["val"], criterion, device, tolerance, args, amp_enabled
        )
        threshold, _ = select_threshold_at_false_alarm_budget(val_frame, args.max_val_false_alarms)
        _, val_metrics = apply_threshold_and_metrics(val_frame, threshold, tolerance)
        scheduler.step(val_losses["total"])
        row = {
            "epoch": epoch, "stage": stage,
            "train_loss": train_losses["total"], "train_fusion_loss": train_losses["fusion"],
            "train_h_aux_loss": train_losses["h_aux"], "train_v_aux_loss": train_losses["v_aux"],
            "val_loss": val_losses["total"], "val_fusion_loss": val_losses["fusion"],
            "val_h_aux_loss": val_losses["h_aux"], "val_v_aux_loss": val_losses["v_aux"],
            "fusion_lr": float(optimizer.param_groups[0]["lr"]),
            "branch_lr": float(optimizer.param_groups[1]["lr"]),
            "val_threshold": threshold,
            "val_joint_pd": float(val_metrics["joint_pd"]),
            "val_pfa": float(val_metrics["pfa"]),
            "val_score_recall": float(val_metrics["score_recall"]),
            "val_auc": float(val_metrics["roc_auc"]),
            "val_range_mae_gates": float(val_metrics["all_positive_range_mae_gates"]),
            "val_velocity_mae_bins": float(val_metrics["all_positive_velocity_mae_bins"]),
        }
        history_rows.append(row)
        key = (float(val_metrics["joint_pd"]), -float(val_metrics["pfa"]), float(val_metrics["roc_auc"]), -float(val_losses["total"]))
        improved = best_key is None or key > best_key
        if improved:
            best_key, best_epoch, stale_epochs = key, epoch, 0
            save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, scheduler, epoch, stage, args, threshold, val_metrics, init_info)
        else:
            stale_epochs += 1
        save_checkpoint(last_path, model, optimizer, scheduler, epoch, stage, args, threshold, val_metrics, init_info)
        print(
            f"Epoch {epoch:03d}/{args.epochs} [{stage}] | train={train_losses['total']:.6f} "
            f"val={val_losses['total']:.6f} | thr={threshold:.5f} "
            f"Pd={float(val_metrics['joint_pd']):.4f} Pfa={float(val_metrics['pfa']):.4f} "
            f"gateLR={optimizer.param_groups[0]['lr']:.2e} branchLR={optimizer.param_groups[1]['lr']:.2e}"
            + (" | best" if improved else "")
        )
        if stage == "full" and stale_epochs >= args.early_stopping_patience:
            print(f"全模型阶段连续{stale_epochs}轮未提升，提前停止。")
            break

    history = pd.DataFrame(history_rows)
    history.to_csv(table_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    if len(history):
        plot_history(history, figure_dir / "training_loss.png")

    best = torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    val_frame, val_losses = collect_predictions(model, loaders["val"], criterion, device, tolerance, args, amp_enabled)
    threshold, threshold_curve = select_threshold_at_false_alarm_budget(val_frame, args.max_val_false_alarms)
    val_results, val_metrics = apply_threshold_and_metrics(val_frame, threshold, tolerance)
    val_results = add_gate_diagnostics(val_results, threshold)
    test_frame, test_losses = collect_predictions(model, loaders["test"], criterion, device, tolerance, args, amp_enabled)
    test_results, test_metrics = apply_threshold_and_metrics(test_frame, threshold, tolerance)
    test_results = add_gate_diagnostics(test_results, threshold)

    val_results.to_csv(table_dir / "val_predictions.csv", index=False, encoding="utf-8-sig")
    test_results.to_csv(table_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    threshold_curve.to_csv(table_dir / "val_threshold_curve.csv", index=False, encoding="utf-8-sig")
    pd.concat([gate_statistics(val_results, "val"), gate_statistics(test_results, "test")], ignore_index=True).to_csv(
        table_dir / "gate_statistics.csv", index=False, encoding="utf-8-sig"
    )
    test_results[(test_results["target_present"] == 1) & ~test_results["correct_detection"]].to_csv(
        table_dir / "test_failure_samples.csv", index=False, encoding="utf-8-sig"
    )
    test_results[test_results["false_alarm"]].to_csv(
        table_dir / "test_false_alarm_samples.csv", index=False, encoding="utf-8-sig"
    )
    test_results[
        (test_results["h_pred_range_index"] != test_results["v_pred_range_index"])
        | (test_results["h_pred_velocity_index"] != test_results["v_pred_velocity_index"])
    ].to_csv(
        table_dir / "test_branch_disagreement.csv", index=False, encoding="utf-8-sig"
    )
    plot_score_histogram(val_results, threshold, figure_dir / "val_score_histogram.png", "DPG-FCN validation scores")
    plot_score_histogram(test_results, threshold, figure_dir / "test_score_histogram.png", "DPG-FCN test scores")

    summary = {
        "experiment_name": args.name,
        "model": "DualBranchGatedFCN",
        "dataset_version": args.dataset_version,
        "manifest_path": args.manifest_path,
        "split_strategy": args.split_strategy,
        "fold_id": int(args.fold_id),
        "raw_data_root": args.data_root,
        "h_checkpoint": args.h_checkpoint,
        "v_checkpoint": args.v_checkpoint,
        "device": str(device), "amp_enabled": amp_enabled,
        "parameter_count": count_total_parameters(model),
        "dataset_sizes": {split: len(dataset) for split, dataset in datasets.items()},
        "best_epoch": int(best["epoch"]), "best_stage": best.get("stage"),
        "validation_threshold": float(threshold),
        "max_val_false_alarms": args.max_val_false_alarms,
        "tolerance": asdict(tolerance),
        "initialization": init_info,
        "validation_losses": val_losses, "test_losses": test_losses,
        "validation_metrics": val_metrics, "test_metrics": test_metrics,
        "elapsed_seconds": time.time() - start_time,
        "config": vars(args),
    }
    (table_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n" + "=" * 78)
    print(f"最佳epoch：{best['epoch']}；阶段：{best.get('stage')}；验证集阈值：{threshold:.6f}")
    print(f"验证集：Pd={float(val_metrics['joint_pd']):.4f}, Pfa={float(val_metrics['pfa']):.4f}, AUC={float(val_metrics['roc_auc']):.4f}")
    print(f"测试集：Pd={float(test_metrics['joint_pd']):.4f}, Pfa={float(test_metrics['pfa']):.4f}, AUC={float(test_metrics['roc_auc']):.4f}")
    print(f"测试定位：距离MAE={float(test_metrics['detected_positive_range_mae_gates']):.3f}门，速度MAE={float(test_metrics['detected_positive_velocity_mae_bins']):.3f}单元")
    print(f"结果目录：{experiment_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
