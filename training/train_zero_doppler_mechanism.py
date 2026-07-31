#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.detection_dataset_v3 import DetectionRadarDatasetV3  # noqa: E402
from models.dual_branch_gated_fcn import DualBranchGatedFCN  # noqa: E402
from models.zero_doppler_mechanisms import (  # noqa: E402
    ClutterAwareSuppressionHead,
    FixedNotchResidualSuppressionHead,
    FixedZeroDopplerNotch,
    SuppressionOutput,
)
from scripts.train_detection_baseline_v2 import (  # noqa: E402
    DetectionTolerance,
    MemoryCachedDataset,
    apply_threshold_and_metrics,
    extract_scores_and_peaks,
    json_safe,
    make_debug_subset,
    make_loader,
    set_seed,
)
from training.zero_doppler_objectives import (  # noqa: E402
    DenseZeroDopplerMSE,
    clutter_aware_detection_loss,
    fixed_residual_detection_loss,
)


MODES = (
    "baseline",
    "fixed_notch",
    "dense_negative",
    "clutter_aware",
    "fixed_residual",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or evaluate one zero-Doppler mechanism on frozen DPG"
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--output-root", default="results/experiments")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--debug-per-class", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--zero-band-radius", type=int, default=7)
    parser.add_argument("--zero-negative-weight", type=float, default=4.0)
    parser.add_argument("--target-guard-level", type=float, default=0.10)
    parser.add_argument("--notch-sigma-bins", type=float, default=4.0)
    parser.add_argument("--notch-floor", type=float, default=0.05)
    parser.add_argument("--maximum-suppression", type=float, default=4.0)
    parser.add_argument("--initial-suppression", type=float, default=0.05)
    parser.add_argument("--residual-hidden-channels", type=int, default=16)
    parser.add_argument("--residual-maximum-suppression", type=float, default=1.5)
    parser.add_argument("--residual-initial-suppression", type=float, default=1e-4)
    parser.add_argument("--residual-zero-sigma-bins", type=float, default=8.0)
    parser.add_argument("--background-peak-weight", type=float, default=2.0)
    parser.add_argument("--background-topk", type=int, default=16)
    parser.add_argument(
        "--residual-allowed-target-probability-drop", type=float, default=0.01
    )
    parser.add_argument("--residual-target-keep-weight", type=float, default=12.0)
    parser.add_argument("--allowed-target-probability-drop", type=float, default=0.02)
    parser.add_argument("--target-keep-weight", type=float, default=8.0)
    parser.add_argument("--suppression-regularization", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-memory-cache", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "epochs",
        "batch_size",
        "learning_rate",
        "gradient_clip_norm",
        "notch_sigma_bins",
        "maximum_suppression",
        "initial_suppression",
        "target_keep_weight",
        "residual_hidden_channels",
        "residual_maximum_suppression",
        "residual_initial_suppression",
        "residual_zero_sigma_bins",
        "background_peak_weight",
        "background_topk",
        "residual_target_keep_weight",
    )
    if any(getattr(args, name) <= 0 for name in positive):
        raise ValueError("positive training arguments must be greater than zero")
    if args.num_workers < 0 or args.debug_per_class < 0 or args.zero_band_radius < 0:
        raise ValueError("count and radius arguments must be nonnegative")
    if args.zero_negative_weight < 1.0:
        raise ValueError("zero-negative-weight must be at least one")
    if not 0.0 < args.notch_floor <= 1.0:
        raise ValueError("notch-floor must be in (0, 1]")
    if not 0.0 <= args.target_guard_level <= 1.0:
        raise ValueError("target-guard-level must be in [0, 1]")
    if not 0.0 <= args.residual_allowed_target_probability_drop <= 1.0:
        raise ValueError(
            "residual-allowed-target-probability-drop must be in [0, 1]"
        )


class ZeroDopplerMechanismDetector(nn.Module):
    def __init__(
        self,
        base: DualBranchGatedFCN,
        mode: str,
        args: argparse.Namespace,
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unsupported mode: {mode}")
        self.base = base
        self.mode = mode
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.fixed_notch: FixedZeroDopplerNotch | None = None
        self.clutter_head: ClutterAwareSuppressionHead | None = None
        self.residual_head: FixedNotchResidualSuppressionHead | None = None
        if mode == "dense_negative":
            for parameter in self.base.fusion_head.parameters():
                parameter.requires_grad_(True)
        elif mode == "fixed_notch":
            self.fixed_notch = FixedZeroDopplerNotch(
                velocity_bins=128,
                sigma_bins=args.notch_sigma_bins,
                floor=args.notch_floor,
            )
        elif mode == "clutter_aware":
            self.clutter_head = ClutterAwareSuppressionHead(
                hidden_channels=16,
                maximum_suppression=args.maximum_suppression,
                initial_suppression=args.initial_suppression,
            )
        elif mode == "fixed_residual":
            self.fixed_notch = FixedZeroDopplerNotch(
                velocity_bins=128,
                sigma_bins=args.notch_sigma_bins,
                floor=args.notch_floor,
            )
            self.residual_head = FixedNotchResidualSuppressionHead(
                velocity_bins=128,
                hidden_channels=args.residual_hidden_channels,
                maximum_suppression=args.residual_maximum_suppression,
                initial_suppression=args.residual_initial_suppression,
                zero_sigma_bins=args.residual_zero_sigma_bins,
            )

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def prepare_training_mode(self) -> None:
        self.eval()
        if self.mode == "dense_negative":
            self.base.fusion_head.train()
        elif self.mode == "clutter_aware":
            if self.clutter_head is None:
                raise RuntimeError("clutter head is unavailable")
            self.clutter_head.train()
        elif self.mode == "fixed_residual":
            if self.residual_head is None:
                raise RuntimeError("fixed residual head is unavailable")
            self.residual_head.train()

    def forward(self, inputs: Tensor) -> tuple[Tensor, SuppressionOutput]:
        base_output = self.base(inputs)
        raw_logits = base_output["fusion_logits"]
        if self.mode in {"baseline", "dense_negative"}:
            output = SuppressionOutput(
                calibrated_logits=raw_logits,
                suppression=torch.zeros_like(raw_logits),
            )
        elif self.mode == "fixed_notch":
            if self.fixed_notch is None:
                raise RuntimeError("fixed notch is unavailable")
            output = self.fixed_notch(raw_logits)
        elif self.mode == "clutter_aware":
            if self.clutter_head is None:
                raise RuntimeError("clutter head is unavailable")
            output = self.clutter_head(raw_logits, inputs)
        else:
            if self.fixed_notch is None or self.residual_head is None:
                raise RuntimeError("fixed residual mechanism is unavailable")
            fixed = self.fixed_notch(raw_logits)
            residual = self.residual_head(fixed.calibrated_logits, inputs)
            output = SuppressionOutput(
                calibrated_logits=residual.calibrated_logits,
                suppression=fixed.suppression + residual.suppression,
            )
        return raw_logits, output

    def fixed_reference(self, raw_logits: Tensor) -> SuppressionOutput:
        if self.fixed_notch is None:
            raise RuntimeError("fixed notch reference is unavailable")
        return self.fixed_notch(raw_logits)


def load_model(
    checkpoint_path: Path,
    mode: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[ZeroDopplerMechanismDetector, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    base = DualBranchGatedFCN(gate_hidden_dim=int(config.get("gate_hidden_dim", 16)))
    result = base.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"base checkpoint mismatch: {result}")
    return ZeroDopplerMechanismDetector(base, mode, args).to(device), checkpoint


def build_dataset(
    manifest_path: Path,
    split: str,
    checkpoint: dict[str, Any],
    args: argparse.Namespace,
) -> Dataset:
    config = checkpoint.get("config", {})
    dataset: Dataset = DetectionRadarDatasetV3(
        manifest_path=manifest_path,
        split=split,
        channel_mode="HV",
        range_sigma=float(config.get("range_sigma", 5.0)),
        velocity_sigma=float(config.get("velocity_sigma", 5.0)),
    )
    dataset = make_debug_subset(dataset, args.debug_per_class)
    if not args.no_memory_cache:
        dataset = MemoryCachedDataset(dataset, label=split)
    return dataset


def compute_loss(
    model: ZeroDopplerMechanismDetector,
    raw_logits: Tensor,
    output: SuppressionOutput,
    target: Tensor,
    present: Tensor,
    criterion: DenseZeroDopplerMSE,
    args: argparse.Namespace,
) -> tuple[Tensor, dict[str, Tensor]]:
    if model.mode == "clutter_aware":
        total, parts = clutter_aware_detection_loss(
            raw_logits=raw_logits,
            calibrated_logits=output.calibrated_logits,
            suppression=output.suppression,
            target=target,
            target_present=present,
            detection_criterion=criterion,
            allowed_target_probability_drop=(
                args.residual_allowed_target_probability_drop
            ),
            target_keep_weight=args.residual_target_keep_weight,
            suppression_regularization=args.suppression_regularization,
        )
        return total, dict(parts)
    if model.mode == "fixed_residual":
        fixed = model.fixed_reference(raw_logits)
        residual_suppression = output.suppression - fixed.suppression
        total, parts = fixed_residual_detection_loss(
            notched_logits=fixed.calibrated_logits,
            calibrated_logits=output.calibrated_logits,
            residual_suppression=residual_suppression,
            target=target,
            target_present=present,
            detection_criterion=criterion,
            allowed_target_probability_drop=args.allowed_target_probability_drop,
            target_keep_weight=args.target_keep_weight,
            suppression_regularization=args.suppression_regularization,
            background_peak_weight=args.background_peak_weight,
            background_topk=args.background_topk,
        )
        return total, dict(parts)
    detection = criterion(output.calibrated_logits, target, present)
    zero = detection.new_zeros(())
    return detection, {
        "detection": detection,
        "target_keep": zero,
        "background_peak": zero,
        "suppression_regularization": output.suppression.mean(),
    }


def train_epoch(
    model: ZeroDopplerMechanismDetector,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: DenseZeroDopplerMSE,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.prepare_training_mode()
    sums = {
        "total": 0.0,
        "detection": 0.0,
        "target_keep": 0.0,
        "background_peak": 0.0,
    }
    sample_count = 0
    for batch in loader:
        inputs = batch["input"].to(device)
        target = batch["target"].to(device)
        present = batch["target_present"].to(device)
        optimizer.zero_grad(set_to_none=True)
        raw_logits, output = model(inputs)
        total, parts = compute_loss(
            model, raw_logits, output, target, present, criterion, args
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            model.trainable_parameters(), args.gradient_clip_norm
        )
        optimizer.step()
        count = inputs.shape[0]
        sums["total"] += float(total.item()) * count
        sums["detection"] += float(parts["detection"].item()) * count
        sums["target_keep"] += float(parts["target_keep"].item()) * count
        background_peak = parts.get("background_peak")
        if background_peak is not None:
            sums["background_peak"] += float(background_peak.detach().item()) * count
        sample_count += count
    if sample_count == 0:
        raise RuntimeError("training loader is empty")
    return {key: value / sample_count for key, value in sums.items()}


def collect_predictions(
    model: ZeroDopplerMechanismDetector,
    loader: torch.utils.data.DataLoader,
    criterion: DenseZeroDopplerMSE,
    device: torch.device,
    tolerance: DetectionTolerance,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, float]:
    model.eval()
    rows: list[dict[str, Any]] = []
    loss_sum = 0.0
    sample_count = 0
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device)
            target = batch["target"].to(device)
            present = batch["target_present"].to(device)
            raw_logits, output = model(inputs)
            total, _ = compute_loss(
                model, raw_logits, output, target, present, criterion, args
            )
            score, pred_range, pred_velocity = extract_scores_and_peaks(
                output.calibrated_logits
            )
            raw_score, _, _ = extract_scores_and_peaks(raw_logits)
            suppression_at_peak = output.suppression[
                torch.arange(len(score), device=device),
                0,
                pred_velocity,
                pred_range,
            ]
            present_np = present.cpu().numpy().astype(np.int64)
            true_range = np.asarray(batch["range_index"], dtype=np.int64)
            true_velocity = np.asarray(batch["velocity_index"], dtype=np.int64)
            for index in range(len(score)):
                if present_np[index]:
                    range_error = abs(
                        int(pred_range[index]) - int(true_range[index])
                    )
                    velocity_error = abs(
                        int(pred_velocity[index]) - int(true_velocity[index])
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
                        "source_file": batch["source_file"][index],
                        "target_present": int(present_np[index]),
                        "score": float(score[index]),
                        "raw_score": float(raw_score[index]),
                        "score_never_increased": bool(
                            score[index] <= raw_score[index] + 1e-7
                        ),
                        "pred_range_index": int(pred_range[index]),
                        "pred_velocity_index": int(pred_velocity[index]),
                        "true_range_index": int(true_range[index]),
                        "true_velocity_index": int(true_velocity[index]),
                        "range_error_gates": range_error,
                        "velocity_error_bins": velocity_error,
                        "localization_ok": bool(localization_ok),
                        "suppression_at_peak": float(suppression_at_peak[index]),
                    }
                )
            count = inputs.shape[0]
            loss_sum += float(total.item()) * count
            sample_count += count
    if sample_count == 0:
        raise RuntimeError("evaluation loader is empty")
    return pd.DataFrame(rows), loss_sum / sample_count


def save_checkpoint(
    path: Path,
    model: ZeroDopplerMechanismDetector,
    epoch: int,
    metrics: dict[str, float | int],
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "mode": model.mode,
            "model_state_dict": model.state_dict(),
            "validation_metrics": metrics,
            "config": vars(args),
        },
        path,
    )


def worst_background_group_pfa(predictions: pd.DataFrame) -> float:
    background = predictions.loc[predictions["target_present"].eq(0)]
    if background.empty:
        return 0.0
    return float(
        background.groupby("source_file", observed=True)["false_alarm"].mean().max()
    )


def main() -> int:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    manifest_path = resolve_path(args.manifest_path)
    checkpoint_path = resolve_path(args.base_checkpoint)
    output_dir = resolve_path(args.output_root) / args.name
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"experiment exists: {output_dir}; use --overwrite")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    table_dir = output_dir / "tables"
    checkpoint_dir = output_dir / "checkpoints"
    table_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, base_checkpoint = load_model(checkpoint_path, args.mode, args, device)
    checkpoint_manifest_text = str(base_checkpoint.get("manifest_path", ""))
    if not checkpoint_manifest_text:
        raise ValueError("base checkpoint does not record its manifest path")
    checkpoint_manifest = resolve_path(checkpoint_manifest_text)
    if checkpoint_manifest != manifest_path:
        raise ValueError(
            "base checkpoint manifest does not match the requested manifest: "
            f"{checkpoint_manifest} != {manifest_path}"
        )
    trainable = model.trainable_parameters()
    frozen_threshold = float(base_checkpoint["threshold"])
    base_config = base_checkpoint.get("config", {})
    tolerance = DetectionTolerance(
        int(base_config.get("range_tolerance_gates", 2)),
        int(base_config.get("velocity_tolerance_bins", 3)),
    )
    required_splits = ("train", "val", "test") if trainable else ("val", "test")
    datasets = {
        split: build_dataset(manifest_path, split, base_checkpoint, args)
        for split in required_splits
    }
    loaders = {
        split: make_loader(
            dataset,
            args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.num_workers,
            device=device,
        )
        for split, dataset in datasets.items()
    }
    criterion = DenseZeroDopplerMSE(
        positive_sample_weight=float(base_config.get("positive_sample_weight", 10.0)),
        zero_band_radius=args.zero_band_radius,
        zero_negative_weight=args.zero_negative_weight,
        target_guard_level=args.target_guard_level,
    )

    history: list[dict[str, Any]] = []
    best_key: tuple[float, float, float, float, float] | None = None
    best_path = checkpoint_dir / "best.pt"
    selected_epoch: int | None = None
    if trainable:
        optimizer = torch.optim.AdamW(
            trainable,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        initial_frame, initial_loss = collect_predictions(
            model, loaders["val"], criterion, device, tolerance, args
        )
        initial_predictions, initial_metrics = apply_threshold_and_metrics(
            initial_frame, frozen_threshold, tolerance
        )
        initial_worst_group_pfa = worst_background_group_pfa(initial_predictions)
        best_key = (
            float(initial_metrics["joint_pd"]),
            -initial_worst_group_pfa,
            -float(initial_metrics["pfa"]),
            float(initial_metrics["roc_auc"]),
            -float(initial_loss),
        )
        history.append(
            {
                "epoch": 0,
                "train_total": math.nan,
                "train_detection": math.nan,
                "train_target_keep": math.nan,
                "val_loss": initial_loss,
                "val_joint_pd": initial_metrics["joint_pd"],
                "val_pfa": initial_metrics["pfa"],
                "val_worst_background_group_pfa": initial_worst_group_pfa,
                "val_auc": initial_metrics["roc_auc"],
            }
        )
        save_checkpoint(best_path, model, 0, initial_metrics, args)
        for epoch in range(1, args.epochs + 1):
            train_losses = train_epoch(
                model, loaders["train"], optimizer, criterion, device, args
            )
            val_frame, val_loss = collect_predictions(
                model, loaders["val"], criterion, device, tolerance, args
            )
            val_predictions_epoch, val_metrics = apply_threshold_and_metrics(
                val_frame, frozen_threshold, tolerance
            )
            val_worst_group_pfa = worst_background_group_pfa(
                val_predictions_epoch
            )
            key = (
                float(val_metrics["joint_pd"]),
                -val_worst_group_pfa,
                -float(val_metrics["pfa"]),
                float(val_metrics["roc_auc"]),
                -float(val_loss),
            )
            history.append(
                {
                    "epoch": epoch,
                    **{f"train_{key}": value for key, value in train_losses.items()},
                    "val_loss": val_loss,
                    "val_joint_pd": val_metrics["joint_pd"],
                    "val_pfa": val_metrics["pfa"],
                    "val_worst_background_group_pfa": val_worst_group_pfa,
                    "val_auc": val_metrics["roc_auc"],
                }
            )
            if key > best_key:
                best_key = key
                save_checkpoint(best_path, model, epoch, val_metrics, args)
        selected = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(selected["model_state_dict"], strict=True)
        selected_epoch = int(selected["epoch"])

    val_frame, val_loss = collect_predictions(
        model, loaders["val"], criterion, device, tolerance, args
    )
    test_frame, test_loss = collect_predictions(
        model, loaders["test"], criterion, device, tolerance, args
    )
    val_predictions, val_metrics = apply_threshold_and_metrics(
        val_frame, frozen_threshold, tolerance
    )
    test_predictions, test_metrics = apply_threshold_and_metrics(
        test_frame, frozen_threshold, tolerance
    )
    val_predictions.to_csv(table_dir / "val_predictions.csv", index=False)
    test_predictions.to_csv(table_dir / "test_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(table_dir / "training_history.csv", index=False)
    scan_metrics = (
        test_predictions.loc[test_predictions["target_present"].eq(0)]
        .groupby("source_file", observed=True)
        .agg(
            sample_count=("sample_id", "size"),
            false_alarm_count=("false_alarm", "sum"),
            pfa=("detected", "mean"),
            score_mean=("score", "mean"),
        )
        .reset_index()
    )
    scan_metrics.to_csv(table_dir / "test_background_scan_metrics.csv", index=False)
    contract_applies = args.mode in {
        "fixed_notch",
        "clutter_aware",
        "fixed_residual",
    }
    summary = {
        "status": "COMPLETE_MECHANICAL_SMOKE" if args.debug_per_class else "COMPLETE_DEVELOPMENT_RUN",
        "experiment_name": args.name,
        "mode": args.mode,
        "device": str(device),
        "seed": args.seed,
        "debug_per_class": args.debug_per_class,
        "epochs": args.epochs if trainable else 0,
        "selected_epoch": selected_epoch,
        "trainable_parameter_count": int(sum(parameter.numel() for parameter in trainable)),
        "dataset_sizes": {split: len(dataset) for split, dataset in datasets.items()},
        "frozen_threshold": frozen_threshold,
        "threshold_source": "base_DPG_validation_checkpoint",
        "validation_loss": val_loss,
        "test_loss": test_loss,
        "validation_metrics": val_metrics,
        "validation_worst_background_group_pfa": worst_background_group_pfa(
            val_predictions
        ),
        "test_metrics": test_metrics,
        "nonincrease_contract_applies": contract_applies,
        "nonincrease_contract_satisfied": (
            bool(test_predictions["score_never_increased"].all())
            if contract_applies
            else None
        ),
        "test_background_scan_metrics": scan_metrics.to_dict(orient="records"),
        "claim_warning": (
            "mechanical smoke only; debug subsets cannot support model comparison"
            if args.debug_per_class
            else "development evidence; outer folds are already consumed"
        ),
        "config": vars(args),
    }
    with (table_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(summary), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
