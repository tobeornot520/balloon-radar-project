from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from datasets.radar_dataset import RadarDataset
from scripts.train_local_refiner import (
    LocalOffsetRefiner,
    extract_local_patches,
)
from scripts.train_notch_dual_fcn import (
    DualViewFCN,
    SoftNotchDataset,
    extract_peaks,
    get_state,
    load_torch,
)


SEED = 42

METRIC_RADII = {
    "strict": {
        "range": 1,
        "velocity": 1,
    },
    "relaxed": {
        "range": 4,
        "velocity": 1,
    },
    "application": {
        "range": 4,
        "velocity": 2,
    },
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "双分支零多普勒抗杂波FCN"
            " + 局部偏移精修器"
        )
    )

    parser.add_argument(
        "--dual-model",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results"
            / "experiments"
            / "sigma_r3_v1_notch_dual"
            / "checkpoints"
            / "best.pt"
        ),
        help="双分支抗杂波FCN模型",
    )

    parser.add_argument(
        "--original-model",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results"
            / "experiments"
            / "sigma_r3_v1"
            / "checkpoints"
            / "best.pt"
        ),
        help="构造双分支网络时使用的原始FCN模型",
    )

    parser.add_argument(
        "--name",
        default="sigma_r3_v1_notch_dual_refiner",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1.0e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1.0e-5,
    )

    parser.add_argument(
        "--crop-size",
        type=int,
        default=17,
    )

    parser.add_argument(
        "--max-range-offset",
        type=float,
        default=4.0,
    )

    parser.add_argument(
        "--max-velocity-offset",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--notch-sigma",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--notch-floor",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--early-stopping",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_trainable_parameters(
    model: nn.Module,
) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def load_frozen_dual_model(
    dual_model_path: Path,
    original_model_path: Path,
    device: torch.device,
) -> DualViewFCN:
    if not dual_model_path.exists():
        raise FileNotFoundError(
            f"找不到双分支模型：{dual_model_path}"
        )

    if not original_model_path.exists():
        raise FileNotFoundError(
            f"找不到原始FCN模型：{original_model_path}"
        )

    original_checkpoint = load_torch(
        original_model_path,
        device,
    )

    original_state = get_state(
        original_checkpoint
    )

    model = DualViewFCN(
        original_state
    ).to(device)

    dual_checkpoint = load_torch(
        dual_model_path,
        device,
    )

    if not isinstance(dual_checkpoint, dict):
        raise TypeError(
            "双分支模型文件不是有效checkpoint。"
        )

    dual_state = dual_checkpoint.get(
        "model_state_dict",
        dual_checkpoint,
    )

    model.load_state_dict(
        dual_state
    )

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    return model


def predict_coordinates(
    backbone: DualViewFCN,
    refiner: LocalOffsetRefiner,
    raw_input: torch.Tensor,
    notch_input: torch.Tensor,
    crop_size: int,
    max_range_offset: float,
    max_velocity_offset: float,
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        (
            raw_logits,
            notch_logits,
            fused_logits,
        ) = backbone(
            raw_input,
            notch_input,
        )

        fused_probability = torch.sigmoid(
            fused_logits
        )

        (
            coarse_range_indices,
            coarse_velocity_indices,
        ) = extract_peaks(
            fused_logits
        )

    # 2个原始H/V通道 + 1个融合概率图
    feature_map = torch.cat(
        [
            raw_input,
            fused_probability,
        ],
        dim=1,
    )

    # extract_local_patches会继续添加：
    # 距离坐标通道 + 速度坐标通道
    # 因此最终输入局部精修器的通道数为5
    local_patches = extract_local_patches(
        feature_map=feature_map,
        center_range_indices=
            coarse_range_indices,
        center_velocity_indices=
            coarse_velocity_indices,
        crop_size=crop_size,
    )

    normalized_offsets = refiner(
        local_patches
    )

    predicted_range_offsets = (
        normalized_offsets[:, 0]
        * max_range_offset
    )

    predicted_velocity_offsets = (
        normalized_offsets[:, 1]
        * max_velocity_offset
    )

    refined_range_indices = (
        coarse_range_indices.float()
        + torch.round(
            predicted_range_offsets
        )
    ).long()

    refined_velocity_indices = (
        coarse_velocity_indices.float()
        + torch.round(
            predicted_velocity_offsets
        )
    ).long()

    refined_range_indices = torch.clamp(
        refined_range_indices,
        min=0,
        max=raw_input.shape[-1] - 1,
    )

    refined_velocity_indices = torch.clamp(
        refined_velocity_indices,
        min=0,
        max=raw_input.shape[-2] - 1,
    )

    return {
        "raw_logits":
            raw_logits,
        "notch_logits":
            notch_logits,
        "fused_logits":
            fused_logits,
        "fused_probability":
            fused_probability,
        "coarse_range_indices":
            coarse_range_indices,
        "coarse_velocity_indices":
            coarse_velocity_indices,
        "normalized_offsets":
            normalized_offsets,
        "predicted_range_offsets":
            predicted_range_offsets,
        "predicted_velocity_offsets":
            predicted_velocity_offsets,
        "refined_range_indices":
            refined_range_indices,
        "refined_velocity_indices":
            refined_velocity_indices,
    }


def new_metric_accumulator() -> dict[str, float]:
    result = {
        "sample_count": 0.0,
        "range_error_sum": 0.0,
        "velocity_error_sum": 0.0,
        "zero_false_peak_count": 0.0,
    }

    for metric_name in METRIC_RADII:
        result[
            f"{metric_name}_hit_count"
        ] = 0.0

    return result


def update_metric_accumulator(
    accumulator: dict[str, float],
    predicted_range_indices: torch.Tensor,
    predicted_velocity_indices: torch.Tensor,
    true_range_indices: torch.Tensor,
    true_velocity_indices: torch.Tensor,
) -> None:
    range_errors = (
        predicted_range_indices
        - true_range_indices
    ).abs()

    velocity_errors = (
        predicted_velocity_indices
        - true_velocity_indices
    ).abs()

    accumulator["sample_count"] += float(
        true_range_indices.numel()
    )

    accumulator["range_error_sum"] += float(
        range_errors.sum().item()
    )

    accumulator["velocity_error_sum"] += float(
        velocity_errors.sum().item()
    )

    for metric_name, radii in METRIC_RADII.items():
        hits = (
            (
                range_errors
                <= radii["range"]
            )
            & (
                velocity_errors
                <= radii["velocity"]
            )
        )

        accumulator[
            f"{metric_name}_hit_count"
        ] += float(
            hits.sum().item()
        )

    zero_center = 64

    true_away_from_zero = (
        true_velocity_indices
        - zero_center
    ).abs() > 6

    predicted_in_zero_band = (
        predicted_velocity_indices
        - zero_center
    ).abs() <= 3

    zero_false_peaks = (
        true_away_from_zero
        & predicted_in_zero_band
    )

    accumulator[
        "zero_false_peak_count"
    ] += float(
        zero_false_peaks.sum().item()
    )


def finalize_metrics(
    accumulator: dict[str, float],
) -> dict[str, float]:
    sample_count = accumulator[
        "sample_count"
    ]

    if sample_count <= 0:
        raise RuntimeError(
            "没有可评价的样本。"
        )

    result = {
        "mean_range_error": (
            accumulator["range_error_sum"]
            / sample_count
        ),
        "mean_velocity_error": (
            accumulator["velocity_error_sum"]
            / sample_count
        ),
        "zero_false_peak_count": (
            accumulator[
                "zero_false_peak_count"
            ]
        ),
    }

    for metric_name in METRIC_RADII:
        result[
            f"{metric_name}_hit_rate"
        ] = (
            accumulator[
                f"{metric_name}_hit_count"
            ]
            / sample_count
        )

    return result


def run_epoch(
    backbone: DualViewFCN,
    refiner: LocalOffsetRefiner,
    data_loader: DataLoader,
    device: torch.device,
    crop_size: int,
    max_range_offset: float,
    max_velocity_offset: float,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, Any]:
    training = optimizer is not None

    if training:
        refiner.train()
    else:
        refiner.eval()

    total_loss = 0.0
    total_samples = 0

    coarse_accumulator = (
        new_metric_accumulator()
    )

    refined_accumulator = (
        new_metric_accumulator()
    )

    context = (
        torch.enable_grad()
        if training
        else torch.no_grad()
    )

    with context:
        for batch in data_loader:
            raw_input = batch[
                "raw_input"
            ].to(
                device,
                non_blocking=True,
            )

            notch_input = batch[
                "notch_input"
            ].to(
                device,
                non_blocking=True,
            )

            true_range_indices = (
                batch["range_index"]
                .long()
                .reshape(-1)
                .to(device)
            )

            true_velocity_indices = (
                batch["velocity_index"]
                .long()
                .reshape(-1)
                .to(device)
            )

            if training:
                optimizer.zero_grad(
                    set_to_none=True
                )

            predictions = predict_coordinates(
                backbone=backbone,
                refiner=refiner,
                raw_input=raw_input,
                notch_input=notch_input,
                crop_size=crop_size,
                max_range_offset=
                    max_range_offset,
                max_velocity_offset=
                    max_velocity_offset,
            )

            coarse_range_indices = predictions[
                "coarse_range_indices"
            ]

            coarse_velocity_indices = predictions[
                "coarse_velocity_indices"
            ]

            true_range_offsets = (
                true_range_indices.float()
                - coarse_range_indices.float()
            )

            true_velocity_offsets = (
                true_velocity_indices.float()
                - coarse_velocity_indices.float()
            )

            normalized_targets = torch.stack(
                [
                    torch.clamp(
                        true_range_offsets
                        / max_range_offset,
                        min=-1.0,
                        max=1.0,
                    ),
                    torch.clamp(
                        true_velocity_offsets
                        / max_velocity_offset,
                        min=-1.0,
                        max=1.0,
                    ),
                ],
                dim=1,
            )

            loss = loss_function(
                predictions[
                    "normalized_offsets"
                ],
                normalized_targets,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "精修损失出现NaN或无穷值。"
                )

            if training:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    refiner.parameters(),
                    max_norm=5.0,
                )

                optimizer.step()

            batch_size = int(
                raw_input.shape[0]
            )

            total_loss += (
                float(loss.item())
                * batch_size
            )

            total_samples += batch_size

            update_metric_accumulator(
                accumulator=
                    coarse_accumulator,
                predicted_range_indices=
                    coarse_range_indices,
                predicted_velocity_indices=
                    coarse_velocity_indices,
                true_range_indices=
                    true_range_indices,
                true_velocity_indices=
                    true_velocity_indices,
            )

            update_metric_accumulator(
                accumulator=
                    refined_accumulator,
                predicted_range_indices=
                    predictions[
                        "refined_range_indices"
                    ],
                predicted_velocity_indices=
                    predictions[
                        "refined_velocity_indices"
                    ],
                true_range_indices=
                    true_range_indices,
                true_velocity_indices=
                    true_velocity_indices,
            )

    return {
        "loss":
            total_loss / total_samples,
        "coarse":
            finalize_metrics(
                coarse_accumulator
            ),
        "refined":
            finalize_metrics(
                refined_accumulator
            ),
    }


def build_model_score(
    validation_metrics: dict[str, Any],
) -> tuple[float, ...]:
    refined = validation_metrics[
        "refined"
    ]

    # 先确保不牺牲100%的邻域命中，
    # 然后最大化严格命中率。
    return (
        refined["application_hit_rate"],
        refined["relaxed_hit_rate"],
        refined["strict_hit_rate"],
        -refined["zero_false_peak_count"],
        -refined["mean_velocity_error"],
        -refined["mean_range_error"],
        -validation_metrics["loss"],
    )


def normalize_sample_ids(
    raw_sample_ids: Any,
    batch_size: int,
    start_index: int,
) -> list[str]:
    if raw_sample_ids is None:
        return [
            f"test_{start_index + index:03d}"
            for index in range(batch_size)
        ]

    if isinstance(raw_sample_ids, str):
        return [raw_sample_ids]

    return [
        str(value)
        for value in raw_sample_ids
    ]


def create_test_details(
    backbone: DualViewFCN,
    refiner: LocalOffsetRefiner,
    data_loader: DataLoader,
    device: torch.device,
    crop_size: int,
    max_range_offset: float,
    max_velocity_offset: float,
) -> pd.DataFrame:
    backbone.eval()
    refiner.eval()

    records = []
    global_index = 0

    with torch.no_grad():
        for batch in data_loader:
            raw_input = batch[
                "raw_input"
            ].to(device)

            notch_input = batch[
                "notch_input"
            ].to(device)

            true_range_indices = (
                batch["range_index"]
                .long()
                .reshape(-1)
                .to(device)
            )

            true_velocity_indices = (
                batch["velocity_index"]
                .long()
                .reshape(-1)
                .to(device)
            )

            predictions = predict_coordinates(
                backbone=backbone,
                refiner=refiner,
                raw_input=raw_input,
                notch_input=notch_input,
                crop_size=crop_size,
                max_range_offset=
                    max_range_offset,
                max_velocity_offset=
                    max_velocity_offset,
            )

            batch_size = int(
                raw_input.shape[0]
            )

            sample_ids = normalize_sample_ids(
                batch.get("sample_id"),
                batch_size,
                global_index,
            )

            for index in range(batch_size):
                true_range = int(
                    true_range_indices[
                        index
                    ].item()
                )

                true_velocity = int(
                    true_velocity_indices[
                        index
                    ].item()
                )

                coarse_range = int(
                    predictions[
                        "coarse_range_indices"
                    ][index].item()
                )

                coarse_velocity = int(
                    predictions[
                        "coarse_velocity_indices"
                    ][index].item()
                )

                refined_range = int(
                    predictions[
                        "refined_range_indices"
                    ][index].item()
                )

                refined_velocity = int(
                    predictions[
                        "refined_velocity_indices"
                    ][index].item()
                )

                coarse_range_error = abs(
                    coarse_range - true_range
                )

                coarse_velocity_error = abs(
                    coarse_velocity
                    - true_velocity
                )

                refined_range_error = abs(
                    refined_range - true_range
                )

                refined_velocity_error = abs(
                    refined_velocity
                    - true_velocity
                )

                record = {
                    "sample_id":
                        sample_ids[index],

                    "true_range_index":
                        true_range,

                    "true_velocity_index":
                        true_velocity,

                    "coarse_range_index":
                        coarse_range,

                    "coarse_velocity_index":
                        coarse_velocity,

                    "coarse_range_error":
                        coarse_range_error,

                    "coarse_velocity_error":
                        coarse_velocity_error,

                    "predicted_range_offset":
                        float(
                            predictions[
                                "predicted_range_offsets"
                            ][index].item()
                        ),

                    "predicted_velocity_offset":
                        float(
                            predictions[
                                "predicted_velocity_offsets"
                            ][index].item()
                        ),

                    "refined_range_index":
                        refined_range,

                    "refined_velocity_index":
                        refined_velocity,

                    "refined_range_error":
                        refined_range_error,

                    "refined_velocity_error":
                        refined_velocity_error,

                    "coarse_in_zero_band":
                        abs(
                            coarse_velocity - 64
                        ) <= 3,

                    "refined_in_zero_band":
                        abs(
                            refined_velocity - 64
                        ) <= 3,
                }

                for metric_name, radii in (
                    METRIC_RADII.items()
                ):
                    record[
                        f"coarse_{metric_name}_hit"
                    ] = (
                        coarse_range_error
                        <= radii["range"]
                        and coarse_velocity_error
                        <= radii["velocity"]
                    )

                    record[
                        f"refined_{metric_name}_hit"
                    ] = (
                        refined_range_error
                        <= radii["range"]
                        and refined_velocity_error
                        <= radii["velocity"]
                    )

                records.append(record)

            global_index += batch_size

    return pd.DataFrame(records)


def main() -> None:
    args = parse_arguments()

    if args.crop_size % 2 == 0:
        raise ValueError(
            "--crop-size必须为奇数。"
        )

    if args.max_range_offset <= 0:
        raise ValueError(
            "--max-range-offset必须大于0。"
        )

    if args.max_velocity_offset <= 0:
        raise ValueError(
            "--max-velocity-offset必须大于0。"
        )

    set_random_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    experiment_dir = (
        PROJECT_ROOT
        / "results"
        / "experiments"
        / args.name
    )

    checkpoint_dir = (
        experiment_dir / "checkpoints"
    )

    table_dir = (
        experiment_dir / "tables"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    table_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_path = (
        checkpoint_dir
        / "best_refiner.pt"
    )

    last_path = (
        checkpoint_dir
        / "last_refiner.pt"
    )

    history_path = (
        table_dir / "history.csv"
    )

    test_details_path = (
        table_dir / "test_details.csv"
    )

    summary_path = (
        table_dir / "summary.json"
    )

    print("=" * 72)
    print("双分支抗杂波FCN + 局部偏移精修器")
    print("=" * 72)

    print(f"设备：{device}")

    if torch.cuda.is_available():
        print(
            "GPU："
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"双分支模型：{args.dual_model}"
    )

    print(
        f"原始FCN模型：{args.original_model}"
    )

    print(
        f"局部裁剪尺寸：{args.crop_size}"
    )

    print(
        "最大修正范围："
        f"距离±{args.max_range_offset}门，"
        f"速度±{args.max_velocity_offset}单元"
    )

    def build_dataset(
        split: str,
    ) -> SoftNotchDataset:
        base_dataset = RadarDataset(
            split=split,
            range_sigma=3.0,
            velocity_sigma=1.0,
        )

        return SoftNotchDataset(
            base=base_dataset,
            sigma=args.notch_sigma,
            floor=args.notch_floor,
        )

    print("\n正在加载数据集……")

    train_dataset = build_dataset(
        "train"
    )

    validation_dataset = build_dataset(
        "val"
    )

    test_dataset = build_dataset(
        "test"
    )

    print(
        "训练/验证/测试："
        f"{len(train_dataset)}/"
        f"{len(validation_dataset)}/"
        f"{len(test_dataset)}"
    )

    pin_memory = (
        device.type == "cuda"
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    backbone = load_frozen_dual_model(
        dual_model_path=args.dual_model,
        original_model_path=
            args.original_model,
        device=device,
    )

    refiner = LocalOffsetRefiner().to(
        device
    )

    print(
        "精修器可训练参数量："
        f"{count_trainable_parameters(refiner):,}"
    )

    loss_function = nn.SmoothL1Loss(
        beta=0.25
    )

    optimizer = torch.optim.AdamW(
        refiner.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=7,
            min_lr=1.0e-6,
        )
    )

    best_score = None
    best_epoch = 0
    stale_epochs = 0
    history = []

    training_start = time.time()

    print("\n开始训练精修器……")

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        train_metrics = run_epoch(
            backbone=backbone,
            refiner=refiner,
            data_loader=train_loader,
            device=device,
            crop_size=args.crop_size,
            max_range_offset=
                args.max_range_offset,
            max_velocity_offset=
                args.max_velocity_offset,
            loss_function=loss_function,
            optimizer=optimizer,
        )

        validation_metrics = run_epoch(
            backbone=backbone,
            refiner=refiner,
            data_loader=
                validation_loader,
            device=device,
            crop_size=args.crop_size,
            max_range_offset=
                args.max_range_offset,
            max_velocity_offset=
                args.max_velocity_offset,
            loss_function=loss_function,
            optimizer=None,
        )

        scheduler.step(
            validation_metrics["loss"]
        )

        learning_rate = float(
            optimizer.param_groups[0]["lr"]
        )

        score = build_model_score(
            validation_metrics
        )

        is_best = (
            best_score is None
            or score > best_score
        )

        configuration = {
            key: (
                str(value)
                if isinstance(value, Path)
                else value
            )
            for key, value in vars(args).items()
        }

        checkpoint = {
            "epoch":
                epoch,
            "refiner_state_dict":
                refiner.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
            "scheduler_state_dict":
                scheduler.state_dict(),
            "train_metrics":
                train_metrics,
            "validation_metrics":
                validation_metrics,
            "configuration":
                configuration,
        }

        torch.save(
            checkpoint,
            last_path,
        )

        if is_best:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0

            torch.save(
                checkpoint,
                best_path,
            )

        else:
            stale_epochs += 1

        train_refined = train_metrics[
            "refined"
        ]

        validation_coarse = (
            validation_metrics[
                "coarse"
            ]
        )

        validation_refined = (
            validation_metrics[
                "refined"
            ]
        )

        history.append(
            {
                "epoch":
                    epoch,
                "learning_rate":
                    learning_rate,
                "train_loss":
                    train_metrics["loss"],
                "train_refined_strict_hit_rate":
                    train_refined[
                        "strict_hit_rate"
                    ],
                "train_refined_relaxed_hit_rate":
                    train_refined[
                        "relaxed_hit_rate"
                    ],
                "val_loss":
                    validation_metrics["loss"],
                "val_coarse_strict_hit_rate":
                    validation_coarse[
                        "strict_hit_rate"
                    ],
                "val_coarse_relaxed_hit_rate":
                    validation_coarse[
                        "relaxed_hit_rate"
                    ],
                "val_coarse_application_hit_rate":
                    validation_coarse[
                        "application_hit_rate"
                    ],
                "val_refined_strict_hit_rate":
                    validation_refined[
                        "strict_hit_rate"
                    ],
                "val_refined_relaxed_hit_rate":
                    validation_refined[
                        "relaxed_hit_rate"
                    ],
                "val_refined_application_hit_rate":
                    validation_refined[
                        "application_hit_rate"
                    ],
                "val_refined_mean_range_error":
                    validation_refined[
                        "mean_range_error"
                    ],
                "val_refined_mean_velocity_error":
                    validation_refined[
                        "mean_velocity_error"
                    ],
                "val_refined_zero_false_peak_count":
                    validation_refined[
                        "zero_false_peak_count"
                    ],
                "is_best":
                    is_best,
            }
        )

        pd.DataFrame(history).to_csv(
            history_path,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"第{epoch:03d}轮｜"
            f"验证损失="
            f"{validation_metrics['loss']:.5f}｜"
            f"粗定位严格="
            f"{validation_coarse['strict_hit_rate']:.2%}｜"
            f"精修严格="
            f"{validation_refined['strict_hit_rate']:.2%}｜"
            f"精修宽松="
            f"{validation_refined['relaxed_hit_rate']:.2%}｜"
            f"精修应用="
            f"{validation_refined['application_hit_rate']:.2%}｜"
            f"距离误差="
            f"{validation_refined['mean_range_error']:.3f}门｜"
            f"速度误差="
            f"{validation_refined['mean_velocity_error']:.3f}单元｜"
            f"零多普勒误峰="
            f"{int(validation_refined['zero_false_peak_count'])}｜"
            f"学习率={learning_rate:.2e}"
            + (
                "｜最佳模型"
                if is_best
                else ""
            )
        )

        if (
            stale_epochs
            >= args.early_stopping
        ):
            print(
                "\n连续"
                f"{args.early_stopping}"
                "轮没有产生更优模型，提前停止。"
            )
            break

    training_seconds = (
        time.time() - training_start
    )

    print("\n加载最佳精修器……")

    best_checkpoint = load_torch(
        best_path,
        device,
    )

    refiner.load_state_dict(
        best_checkpoint[
            "refiner_state_dict"
        ]
    )

    print("正在评价测试集……")

    test_metrics = run_epoch(
        backbone=backbone,
        refiner=refiner,
        data_loader=test_loader,
        device=device,
        crop_size=args.crop_size,
        max_range_offset=
            args.max_range_offset,
        max_velocity_offset=
            args.max_velocity_offset,
        loss_function=loss_function,
        optimizer=None,
    )

    test_details = create_test_details(
        backbone=backbone,
        refiner=refiner,
        data_loader=test_loader,
        device=device,
        crop_size=args.crop_size,
        max_range_offset=
            args.max_range_offset,
        max_velocity_offset=
            args.max_velocity_offset,
    )

    test_details.to_csv(
        test_details_path,
        index=False,
        encoding="utf-8-sig",
    )

    target_sample = test_details[
        test_details["sample_id"].astype(str)
        == "20260202_144849_beam4"
    ]

    summary = {
        "experiment_name":
            args.name,
        "best_epoch":
            best_epoch,
        "training_seconds":
            training_seconds,
        "best_validation_metrics":
            best_checkpoint[
                "validation_metrics"
            ],
        "test_metrics":
            test_metrics,
        "target_sample_result":
            target_sample.to_dict(
                orient="records"
            ),
        "configuration":
            configuration,
    }

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    test_coarse = test_metrics[
        "coarse"
    ]

    test_refined = test_metrics[
        "refined"
    ]

    print("\n" + "=" * 72)
    print("双分支FCN局部精修训练完成")
    print("=" * 72)

    print(f"最佳轮次：{best_epoch}")

    print(
        "\n========== 测试集粗定位 =========="
    )

    print(
        "平均距离误差："
        f"{test_coarse['mean_range_error']:.3f}门"
    )

    print(
        "平均速度误差："
        f"{test_coarse['mean_velocity_error']:.3f}单元"
    )

    print(
        "严格命中率："
        f"{test_coarse['strict_hit_rate']:.2%}"
    )

    print(
        "宽松命中率："
        f"{test_coarse['relaxed_hit_rate']:.2%}"
    )

    print(
        "应用命中率："
        f"{test_coarse['application_hit_rate']:.2%}"
    )

    print(
        "零多普勒误峰："
        f"{int(test_coarse['zero_false_peak_count'])}"
    )

    print(
        "\n========== 测试集精修后 =========="
    )

    print(
        "平均距离误差："
        f"{test_refined['mean_range_error']:.3f}门"
    )

    print(
        "平均速度误差："
        f"{test_refined['mean_velocity_error']:.3f}单元"
    )

    print(
        "严格命中率："
        f"{test_refined['strict_hit_rate']:.2%}"
    )

    print(
        "宽松命中率："
        f"{test_refined['relaxed_hit_rate']:.2%}"
    )

    print(
        "应用命中率："
        f"{test_refined['application_hit_rate']:.2%}"
    )

    print(
        "零多普勒误峰："
        f"{int(test_refined['zero_false_peak_count'])}"
    )

    if not target_sample.empty:
        print(
            "\n========== beam4关键样本 =========="
        )

        print(
            target_sample.to_string(
                index=False
            )
        )

    print(
        f"\n最佳精修器：{best_path}"
    )

    print(
        f"测试明细：{test_details_path}"
    )

    print(
        f"结果汇总：{summary_path}"
    )


if __name__ == "__main__":
    main()
