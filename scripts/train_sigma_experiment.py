from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from datasets.radar_dataset import RadarDataset
from models.simple_fcn import SimpleRadarFCN
from utils.plot_config import configure_chinese_font


# ============================================================
# 固定参数
# ============================================================

RANDOM_SEED = 42

LEARNING_RATE = 1.0e-3
MIN_LEARNING_RATE = 1.0e-6
WEIGHT_DECAY = 1.0e-5

POSITIVE_WEIGHT = 30.0
DICE_WEIGHT = 0.2

SCHEDULER_PATIENCE = 10
EARLY_STOPPING_PATIENCE = 40
GRADIENT_CLIP_NORM = 5.0

# 三套评价标准
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


# ============================================================
# 基础工具
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple FCN标签Sigma对照实验"
    )

    parser.add_argument(
        "--name",
        required=True,
        help="实验名称，例如 sigma_r3_v15",
    )

    parser.add_argument(
        "--channel",
        choices=["H", "V", "HV"],
        default="HV",
        help=(
            "输入通道模式：H、V或HV；"
            "默认HV，兼容原有双通道实验"
        ),
    )

    parser.add_argument(
        "--range-sigma",
        type=float,
        required=True,
        help="距离方向高斯标签sigma",
    )

    parser.add_argument(
        "--velocity-sigma",
        type=float,
        required=True,
        help="速度方向高斯标签sigma",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--prediction-count",
        type=int,
        default=8,
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


def count_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def scalar_to_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())

    return int(value)


def get_device() -> torch.device:
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


# ============================================================
# 损失函数
# ============================================================

class WeightedHeatmapLoss(nn.Module):
    def __init__(
        self,
        positive_weight: float = 30.0,
        dice_weight: float = 0.2,
    ) -> None:
        super().__init__()

        self.positive_weight = float(positive_weight)
        self.dice_weight = float(dice_weight)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        prediction = torch.sigmoid(logits)

        pixel_weight = (
            1.0
            + self.positive_weight * target
        )

        weighted_mse = (
            pixel_weight
            * (prediction - target).pow(2)
        ).mean()

        prediction_flat = prediction.flatten(start_dim=1)
        target_flat = target.flatten(start_dim=1)

        intersection = (
            prediction_flat * target_flat
        ).sum(dim=1)

        dice_score = (
            2.0 * intersection + 1.0
        ) / (
            prediction_flat.sum(dim=1)
            + target_flat.sum(dim=1)
            + 1.0
        )

        dice_loss = (
            1.0 - dice_score
        ).mean()

        return (
            weighted_mse
            + self.dice_weight * dice_loss
        )


# ============================================================
# 坐标与指标
# ============================================================

def extract_peak_coordinates(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    prediction = torch.sigmoid(logits)

    batch_size = prediction.shape[0]
    range_size = prediction.shape[-1]

    flat_indices = (
        prediction
        .reshape(batch_size, -1)
        .argmax(dim=1)
    )

    velocity_indices = (
        flat_indices // range_size
    )

    range_indices = (
        flat_indices % range_size
    )

    return range_indices, velocity_indices


def calculate_batch_statistics(
    logits: torch.Tensor,
    true_range_indices: torch.Tensor,
    true_velocity_indices: torch.Tensor,
) -> dict[str, float]:
    (
        predicted_range_indices,
        predicted_velocity_indices,
    ) = extract_peak_coordinates(logits)

    true_range_indices = (
        true_range_indices
        .long()
        .reshape(-1)
    )

    true_velocity_indices = (
        true_velocity_indices
        .long()
        .reshape(-1)
    )

    range_errors = (
        predicted_range_indices
        - true_range_indices
    ).abs()

    velocity_errors = (
        predicted_velocity_indices
        - true_velocity_indices
    ).abs()

    result = {
        "sample_count": float(logits.shape[0]),
        "range_error_sum": float(
            range_errors.sum().item()
        ),
        "velocity_error_sum": float(
            velocity_errors.sum().item()
        ),
    }

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

        result[f"{metric_name}_hit_count"] = float(
            hits.sum().item()
        )

    return result


def initialize_accumulator() -> dict[str, float]:
    return {
        "loss_sum": 0.0,
        "sample_count": 0.0,
        "range_error_sum": 0.0,
        "velocity_error_sum": 0.0,
        "strict_hit_count": 0.0,
        "relaxed_hit_count": 0.0,
        "application_hit_count": 0.0,
    }


def finalize_accumulator(
    accumulator: dict[str, float],
) -> dict[str, float]:
    sample_count = accumulator["sample_count"]

    if sample_count <= 0:
        raise RuntimeError("数据加载器中没有样本。")

    return {
        "loss": (
            accumulator["loss_sum"]
            / sample_count
        ),
        "mean_range_error": (
            accumulator["range_error_sum"]
            / sample_count
        ),
        "mean_velocity_error": (
            accumulator["velocity_error_sum"]
            / sample_count
        ),
        "strict_hit_rate": (
            accumulator["strict_hit_count"]
            / sample_count
        ),
        "relaxed_hit_rate": (
            accumulator["relaxed_hit_count"]
            / sample_count
        ),
        "application_hit_rate": (
            accumulator["application_hit_count"]
            / sample_count
        ),
    }


# ============================================================
# 单轮训练或评价
# ============================================================

def run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    accumulator = initialize_accumulator()

    context = (
        torch.enable_grad()
        if training
        else torch.no_grad()
    )

    with context:
        for batch in data_loader:
            input_tensor = batch["input"].to(
                device,
                non_blocking=True,
            )

            target_tensor = batch["target"].to(
                device,
                non_blocking=True,
            )

            true_range_indices = batch[
                "range_index"
            ].to(
                device,
                non_blocking=True,
            )

            true_velocity_indices = batch[
                "velocity_index"
            ].to(
                device,
                non_blocking=True,
            )

            if training:
                optimizer.zero_grad(
                    set_to_none=True
                )

            logits = model(input_tensor)

            loss = loss_function(
                logits,
                target_tensor,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "损失出现NaN或无穷值。"
                )

            if training:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=GRADIENT_CLIP_NORM,
                )

                optimizer.step()

            batch_size = int(
                input_tensor.shape[0]
            )

            statistics = calculate_batch_statistics(
                logits.detach(),
                true_range_indices,
                true_velocity_indices,
            )

            accumulator["loss_sum"] += (
                float(loss.item()) * batch_size
            )

            accumulator["sample_count"] += (
                statistics["sample_count"]
            )

            accumulator["range_error_sum"] += (
                statistics["range_error_sum"]
            )

            accumulator["velocity_error_sum"] += (
                statistics["velocity_error_sum"]
            )

            for metric_name in METRIC_RADII:
                accumulator[
                    f"{metric_name}_hit_count"
                ] += statistics[
                    f"{metric_name}_hit_count"
                ]

    return finalize_accumulator(accumulator)


# ============================================================
# 最佳模型选择
# ============================================================

def build_model_score(
    metrics: dict[str, float],
) -> tuple[float, ...]:
    """
    优先选择应用邻域命中率较高的模型。

    排序顺序：
    1. 应用邻域命中率
    2. 宽松命中率
    3. 严格命中率
    4. 平均速度误差
    5. 平均距离误差
    6. 验证损失
    """

    return (
        metrics["application_hit_rate"],
        metrics["relaxed_hit_rate"],
        metrics["strict_hit_rate"],
        -metrics["mean_velocity_error"],
        -metrics["mean_range_error"],
        -metrics["loss"],
    )


# ============================================================
# 测试集逐样本评价
# ============================================================

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

    if isinstance(raw_sample_ids, (list, tuple)):
        return [
            str(value)
            for value in raw_sample_ids
        ]

    return [
        str(value)
        for value in raw_sample_ids
    ]


def create_test_details(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()

    records: list[dict[str, Any]] = []
    global_index = 0

    with torch.no_grad():
        for batch in data_loader:
            input_tensor = batch["input"].to(
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

            logits = model(input_tensor)
            probabilities = torch.sigmoid(logits)

            (
                predicted_range_indices,
                predicted_velocity_indices,
            ) = extract_peak_coordinates(logits)

            batch_size = int(
                input_tensor.shape[0]
            )

            sample_ids = normalize_sample_ids(
                batch.get("sample_id"),
                batch_size,
                global_index,
            )

            for index in range(batch_size):
                true_range = int(
                    true_range_indices[index].item()
                )

                true_velocity = int(
                    true_velocity_indices[index].item()
                )

                predicted_range = int(
                    predicted_range_indices[index].item()
                )

                predicted_velocity = int(
                    predicted_velocity_indices[index].item()
                )

                range_error = abs(
                    predicted_range - true_range
                )

                velocity_error = abs(
                    predicted_velocity - true_velocity
                )

                record = {
                    "sample_id": sample_ids[index],
                    "true_range_index": true_range,
                    "true_velocity_index": true_velocity,
                    "predicted_range_index":
                        predicted_range,
                    "predicted_velocity_index":
                        predicted_velocity,
                    "range_error_gates":
                        range_error,
                    "velocity_error_bins":
                        velocity_error,
                    "peak_probability": float(
                        probabilities[
                            index,
                            0,
                            predicted_velocity,
                            predicted_range,
                        ].item()
                    ),
                }

                for metric_name, radii in (
                    METRIC_RADII.items()
                ):
                    record[f"{metric_name}_hit"] = (
                        range_error <= radii["range"]
                        and velocity_error
                        <= radii["velocity"]
                    )

                records.append(record)

            global_index += batch_size

    return pd.DataFrame(records)


# ============================================================
# 曲线绘制
# ============================================================

def save_curves(
    history_df: pd.DataFrame,
    figure_dir: Path,
) -> None:
    configure_chinese_font()

    # 损失曲线
    figure = plt.figure(figsize=(9, 5.5))

    plt.plot(
        history_df["epoch"],
        history_df["train_loss"],
        label="训练损失",
    )

    plt.plot(
        history_df["epoch"],
        history_df["val_loss"],
        label="验证损失",
    )

    plt.xlabel("训练轮次")
    plt.ylabel("损失值")
    plt.title("训练与验证损失")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    figure.savefig(
        figure_dir / "loss_curve.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(figure)

    # 三种命中率
    figure = plt.figure(figsize=(9, 5.5))

    plt.plot(
        history_df["epoch"],
        history_df["val_strict_hit_rate"] * 100,
        label="严格命中率：±1门、±1单元",
    )

    plt.plot(
        history_df["epoch"],
        history_df["val_relaxed_hit_rate"] * 100,
        label="宽松命中率：±4门、±1单元",
    )

    plt.plot(
        history_df["epoch"],
        history_df[
            "val_application_hit_rate"
        ] * 100,
        label="应用邻域：±4门、±2单元",
    )

    plt.xlabel("训练轮次")
    plt.ylabel("命中率（%）")
    plt.title("验证集定位命中率")
    plt.ylim(0, 105)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    figure.savefig(
        figure_dir / "hit_rate_curve.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(figure)

    # 定位误差
    figure = plt.figure(figsize=(9, 5.5))

    plt.plot(
        history_df["epoch"],
        history_df["val_mean_range_error"],
        label="平均距离误差（门）",
    )

    plt.plot(
        history_df["epoch"],
        history_df["val_mean_velocity_error"],
        label="平均速度误差（单元）",
    )

    plt.xlabel("训练轮次")
    plt.ylabel("平均下标误差")
    plt.title("验证集定位误差")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    figure.savefig(
        figure_dir / "error_curve.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# 测试预测图
# ============================================================

def save_prediction_figures(
    model: nn.Module,
    dataset: RadarDataset,
    device: torch.device,
    output_dir: Path,
    sample_count: int,
) -> None:
    configure_chinese_font()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.eval()

    with torch.no_grad():
        for sample_index in range(
            min(sample_count, len(dataset))
        ):
            sample = dataset[sample_index]

            input_tensor = (
                sample["input"]
                .unsqueeze(0)
                .to(device)
            )

            logits = model(input_tensor)

            prediction = (
                torch.sigmoid(logits)[0, 0]
                .cpu()
                .numpy()
            )

            target = (
                sample["target"][0]
                .cpu()
                .numpy()
            )

            input_image = (
                sample["input"]
                .mean(dim=0)
                .cpu()
                .numpy()
            )

            true_range = scalar_to_int(
                sample["range_index"]
            )

            true_velocity = scalar_to_int(
                sample["velocity_index"]
            )

            flat_index = int(
                np.argmax(prediction)
            )

            (
                predicted_velocity,
                predicted_range,
            ) = np.unravel_index(
                flat_index,
                prediction.shape,
            )

            range_error = abs(
                predicted_range - true_range
            )

            velocity_error = abs(
                predicted_velocity - true_velocity
            )

            sample_id = str(
                sample.get(
                    "sample_id",
                    f"test_{sample_index:03d}",
                )
            )

            figure, axes = plt.subplots(
                1,
                3,
                figsize=(16, 5),
            )

            axes[0].imshow(
                input_image,
                aspect="auto",
                cmap="jet",
            )

            axes[0].plot(
                true_range,
                true_velocity,
                marker="+",
                color="red",
                markersize=15,
                markeredgewidth=2,
            )

            axes[0].set_title(
                "H/V平均归一化RD图"
            )

            axes[1].imshow(
                target,
                aspect="auto",
                cmap="jet",
                vmin=0,
                vmax=1,
            )

            axes[1].plot(
                true_range,
                true_velocity,
                marker="+",
                color="white",
                markersize=15,
                markeredgewidth=2,
            )

            axes[1].set_title(
                "真实热力图标签"
            )

            axes[2].imshow(
                prediction,
                aspect="auto",
                cmap="jet",
                vmin=0,
                vmax=1,
            )

            axes[2].plot(
                true_range,
                true_velocity,
                marker="+",
                color="white",
                markersize=15,
                markeredgewidth=2,
                label="真实位置",
            )

            axes[2].plot(
                predicted_range,
                predicted_velocity,
                marker="x",
                color="red",
                markersize=12,
                markeredgewidth=2,
                label="预测峰值",
            )

            axes[2].set_title(
                "FCN预测热力图"
            )

            axes[2].legend()

            for axis in axes:
                axis.set_xlabel("距离门下标")
                axis.set_ylabel("速度单元下标")

            figure.suptitle(
                (
                    f"{sample_id}　"
                    f"距离误差={range_error}门　"
                    f"速度误差={velocity_error}单元"
                )
            )

            figure.tight_layout()

            figure.savefig(
                output_dir
                / f"prediction_{sample_index + 1:02d}.png",
                dpi=220,
                bbox_inches="tight",
            )

            plt.close(figure)


# ============================================================
# 主程序
# ============================================================

def main() -> None:
    args = parse_arguments()

    set_random_seed(RANDOM_SEED)

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

    figure_dir = (
        experiment_dir / "figures"
    )

    prediction_dir = (
        figure_dir / "test_predictions"
    )

    for directory in (
        checkpoint_dir,
        table_dir,
        figure_dir,
        prediction_dir,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    best_model_path = (
        checkpoint_dir / "best.pt"
    )

    last_model_path = (
        checkpoint_dir / "last.pt"
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

    configure_chinese_font()

    device = get_device()

    print("=" * 62)
    print("Simple FCN标签Sigma对照实验")
    print("=" * 62)
    print(f"实验名称：{args.name}")
    print(f"输入模式：{args.channel}")
    print(f"运行设备：{device}")

    if torch.cuda.is_available():
        print(
            "GPU："
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        "标签参数："
        f"range_sigma={args.range_sigma}，"
        f"velocity_sigma={args.velocity_sigma}"
    )

    print(
        "严格指标：距离±1门、速度±1单元"
    )

    print(
        "宽松指标：距离±4门、速度±1单元"
    )

    print(
        "应用指标：距离±4门、速度±2单元"
    )

    print("\n正在加载数据集……")

    train_dataset = RadarDataset(
        split="train",
        channel_mode=args.channel,
        range_sigma=args.range_sigma,
        velocity_sigma=args.velocity_sigma,
    )

    validation_dataset = RadarDataset(
        split="val",
        channel_mode=args.channel,
        range_sigma=args.range_sigma,
        velocity_sigma=args.velocity_sigma,
    )

    test_dataset = RadarDataset(
        split="test",
        channel_mode=args.channel,
        range_sigma=args.range_sigma,
        velocity_sigma=args.velocity_sigma,
    )

    print(f"训练集：{len(train_dataset)}")
    print(f"验证集：{len(validation_dataset)}")
    print(f"测试集：{len(test_dataset)}")

    pin_memory = device.type == "cuda"

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

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

    input_channels = (
        2
        if args.channel == "HV"
        else 1
    )

    model = SimpleRadarFCN(
        in_channels=input_channels,
    ).to(device)

    print(
        f"可训练参数量：{count_parameters(model):,}"
    )

    loss_function = WeightedHeatmapLoss(
        positive_weight=POSITIVE_WEIGHT,
        dice_weight=DICE_WEIGHT,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=SCHEDULER_PATIENCE,
            min_lr=MIN_LEARNING_RATE,
        )
    )

    history: list[dict[str, Any]] = []

    best_score: tuple[float, ...] | None = None
    best_epoch = 0
    epochs_without_improvement = 0

    start_time = time.time()

    print("\n开始训练……")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            data_loader=train_loader,
            loss_function=loss_function,
            device=device,
            optimizer=optimizer,
        )

        validation_metrics = run_epoch(
            model=model,
            data_loader=validation_loader,
            loss_function=loss_function,
            device=device,
            optimizer=None,
        )

        scheduler.step(
            validation_metrics["loss"]
        )

        learning_rate = float(
            optimizer.param_groups[0]["lr"]
        )

        current_score = build_model_score(
            validation_metrics
        )

        is_best = (
            best_score is None
            or current_score > best_score
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict":
                model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
            "scheduler_state_dict":
                scheduler.state_dict(),
            "train_metrics":
                train_metrics,
            "validation_metrics":
                validation_metrics,
            "configuration": {
                "experiment_name":
                    args.name,
                "channel_mode":
                    args.channel,
                "input_channels":
                    input_channels,
                "range_sigma":
                    args.range_sigma,
                "velocity_sigma":
                    args.velocity_sigma,
                "metric_radii":
                    METRIC_RADII,
                "random_seed":
                    RANDOM_SEED,
            },
        }

        torch.save(
            checkpoint,
            last_model_path,
        )

        if is_best:
            best_score = current_score
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                checkpoint,
                best_model_path,
            )

        else:
            epochs_without_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_loss":
                    train_metrics["loss"],
                "train_mean_range_error":
                    train_metrics[
                        "mean_range_error"
                    ],
                "train_mean_velocity_error":
                    train_metrics[
                        "mean_velocity_error"
                    ],
                "train_strict_hit_rate":
                    train_metrics[
                        "strict_hit_rate"
                    ],
                "train_relaxed_hit_rate":
                    train_metrics[
                        "relaxed_hit_rate"
                    ],
                "train_application_hit_rate":
                    train_metrics[
                        "application_hit_rate"
                    ],
                "val_loss":
                    validation_metrics["loss"],
                "val_mean_range_error":
                    validation_metrics[
                        "mean_range_error"
                    ],
                "val_mean_velocity_error":
                    validation_metrics[
                        "mean_velocity_error"
                    ],
                "val_strict_hit_rate":
                    validation_metrics[
                        "strict_hit_rate"
                    ],
                "val_relaxed_hit_rate":
                    validation_metrics[
                        "relaxed_hit_rate"
                    ],
                "val_application_hit_rate":
                    validation_metrics[
                        "application_hit_rate"
                    ],
                "is_best": is_best,
            }
        )

        pd.DataFrame(history).to_csv(
            history_path,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"第{epoch:03d}轮｜"
            f"训练损失={train_metrics['loss']:.6f}｜"
            f"验证损失={validation_metrics['loss']:.6f}｜"
            f"距离误差="
            f"{validation_metrics['mean_range_error']:.3f}门｜"
            f"速度误差="
            f"{validation_metrics['mean_velocity_error']:.3f}单元｜"
            f"严格="
            f"{validation_metrics['strict_hit_rate']:.2%}｜"
            f"宽松="
            f"{validation_metrics['relaxed_hit_rate']:.2%}｜"
            f"应用="
            f"{validation_metrics['application_hit_rate']:.2%}｜"
            f"学习率={learning_rate:.2e}"
            + ("｜最佳模型" if is_best else "")
        )

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            print(
                "\n连续"
                f"{EARLY_STOPPING_PATIENCE}"
                "轮未产生更优模型，提前停止。"
            )
            break

    training_seconds = time.time() - start_time

    history_df = pd.DataFrame(history)

    save_curves(
        history_df,
        figure_dir,
    )

    print("\n加载最佳模型并评价测试集……")

    best_checkpoint = torch.load(
        best_model_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        best_checkpoint["model_state_dict"]
    )

    test_metrics = run_epoch(
        model=model,
        data_loader=test_loader,
        loss_function=loss_function,
        device=device,
        optimizer=None,
    )

    test_details = create_test_details(
        model,
        test_loader,
        device,
    )

    test_details.to_csv(
        test_details_path,
        index=False,
        encoding="utf-8-sig",
    )

    save_prediction_figures(
        model=model,
        dataset=test_dataset,
        device=device,
        output_dir=prediction_dir,
        sample_count=args.prediction_count,
    )

    summary = {
        "experiment_name":
            args.name,
        "channel_mode":
            args.channel,
        "input_channels":
            input_channels,
        "range_sigma":
            args.range_sigma,
        "velocity_sigma":
            args.velocity_sigma,
        "train_sample_count":
            len(train_dataset),
        "validation_sample_count":
            len(validation_dataset),
        "test_sample_count":
            len(test_dataset),
        "trainable_parameters":
            count_parameters(model),
        "best_epoch":
            best_epoch,
        "training_seconds":
            training_seconds,
        "metric_radii":
            METRIC_RADII,
        "best_validation_metrics":
            best_checkpoint[
                "validation_metrics"
            ],
        "test_metrics":
            test_metrics,
    }

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 62)
    print("实验完成")
    print("=" * 62)

    print(f"实验名称：{args.name}")
    print(f"最佳轮次：{best_epoch}")

    print(
        "\n========== 最佳验证集 =========="
    )

    validation_metrics = best_checkpoint[
        "validation_metrics"
    ]

    print(
        f"验证损失："
        f"{validation_metrics['loss']:.6f}"
    )

    print(
        f"平均距离误差："
        f"{validation_metrics['mean_range_error']:.3f}门"
    )

    print(
        f"平均速度误差："
        f"{validation_metrics['mean_velocity_error']:.3f}单元"
    )

    print(
        f"严格命中率："
        f"{validation_metrics['strict_hit_rate']:.2%}"
    )

    print(
        f"宽松命中率："
        f"{validation_metrics['relaxed_hit_rate']:.2%}"
    )

    print(
        f"应用邻域命中率："
        f"{validation_metrics['application_hit_rate']:.2%}"
    )

    print(
        "\n========== 独立测试集 =========="
    )

    print(
        f"测试损失：{test_metrics['loss']:.6f}"
    )

    print(
        f"平均距离误差："
        f"{test_metrics['mean_range_error']:.3f}门"
    )

    print(
        f"平均速度误差："
        f"{test_metrics['mean_velocity_error']:.3f}单元"
    )

    print(
        f"严格命中率："
        f"{test_metrics['strict_hit_rate']:.2%}"
    )

    print(
        f"宽松命中率："
        f"{test_metrics['relaxed_hit_rate']:.2%}"
    )

    print(
        f"应用邻域命中率："
        f"{test_metrics['application_hit_rate']:.2%}"
    )

    print(
        "\n输出目录："
        f"{experiment_dir}"
    )


if __name__ == "__main__":
    main()