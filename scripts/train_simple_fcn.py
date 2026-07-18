from pathlib import Path
import json
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader


# ============================================================
# 项目路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from datasets.radar_dataset import RadarDataset
from models.simple_fcn import (
    SimpleRadarFCN,
    count_trainable_parameters,
)
from utils.plot_config import configure_chinese_font


# ============================================================
# 随机种子
# ============================================================

RANDOM_SEED = 42


# ============================================================
# 数据参数
# ============================================================

RANGE_SIGMA = 2.0
VELOCITY_SIGMA = 1.0

BATCH_SIZE = 8
NUM_WORKERS = 0


# ============================================================
# 训练参数
# ============================================================

EPOCH_COUNT = 200
LEARNING_RATE = 1.0e-3
MIN_LEARNING_RATE = 1.0e-6
WEIGHT_DECAY = 1.0e-5

POSITIVE_WEIGHT = 30.0
DICE_WEIGHT = 0.2

EARLY_STOPPING_PATIENCE = 40
SCHEDULER_PATIENCE = 10
GRADIENT_CLIP_NORM = 5.0


# ============================================================
# 定位评价标准
# ============================================================

STRICT_RANGE_RADIUS = 1
STRICT_VELOCITY_RADIUS = 1

RELAXED_RANGE_RADIUS = 4
RELAXED_VELOCITY_RADIUS = 1


# ============================================================
# 输出路径
# ============================================================

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "results"
    / "checkpoints"
)

TABLE_DIR = (
    PROJECT_ROOT
    / "results"
    / "tables"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "simple_fcn_formal"
)

BEST_MODEL_PATH = (
    CHECKPOINT_DIR
    / "simple_fcn_best.pt"
)

LAST_MODEL_PATH = (
    CHECKPOINT_DIR
    / "simple_fcn_last.pt"
)

HISTORY_PATH = (
    TABLE_DIR
    / "simple_fcn_training_history.csv"
)

TEST_RESULT_PATH = (
    TABLE_DIR
    / "simple_fcn_test_results.csv"
)

SUMMARY_PATH = (
    TABLE_DIR
    / "simple_fcn_summary.json"
)


# ============================================================
# 基础函数
# ============================================================

def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_output_directories() -> None:
    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


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
    """
    加权热力图损失。

    由两部分组成：

    1. 加权均方误差：
       对目标高斯区域赋予更高权重，降低背景像素过多造成的影响。

    2. 软Dice损失：
       促进预测热力图和真实热力图区域重合。
    """

    def __init__(
        self,
        positive_weight: float = 30.0,
        dice_weight: float = 0.2,
    ) -> None:
        super().__init__()

        self.positive_weight = float(
            positive_weight
        )

        self.dice_weight = float(
            dice_weight
        )

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        prediction = torch.sigmoid(logits)

        pixel_weight = (
            1.0
            + self.positive_weight
            * target
        )

        weighted_mse = (
            pixel_weight
            * (
                prediction - target
            ).pow(2)
        ).mean()

        prediction_flat = prediction.flatten(
            start_dim=1
        )

        target_flat = target.flatten(
            start_dim=1
        )

        intersection = (
            prediction_flat
            * target_flat
        ).sum(dim=1)

        dice_score = (
            2.0 * intersection
            + 1.0
        ) / (
            prediction_flat.sum(dim=1)
            + target_flat.sum(dim=1)
            + 1.0
        )

        dice_loss = (
            1.0 - dice_score
        ).mean()

        total_loss = (
            weighted_mse
            + self.dice_weight
            * dice_loss
        )

        return total_loss


# ============================================================
# 坐标与指标
# ============================================================

def extract_peak_coordinates(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    从网络输出热力图中提取峰值下标。

    返回：
        predicted_range_indices
        predicted_velocity_indices
    """

    prediction = torch.sigmoid(logits)

    batch_size = prediction.shape[0]
    range_size = prediction.shape[-1]

    flat_indices = (
        prediction
        .reshape(batch_size, -1)
        .argmax(dim=1)
    )

    predicted_velocity_indices = (
        flat_indices // range_size
    )

    predicted_range_indices = (
        flat_indices % range_size
    )

    return (
        predicted_range_indices,
        predicted_velocity_indices,
    )


def calculate_batch_metrics(
    logits: torch.Tensor,
    true_range_indices: torch.Tensor,
    true_velocity_indices: torch.Tensor,
) -> dict:
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

    strict_hits = (
        (
            range_errors
            <= STRICT_RANGE_RADIUS
        )
        & (
            velocity_errors
            <= STRICT_VELOCITY_RADIUS
        )
    )

    relaxed_hits = (
        (
            range_errors
            <= RELAXED_RANGE_RADIUS
        )
        & (
            velocity_errors
            <= RELAXED_VELOCITY_RADIUS
        )
    )

    return {
        "sample_count": int(
            logits.shape[0]
        ),
        "range_error_sum": float(
            range_errors.sum().item()
        ),
        "velocity_error_sum": float(
            velocity_errors.sum().item()
        ),
        "strict_hit_count": int(
            strict_hits.sum().item()
        ),
        "relaxed_hit_count": int(
            relaxed_hits.sum().item()
        ),
    }


# ============================================================
# 单轮训练
# ============================================================

def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    model.train()

    total_loss = 0.0
    total_samples = 0

    total_range_error = 0.0
    total_velocity_error = 0.0

    total_strict_hits = 0
    total_relaxed_hits = 0

    for batch in data_loader:
        input_tensor = (
            batch["input"]
            .to(
                device,
                non_blocking=True,
            )
        )

        target_tensor = (
            batch["target"]
            .to(
                device,
                non_blocking=True,
            )
        )

        true_range_indices = (
            batch["range_index"]
            .to(
                device,
                non_blocking=True,
            )
        )

        true_velocity_indices = (
            batch["velocity_index"]
            .to(
                device,
                non_blocking=True,
            )
        )

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
                "训练损失出现 NaN 或无穷值。"
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRADIENT_CLIP_NORM,
        )

        optimizer.step()

        batch_size = int(
            input_tensor.shape[0]
        )

        total_loss += (
            float(loss.item())
            * batch_size
        )

        total_samples += batch_size

        batch_metrics = calculate_batch_metrics(
            logits.detach(),
            true_range_indices,
            true_velocity_indices,
        )

        total_range_error += (
            batch_metrics[
                "range_error_sum"
            ]
        )

        total_velocity_error += (
            batch_metrics[
                "velocity_error_sum"
            ]
        )

        total_strict_hits += (
            batch_metrics[
                "strict_hit_count"
            ]
        )

        total_relaxed_hits += (
            batch_metrics[
                "relaxed_hit_count"
            ]
        )

    return {
        "loss": (
            total_loss
            / total_samples
        ),
        "mean_range_error": (
            total_range_error
            / total_samples
        ),
        "mean_velocity_error": (
            total_velocity_error
            / total_samples
        ),
        "strict_hit_rate": (
            total_strict_hits
            / total_samples
        ),
        "relaxed_hit_rate": (
            total_relaxed_hits
            / total_samples
        ),
    }


# ============================================================
# 验证与测试
# ============================================================

def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()

    total_loss = 0.0
    total_samples = 0

    total_range_error = 0.0
    total_velocity_error = 0.0

    total_strict_hits = 0
    total_relaxed_hits = 0

    with torch.no_grad():
        for batch in data_loader:
            input_tensor = (
                batch["input"]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            target_tensor = (
                batch["target"]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            true_range_indices = (
                batch["range_index"]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            true_velocity_indices = (
                batch["velocity_index"]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            logits = model(input_tensor)

            loss = loss_function(
                logits,
                target_tensor,
            )

            batch_size = int(
                input_tensor.shape[0]
            )

            total_loss += (
                float(loss.item())
                * batch_size
            )

            total_samples += batch_size

            batch_metrics = calculate_batch_metrics(
                logits,
                true_range_indices,
                true_velocity_indices,
            )

            total_range_error += (
                batch_metrics[
                    "range_error_sum"
                ]
            )

            total_velocity_error += (
                batch_metrics[
                    "velocity_error_sum"
                ]
            )

            total_strict_hits += (
                batch_metrics[
                    "strict_hit_count"
                ]
            )

            total_relaxed_hits += (
                batch_metrics[
                    "relaxed_hit_count"
                ]
            )

    return {
        "loss": (
            total_loss
            / total_samples
        ),
        "mean_range_error": (
            total_range_error
            / total_samples
        ),
        "mean_velocity_error": (
            total_velocity_error
            / total_samples
        ),
        "strict_hit_rate": (
            total_strict_hits
            / total_samples
        ),
        "relaxed_hit_rate": (
            total_relaxed_hits
            / total_samples
        ),
    }


# ============================================================
# 模型保存
# ============================================================

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    train_metrics: dict,
    validation_metrics: dict,
) -> None:
    checkpoint = {
        "epoch": int(epoch),
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
            "random_seed":
                RANDOM_SEED,
            "range_sigma":
                RANGE_SIGMA,
            "velocity_sigma":
                VELOCITY_SIGMA,
            "batch_size":
                BATCH_SIZE,
            "learning_rate":
                LEARNING_RATE,
            "weight_decay":
                WEIGHT_DECAY,
            "positive_weight":
                POSITIVE_WEIGHT,
            "dice_weight":
                DICE_WEIGHT,
            "strict_range_radius":
                STRICT_RANGE_RADIUS,
            "strict_velocity_radius":
                STRICT_VELOCITY_RADIUS,
            "relaxed_range_radius":
                RELAXED_RANGE_RADIUS,
            "relaxed_velocity_radius":
                RELAXED_VELOCITY_RADIUS,
        },
    }

    torch.save(
        checkpoint,
        path,
    )


# ============================================================
# 最佳模型判定
# ============================================================

def build_model_score(
    metrics: dict,
) -> tuple:
    """
    最佳模型选择优先级：

    1. 宽松命中率越高越好；
    2. 严格命中率越高越好；
    3. 平均速度误差越低越好；
    4. 平均距离误差越低越好；
    5. 验证损失越低越好。
    """

    return (
        float(
            metrics[
                "relaxed_hit_rate"
            ]
        ),
        float(
            metrics[
                "strict_hit_rate"
            ]
        ),
        -float(
            metrics[
                "mean_velocity_error"
            ]
        ),
        -float(
            metrics[
                "mean_range_error"
            ]
        ),
        -float(
            metrics["loss"]
        ),
    )


# ============================================================
# 测试集逐样本结果
# ============================================================

def get_sample_ids(
    batch: dict,
    batch_size: int,
    start_index: int,
) -> list[str]:
    if "sample_id" not in batch:
        return [
            f"test_{start_index + index:03d}"
            for index in range(batch_size)
        ]

    sample_ids = batch["sample_id"]

    if isinstance(sample_ids, str):
        return [sample_ids]

    if isinstance(sample_ids, tuple):
        return [
            str(value)
            for value in sample_ids
        ]

    if isinstance(sample_ids, list):
        return [
            str(value)
            for value in sample_ids
        ]

    return [
        str(value)
        for value in sample_ids
    ]


def evaluate_test_samples(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()

    records = []
    global_sample_index = 0

    with torch.no_grad():
        for batch in data_loader:
            input_tensor = (
                batch["input"]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            true_range_indices = (
                batch["range_index"]
                .long()
                .reshape(-1)
                .to(
                    device,
                    non_blocking=True,
                )
            )

            true_velocity_indices = (
                batch["velocity_index"]
                .long()
                .reshape(-1)
                .to(
                    device,
                    non_blocking=True,
                )
            )

            logits = model(input_tensor)

            prediction = torch.sigmoid(logits)

            (
                predicted_range_indices,
                predicted_velocity_indices,
            ) = extract_peak_coordinates(
                logits
            )

            batch_size = int(
                input_tensor.shape[0]
            )

            sample_ids = get_sample_ids(
                batch,
                batch_size,
                global_sample_index,
            )

            for local_index in range(
                batch_size
            ):
                true_range_index = int(
                    true_range_indices[
                        local_index
                    ].item()
                )

                true_velocity_index = int(
                    true_velocity_indices[
                        local_index
                    ].item()
                )

                predicted_range_index = int(
                    predicted_range_indices[
                        local_index
                    ].item()
                )

                predicted_velocity_index = int(
                    predicted_velocity_indices[
                        local_index
                    ].item()
                )

                range_error = abs(
                    predicted_range_index
                    - true_range_index
                )

                velocity_error = abs(
                    predicted_velocity_index
                    - true_velocity_index
                )

                strict_hit = (
                    range_error
                    <= STRICT_RANGE_RADIUS
                    and velocity_error
                    <= STRICT_VELOCITY_RADIUS
                )

                relaxed_hit = (
                    range_error
                    <= RELAXED_RANGE_RADIUS
                    and velocity_error
                    <= RELAXED_VELOCITY_RADIUS
                )

                peak_probability = float(
                    prediction[
                        local_index,
                        0,
                        predicted_velocity_index,
                        predicted_range_index,
                    ].item()
                )

                records.append(
                    {
                        "sample_id":
                            sample_ids[
                                local_index
                            ],
                        "true_range_index":
                            true_range_index,
                        "true_velocity_index":
                            true_velocity_index,
                        "predicted_range_index":
                            predicted_range_index,
                        "predicted_velocity_index":
                            predicted_velocity_index,
                        "range_error_gates":
                            range_error,
                        "velocity_error_bins":
                            velocity_error,
                        "strict_hit":
                            strict_hit,
                        "relaxed_hit":
                            relaxed_hit,
                        "peak_probability":
                            peak_probability,
                    }
                )

            global_sample_index += (
                batch_size
            )

    return pd.DataFrame(records)


# ============================================================
# 训练曲线
# ============================================================

def save_training_curves(
    history_df: pd.DataFrame,
) -> None:
    configure_chinese_font()

    # 损失曲线
    figure = plt.figure(
        figsize=(9, 5.5)
    )

    plt.plot(
        history_df["epoch"],
        history_df["train_loss"],
        label="训练集损失",
    )

    plt.plot(
        history_df["epoch"],
        history_df["val_loss"],
        label="验证集损失",
    )

    plt.xlabel("训练轮次")
    plt.ylabel("损失值")
    plt.title("Simple FCN训练与验证损失")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    figure.savefig(
        FIGURE_DIR
        / "loss_curve.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(figure)

    # 命中率曲线
    figure = plt.figure(
        figsize=(9, 5.5)
    )

    plt.plot(
        history_df["epoch"],
        history_df[
            "train_strict_hit_rate"
        ] * 100.0,
        label="训练集严格命中率",
    )

    plt.plot(
        history_df["epoch"],
        history_df[
            "val_strict_hit_rate"
        ] * 100.0,
        label="验证集严格命中率",
    )

    plt.plot(
        history_df["epoch"],
        history_df[
            "val_relaxed_hit_rate"
        ] * 100.0,
        label="验证集宽松命中率",
    )

    plt.xlabel("训练轮次")
    plt.ylabel("命中率（%）")
    plt.title("Simple FCN定位命中率")
    plt.ylim(0.0, 105.0)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    figure.savefig(
        FIGURE_DIR
        / "hit_rate_curve.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(figure)

    # 平均误差曲线
    figure = plt.figure(
        figsize=(9, 5.5)
    )

    plt.plot(
        history_df["epoch"],
        history_df[
            "val_mean_range_error"
        ],
        label="验证集平均距离误差",
    )

    plt.plot(
        history_df["epoch"],
        history_df[
            "val_mean_velocity_error"
        ],
        label="验证集平均速度误差",
    )

    plt.xlabel("训练轮次")
    plt.ylabel("下标误差")
    plt.title("Simple FCN验证集定位误差")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    figure.savefig(
        FIGURE_DIR
        / "validation_error_curve.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# 测试集预测图
# ============================================================

def save_test_prediction_figures(
    model: nn.Module,
    dataset: RadarDataset,
    device: torch.device,
    sample_count: int = 8,
) -> None:
    configure_chinese_font()

    prediction_dir = (
        FIGURE_DIR
        / "test_predictions"
    )

    prediction_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.eval()

    with torch.no_grad():
        for sample_index in range(
            min(
                sample_count,
                len(dataset),
            )
        ):
            sample = dataset[
                sample_index
            ]

            input_tensor = (
                sample["input"]
                .unsqueeze(0)
                .to(device)
            )

            logits = model(input_tensor)

            prediction = (
                torch.sigmoid(logits)[
                    0,
                    0,
                ]
                .cpu()
                .numpy()
            )

            target = (
                sample["target"][0]
                .cpu()
                .numpy()
            )

            input_mean = (
                sample["input"]
                .mean(dim=0)
                .cpu()
                .numpy()
            )

            true_range_index = int(
                sample["range_index"]
            )

            true_velocity_index = int(
                sample["velocity_index"]
            )

            flat_index = int(
                np.argmax(prediction)
            )

            (
                predicted_velocity_index,
                predicted_range_index,
            ) = np.unravel_index(
                flat_index,
                prediction.shape,
            )

            range_error = abs(
                predicted_range_index
                - true_range_index
            )

            velocity_error = abs(
                predicted_velocity_index
                - true_velocity_index
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
                input_mean,
                aspect="auto",
                cmap="jet",
            )

            axes[0].plot(
                true_range_index,
                true_velocity_index,
                marker="+",
                color="red",
                markersize=15,
                markeredgewidth=2,
            )

            axes[0].set_title(
                "H/V平均归一化RD图"
            )

            axes[0].set_xlabel(
                "距离门下标"
            )

            axes[0].set_ylabel(
                "速度单元下标"
            )

            axes[1].imshow(
                target,
                aspect="auto",
                cmap="jet",
                vmin=0.0,
                vmax=1.0,
            )

            axes[1].plot(
                true_range_index,
                true_velocity_index,
                marker="+",
                color="white",
                markersize=15,
                markeredgewidth=2,
            )

            axes[1].set_title(
                "真实目标热力图"
            )

            axes[1].set_xlabel(
                "距离门下标"
            )

            axes[1].set_ylabel(
                "速度单元下标"
            )

            axes[2].imshow(
                prediction,
                aspect="auto",
                cmap="jet",
                vmin=0.0,
                vmax=1.0,
            )

            axes[2].plot(
                true_range_index,
                true_velocity_index,
                marker="+",
                color="white",
                markersize=15,
                markeredgewidth=2,
                label="真实标签",
            )

            axes[2].plot(
                predicted_range_index,
                predicted_velocity_index,
                marker="x",
                color="red",
                markersize=12,
                markeredgewidth=2,
                label="预测峰值",
            )

            axes[2].set_title(
                "Simple FCN预测热力图"
            )

            axes[2].set_xlabel(
                "距离门下标"
            )

            axes[2].set_ylabel(
                "速度单元下标"
            )

            axes[2].legend()

            figure.suptitle(
                (
                    f"测试样本：{sample_id}　"
                    f"距离误差：{range_error}门　"
                    f"速度误差：{velocity_error}单元"
                ),
                fontsize=13,
            )

            figure.tight_layout()

            output_path = (
                prediction_dir
                / (
                    f"prediction_"
                    f"{sample_index + 1:02d}.png"
                )
            )

            figure.savefig(
                output_path,
                dpi=220,
                bbox_inches="tight",
            )

            plt.close(figure)


# ============================================================
# 主程序
# ============================================================

def main() -> None:
    set_random_seed(
        RANDOM_SEED
    )

    prepare_output_directories()

    configure_chinese_font()

    device = get_device()

    print(
        "=========================================="
    )
    print(
        "Simple FCN正式训练"
    )
    print(
        "=========================================="
    )

    print(
        f"项目目录：{PROJECT_ROOT}"
    )

    print(
        f"运行设备：{device}"
    )

    if torch.cuda.is_available():
        print(
            "GPU："
            f"{torch.cuda.get_device_name(0)}"
        )

        gpu_memory_gb = (
            torch.cuda.get_device_properties(
                0
            ).total_memory
            / 1024 ** 3
        )

        print(
            f"GPU显存：{gpu_memory_gb:.2f} GB"
        )

    # --------------------------------------------------------
    # 数据集
    # --------------------------------------------------------

    print(
        "\n正在创建训练、验证和测试数据集……"
    )

    train_dataset = RadarDataset(
        split="train",
        range_sigma=RANGE_SIGMA,
        velocity_sigma=VELOCITY_SIGMA,
    )

    validation_dataset = RadarDataset(
        split="val",
        range_sigma=RANGE_SIGMA,
        velocity_sigma=VELOCITY_SIGMA,
    )

    test_dataset = RadarDataset(
        split="test",
        range_sigma=RANGE_SIGMA,
        velocity_sigma=VELOCITY_SIGMA,
    )

    print(
        f"训练集：{len(train_dataset)}"
    )

    print(
        f"验证集：{len(validation_dataset)}"
    )

    print(
        f"测试集：{len(test_dataset)}"
    )

    if len(train_dataset) == 0:
        raise RuntimeError(
            "训练集为空。"
        )

    if len(validation_dataset) == 0:
        raise RuntimeError(
            "验证集为空。"
        )

    if len(test_dataset) == 0:
        raise RuntimeError(
            "测试集为空。"
        )

    pin_memory = (
        device.type == "cuda"
    )

    train_generator = (
        torch.Generator()
    )

    train_generator.manual_seed(
        RANDOM_SEED
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=False,
        generator=train_generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=False,
    )

    # --------------------------------------------------------
    # 模型与优化器
    # --------------------------------------------------------

    model = SimpleRadarFCN().to(
        device
    )

    print(
        "可训练参数量："
        f"{count_trainable_parameters(model):,}"
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

    # --------------------------------------------------------
    # 正式训练
    # --------------------------------------------------------

    history = []

    best_score = None
    best_epoch = 0
    epochs_without_improvement = 0

    training_start_time = time.time()

    print(
        "\n开始正式训练……"
    )

    for epoch in range(
        1,
        EPOCH_COUNT + 1,
    ):
        epoch_start_time = time.time()

        train_metrics = train_one_epoch(
            model,
            train_loader,
            loss_function,
            optimizer,
            device,
        )

        validation_metrics = evaluate_model(
            model,
            validation_loader,
            loss_function,
            device,
        )

        scheduler.step(
            validation_metrics[
                "loss"
            ]
        )

        current_learning_rate = float(
            optimizer.param_groups[0]["lr"]
        )

        current_score = build_model_score(
            validation_metrics
        )

        is_best = (
            best_score is None
            or current_score > best_score
        )

        if is_best:
            best_score = current_score
            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                BEST_MODEL_PATH,
                model,
                optimizer,
                scheduler,
                epoch,
                train_metrics,
                validation_metrics,
            )

        else:
            epochs_without_improvement += 1

        save_checkpoint(
            LAST_MODEL_PATH,
            model,
            optimizer,
            scheduler,
            epoch,
            train_metrics,
            validation_metrics,
        )

        epoch_time = (
            time.time()
            - epoch_start_time
        )

        history.append(
            {
                "epoch": epoch,
                "learning_rate":
                    current_learning_rate,
                "epoch_time_seconds":
                    epoch_time,

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
                "is_best":
                    is_best,
            }
        )

        print(
            f"第{epoch:03d}轮｜"
            f"训练损失={train_metrics['loss']:.6f}｜"
            f"验证损失={validation_metrics['loss']:.6f}｜"
            f"验证距离误差="
            f"{validation_metrics['mean_range_error']:.3f}门｜"
            f"验证速度误差="
            f"{validation_metrics['mean_velocity_error']:.3f}单元｜"
            f"严格命中率="
            f"{validation_metrics['strict_hit_rate']:.2%}｜"
            f"宽松命中率="
            f"{validation_metrics['relaxed_hit_rate']:.2%}｜"
            f"学习率={current_learning_rate:.2e}"
            + (
                "｜最佳模型"
                if is_best
                else ""
            )
        )

        history_df = pd.DataFrame(
            history
        )

        history_df.to_csv(
            HISTORY_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            print(
                "\n验证集连续"
                f"{EARLY_STOPPING_PATIENCE}"
                "轮没有改进，提前停止训练。"
            )
            break

    total_training_time = (
        time.time()
        - training_start_time
    )

    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(
        HISTORY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    save_training_curves(
        history_df
    )

    # --------------------------------------------------------
    # 加载最佳模型
    # --------------------------------------------------------

    print(
        "\n正在加载最佳验证模型……"
    )

    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    best_validation_metrics = (
        checkpoint[
            "validation_metrics"
        ]
    )

    # --------------------------------------------------------
    # 测试集评价
    # --------------------------------------------------------

    print(
        "正在评价测试集……"
    )

    test_metrics = evaluate_model(
        model,
        test_loader,
        loss_function,
        device,
    )

    test_result_df = evaluate_test_samples(
        model,
        test_loader,
        device,
    )

    test_result_df.to_csv(
        TEST_RESULT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    save_test_prediction_figures(
        model,
        test_dataset,
        device,
        sample_count=8,
    )

    # --------------------------------------------------------
    # 汇总保存
    # --------------------------------------------------------

    summary = {
        "model":
            "SimpleRadarFCN",
        "train_sample_count":
            len(train_dataset),
        "validation_sample_count":
            len(validation_dataset),
        "test_sample_count":
            len(test_dataset),
        "trainable_parameters":
            count_trainable_parameters(
                model
            ),
        "best_epoch":
            int(best_epoch),
        "total_training_time_seconds":
            float(total_training_time),
        "best_validation_metrics":
            best_validation_metrics,
        "test_metrics":
            test_metrics,
        "cfar_baseline": {
            "strict_hit_rate":
                0.9371,
            "relaxed_hit_rate":
                0.9528,
            "average_false_candidates":
                6.48,
        },
    }

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # 最终输出
    # --------------------------------------------------------

    print(
        "\n=========================================="
    )
    print(
        "正式训练完成"
    )
    print(
        "=========================================="
    )

    print(
        f"最佳轮次：{best_epoch}"
    )

    print(
        "最佳验证集损失："
        f"{best_validation_metrics['loss']:.6f}"
    )

    print(
        "最佳验证集严格命中率："
        f"{best_validation_metrics['strict_hit_rate']:.2%}"
    )

    print(
        "最佳验证集宽松命中率："
        f"{best_validation_metrics['relaxed_hit_rate']:.2%}"
    )

    print(
        "\n========== 测试集结果 =========="
    )

    print(
        f"测试损失：{test_metrics['loss']:.6f}"
    )

    print(
        "平均距离误差："
        f"{test_metrics['mean_range_error']:.3f}门"
    )

    print(
        "平均速度误差："
        f"{test_metrics['mean_velocity_error']:.3f}单元"
    )

    print(
        "严格命中率："
        f"{test_metrics['strict_hit_rate']:.2%}"
    )

    print(
        "宽松命中率："
        f"{test_metrics['relaxed_hit_rate']:.2%}"
    )

    print(
        "\n========== CA-CFAR参考 =========="
    )

    print(
        "严格命中率：93.71%"
    )

    print(
        "宽松命中率：95.28%"
    )

    print(
        "平均虚警候选：6.48个/帧"
    )

    print(
        "\n========== 文件输出 =========="
    )

    print(
        f"最佳模型：{BEST_MODEL_PATH}"
    )

    print(
        f"最后模型：{LAST_MODEL_PATH}"
    )

    print(
        f"训练记录：{HISTORY_PATH}"
    )

    print(
        f"测试明细：{TEST_RESULT_PATH}"
    )

    print(
        f"结果汇总：{SUMMARY_PATH}"
    )

    print(
        f"结果图片：{FIGURE_DIR}"
    )


if __name__ == "__main__":
    main()