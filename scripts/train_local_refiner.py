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
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


# ============================================================
# 项目路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from datasets.radar_dataset import RadarDataset
from models.simple_fcn import SimpleRadarFCN
from utils.plot_config import configure_chinese_font


# ============================================================
# 默认参数
# ============================================================

RANDOM_SEED = 42

DEFAULT_BACKBONE_PATH = (
    PROJECT_ROOT
    / "results"
    / "experiments"
    / "sigma_r3_v1"
    / "checkpoints"
    / "best.pt"
)

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
# 参数读取
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FCN粗定位后的局部偏移精修训练"
    )

    parser.add_argument(
        "--backbone",
        type=Path,
        default=DEFAULT_BACKBONE_PATH,
        help="sigma_r3_v1最佳模型路径",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="sigma_r3_v1_refiner",
        help="实验名称",
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
        help="以FCN峰值为中心裁剪的局部区域大小，必须为奇数",
    )

    parser.add_argument(
        "--max-range-offset",
        type=float,
        default=4.0,
        help="局部回归器允许修正的最大距离门数",
    )

    parser.add_argument(
        "--max-velocity-offset",
        type=float,
        default=2.0,
        help="局部回归器允许修正的最大速度单元数",
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

    parser.add_argument(
        "--prediction-count",
        type=int,
        default=10,
    )

    return parser.parse_args()


# ============================================================
# 基础工具
# ============================================================

def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def load_torch_file(
    path: Path,
    device: torch.device,
) -> Any:
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=device,
        )


def scalar_to_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())

    return int(value)


# ============================================================
# 局部偏移回归器
# ============================================================

class LocalOffsetRefiner(nn.Module):
    """
    输入通道：

    1. H通道归一化RD图
    2. V通道归一化RD图
    3. FCN预测概率图
    4. 局部距离坐标
    5. 局部速度坐标

    输出：

    [距离偏移归一化值, 速度偏移归一化值]

    两个输出均限制在 [-1, 1]。
    """

    def __init__(self) -> None:
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                5,
                32,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                48,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(8, 48),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                48,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(64, 2),
            nn.Tanh(),
        )

    def forward(
        self,
        local_patch: torch.Tensor,
    ) -> torch.Tensor:
        features = self.feature_extractor(
            local_patch
        )

        return self.regressor(features)


# ============================================================
# FCN加载
# ============================================================

def load_frozen_backbone(
    checkpoint_path: Path,
    device: torch.device,
) -> SimpleRadarFCN:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"找不到FCN模型：{checkpoint_path}"
        )

    checkpoint = load_torch_file(
        checkpoint_path,
        device,
    )

    model_state_dict = checkpoint

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            model_state_dict = checkpoint[
                "model_state_dict"
            ]

    backbone = SimpleRadarFCN()

    backbone.load_state_dict(
        model_state_dict
    )

    backbone.to(device)
    backbone.eval()

    for parameter in backbone.parameters():
        parameter.requires_grad = False

    return backbone


# ============================================================
# 粗定位峰值
# ============================================================

def extract_peak_coordinates(
    probability_map: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    probability_map形状：

    [B, 1, velocity_size, range_size]
    """

    batch_size = probability_map.shape[0]
    range_size = probability_map.shape[-1]

    flat_indices = (
        probability_map
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


# ============================================================
# 局部裁剪
# ============================================================

def extract_local_patches(
    feature_map: torch.Tensor,
    center_range_indices: torch.Tensor,
    center_velocity_indices: torch.Tensor,
    crop_size: int,
) -> torch.Tensor:
    """
    feature_map形状：

    [B, C, velocity_size, range_size]
    """

    if crop_size % 2 == 0:
        raise ValueError(
            "crop_size必须为奇数。"
        )

    half_size = crop_size // 2

    padded = F.pad(
        feature_map,
        pad=(
            half_size,
            half_size,
            half_size,
            half_size,
        ),
        mode="replicate",
    )

    patches = []

    batch_size = feature_map.shape[0]

    for batch_index in range(batch_size):
        range_index = int(
            center_range_indices[
                batch_index
            ].item()
        )

        velocity_index = int(
            center_velocity_indices[
                batch_index
            ].item()
        )

        patch = padded[
            batch_index,
            :,
            velocity_index:
                velocity_index + crop_size,
            range_index:
                range_index + crop_size,
        ]

        patches.append(patch)

    patches_tensor = torch.stack(
        patches,
        dim=0,
    )

    coordinate_axis = torch.linspace(
        -1.0,
        1.0,
        crop_size,
        device=feature_map.device,
        dtype=feature_map.dtype,
    )

    velocity_grid, range_grid = torch.meshgrid(
        coordinate_axis,
        coordinate_axis,
        indexing="ij",
    )

    range_grid = (
        range_grid
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(batch_size, 1, -1, -1)
    )

    velocity_grid = (
        velocity_grid
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(batch_size, 1, -1, -1)
    )

    return torch.cat(
        [
            patches_tensor,
            range_grid,
            velocity_grid,
        ],
        dim=1,
    )


# ============================================================
# 粗定位和精修定位
# ============================================================

def predict_coordinates(
    backbone: nn.Module,
    refiner: nn.Module,
    input_tensor: torch.Tensor,
    crop_size: int,
    max_range_offset: float,
    max_velocity_offset: float,
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        backbone_logits = backbone(
            input_tensor
        )

        probability_map = torch.sigmoid(
            backbone_logits
        )

        (
            coarse_range_indices,
            coarse_velocity_indices,
        ) = extract_peak_coordinates(
            probability_map
        )

    combined_feature_map = torch.cat(
        [
            input_tensor,
            probability_map,
        ],
        dim=1,
    )

    local_patches = extract_local_patches(
        feature_map=combined_feature_map,
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

    range_size = input_tensor.shape[-1]
    velocity_size = input_tensor.shape[-2]

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
        max=range_size - 1,
    )

    refined_velocity_indices = torch.clamp(
        refined_velocity_indices,
        min=0,
        max=velocity_size - 1,
    )

    return {
        "probability_map":
            probability_map,
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


# ============================================================
# 指标
# ============================================================

def empty_metric_accumulator() -> dict[str, float]:
    result = {
        "sample_count": 0.0,
        "range_error_sum": 0.0,
        "velocity_error_sum": 0.0,
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

    for metric_name, radii in (
        METRIC_RADII.items()
    ):
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


def finalize_metrics(
    accumulator: dict[str, float],
) -> dict[str, float]:
    sample_count = accumulator[
        "sample_count"
    ]

    if sample_count <= 0:
        raise RuntimeError(
            "评价样本数为0。"
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
    }

    for metric_name in METRIC_RADII:
        result[f"{metric_name}_hit_rate"] = (
            accumulator[
                f"{metric_name}_hit_count"
            ]
            / sample_count
        )

    return result


# ============================================================
# 单轮训练或评价
# ============================================================

def run_epoch(
    backbone: nn.Module,
    refiner: nn.Module,
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

    loss_sum = 0.0
    sample_count = 0

    coarse_accumulator = (
        empty_metric_accumulator()
    )

    refined_accumulator = (
        empty_metric_accumulator()
    )

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
                input_tensor=input_tensor,
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

            normalized_offsets = predictions[
                "normalized_offsets"
            ]

            true_range_offsets = (
                true_range_indices.float()
                - coarse_range_indices.float()
            )

            true_velocity_offsets = (
                true_velocity_indices.float()
                - coarse_velocity_indices.float()
            )

            normalized_range_targets = (
                true_range_offsets
                / max_range_offset
            )

            normalized_velocity_targets = (
                true_velocity_offsets
                / max_velocity_offset
            )

            normalized_targets = torch.stack(
                [
                    torch.clamp(
                        normalized_range_targets,
                        min=-1.0,
                        max=1.0,
                    ),
                    torch.clamp(
                        normalized_velocity_targets,
                        min=-1.0,
                        max=1.0,
                    ),
                ],
                dim=1,
            )

            loss = loss_function(
                normalized_offsets,
                normalized_targets,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "偏移回归损失出现NaN或无穷值。"
                )

            if training:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    refiner.parameters(),
                    max_norm=5.0,
                )

                optimizer.step()

            current_batch_size = int(
                input_tensor.shape[0]
            )

            loss_sum += (
                float(loss.item())
                * current_batch_size
            )

            sample_count += current_batch_size

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
        "loss": loss_sum / sample_count,
        "coarse": finalize_metrics(
            coarse_accumulator
        ),
        "refined": finalize_metrics(
            refined_accumulator
        ),
    }


# ============================================================
# 最佳模型规则
# ============================================================

def build_model_score(
    metrics: dict[str, Any],
) -> tuple[float, ...]:
    refined = metrics["refined"]

    return (
        refined["relaxed_hit_rate"],
        refined["strict_hit_rate"],
        refined["application_hit_rate"],
        -refined["mean_velocity_error"],
        -refined["mean_range_error"],
        -metrics["loss"],
    )


# ============================================================
# 测试集逐样本结果
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

    if isinstance(
        raw_sample_ids,
        (list, tuple),
    ):
        return [
            str(value)
            for value in raw_sample_ids
        ]

    return [
        str(value)
        for value in raw_sample_ids
    ]


def create_test_details(
    backbone: nn.Module,
    refiner: nn.Module,
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

            predictions = predict_coordinates(
                backbone=backbone,
                refiner=refiner,
                input_tensor=input_tensor,
                crop_size=crop_size,
                max_range_offset=
                    max_range_offset,
                max_velocity_offset=
                    max_velocity_offset,
            )

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


# ============================================================
# 曲线
# ============================================================

def save_training_curves(
    history_df: pd.DataFrame,
    figure_dir: Path,
) -> None:
    configure_chinese_font()

    figure = plt.figure(
        figsize=(9, 5.5)
    )

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
    plt.ylabel("偏移回归损失")
    plt.title("局部偏移回归器损失")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    figure.savefig(
        figure_dir / "loss_curve.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(figure)

    figure = plt.figure(
        figsize=(9, 5.5)
    )

    plt.plot(
        history_df["epoch"],
        history_df[
            "val_coarse_strict_hit_rate"
        ] * 100,
        label="FCN粗定位严格命中率",
    )

    plt.plot(
        history_df["epoch"],
        history_df[
            "val_refined_strict_hit_rate"
        ] * 100,
        label="精修后严格命中率",
    )

    plt.plot(
        history_df["epoch"],
        history_df[
            "val_refined_relaxed_hit_rate"
        ] * 100,
        label="精修后宽松命中率",
    )

    plt.xlabel("训练轮次")
    plt.ylabel("命中率（%）")
    plt.title("局部偏移精修验证集命中率")
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

    figure = plt.figure(
        figsize=(9, 5.5)
    )

    plt.plot(
        history_df["epoch"],
        history_df[
            "val_coarse_mean_range_error"
        ],
        label="粗定位距离误差",
    )

    plt.plot(
        history_df["epoch"],
        history_df[
            "val_refined_mean_range_error"
        ],
        label="精修后距离误差",
    )

    plt.plot(
        history_df["epoch"],
        history_df[
            "val_refined_mean_velocity_error"
        ],
        label="精修后速度误差",
    )

    plt.xlabel("训练轮次")
    plt.ylabel("平均下标误差")
    plt.title("局部偏移精修定位误差")
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
# 测试样本图片
# ============================================================

def save_prediction_figures(
    backbone: nn.Module,
    refiner: nn.Module,
    dataset: RadarDataset,
    device: torch.device,
    output_dir: Path,
    crop_size: int,
    max_range_offset: float,
    max_velocity_offset: float,
    sample_count: int,
) -> None:
    configure_chinese_font()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    backbone.eval()
    refiner.eval()

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

            predictions = predict_coordinates(
                backbone=backbone,
                refiner=refiner,
                input_tensor=input_tensor,
                crop_size=crop_size,
                max_range_offset=
                    max_range_offset,
                max_velocity_offset=
                    max_velocity_offset,
            )

            probability_map = (
                predictions[
                    "probability_map"
                ][0, 0]
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

            coarse_range = int(
                predictions[
                    "coarse_range_indices"
                ][0].item()
            )

            coarse_velocity = int(
                predictions[
                    "coarse_velocity_indices"
                ][0].item()
            )

            refined_range = int(
                predictions[
                    "refined_range_indices"
                ][0].item()
            )

            refined_velocity = int(
                predictions[
                    "refined_velocity_indices"
                ][0].item()
            )

            sample_id = str(
                sample.get(
                    "sample_id",
                    f"test_{sample_index:03d}",
                )
            )

            figure, axes = plt.subplots(
                1,
                2,
                figsize=(12, 5),
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
                color="white",
                markersize=16,
                markeredgewidth=2,
                label="真实标签",
            )

            axes[0].plot(
                coarse_range,
                coarse_velocity,
                marker="x",
                color="red",
                markersize=13,
                markeredgewidth=2,
                label="FCN粗定位",
            )

            axes[0].plot(
                refined_range,
                refined_velocity,
                marker="o",
                markerfacecolor="none",
                markeredgecolor="lime",
                markersize=13,
                markeredgewidth=2,
                label="精修定位",
            )

            axes[0].set_title(
                "RD图定位结果"
            )

            axes[0].set_xlabel(
                "距离门下标"
            )

            axes[0].set_ylabel(
                "速度单元下标"
            )

            axes[0].legend()

            axes[1].imshow(
                probability_map,
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
                markersize=16,
                markeredgewidth=2,
                label="真实标签",
            )

            axes[1].plot(
                coarse_range,
                coarse_velocity,
                marker="x",
                color="red",
                markersize=13,
                markeredgewidth=2,
                label="FCN粗定位",
            )

            axes[1].plot(
                refined_range,
                refined_velocity,
                marker="o",
                markerfacecolor="none",
                markeredgecolor="lime",
                markersize=13,
                markeredgewidth=2,
                label="精修定位",
            )

            axes[1].set_title(
                "FCN概率图与精修结果"
            )

            axes[1].set_xlabel(
                "距离门下标"
            )

            axes[1].set_ylabel(
                "速度单元下标"
            )

            axes[1].legend()

            coarse_range_error = abs(
                coarse_range - true_range
            )

            coarse_velocity_error = abs(
                coarse_velocity - true_velocity
            )

            refined_range_error = abs(
                refined_range - true_range
            )

            refined_velocity_error = abs(
                refined_velocity - true_velocity
            )

            figure.suptitle(
                (
                    f"{sample_id}\n"
                    f"粗定位误差："
                    f"{coarse_range_error}门、"
                    f"{coarse_velocity_error}单元；"
                    f"精修误差："
                    f"{refined_range_error}门、"
                    f"{refined_velocity_error}单元"
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

    if args.crop_size % 2 == 0:
        raise ValueError(
            "--crop-size必须是奇数。"
        )

    if args.max_range_offset <= 0:
        raise ValueError(
            "--max-range-offset必须大于0。"
        )

    if args.max_velocity_offset <= 0:
        raise ValueError(
            "--max-velocity-offset必须大于0。"
        )

    set_random_seed(RANDOM_SEED)
    configure_chinese_font()

    device = get_device()

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

    best_refiner_path = (
        checkpoint_dir
        / "best_refiner.pt"
    )

    last_refiner_path = (
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

    print("=" * 68)
    print("FCN粗定位 + 局部偏移回归精修")
    print("=" * 68)

    print(f"运行设备：{device}")

    if torch.cuda.is_available():
        print(
            "GPU："
            f"{torch.cuda.get_device_name(0)}"
        )

    print(f"FCN模型：{args.backbone}")
    print(f"局部裁剪尺寸：{args.crop_size}")
    print(
        "最大距离修正："
        f"±{args.max_range_offset}门"
    )
    print(
        "最大速度修正："
        f"±{args.max_velocity_offset}单元"
    )

    # --------------------------------------------------------
    # 数据集
    # --------------------------------------------------------

    print("\n正在加载数据集……")

    train_dataset = RadarDataset(
        split="train",
        range_sigma=3.0,
        velocity_sigma=1.0,
    )

    validation_dataset = RadarDataset(
        split="val",
        range_sigma=3.0,
        velocity_sigma=1.0,
    )

    test_dataset = RadarDataset(
        split="test",
        range_sigma=3.0,
        velocity_sigma=1.0,
    )

    print(f"训练集：{len(train_dataset)}")
    print(f"验证集：{len(validation_dataset)}")
    print(f"测试集：{len(test_dataset)}")

    pin_memory = device.type == "cuda"

    train_generator = torch.Generator()
    train_generator.manual_seed(
        RANDOM_SEED
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        generator=train_generator,
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

    # --------------------------------------------------------
    # 模型
    # --------------------------------------------------------

    backbone = load_frozen_backbone(
        checkpoint_path=args.backbone,
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

    # --------------------------------------------------------
    # 训练
    # --------------------------------------------------------

    history = []

    best_score = None
    best_epoch = 0
    epochs_without_improvement = 0

    training_start_time = time.time()

    print("\n开始训练局部偏移回归器……")

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
            data_loader=validation_loader,
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

        checkpoint = {
            "epoch": epoch,
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
            "configuration": {
                "backbone_path":
                    str(args.backbone),
                "crop_size":
                    args.crop_size,
                "max_range_offset":
                    args.max_range_offset,
                "max_velocity_offset":
                    args.max_velocity_offset,
                "range_sigma":
                    3.0,
                "velocity_sigma":
                    1.0,
                "metric_radii":
                    METRIC_RADII,
                "random_seed":
                    RANDOM_SEED,
            },
        }

        torch.save(
            checkpoint,
            last_refiner_path,
        )

        if is_best:
            best_score = current_score
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                checkpoint,
                best_refiner_path,
            )

        else:
            epochs_without_improvement += 1

        train_refined = train_metrics[
            "refined"
        ]

        val_coarse = validation_metrics[
            "coarse"
        ]

        val_refined = validation_metrics[
            "refined"
        ]

        history.append(
            {
                "epoch":
                    epoch,
                "learning_rate":
                    current_learning_rate,

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

                "val_coarse_mean_range_error":
                    val_coarse[
                        "mean_range_error"
                    ],
                "val_coarse_mean_velocity_error":
                    val_coarse[
                        "mean_velocity_error"
                    ],
                "val_coarse_strict_hit_rate":
                    val_coarse[
                        "strict_hit_rate"
                    ],
                "val_coarse_relaxed_hit_rate":
                    val_coarse[
                        "relaxed_hit_rate"
                    ],
                "val_coarse_application_hit_rate":
                    val_coarse[
                        "application_hit_rate"
                    ],

                "val_refined_mean_range_error":
                    val_refined[
                        "mean_range_error"
                    ],
                "val_refined_mean_velocity_error":
                    val_refined[
                        "mean_velocity_error"
                    ],
                "val_refined_strict_hit_rate":
                    val_refined[
                        "strict_hit_rate"
                    ],
                "val_refined_relaxed_hit_rate":
                    val_refined[
                        "relaxed_hit_rate"
                    ],
                "val_refined_application_hit_rate":
                    val_refined[
                        "application_hit_rate"
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
            f"损失={validation_metrics['loss']:.5f}｜"
            f"粗定位严格="
            f"{val_coarse['strict_hit_rate']:.2%}｜"
            f"精修严格="
            f"{val_refined['strict_hit_rate']:.2%}｜"
            f"精修宽松="
            f"{val_refined['relaxed_hit_rate']:.2%}｜"
            f"精修应用="
            f"{val_refined['application_hit_rate']:.2%}｜"
            f"距离误差="
            f"{val_refined['mean_range_error']:.3f}门｜"
            f"速度误差="
            f"{val_refined['mean_velocity_error']:.3f}单元｜"
            f"学习率={current_learning_rate:.2e}"
            + (
                "｜最佳模型"
                if is_best
                else ""
            )
        )

        if (
            epochs_without_improvement
            >= args.early_stopping
        ):
            print(
                "\n验证集连续"
                f"{args.early_stopping}"
                "轮没有产生更优模型，提前停止。"
            )
            break

    training_seconds = (
        time.time()
        - training_start_time
    )

    history_df = pd.DataFrame(history)

    save_training_curves(
        history_df,
        figure_dir,
    )

    # --------------------------------------------------------
    # 加载最佳精修器
    # --------------------------------------------------------

    print("\n加载最佳精修器……")

    best_checkpoint = load_torch_file(
        best_refiner_path,
        device,
    )

    refiner.load_state_dict(
        best_checkpoint[
            "refiner_state_dict"
        ]
    )

    # --------------------------------------------------------
    # 测试
    # --------------------------------------------------------

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

    save_prediction_figures(
        backbone=backbone,
        refiner=refiner,
        dataset=test_dataset,
        device=device,
        output_dir=prediction_dir,
        crop_size=args.crop_size,
        max_range_offset=
            args.max_range_offset,
        max_velocity_offset=
            args.max_velocity_offset,
        sample_count=
            args.prediction_count,
    )

    summary = {
        "experiment_name":
            args.name,
        "backbone_path":
            str(args.backbone),
        "best_epoch":
            best_epoch,
        "training_seconds":
            training_seconds,
        "trainable_parameters":
            count_trainable_parameters(
                refiner
            ),
        "configuration": {
            "crop_size":
                args.crop_size,
            "max_range_offset":
                args.max_range_offset,
            "max_velocity_offset":
                args.max_velocity_offset,
            "range_sigma":
                3.0,
            "velocity_sigma":
                1.0,
            "metric_radii":
                METRIC_RADII,
        },
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

    # --------------------------------------------------------
    # 最终输出
    # --------------------------------------------------------

    validation_coarse = best_checkpoint[
        "validation_metrics"
    ]["coarse"]

    validation_refined = best_checkpoint[
        "validation_metrics"
    ]["refined"]

    test_coarse = test_metrics[
        "coarse"
    ]

    test_refined = test_metrics[
        "refined"
    ]

    print("\n" + "=" * 68)
    print("局部偏移精修训练完成")
    print("=" * 68)

    print(f"最佳轮次：{best_epoch}")

    print(
        "\n========== 最佳验证集 =========="
    )

    print(
        "FCN粗定位严格命中率："
        f"{validation_coarse['strict_hit_rate']:.2%}"
    )

    print(
        "精修后严格命中率："
        f"{validation_refined['strict_hit_rate']:.2%}"
    )

    print(
        "精修后宽松命中率："
        f"{validation_refined['relaxed_hit_rate']:.2%}"
    )

    print(
        "精修后应用邻域命中率："
        f"{validation_refined['application_hit_rate']:.2%}"
    )

    print(
        "\n========== 独立测试集 =========="
    )

    print(
        "FCN粗定位平均距离误差："
        f"{test_coarse['mean_range_error']:.3f}门"
    )

    print(
        "精修后平均距离误差："
        f"{test_refined['mean_range_error']:.3f}门"
    )

    print(
        "FCN粗定位平均速度误差："
        f"{test_coarse['mean_velocity_error']:.3f}单元"
    )

    print(
        "精修后平均速度误差："
        f"{test_refined['mean_velocity_error']:.3f}单元"
    )

    print(
        "FCN粗定位严格命中率："
        f"{test_coarse['strict_hit_rate']:.2%}"
    )

    print(
        "精修后严格命中率："
        f"{test_refined['strict_hit_rate']:.2%}"
    )

    print(
        "FCN粗定位宽松命中率："
        f"{test_coarse['relaxed_hit_rate']:.2%}"
    )

    print(
        "精修后宽松命中率："
        f"{test_refined['relaxed_hit_rate']:.2%}"
    )

    print(
        "精修后应用邻域命中率："
        f"{test_refined['application_hit_rate']:.2%}"
    )

    print(
        "\n最佳精修器："
        f"{best_refiner_path}"
    )

    print(
        "测试结果明细："
        f"{test_details_path}"
    )

    print(
        "实验汇总："
        f"{summary_path}"
    )

    print(
        "结果图片："
        f"{figure_dir}"
    )


if __name__ == "__main__":
    main()