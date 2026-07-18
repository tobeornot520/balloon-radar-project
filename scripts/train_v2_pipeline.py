from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import (
    DataLoader,
    Dataset,
    WeightedRandomSampler,
)


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
    HeatmapLoss,
    extract_peaks,
    get_state,
    load_torch,
)


METRIC_RADII = {
    "strict": (1, 1),
    "relaxed": (4, 1),
    "application": (4, 2),
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "v2完整定位管线："
            "难负峰损失 + 平移增强 + 平衡采样"
            " + 有效局部精修"
        )
    )

    parser.add_argument(
        "--pretrained",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results"
            / "experiments"
            / "sigma_r3_v1"
            / "checkpoints"
            / "best.pt"
        ),
        help="当前划分下训练得到的sigma_r3_v1最佳模型",
    )

    parser.add_argument(
        "--name",
        default="sigma_r3_v1_v2_pipeline",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--dual-epochs",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--refiner-epochs",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--refiner-batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--dual-learning-rate",
        type=float,
        default=3.0e-4,
    )

    parser.add_argument(
        "--refiner-learning-rate",
        type=float,
        default=1.0e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1.0e-5,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--dual-early-stopping",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--refiner-early-stopping",
        type=int,
        default=25,
    )

    # --------------------------------------------------------
    # 零多普勒软抑制
    # --------------------------------------------------------

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
        "--zero-band-half-width",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--target-exclusion-margin",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--zero-ranking-margin",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--zero-ranking-weight",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--zero-band-weight",
        type=float,
        default=0.10,
    )

    # --------------------------------------------------------
    # 全局最难负峰损失
    # --------------------------------------------------------

    parser.add_argument(
        "--positive-range-radius",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--positive-velocity-radius",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--hard-negative-margin",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--hard-negative-weight",
        type=float,
        default=1.00,
    )

    parser.add_argument(
        "--notch-aux-weight",
        type=float,
        default=0.30,
    )

    # --------------------------------------------------------
    # 平移增强
    # --------------------------------------------------------

    parser.add_argument(
        "--shift-probability",
        type=float,
        default=0.80,
    )

    parser.add_argument(
        "--max-range-shift",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--max-velocity-shift",
        type=int,
        default=12,
    )

    # --------------------------------------------------------
    # 局部精修
    # --------------------------------------------------------

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
        "--invalid-refiner-weight",
        type=float,
        default=0.0,
        help=(
            "超出局部精修范围样本的回归损失权重；"
            "建议保持0"
        ),
    )

    return parser.parse_args()


def seed_everything(seed: int) -> None:
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


def shift_zero_fill(
    tensor: torch.Tensor,
    velocity_shift: int,
    range_shift: int,
) -> torch.Tensor:
    """
    对[C, V, R]张量做零填充平移。

    velocity_shift > 0：
        内容向速度下标增大的方向移动。

    range_shift > 0：
        内容向距离下标增大的方向移动。

    不使用torch.roll，避免首尾回绕。
    """

    if tensor.ndim != 3:
        raise ValueError(
            "shift_zero_fill要求输入形状为[C,V,R]。"
        )

    channel_count = tensor.shape[0]
    velocity_size = tensor.shape[1]
    range_size = tensor.shape[2]

    output = torch.zeros_like(tensor)

    source_velocity_start = max(
        0,
        -velocity_shift,
    )

    source_velocity_end = min(
        velocity_size,
        velocity_size - velocity_shift,
    )

    destination_velocity_start = max(
        0,
        velocity_shift,
    )

    destination_velocity_end = min(
        velocity_size,
        velocity_size + velocity_shift,
    )

    source_range_start = max(
        0,
        -range_shift,
    )

    source_range_end = min(
        range_size,
        range_size - range_shift,
    )

    destination_range_start = max(
        0,
        range_shift,
    )

    destination_range_end = min(
        range_size,
        range_size + range_shift,
    )

    if (
        source_velocity_end
        <= source_velocity_start
        or source_range_end
        <= source_range_start
    ):
        return output

    output[
        :channel_count,
        destination_velocity_start:
            destination_velocity_end,
        destination_range_start:
            destination_range_end,
    ] = tensor[
        :channel_count,
        source_velocity_start:
            source_velocity_end,
        source_range_start:
            source_range_end,
    ]

    return output


class V2RadarDataset(Dataset):
    """
    训练集：
        随机距离—速度平移增强。

    验证集/测试集：
        不增强。

    每个样本同时生成：
        raw_input
        notch_input
    """

    def __init__(
        self,
        base_dataset: RadarDataset,
        training: bool,
        notch_sigma: float,
        notch_floor: float,
        shift_probability: float,
        max_range_shift: int,
        max_velocity_shift: int,
    ):
        self.base_dataset = base_dataset
        self.training = bool(training)

        self.notch_sigma = float(
            notch_sigma
        )

        self.notch_floor = float(
            notch_floor
        )

        self.shift_probability = float(
            shift_probability
        )

        self.max_range_shift = int(
            max_range_shift
        )

        self.max_velocity_shift = int(
            max_velocity_shift
        )

        if self.notch_sigma <= 0:
            raise ValueError(
                "notch_sigma必须大于0。"
            )

        if not 0 <= self.notch_floor <= 1:
            raise ValueError(
                "notch_floor必须在[0,1]之间。"
            )

        if not 0 <= self.shift_probability <= 1:
            raise ValueError(
                "shift_probability必须在[0,1]之间。"
            )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def sample_id_at(
        self,
        index: int,
    ) -> str:
        sample = self.base_dataset[
            index
        ]

        return str(
            sample.get(
                "sample_id",
                f"sample_{index:04d}",
            )
        )

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        original = self.base_dataset[
            index
        ]

        sample = dict(original)

        raw_input = (
            original["input"]
            .float()
            .clone()
        )

        target = (
            original["target"]
            .float()
            .clone()
        )

        true_range = int(
            original["range_index"]
            .item()
            if isinstance(
                original["range_index"],
                torch.Tensor,
            )
            else original["range_index"]
        )

        true_velocity = int(
            original["velocity_index"]
            .item()
            if isinstance(
                original["velocity_index"],
                torch.Tensor,
            )
            else original["velocity_index"]
        )

        velocity_size = raw_input.shape[-2]
        range_size = raw_input.shape[-1]

        range_shift = 0
        velocity_shift = 0

        if (
            self.training
            and random.random()
            < self.shift_probability
        ):
            minimum_range_shift = max(
                -self.max_range_shift,
                -true_range,
            )

            maximum_range_shift = min(
                self.max_range_shift,
                range_size - 1 - true_range,
            )

            minimum_velocity_shift = max(
                -self.max_velocity_shift,
                -true_velocity,
            )

            maximum_velocity_shift = min(
                self.max_velocity_shift,
                velocity_size - 1 - true_velocity,
            )

            range_shift = random.randint(
                minimum_range_shift,
                maximum_range_shift,
            )

            velocity_shift = random.randint(
                minimum_velocity_shift,
                maximum_velocity_shift,
            )

            raw_input = shift_zero_fill(
                raw_input,
                velocity_shift=
                    velocity_shift,
                range_shift=
                    range_shift,
            )

            target = shift_zero_fill(
                target,
                velocity_shift=
                    velocity_shift,
                range_shift=
                    range_shift,
            )

            true_range += range_shift
            true_velocity += velocity_shift

        velocity_axis = torch.arange(
            velocity_size,
            dtype=raw_input.dtype,
        )

        zero_center = (
            velocity_size // 2
        )

        zero_gaussian = torch.exp(
            -0.5
            * (
                (
                    velocity_axis
                    - zero_center
                )
                / self.notch_sigma
            ).pow(2)
        )

        attenuation = (
            1.0
            - (
                1.0
                - self.notch_floor
            )
            * zero_gaussian
        ).view(
            1,
            velocity_size,
            1,
        )

        notch_input = (
            raw_input
            * attenuation
        )

        sample["input"] = raw_input
        sample["raw_input"] = raw_input
        sample["notch_input"] = (
            notch_input
        )
        sample["target"] = target

        sample["range_index"] = (
            torch.tensor(
                true_range,
                dtype=torch.long,
            )
        )

        sample["velocity_index"] = (
            torch.tensor(
                true_velocity,
                dtype=torch.long,
            )
        )

        sample["range_shift"] = (
            torch.tensor(
                range_shift,
                dtype=torch.long,
            )
        )

        sample["velocity_shift"] = (
            torch.tensor(
                velocity_shift,
                dtype=torch.long,
            )
        )

        return sample


def parse_session_and_beam(
    sample_id: str,
) -> tuple[str, str]:
    sample_id = str(sample_id)

    session_id = re.sub(
        r"_beam\d+$",
        "",
        sample_id,
    )

    match = re.search(
        r"_(beam\d+)$",
        sample_id,
    )

    beam = (
        match.group(1)
        if match
        else "unknown"
    )

    return session_id, beam


def create_balanced_sampler(
    dataset: V2RadarDataset,
    seed: int,
) -> WeightedRandomSampler:
    sample_ids = [
        dataset.sample_id_at(index)
        for index in range(len(dataset))
    ]

    parsed = [
        parse_session_and_beam(
            sample_id
        )
        for sample_id in sample_ids
    ]

    session_counts = Counter(
        session_id
        for session_id, unused_beam
        in parsed
    )

    beam_counts = Counter(
        beam
        for unused_session, beam
        in parsed
    )

    weights = []

    for session_id, beam in parsed:
        weight = (
            1.0
            / float(
                session_counts[
                    session_id
                ]
            )
            / float(
                beam_counts[
                    beam
                ]
            )
        )

        weights.append(weight)

    weight_tensor = torch.tensor(
        weights,
        dtype=torch.double,
    )

    # 归一化不影响WeightedRandomSampler，
    # 但便于检查数值。
    weight_tensor = (
        weight_tensor
        / weight_tensor.mean()
    )

    generator = (
        torch.Generator()
        .manual_seed(seed)
    )

    sampler = WeightedRandomSampler(
        weights=weight_tensor,
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )

    print(
        "训练平衡采样："
        f"{len(session_counts)}个session，"
        f"{len(beam_counts)}类beam"
    )

    print(
        "采样权重范围："
        f"{weight_tensor.min().item():.4f}"
        "～"
        f"{weight_tensor.max().item():.4f}"
    )

    return sampler


class V2DualLoss(nn.Module):
    """
    损失组成：

    1. 融合热力图基础损失
    2. 陷波分支辅助损失
    3. 全图最强错误峰排名损失
    4. 零多普勒错误峰排名损失
    5. 零多普勒带能量惩罚
    """

    def __init__(
        self,
        args: argparse.Namespace,
    ):
        super().__init__()

        self.base_loss = HeatmapLoss()

        self.positive_range_radius = int(
            args.positive_range_radius
        )

        self.positive_velocity_radius = int(
            args.positive_velocity_radius
        )

        self.hard_negative_margin = float(
            args.hard_negative_margin
        )

        self.hard_negative_weight = float(
            args.hard_negative_weight
        )

        self.zero_half_width = int(
            args.zero_band_half_width
        )

        self.target_exclusion_margin = int(
            args.target_exclusion_margin
        )

        self.zero_ranking_margin = float(
            args.zero_ranking_margin
        )

        self.zero_ranking_weight = float(
            args.zero_ranking_weight
        )

        self.zero_band_weight = float(
            args.zero_band_weight
        )

        self.notch_aux_weight = float(
            args.notch_aux_weight
        )

    def global_hard_negative_loss(
        self,
        probability: torch.Tensor,
        true_range: torch.Tensor,
        true_velocity: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = probability.shape[0]
        velocity_size = probability.shape[-2]
        range_size = probability.shape[-1]

        ranking_losses = []

        for index in range(batch_size):
            center_range = int(
                true_range[index].item()
            )

            center_velocity = int(
                true_velocity[index].item()
            )

            range_start = max(
                0,
                center_range
                - self.positive_range_radius,
            )

            range_end = min(
                range_size,
                center_range
                + self.positive_range_radius
                + 1,
            )

            velocity_start = max(
                0,
                center_velocity
                - self.positive_velocity_radius,
            )

            velocity_end = min(
                velocity_size,
                center_velocity
                + self.positive_velocity_radius
                + 1,
            )

            current_map = probability[
                index,
                0,
            ]

            positive_score = current_map[
                velocity_start:
                    velocity_end,
                range_start:
                    range_end,
            ].amax()

            positive_mask = torch.zeros_like(
                current_map,
                dtype=torch.bool,
            )

            positive_mask[
                velocity_start:
                    velocity_end,
                range_start:
                    range_end,
            ] = True

            negative_map = (
                current_map.masked_fill(
                    positive_mask,
                    -1.0,
                )
            )

            hardest_negative_score = (
                negative_map.amax()
            )

            ranking_losses.append(
                F.relu(
                    self.hard_negative_margin
                    + hardest_negative_score
                    - positive_score
                )
            )

        return torch.stack(
            ranking_losses
        ).mean()

    def zero_doppler_losses(
        self,
        probability: torch.Tensor,
        true_range: torch.Tensor,
        true_velocity: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        velocity_size = probability.shape[-2]
        zero_center = velocity_size // 2

        lower = max(
            0,
            zero_center
            - self.zero_half_width,
        )

        upper = min(
            velocity_size,
            zero_center
            + self.zero_half_width
            + 1,
        )

        eligible = (
            true_velocity
            .sub(zero_center)
            .abs()
            > self.target_exclusion_margin
        )

        eligible_indices = (
            eligible.nonzero(
                as_tuple=False
            ).flatten()
        )

        if eligible_indices.numel() == 0:
            zero = probability.new_zeros(
                ()
            )
            return zero, zero

        zero_band = probability[
            eligible_indices,
            0,
            lower:upper,
            :,
        ].flatten(
            start_dim=1
        )

        zero_band_max = (
            zero_band
            .amax(dim=1)
        )

        target_scores = probability[
            eligible_indices,
            0,
            true_velocity[
                eligible_indices
            ],
            true_range[
                eligible_indices
            ],
        ]

        ranking_loss = F.relu(
            self.zero_ranking_margin
            + zero_band_max
            - target_scores
        ).mean()

        band_energy_loss = (
            zero_band.pow(2).mean()
        )

        return (
            ranking_loss,
            band_energy_loss,
        )

    def forward(
        self,
        fused_logits: torch.Tensor,
        notch_logits: torch.Tensor,
        target: torch.Tensor,
        true_range: torch.Tensor,
        true_velocity: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        fused_base = self.base_loss(
            fused_logits,
            target,
        )

        notch_base = self.base_loss(
            notch_logits,
            target,
        )

        probability = torch.sigmoid(
            fused_logits
        )

        hard_negative = (
            self.global_hard_negative_loss(
                probability,
                true_range,
                true_velocity,
            )
        )

        (
            zero_ranking,
            zero_band_energy,
        ) = self.zero_doppler_losses(
            probability,
            true_range,
            true_velocity,
        )

        total = (
            fused_base
            + self.notch_aux_weight
            * notch_base
            + self.hard_negative_weight
            * hard_negative
            + self.zero_ranking_weight
            * zero_ranking
            + self.zero_band_weight
            * zero_band_energy
        )

        components = {
            "fused_base":
                fused_base.detach(),
            "notch_base":
                notch_base.detach(),
            "hard_negative":
                hard_negative.detach(),
            "zero_ranking":
                zero_ranking.detach(),
            "zero_band_energy":
                zero_band_energy.detach(),
        }

        return total, components


def new_metric_accumulator() -> dict[str, float]:
    result = {
        "sample_count": 0.0,
        "range_error_sum": 0.0,
        "velocity_error_sum": 0.0,
        "zero_eligible_count": 0.0,
        "zero_false_peak_count": 0.0,
    }

    for metric_name in METRIC_RADII:
        result[
            f"{metric_name}_hit_count"
        ] = 0.0

    return result


def update_metrics(
    accumulator: dict[str, float],
    logits: torch.Tensor,
    true_range: torch.Tensor,
    true_velocity: torch.Tensor,
    zero_half_width: int,
    zero_exclusion: int,
) -> None:
    (
        predicted_range,
        predicted_velocity,
    ) = extract_peaks(logits)

    true_range = (
        true_range.long().reshape(-1)
    )

    true_velocity = (
        true_velocity.long().reshape(-1)
    )

    range_error = (
        predicted_range
        - true_range
    ).abs()

    velocity_error = (
        predicted_velocity
        - true_velocity
    ).abs()

    sample_count = int(
        true_range.numel()
    )

    accumulator[
        "sample_count"
    ] += sample_count

    accumulator[
        "range_error_sum"
    ] += float(
        range_error.sum().item()
    )

    accumulator[
        "velocity_error_sum"
    ] += float(
        velocity_error.sum().item()
    )

    for metric_name, radii in (
        METRIC_RADII.items()
    ):
        range_radius = radii[0]
        velocity_radius = radii[1]

        hits = (
            (
                range_error
                <= range_radius
            )
            & (
                velocity_error
                <= velocity_radius
            )
        )

        accumulator[
            f"{metric_name}_hit_count"
        ] += float(
            hits.sum().item()
        )

    zero_center = (
        logits.shape[-2] // 2
    )

    eligible = (
        true_velocity
        .sub(zero_center)
        .abs()
        > zero_exclusion
    )

    false_zero_peak = (
        eligible
        & (
            predicted_velocity
            .sub(zero_center)
            .abs()
            <= zero_half_width
        )
    )

    accumulator[
        "zero_eligible_count"
    ] += float(
        eligible.sum().item()
    )

    accumulator[
        "zero_false_peak_count"
    ] += float(
        false_zero_peak.sum().item()
    )


def finalize_metrics(
    accumulator: dict[str, float],
) -> dict[str, float]:
    sample_count = accumulator[
        "sample_count"
    ]

    if sample_count <= 0:
        raise RuntimeError(
            "没有可评价样本。"
        )

    eligible_count = accumulator[
        "zero_eligible_count"
    ]

    result = {
        "mean_range_error": (
            accumulator[
                "range_error_sum"
            ]
            / sample_count
        ),
        "mean_velocity_error": (
            accumulator[
                "velocity_error_sum"
            ]
            / sample_count
        ),
        "zero_false_peak_count": (
            accumulator[
                "zero_false_peak_count"
            ]
        ),
        "zero_eligible_count":
            eligible_count,
        "zero_false_peak_rate": (
            accumulator[
                "zero_false_peak_count"
            ]
            / eligible_count
            if eligible_count > 0
            else 0.0
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


def dual_model_score(
    metrics: dict[str, float],
) -> tuple[float, ...]:
    return (
        metrics[
            "application_hit_rate"
        ],
        metrics[
            "relaxed_hit_rate"
        ],
        metrics[
            "strict_hit_rate"
        ],
        -metrics[
            "zero_false_peak_rate"
        ],
        -metrics[
            "mean_velocity_error"
        ],
        -metrics[
            "mean_range_error"
        ],
        -metrics["loss"],
    )


def run_dual_epoch(
    model: DualViewFCN,
    data_loader: DataLoader,
    loss_function: V2DualLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    args: argparse.Namespace,
) -> dict[str, float]:
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    accumulator = (
        new_metric_accumulator()
    )

    loss_sum = 0.0

    component_sums = {
        "fused_base": 0.0,
        "notch_base": 0.0,
        "hard_negative": 0.0,
        "zero_ranking": 0.0,
        "zero_band_energy": 0.0,
    }

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

            target = batch[
                "target"
            ].to(
                device,
                non_blocking=True,
            )

            true_range = (
                batch["range_index"]
                .long()
                .reshape(-1)
                .to(device)
            )

            true_velocity = (
                batch["velocity_index"]
                .long()
                .reshape(-1)
                .to(device)
            )

            if training:
                optimizer.zero_grad(
                    set_to_none=True
                )

            (
                unused_raw_logits,
                notch_logits,
                fused_logits,
            ) = model(
                raw_input,
                notch_input,
            )

            loss, components = (
                loss_function(
                    fused_logits,
                    notch_logits,
                    target,
                    true_range,
                    true_velocity,
                )
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "双分支损失出现NaN或无穷值。"
                )

            if training:
                loss.backward()

                trainable_parameters = [
                    parameter
                    for parameter
                    in model.parameters()
                    if parameter.requires_grad
                ]

                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    max_norm=5.0,
                )

                optimizer.step()

            batch_size = int(
                raw_input.shape[0]
            )

            loss_sum += (
                float(loss.item())
                * batch_size
            )

            for key in component_sums:
                component_sums[key] += (
                    float(
                        components[
                            key
                        ].item()
                    )
                    * batch_size
                )

            update_metrics(
                accumulator,
                fused_logits.detach(),
                true_range,
                true_velocity,
                zero_half_width=
                    args.zero_band_half_width,
                zero_exclusion=
                    args.target_exclusion_margin,
            )

    metrics = finalize_metrics(
        accumulator
    )

    sample_count = accumulator[
        "sample_count"
    ]

    metrics["loss"] = (
        loss_sum / sample_count
    )

    for key, value in (
        component_sums.items()
    ):
        metrics[
            f"{key}_loss"
        ] = value / sample_count

    return metrics


def load_initial_dual_model(
    pretrained_path: Path,
    device: torch.device,
) -> DualViewFCN:
    if not pretrained_path.exists():
        raise FileNotFoundError(
            f"找不到预训练模型：{pretrained_path}"
        )

    checkpoint = load_torch(
        pretrained_path,
        device,
    )

    pretrained_state = get_state(
        checkpoint
    )

    return DualViewFCN(
        pretrained_state
    ).to(device)


def train_dual_model(
    args: argparse.Namespace,
    device: torch.device,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    checkpoint_dir: Path,
    table_dir: Path,
) -> tuple[
    DualViewFCN,
    dict[str, Any],
]:
    model = load_initial_dual_model(
        args.pretrained,
        device,
    )

    loss_function = V2DualLoss(
        args
    )

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.dual_learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=8,
            min_lr=1.0e-6,
        )
    )

    best_path = (
        checkpoint_dir
        / "best_dual.pt"
    )

    last_path = (
        checkpoint_dir
        / "last_dual.pt"
    )

    history_path = (
        table_dir
        / "dual_history.csv"
    )

    best_score = None
    best_epoch = 0
    stale_epochs = 0
    history = []

    print("\n" + "=" * 74)
    print("阶段一：训练双分支抗杂波v2模型")
    print("=" * 74)

    print(
        "双分支可训练参数量："
        f"{count_trainable_parameters(model):,}"
    )

    for epoch in range(
        1,
        args.dual_epochs + 1,
    ):
        train_metrics = run_dual_epoch(
            model=model,
            data_loader=train_loader,
            loss_function=loss_function,
            device=device,
            optimizer=optimizer,
            args=args,
        )

        validation_metrics = (
            run_dual_epoch(
                model=model,
                data_loader=
                    validation_loader,
                loss_function=
                    loss_function,
                device=device,
                optimizer=None,
                args=args,
            )
        )

        scheduler.step(
            validation_metrics["loss"]
        )

        current_score = (
            dual_model_score(
                validation_metrics
            )
        )

        is_best = (
            best_score is None
            or current_score > best_score
        )

        learning_rate = float(
            optimizer.param_groups[
                0
            ]["lr"]
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
            "configuration":
                vars(args),
        }

        torch.save(
            checkpoint,
            last_path,
        )

        if is_best:
            best_score = current_score
            best_epoch = epoch
            stale_epochs = 0

            torch.save(
                checkpoint,
                best_path,
            )
        else:
            stale_epochs += 1

        history.append(
            {
                "epoch": epoch,
                "learning_rate":
                    learning_rate,
                **{
                    f"train_{key}":
                        value
                    for key, value
                    in train_metrics.items()
                },
                **{
                    f"val_{key}":
                        value
                    for key, value
                    in validation_metrics.items()
                },
                "is_best": is_best,
            }
        )

        pd.DataFrame(
            history
        ).to_csv(
            history_path,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"双分支第{epoch:03d}轮｜"
            f"验证损失="
            f"{validation_metrics['loss']:.5f}｜"
            f"严格="
            f"{validation_metrics['strict_hit_rate']:.2%}｜"
            f"宽松="
            f"{validation_metrics['relaxed_hit_rate']:.2%}｜"
            f"应用="
            f"{validation_metrics['application_hit_rate']:.2%}｜"
            f"距离="
            f"{validation_metrics['mean_range_error']:.3f}门｜"
            f"速度="
            f"{validation_metrics['mean_velocity_error']:.3f}单元｜"
            f"难负峰损失="
            f"{validation_metrics['hard_negative_loss']:.4f}｜"
            f"零多普勒误峰="
            f"{int(validation_metrics['zero_false_peak_count'])}｜"
            f"学习率={learning_rate:.2e}"
            + (
                "｜最佳模型"
                if is_best
                else ""
            )
        )

        if (
            stale_epochs
            >= args.dual_early_stopping
        ):
            print(
                "\n双分支模型连续"
                f"{args.dual_early_stopping}"
                "轮没有改进，提前停止。"
            )
            break

    best_checkpoint = load_torch(
        best_path,
        device,
    )

    model.load_state_dict(
        best_checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    print(
        f"\n双分支最佳轮次：{best_epoch}"
    )

    return model, best_checkpoint


def predict_refined_coordinates(
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
            coarse_range,
            coarse_velocity,
        ) = extract_peaks(
            fused_logits
        )

    feature_map = torch.cat(
        [
            raw_input,
            fused_probability,
        ],
        dim=1,
    )

    local_patches = extract_local_patches(
        feature_map=feature_map,
        center_range_indices=
            coarse_range,
        center_velocity_indices=
            coarse_velocity,
        crop_size=crop_size,
    )

    normalized_offsets = refiner(
        local_patches
    )

    range_offsets = (
        normalized_offsets[:, 0]
        * max_range_offset
    )

    velocity_offsets = (
        normalized_offsets[:, 1]
        * max_velocity_offset
    )

    refined_range = (
        coarse_range.float()
        + torch.round(
            range_offsets
        )
    ).long()

    refined_velocity = (
        coarse_velocity.float()
        + torch.round(
            velocity_offsets
        )
    ).long()

    refined_range = torch.clamp(
        refined_range,
        min=0,
        max=raw_input.shape[-1] - 1,
    )

    refined_velocity = torch.clamp(
        refined_velocity,
        min=0,
        max=raw_input.shape[-2] - 1,
    )

    return {
        "raw_logits": raw_logits,
        "notch_logits": notch_logits,
        "fused_logits": fused_logits,
        "fused_probability":
            fused_probability,
        "coarse_range":
            coarse_range,
        "coarse_velocity":
            coarse_velocity,
        "normalized_offsets":
            normalized_offsets,
        "range_offsets":
            range_offsets,
        "velocity_offsets":
            velocity_offsets,
        "refined_range":
            refined_range,
        "refined_velocity":
            refined_velocity,
    }


def update_coordinate_metrics(
    accumulator: dict[str, float],
    predicted_range: torch.Tensor,
    predicted_velocity: torch.Tensor,
    true_range: torch.Tensor,
    true_velocity: torch.Tensor,
    zero_half_width: int,
    zero_exclusion: int,
) -> None:
    range_error = (
        predicted_range
        - true_range
    ).abs()

    velocity_error = (
        predicted_velocity
        - true_velocity
    ).abs()

    sample_count = int(
        true_range.numel()
    )

    accumulator[
        "sample_count"
    ] += sample_count

    accumulator[
        "range_error_sum"
    ] += float(
        range_error.sum().item()
    )

    accumulator[
        "velocity_error_sum"
    ] += float(
        velocity_error.sum().item()
    )

    for metric_name, radii in (
        METRIC_RADII.items()
    ):
        hits = (
            (
                range_error
                <= radii[0]
            )
            & (
                velocity_error
                <= radii[1]
            )
        )

        accumulator[
            f"{metric_name}_hit_count"
        ] += float(
            hits.sum().item()
        )

    zero_center = 64

    eligible = (
        true_velocity
        .sub(zero_center)
        .abs()
        > zero_exclusion
    )

    zero_false_peak = (
        eligible
        & (
            predicted_velocity
            .sub(zero_center)
            .abs()
            <= zero_half_width
        )
    )

    accumulator[
        "zero_eligible_count"
    ] += float(
        eligible.sum().item()
    )

    accumulator[
        "zero_false_peak_count"
    ] += float(
        zero_false_peak.sum().item()
    )


def run_refiner_epoch(
    backbone: DualViewFCN,
    refiner: LocalOffsetRefiner,
    data_loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    training = optimizer is not None

    backbone.eval()

    if training:
        refiner.train()
    else:
        refiner.eval()

    coarse_accumulator = (
        new_metric_accumulator()
    )

    refined_accumulator = (
        new_metric_accumulator()
    )

    valid_loss_sum = 0.0
    valid_sample_count = 0
    invalid_sample_count = 0

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

            true_range = (
                batch["range_index"]
                .long()
                .reshape(-1)
                .to(device)
            )

            true_velocity = (
                batch["velocity_index"]
                .long()
                .reshape(-1)
                .to(device)
            )

            if training:
                optimizer.zero_grad(
                    set_to_none=True
                )

            prediction = (
                predict_refined_coordinates(
                    backbone=backbone,
                    refiner=refiner,
                    raw_input=raw_input,
                    notch_input=
                        notch_input,
                    crop_size=
                        args.crop_size,
                    max_range_offset=
                        args.max_range_offset,
                    max_velocity_offset=
                        args.max_velocity_offset,
                )
            )

            true_range_offset = (
                true_range.float()
                - prediction[
                    "coarse_range"
                ].float()
            )

            true_velocity_offset = (
                true_velocity.float()
                - prediction[
                    "coarse_velocity"
                ].float()
            )

            valid_mask = (
                (
                    true_range_offset.abs()
                    <= args.max_range_offset
                )
                & (
                    true_velocity_offset.abs()
                    <= args.max_velocity_offset
                )
            )

            normalized_target = torch.stack(
                [
                    true_range_offset
                    / args.max_range_offset,
                    true_velocity_offset
                    / args.max_velocity_offset,
                ],
                dim=1,
            )

            normalized_target = (
                torch.clamp(
                    normalized_target,
                    min=-1.0,
                    max=1.0,
                )
            )

            per_element_loss = (
                F.smooth_l1_loss(
                    prediction[
                        "normalized_offsets"
                    ],
                    normalized_target,
                    beta=0.25,
                    reduction="none",
                )
            )

            per_sample_loss = (
                per_element_loss.mean(
                    dim=1
                )
            )

            valid_weights = (
                valid_mask.float()
            )

            if (
                args.invalid_refiner_weight
                > 0
            ):
                valid_weights = torch.where(
                    valid_mask,
                    torch.ones_like(
                        valid_weights
                    ),
                    torch.full_like(
                        valid_weights,
                        args.invalid_refiner_weight,
                    ),
                )

            weight_sum = (
                valid_weights.sum()
            )

            if weight_sum.item() > 0:
                loss = (
                    per_sample_loss
                    * valid_weights
                ).sum() / weight_sum

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        "精修器损失出现NaN或无穷值。"
                    )

                if training:
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        refiner.parameters(),
                        max_norm=5.0,
                    )

                    optimizer.step()

                true_valid_count = int(
                    valid_mask.sum().item()
                )

                valid_loss_sum += (
                    float(loss.item())
                    * max(
                        true_valid_count,
                        1,
                    )
                )

                valid_sample_count += (
                    true_valid_count
                )

            invalid_sample_count += int(
                (
                    ~valid_mask
                ).sum().item()
            )

            update_coordinate_metrics(
                accumulator=
                    coarse_accumulator,
                predicted_range=
                    prediction[
                        "coarse_range"
                    ],
                predicted_velocity=
                    prediction[
                        "coarse_velocity"
                    ],
                true_range=true_range,
                true_velocity=
                    true_velocity,
                zero_half_width=
                    args.zero_band_half_width,
                zero_exclusion=
                    args.target_exclusion_margin,
            )

            update_coordinate_metrics(
                accumulator=
                    refined_accumulator,
                predicted_range=
                    prediction[
                        "refined_range"
                    ],
                predicted_velocity=
                    prediction[
                        "refined_velocity"
                    ],
                true_range=true_range,
                true_velocity=
                    true_velocity,
                zero_half_width=
                    args.zero_band_half_width,
                zero_exclusion=
                    args.target_exclusion_margin,
            )

    coarse_metrics = finalize_metrics(
        coarse_accumulator
    )

    refined_metrics = finalize_metrics(
        refined_accumulator
    )

    loss = (
        valid_loss_sum
        / valid_sample_count
        if valid_sample_count > 0
        else 0.0
    )

    return {
        "loss": loss,
        "valid_sample_count":
            valid_sample_count,
        "invalid_sample_count":
            invalid_sample_count,
        "coarse": coarse_metrics,
        "refined": refined_metrics,
    }


def refiner_model_score(
    metrics: dict[str, Any],
) -> tuple[float, ...]:
    refined = metrics["refined"]

    return (
        refined[
            "application_hit_rate"
        ],
        refined[
            "relaxed_hit_rate"
        ],
        refined[
            "strict_hit_rate"
        ],
        -refined[
            "zero_false_peak_rate"
        ],
        -refined[
            "mean_velocity_error"
        ],
        -refined[
            "mean_range_error"
        ],
        -metrics["loss"],
    )


def train_refiner(
    backbone: DualViewFCN,
    args: argparse.Namespace,
    device: torch.device,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    checkpoint_dir: Path,
    table_dir: Path,
) -> tuple[
    LocalOffsetRefiner,
    dict[str, Any],
]:
    refiner = LocalOffsetRefiner().to(
        device
    )

    optimizer = torch.optim.AdamW(
        refiner.parameters(),
        lr=args.refiner_learning_rate,
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

    best_path = (
        checkpoint_dir
        / "best_refiner.pt"
    )

    last_path = (
        checkpoint_dir
        / "last_refiner.pt"
    )

    history_path = (
        table_dir
        / "refiner_history.csv"
    )

    best_score = None
    best_epoch = 0
    stale_epochs = 0
    history = []

    print("\n" + "=" * 74)
    print("阶段二：训练有效范围局部精修器v2")
    print("=" * 74)

    print(
        "精修器可训练参数量："
        f"{count_trainable_parameters(refiner):,}"
    )

    for epoch in range(
        1,
        args.refiner_epochs + 1,
    ):
        train_metrics = (
            run_refiner_epoch(
                backbone=backbone,
                refiner=refiner,
                data_loader=train_loader,
                device=device,
                optimizer=optimizer,
                args=args,
            )
        )

        validation_metrics = (
            run_refiner_epoch(
                backbone=backbone,
                refiner=refiner,
                data_loader=
                    validation_loader,
                device=device,
                optimizer=None,
                args=args,
            )
        )

        scheduler.step(
            validation_metrics["loss"]
        )

        score = refiner_model_score(
            validation_metrics
        )

        is_best = (
            best_score is None
            or score > best_score
        )

        learning_rate = float(
            optimizer.param_groups[
                0
            ]["lr"]
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
            "configuration":
                vars(args),
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
                "epoch": epoch,
                "learning_rate":
                    learning_rate,
                "train_loss":
                    train_metrics["loss"],
                "train_valid_sample_count":
                    train_metrics[
                        "valid_sample_count"
                    ],
                "train_invalid_sample_count":
                    train_metrics[
                        "invalid_sample_count"
                    ],
                "train_refined_strict":
                    train_refined[
                        "strict_hit_rate"
                    ],
                "train_refined_relaxed":
                    train_refined[
                        "relaxed_hit_rate"
                    ],
                "val_loss":
                    validation_metrics[
                        "loss"
                    ],
                "val_valid_sample_count":
                    validation_metrics[
                        "valid_sample_count"
                    ],
                "val_invalid_sample_count":
                    validation_metrics[
                        "invalid_sample_count"
                    ],
                "val_coarse_strict":
                    validation_coarse[
                        "strict_hit_rate"
                    ],
                "val_coarse_relaxed":
                    validation_coarse[
                        "relaxed_hit_rate"
                    ],
                "val_refined_strict":
                    validation_refined[
                        "strict_hit_rate"
                    ],
                "val_refined_relaxed":
                    validation_refined[
                        "relaxed_hit_rate"
                    ],
                "val_refined_application":
                    validation_refined[
                        "application_hit_rate"
                    ],
                "val_refined_range_error":
                    validation_refined[
                        "mean_range_error"
                    ],
                "val_refined_velocity_error":
                    validation_refined[
                        "mean_velocity_error"
                    ],
                "val_zero_false_peak_count":
                    validation_refined[
                        "zero_false_peak_count"
                    ],
                "is_best": is_best,
            }
        )

        pd.DataFrame(
            history
        ).to_csv(
            history_path,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"精修第{epoch:03d}轮｜"
            f"验证损失="
            f"{validation_metrics['loss']:.5f}｜"
            f"有效/无效="
            f"{validation_metrics['valid_sample_count']}/"
            f"{validation_metrics['invalid_sample_count']}｜"
            f"粗定位严格="
            f"{validation_coarse['strict_hit_rate']:.2%}｜"
            f"精修严格="
            f"{validation_refined['strict_hit_rate']:.2%}｜"
            f"精修宽松="
            f"{validation_refined['relaxed_hit_rate']:.2%}｜"
            f"应用="
            f"{validation_refined['application_hit_rate']:.2%}｜"
            f"距离="
            f"{validation_refined['mean_range_error']:.3f}门｜"
            f"速度="
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
            >= args.refiner_early_stopping
        ):
            print(
                "\n精修器连续"
                f"{args.refiner_early_stopping}"
                "轮没有改进，提前停止。"
            )
            break

    best_checkpoint = load_torch(
        best_path,
        device,
    )

    refiner.load_state_dict(
        best_checkpoint[
            "refiner_state_dict"
        ]
    )

    refiner.eval()

    print(
        f"\n精修器最佳轮次：{best_epoch}"
    )

    return refiner, best_checkpoint


def normalize_sample_ids(
    raw_sample_ids: Any,
    batch_size: int,
    start_index: int,
) -> list[str]:
    if raw_sample_ids is None:
        return [
            f"test_{start_index + index:04d}"
            for index in range(batch_size)
        ]

    if isinstance(
        raw_sample_ids,
        str,
    ):
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
    args: argparse.Namespace,
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

            true_range = (
                batch["range_index"]
                .long()
                .reshape(-1)
                .to(device)
            )

            true_velocity = (
                batch["velocity_index"]
                .long()
                .reshape(-1)
                .to(device)
            )

            prediction = (
                predict_refined_coordinates(
                    backbone=backbone,
                    refiner=refiner,
                    raw_input=raw_input,
                    notch_input=
                        notch_input,
                    crop_size=
                        args.crop_size,
                    max_range_offset=
                        args.max_range_offset,
                    max_velocity_offset=
                        args.max_velocity_offset,
                )
            )

            batch_size = int(
                raw_input.shape[0]
            )

            sample_ids = (
                normalize_sample_ids(
                    batch.get(
                        "sample_id"
                    ),
                    batch_size,
                    global_index,
                )
            )

            for index in range(
                batch_size
            ):
                sample_id = sample_ids[
                    index
                ]

                session_id, beam = (
                    parse_session_and_beam(
                        sample_id
                    )
                )

                true_r = int(
                    true_range[
                        index
                    ].item()
                )

                true_v = int(
                    true_velocity[
                        index
                    ].item()
                )

                coarse_r = int(
                    prediction[
                        "coarse_range"
                    ][index].item()
                )

                coarse_v = int(
                    prediction[
                        "coarse_velocity"
                    ][index].item()
                )

                refined_r = int(
                    prediction[
                        "refined_range"
                    ][index].item()
                )

                refined_v = int(
                    prediction[
                        "refined_velocity"
                    ][index].item()
                )

                coarse_range_error = abs(
                    coarse_r - true_r
                )

                coarse_velocity_error = abs(
                    coarse_v - true_v
                )

                refined_range_error = abs(
                    refined_r - true_r
                )

                refined_velocity_error = abs(
                    refined_v - true_v
                )

                refiner_valid = (
                    coarse_range_error
                    <= args.max_range_offset
                    and coarse_velocity_error
                    <= args.max_velocity_offset
                )

                record = {
                    "sample_id":
                        sample_id,
                    "session_id":
                        session_id,
                    "beam":
                        beam,
                    "true_range_index":
                        true_r,
                    "true_velocity_index":
                        true_v,
                    "coarse_range_index":
                        coarse_r,
                    "coarse_velocity_index":
                        coarse_v,
                    "coarse_range_error":
                        coarse_range_error,
                    "coarse_velocity_error":
                        coarse_velocity_error,
                    "refiner_valid":
                        refiner_valid,
                    "predicted_range_offset":
                        float(
                            prediction[
                                "range_offsets"
                            ][index].item()
                        ),
                    "predicted_velocity_offset":
                        float(
                            prediction[
                                "velocity_offsets"
                            ][index].item()
                        ),
                    "refined_range_index":
                        refined_r,
                    "refined_velocity_index":
                        refined_v,
                    "refined_range_error":
                        refined_range_error,
                    "refined_velocity_error":
                        refined_velocity_error,
                    "coarse_in_zero_band":
                        abs(
                            coarse_v - 64
                        ) <= 3,
                    "refined_in_zero_band":
                        abs(
                            refined_v - 64
                        ) <= 3,
                }

                for metric_name, radii in (
                    METRIC_RADII.items()
                ):
                    record[
                        f"coarse_{metric_name}_hit"
                    ] = (
                        coarse_range_error
                        <= radii[0]
                        and coarse_velocity_error
                        <= radii[1]
                    )

                    record[
                        f"refined_{metric_name}_hit"
                    ] = (
                        refined_range_error
                        <= radii[0]
                        and refined_velocity_error
                        <= radii[1]
                    )

                records.append(record)

            global_index += batch_size

    return pd.DataFrame(records)


def serialize_arguments(
    args: argparse.Namespace,
) -> dict[str, Any]:
    result = {}

    for key, value in vars(
        args
    ).items():
        result[key] = (
            str(value)
            if isinstance(value, Path)
            else value
        )

    return result


def main() -> None:
    args = parse_arguments()

    if args.crop_size % 2 == 0:
        raise ValueError(
            "crop-size必须是奇数。"
        )

    if args.max_range_offset <= 0:
        raise ValueError(
            "max-range-offset必须大于0。"
        )

    if args.max_velocity_offset <= 0:
        raise ValueError(
            "max-velocity-offset必须大于0。"
        )

    seed_everything(
        args.seed
    )

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
        experiment_dir
        / "checkpoints"
    )

    table_dir = (
        experiment_dir
        / "tables"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    table_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print("v2雷达目标定位完整训练管线")
    print("=" * 78)

    print(f"设备：{device}")

    if torch.cuda.is_available():
        print(
            "GPU："
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Sigma预训练模型：{args.pretrained}"
    )

    print(
        "训练增强："
        f"距离±{args.max_range_shift}门，"
        f"速度±{args.max_velocity_shift}单元，"
        f"概率={args.shift_probability}"
    )

    print(
        "难负峰正区域："
        f"距离±{args.positive_range_radius}门，"
        f"速度±{args.positive_velocity_radius}单元"
    )

    def create_dataset(
        split: str,
        training: bool,
    ) -> V2RadarDataset:
        base_dataset = RadarDataset(
            split=split,
            range_sigma=3.0,
            velocity_sigma=1.0,
        )

        return V2RadarDataset(
            base_dataset=
                base_dataset,
            training=training,
            notch_sigma=
                args.notch_sigma,
            notch_floor=
                args.notch_floor,
            shift_probability=
                args.shift_probability,
            max_range_shift=
                args.max_range_shift,
            max_velocity_shift=
                args.max_velocity_shift,
        )

    print("\n正在加载数据集……")

    train_dataset = create_dataset(
        "train",
        training=True,
    )

    validation_dataset = create_dataset(
        "val",
        training=False,
    )

    test_dataset = create_dataset(
        "test",
        training=False,
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

    dual_sampler = (
        create_balanced_sampler(
            train_dataset,
            seed=args.seed,
        )
    )

    dual_train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=dual_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
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

    training_start = time.time()

    backbone, best_dual_checkpoint = (
        train_dual_model(
            args=args,
            device=device,
            train_loader=
                dual_train_loader,
            validation_loader=
                validation_loader,
            checkpoint_dir=
                checkpoint_dir,
            table_dir=table_dir,
        )
    )

    # 精修阶段重新创建Sampler，
    # 避免沿用双分支阶段已经消耗的随机状态。
    refiner_sampler = (
        create_balanced_sampler(
            train_dataset,
            seed=args.seed + 1000,
        )
    )

    refiner_train_loader = DataLoader(
        train_dataset,
        batch_size=
            args.refiner_batch_size,
        sampler=refiner_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    refiner_validation_loader = (
        DataLoader(
            validation_dataset,
            batch_size=
                args.refiner_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
    )

    refiner_test_loader = DataLoader(
        test_dataset,
        batch_size=
            args.refiner_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    refiner, best_refiner_checkpoint = (
        train_refiner(
            backbone=backbone,
            args=args,
            device=device,
            train_loader=
                refiner_train_loader,
            validation_loader=
                refiner_validation_loader,
            checkpoint_dir=
                checkpoint_dir,
            table_dir=table_dir,
        )
    )

    print("\n正在评价测试集……")

    dual_loss = V2DualLoss(
        args
    )

    test_dual_metrics = (
        run_dual_epoch(
            model=backbone,
            data_loader=test_loader,
            loss_function=dual_loss,
            device=device,
            optimizer=None,
            args=args,
        )
    )

    test_refiner_metrics = (
        run_refiner_epoch(
            backbone=backbone,
            refiner=refiner,
            data_loader=
                refiner_test_loader,
            device=device,
            optimizer=None,
            args=args,
        )
    )

    test_details = create_test_details(
        backbone=backbone,
        refiner=refiner,
        data_loader=
            refiner_test_loader,
        device=device,
        args=args,
    )

    test_details_path = (
        table_dir
        / "test_details.csv"
    )

    test_details.to_csv(
        test_details_path,
        index=False,
        encoding="utf-8-sig",
    )

    failures = test_details[
        ~test_details[
            "refined_relaxed_hit"
        ].astype(bool)
    ].copy()

    failures_path = (
        table_dir
        / "relaxed_failures.csv"
    )

    failures.to_csv(
        failures_path,
        index=False,
        encoding="utf-8-sig",
    )

    elapsed_seconds = (
        time.time()
        - training_start
    )

    summary = {
        "experiment_name":
            args.name,
        "training_seconds":
            elapsed_seconds,
        "best_dual_epoch":
            best_dual_checkpoint[
                "epoch"
            ],
        "best_refiner_epoch":
            best_refiner_checkpoint[
                "epoch"
            ],
        "best_dual_validation":
            best_dual_checkpoint[
                "validation_metrics"
            ],
        "best_refiner_validation":
            best_refiner_checkpoint[
                "validation_metrics"
            ],
        "test_dual_metrics":
            test_dual_metrics,
        "test_refiner_metrics":
            test_refiner_metrics,
        "relaxed_failure_count":
            int(len(failures)),
        "configuration":
            serialize_arguments(args),
    }

    summary_path = (
        table_dir
        / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    coarse = test_refiner_metrics[
        "coarse"
    ]

    refined = test_refiner_metrics[
        "refined"
    ]

    print("\n" + "=" * 78)
    print("v2完整训练结束")
    print("=" * 78)

    print(
        "双分支最佳轮次："
        f"{best_dual_checkpoint['epoch']}"
    )

    print(
        "精修器最佳轮次："
        f"{best_refiner_checkpoint['epoch']}"
    )

    print(
        "\n========== 测试集粗定位 =========="
    )

    print(
        "平均距离误差："
        f"{coarse['mean_range_error']:.3f}门"
    )

    print(
        "平均速度误差："
        f"{coarse['mean_velocity_error']:.3f}单元"
    )

    print(
        "严格命中率："
        f"{coarse['strict_hit_rate']:.2%}"
    )

    print(
        "宽松命中率："
        f"{coarse['relaxed_hit_rate']:.2%}"
    )

    print(
        "应用命中率："
        f"{coarse['application_hit_rate']:.2%}"
    )

    print(
        "零多普勒误峰："
        f"{int(coarse['zero_false_peak_count'])}"
    )

    print(
        "\n========== 测试集精修后 =========="
    )

    print(
        "平均距离误差："
        f"{refined['mean_range_error']:.3f}门"
    )

    print(
        "平均速度误差："
        f"{refined['mean_velocity_error']:.3f}单元"
    )

    print(
        "严格命中率："
        f"{refined['strict_hit_rate']:.2%}"
    )

    print(
        "宽松命中率："
        f"{refined['relaxed_hit_rate']:.2%}"
    )

    print(
        "应用命中率："
        f"{refined['application_hit_rate']:.2%}"
    )

    print(
        "零多普勒误峰："
        f"{int(refined['zero_false_peak_count'])}"
    )

    print(
        "超出局部精修范围样本："
        f"{test_refiner_metrics['invalid_sample_count']}"
    )

    print(
        "宽松失败样本："
        f"{len(failures)}"
    )

    if not failures.empty:
        display_columns = [
            "sample_id",
            "true_range_index",
            "true_velocity_index",
            "coarse_range_index",
            "coarse_velocity_index",
            "refined_range_index",
            "refined_velocity_index",
            "refined_range_error",
            "refined_velocity_error",
        ]

        print(
            "\n========== 宽松失败明细 =========="
        )

        print(
            failures[
                display_columns
            ].to_string(
                index=False
            )
        )

    print(
        "\n最佳双分支模型："
        f"{checkpoint_dir / 'best_dual.pt'}"
    )

    print(
        "最佳精修器："
        f"{checkpoint_dir / 'best_refiner.pt'}"
    )

    print(
        f"测试明细：{test_details_path}"
    )

    print(
        f"失败明细：{failures_path}"
    )

    print(
        f"实验汇总：{summary_path}"
    )


if __name__ == "__main__":
    main()
