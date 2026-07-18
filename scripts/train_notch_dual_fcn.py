from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from datasets.radar_dataset import RadarDataset
from models.simple_fcn import SimpleRadarFCN


SEED = 42

RADII = {
    "strict": (1, 1),
    "relaxed": (4, 1),
    "application": (4, 2),
}


def get_args():
    parser = argparse.ArgumentParser(
        description="原始RD + 零多普勒软抑制双分支FCN"
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
    )

    parser.add_argument(
        "--name",
        default="sigma_r3_v1_notch_dual",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--early-stopping",
        type=int,
        default=30,
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
        "--ranking-margin",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--ranking-weight",
        type=float,
        default=0.80,
    )

    parser.add_argument(
        "--band-weight",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--notch-aux-weight",
        type=float,
        default=0.30,
    )

    return parser.parse_args()


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_torch(path, device):
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


def get_state(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError(
            "预训练模型必须是state_dict或checkpoint字典。"
        )

    return checkpoint.get(
        "model_state_dict",
        checkpoint,
    )


def trainable_count(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


class SoftNotchDataset(Dataset):
    """
    在原始H/V输入之外，生成零多普勒软抑制后的H/V输入。
    """

    def __init__(
        self,
        base,
        sigma=2.0,
        floor=0.05,
    ):
        if sigma <= 0:
            raise ValueError(
                "notch_sigma必须大于0。"
            )

        if not 0 <= floor <= 1:
            raise ValueError(
                "notch_floor必须在[0,1]之间。"
            )

        self.base = base
        self.sigma = float(sigma)
        self.floor = float(floor)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = dict(
            self.base[index]
        )

        raw = sample[
            "input"
        ].float()

        velocity_size = raw.shape[-2]
        center = velocity_size // 2

        velocity_axis = torch.arange(
            velocity_size,
            dtype=raw.dtype,
        )

        gaussian = torch.exp(
            -0.5
            * (
                (
                    velocity_axis
                    - center
                )
                / self.sigma
            ).pow(2)
        )

        attenuation = (
            1.0
            - (
                1.0
                - self.floor
            )
            * gaussian
        )

        attenuation = attenuation.view(
            1,
            velocity_size,
            1,
        )

        sample[
            "raw_input"
        ] = raw

        sample[
            "notch_input"
        ] = (
            raw
            * attenuation
        )

        return sample


class DualViewFCN(nn.Module):
    """
    raw_branch：
        冻结原始sigma_r3_v1模型。

    notch_branch：
        使用同一权重初始化，
        输入零多普勒软抑制RD图并继续训练。

    fusion：
        学习融合两个分支输出的热力图。
    """

    def __init__(
        self,
        pretrained_state,
    ):
        super().__init__()

        self.raw_branch = (
            SimpleRadarFCN()
        )

        self.notch_branch = (
            SimpleRadarFCN()
        )

        self.raw_branch.load_state_dict(
            pretrained_state
        )

        self.notch_branch.load_state_dict(
            pretrained_state
        )

        for parameter in (
            self.raw_branch.parameters()
        ):
            parameter.requires_grad = False

        self.fusion = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=1,
            bias=True,
        )

        with torch.no_grad():
            self.fusion.weight.zero_()

            self.fusion.weight[
                0,
                0,
                0,
                0,
            ] = 1.0

            self.fusion.bias.zero_()

    def train(
        self,
        mode=True,
    ):
        super().train(mode)

        self.raw_branch.eval()

        return self

    def forward(
        self,
        raw,
        notch,
    ):
        with torch.no_grad():
            raw_logits = (
                self.raw_branch(raw)
            )

        notch_logits = (
            self.notch_branch(
                notch
            )
        )

        fused_logits = (
            self.fusion(
                torch.cat(
                    [
                        raw_logits,
                        notch_logits,
                    ],
                    dim=1,
                )
            )
        )

        return (
            raw_logits,
            notch_logits,
            fused_logits,
        )


class HeatmapLoss(nn.Module):
    def __init__(
        self,
        positive_weight=30.0,
        dice_weight=0.2,
    ):
        super().__init__()

        self.positive_weight = (
            positive_weight
        )

        self.dice_weight = (
            dice_weight
        )

    def forward(
        self,
        logits,
        target,
    ):
        prediction = torch.sigmoid(
            logits
        )

        weighted_mse = (
            (
                1
                + self.positive_weight
                * target
            )
            * (
                prediction
                - target
            ).pow(2)
        ).mean()

        prediction_flat = (
            prediction.flatten(
                start_dim=1
            )
        )

        target_flat = (
            target.flatten(
                start_dim=1
            )
        )

        intersection = (
            prediction_flat
            * target_flat
        ).sum(dim=1)

        dice_score = (
            2
            * intersection
            + 1
        ) / (
            prediction_flat.sum(
                dim=1
            )
            + target_flat.sum(
                dim=1
            )
            + 1
        )

        dice_loss = (
            1
            - dice_score
        ).mean()

        return (
            weighted_mse
            + self.dice_weight
            * dice_loss
        )


class RobustLoss(nn.Module):
    def __init__(
        self,
        args,
    ):
        super().__init__()

        self.base = HeatmapLoss()

        self.half_width = (
            args.zero_band_half_width
        )

        self.exclusion = (
            args.target_exclusion_margin
        )

        self.margin = (
            args.ranking_margin
        )

        self.ranking_weight = (
            args.ranking_weight
        )

        self.band_weight = (
            args.band_weight
        )

        self.notch_aux_weight = (
            args.notch_aux_weight
        )

    def forward(
        self,
        fused_logits,
        notch_logits,
        target,
        true_range,
        true_velocity,
    ):
        fused_base = self.base(
            fused_logits,
            target,
        )

        notch_base = self.base(
            notch_logits,
            target,
        )

        probability = torch.sigmoid(
            fused_logits
        )

        velocity_size = (
            probability.shape[-2]
        )

        center = (
            velocity_size // 2
        )

        lower = max(
            0,
            center
            - self.half_width,
        )

        upper = min(
            velocity_size,
            center
            + self.half_width
            + 1,
        )

        true_range = (
            true_range
            .long()
            .reshape(-1)
        )

        true_velocity = (
            true_velocity
            .long()
            .reshape(-1)
        )

        eligible = (
            (
                true_velocity
                - center
            ).abs()
            > self.exclusion
        )

        eligible_indices = (
            eligible
            .nonzero(
                as_tuple=False
            )
            .reshape(-1)
        )

        if (
            eligible_indices.numel()
            > 0
        ):
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
                .max(dim=1)
                .values
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
                self.margin
                + zero_band_max
                - target_scores
            ).mean()

            band_energy_loss = (
                zero_band
                .pow(2)
                .mean()
            )

        else:
            ranking_loss = (
                probability
                .new_zeros(())
            )

            band_energy_loss = (
                probability
                .new_zeros(())
            )

        total_loss = (
            fused_base
            + self.notch_aux_weight
            * notch_base
            + self.ranking_weight
            * ranking_loss
            + self.band_weight
            * band_energy_loss
        )

        components = {
            "fused_base":
                fused_base.detach(),

            "notch_base":
                notch_base.detach(),

            "ranking":
                ranking_loss.detach(),

            "band_energy":
                band_energy_loss.detach(),
        }

        return (
            total_loss,
            components,
        )


def extract_peaks(
    logits,
):
    probability = torch.sigmoid(
        logits
    )

    batch_size = (
        probability.shape[0]
    )

    range_size = (
        probability.shape[-1]
    )

    flat_indices = (
        probability
        .reshape(
            batch_size,
            -1,
        )
        .argmax(dim=1)
    )

    range_indices = (
        flat_indices
        % range_size
    )

    velocity_indices = (
        flat_indices
        // range_size
    )

    return (
        range_indices,
        velocity_indices,
    )


def new_accumulator():
    result = {
        "loss_sum": 0.0,
        "sample_count": 0.0,
        "range_error_sum": 0.0,
        "velocity_error_sum": 0.0,
        "eligible_zero_count": 0.0,
        "false_zero_peak_count": 0.0,
        "fused_base_sum": 0.0,
        "notch_base_sum": 0.0,
        "ranking_sum": 0.0,
        "band_energy_sum": 0.0,
    }

    for metric_name in RADII:
        result[
            f"{metric_name}_hit_count"
        ] = 0.0

    return result


def update_metrics(
    accumulator,
    logits,
    true_range,
    true_velocity,
    half_width,
    exclusion,
):
    (
        predicted_range,
        predicted_velocity,
    ) = extract_peaks(
        logits
    )

    true_range = (
        true_range
        .long()
        .reshape(-1)
    )

    true_velocity = (
        true_velocity
        .long()
        .reshape(-1)
    )

    range_error = (
        predicted_range
        - true_range
    ).abs()

    velocity_error = (
        predicted_velocity
        - true_velocity
    ).abs()

    accumulator[
        "range_error_sum"
    ] += float(
        range_error
        .sum()
        .item()
    )

    accumulator[
        "velocity_error_sum"
    ] += float(
        velocity_error
        .sum()
        .item()
    )

    for (
        metric_name,
        radii,
    ) in RADII.items():
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
            hits
            .sum()
            .item()
        )

    center = (
        logits.shape[-2]
        // 2
    )

    eligible = (
        (
            true_velocity
            - center
        ).abs()
        > exclusion
    )

    false_zero_peak = (
        eligible
        & (
            (
                predicted_velocity
                - center
            ).abs()
            <= half_width
        )
    )

    accumulator[
        "eligible_zero_count"
    ] += float(
        eligible
        .sum()
        .item()
    )

    accumulator[
        "false_zero_peak_count"
    ] += float(
        false_zero_peak
        .sum()
        .item()
    )


def finish_metrics(
    accumulator,
):
    sample_count = (
        accumulator[
            "sample_count"
        ]
    )

    eligible_count = (
        accumulator[
            "eligible_zero_count"
        ]
    )

    if sample_count <= 0:
        raise RuntimeError(
            "没有可评价的样本。"
        )

    result = {
        "loss":
            accumulator[
                "loss_sum"
            ]
            / sample_count,

        "mean_range_error":
            accumulator[
                "range_error_sum"
            ]
            / sample_count,

        "mean_velocity_error":
            accumulator[
                "velocity_error_sum"
            ]
            / sample_count,

        "zero_false_peak_rate":
            (
                accumulator[
                    "false_zero_peak_count"
                ]
                / eligible_count
                if eligible_count > 0
                else 0.0
            ),

        "zero_false_peak_count":
            accumulator[
                "false_zero_peak_count"
            ],

        "zero_eligible_count":
            eligible_count,

        "fused_base_loss":
            accumulator[
                "fused_base_sum"
            ]
            / sample_count,

        "notch_base_loss":
            accumulator[
                "notch_base_sum"
            ]
            / sample_count,

        "ranking_loss":
            accumulator[
                "ranking_sum"
            ]
            / sample_count,

        "band_energy_loss":
            accumulator[
                "band_energy_sum"
            ]
            / sample_count,
    }

    for metric_name in RADII:
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
    model,
    data_loader,
    loss_function,
    device,
    optimizer,
    args,
):
    training = (
        optimizer is not None
    )

    if training:
        model.train()
    else:
        model.eval()

    accumulator = (
        new_accumulator()
    )

    context = (
        torch.enable_grad()
        if training
        else torch.no_grad()
    )

    with context:
        for batch in data_loader:
            raw_input = (
                batch[
                    "raw_input"
                ]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            notch_input = (
                batch[
                    "notch_input"
                ]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            target = (
                batch[
                    "target"
                ]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            true_range = (
                batch[
                    "range_index"
                ]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            true_velocity = (
                batch[
                    "velocity_index"
                ]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            if training:
                optimizer.zero_grad(
                    set_to_none=True
                )

            (
                raw_logits,
                notch_logits,
                fused_logits,
            ) = model(
                raw_input,
                notch_input,
            )

            (
                loss,
                components,
            ) = loss_function(
                fused_logits,
                notch_logits,
                target,
                true_range,
                true_velocity,
            )

            if not torch.isfinite(
                loss
            ):
                raise RuntimeError(
                    "损失出现NaN或无穷值。"
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

            accumulator[
                "sample_count"
            ] += batch_size

            accumulator[
                "loss_sum"
            ] += (
                float(
                    loss.item()
                )
                * batch_size
            )

            accumulator[
                "fused_base_sum"
            ] += (
                float(
                    components[
                        "fused_base"
                    ].item()
                )
                * batch_size
            )

            accumulator[
                "notch_base_sum"
            ] += (
                float(
                    components[
                        "notch_base"
                    ].item()
                )
                * batch_size
            )

            accumulator[
                "ranking_sum"
            ] += (
                float(
                    components[
                        "ranking"
                    ].item()
                )
                * batch_size
            )

            accumulator[
                "band_energy_sum"
            ] += (
                float(
                    components[
                        "band_energy"
                    ].item()
                )
                * batch_size
            )

            update_metrics(
                accumulator=
                    accumulator,

                logits=
                    fused_logits.detach(),

                true_range=
                    true_range,

                true_velocity=
                    true_velocity,

                half_width=
                    args.zero_band_half_width,

                exclusion=
                    args.target_exclusion_margin,
            )

    return finish_metrics(
        accumulator
    )


def model_score(
    metrics,
):
    return (
        metrics[
            "relaxed_hit_rate"
        ],

        metrics[
            "application_hit_rate"
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

        -metrics[
            "loss"
        ],
    )


def normalize_sample_ids(
    raw_ids,
    batch_size,
    start_index,
):
    if raw_ids is None:
        return [
            f"test_{start_index + index:03d}"
            for index
            in range(batch_size)
        ]

    if isinstance(
        raw_ids,
        str,
    ):
        return [raw_ids]

    return [
        str(value)
        for value
        in raw_ids
    ]


def create_test_details(
    model,
    data_loader,
    device,
    args,
):
    model.eval()

    records = []
    global_index = 0

    with torch.no_grad():
        for batch in data_loader:
            raw_input = (
                batch[
                    "raw_input"
                ].to(device)
            )

            notch_input = (
                batch[
                    "notch_input"
                ].to(device)
            )

            true_range = (
                batch[
                    "range_index"
                ]
                .long()
                .reshape(-1)
                .to(device)
            )

            true_velocity = (
                batch[
                    "velocity_index"
                ]
                .long()
                .reshape(-1)
                .to(device)
            )

            (
                raw_logits,
                notch_logits,
                fused_logits,
            ) = model(
                raw_input,
                notch_input,
            )

            (
                raw_range,
                raw_velocity,
            ) = extract_peaks(
                raw_logits
            )

            (
                notch_range,
                notch_velocity,
            ) = extract_peaks(
                notch_logits
            )

            (
                predicted_range,
                predicted_velocity,
            ) = extract_peaks(
                fused_logits
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

            center = (
                raw_input.shape[-2]
                // 2
            )

            for index in range(
                batch_size
            ):
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

                predicted_r = int(
                    predicted_range[
                        index
                    ].item()
                )

                predicted_v = int(
                    predicted_velocity[
                        index
                    ].item()
                )

                range_error = abs(
                    predicted_r
                    - true_r
                )

                velocity_error = abs(
                    predicted_v
                    - true_v
                )

                record = {
                    "sample_id":
                        sample_ids[
                            index
                        ],

                    "true_range_index":
                        true_r,

                    "true_velocity_index":
                        true_v,

                    "raw_range_index":
                        int(
                            raw_range[
                                index
                            ].item()
                        ),

                    "raw_velocity_index":
                        int(
                            raw_velocity[
                                index
                            ].item()
                        ),

                    "notch_range_index":
                        int(
                            notch_range[
                                index
                            ].item()
                        ),

                    "notch_velocity_index":
                        int(
                            notch_velocity[
                                index
                            ].item()
                        ),

                    "predicted_range_index":
                        predicted_r,

                    "predicted_velocity_index":
                        predicted_v,

                    "range_error_gates":
                        range_error,

                    "velocity_error_bins":
                        velocity_error,

                    "predicted_in_zero_band":
                        (
                            abs(
                                predicted_v
                                - center
                            )
                            <= args.zero_band_half_width
                        ),
                }

                for (
                    metric_name,
                    radii,
                ) in RADII.items():
                    range_radius = (
                        radii[0]
                    )

                    velocity_radius = (
                        radii[1]
                    )

                    record[
                        f"{metric_name}_hit"
                    ] = (
                        range_error
                        <= range_radius
                        and velocity_error
                        <= velocity_radius
                    )

                records.append(
                    record
                )

            global_index += (
                batch_size
            )

    return pd.DataFrame(
        records
    )


def main():
    args = get_args()

    seed_everything()

    if not args.pretrained.exists():
        raise FileNotFoundError(
            f"找不到预训练模型：{args.pretrained}"
        )

    if (
        args.target_exclusion_margin
        < args.zero_band_half_width
    ):
        raise ValueError(
            "target-exclusion-margin不能小于"
            "zero-band-half-width。"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    experiment_root = (
        PROJECT_ROOT
        / "results"
        / "experiments"
        / args.name
    )

    checkpoint_dir = (
        experiment_root
        / "checkpoints"
    )

    table_dir = (
        experiment_root
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

    best_path = (
        checkpoint_dir
        / "best.pt"
    )

    last_path = (
        checkpoint_dir
        / "last.pt"
    )

    history_path = (
        table_dir
        / "history.csv"
    )

    test_details_path = (
        table_dir
        / "test_details.csv"
    )

    summary_path = (
        table_dir
        / "summary.json"
    )

    print(
        "=" * 70
    )

    print(
        "双分支零多普勒鲁棒FCN"
    )

    print(
        "=" * 70
    )

    print(
        f"设备：{device}"
    )

    if torch.cuda.is_available():
        print(
            "GPU："
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"预训练模型：{args.pretrained}"
    )

    print(
        "软陷波："
        f"sigma={args.notch_sigma}，"
        f"中心保留={args.notch_floor}"
    )

    def build_dataset(
        split,
    ):
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
        device.type
        == "cuda"
    )

    generator = (
        torch.Generator()
        .manual_seed(SEED)
    )

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

    pretrained_checkpoint = (
        load_torch(
            args.pretrained,
            device,
        )
    )

    pretrained_state = (
        get_state(
            pretrained_checkpoint
        )
    )

    model = DualViewFCN(
        pretrained_state
    ).to(device)

    print(
        "可训练参数量："
        f"{trainable_count(model):,}"
    )

    loss_function = RobustLoss(
        args
    )

    trainable_parameters = [
        parameter
        for parameter
        in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = (
        torch.optim.AdamW(
            trainable_parameters,
            lr=args.learning_rate,
            weight_decay=
                args.weight_decay,
        )
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=8,
            min_lr=1e-6,
        )
    )

    history = []

    best_score = None
    best_epoch = 0
    stale_epochs = 0

    training_start = (
        time.time()
    )

    print(
        "\n开始训练……"
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        train_metrics = run_epoch(
            model=model,
            data_loader=train_loader,
            loss_function=loss_function,
            device=device,
            optimizer=optimizer,
            args=args,
        )

        validation_metrics = run_epoch(
            model=model,
            data_loader=
                validation_loader,
            loss_function=loss_function,
            device=device,
            optimizer=None,
            args=args,
        )

        scheduler.step(
            validation_metrics[
                "loss"
            ]
        )

        learning_rate = float(
            optimizer.param_groups[
                0
            ]["lr"]
        )

        current_score = (
            model_score(
                validation_metrics
            )
        )

        is_best = (
            best_score is None
            or current_score
            > best_score
        )

        configuration = {
            key:
                str(value)
                if isinstance(
                    value,
                    Path,
                )
                else value
            for key, value
            in vars(args).items()
        }

        checkpoint = {
            "epoch":
                epoch,

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
                configuration,
        }

        torch.save(
            checkpoint,
            last_path,
        )

        if is_best:
            best_score = (
                current_score
            )

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
                "epoch":
                    epoch,

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

                "is_best":
                    is_best,
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
            f"第{epoch:03d}轮｜"
            f"损失="
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
            f"零多普勒误峰="
            f"{int(validation_metrics['zero_false_peak_count'])}/"
            f"{int(validation_metrics['zero_eligible_count'])}｜"
            f"学习率="
            f"{learning_rate:.2e}"
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
                "轮无改进，提前停止。"
            )

            break

    training_seconds = (
        time.time()
        - training_start
    )

    print(
        "\n加载最佳模型并评价测试集……"
    )

    best_checkpoint = load_torch(
        best_path,
        device,
    )

    model.load_state_dict(
        best_checkpoint[
            "model_state_dict"
        ]
    )

    test_metrics = run_epoch(
        model=model,
        data_loader=test_loader,
        loss_function=loss_function,
        device=device,
        optimizer=None,
        args=args,
    )

    test_details = (
        create_test_details(
            model=model,
            data_loader=test_loader,
            device=device,
            args=args,
        )
    )

    test_details.to_csv(
        test_details_path,
        index=False,
        encoding="utf-8-sig",
    )

    target_sample = test_details[
        test_details[
            "sample_id"
        ].astype(str)
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

    print(
        "\n"
        + "=" * 70
    )

    print(
        "训练完成"
    )

    print(
        "=" * 70
    )

    print(
        f"最佳轮次：{best_epoch}"
    )

    print(
        "测试距离误差："
        f"{test_metrics['mean_range_error']:.3f}门"
    )

    print(
        "测试速度误差："
        f"{test_metrics['mean_velocity_error']:.3f}单元"
    )

    print(
        "测试严格命中率："
        f"{test_metrics['strict_hit_rate']:.2%}"
    )

    print(
        "测试宽松命中率："
        f"{test_metrics['relaxed_hit_rate']:.2%}"
    )

    print(
        "测试应用命中率："
        f"{test_metrics['application_hit_rate']:.2%}"
    )

    print(
        "测试零多普勒误峰："
        f"{int(test_metrics['zero_false_peak_count'])}/"
        f"{int(test_metrics['zero_eligible_count'])}"
    )

    if not target_sample.empty:
        print(
            "\n20260202_144849_beam4："
        )

        print(
            target_sample.to_string(
                index=False
            )
        )

    print(
        f"\n最佳模型：{best_path}"
    )

    print(
        f"测试明细：{test_details_path}"
    )

    print(
        f"结果汇总：{summary_path}"
    )


if __name__ == "__main__":
    main()
