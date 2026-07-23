from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLIT_FILE = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "samples_split.csv"
)

FOLD_SPLIT_DIR = (
    PROJECT_ROOT
    / "results"
    / "cross_validation"
    / "grouped_5fold"
    / "fold_splits"
)

EXPERIMENT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "experiments"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "cross_validation"
    / "grouped_5fold_v2"
)

V1_SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "cross_validation"
    / "grouped_5fold"
    / "summary.json"
)

FOLD_COUNT = 5


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "使用固定session分组五折划分，"
            "运行v2雷达定位完整训练管线"
        )
    )

    parser.add_argument(
        "--start-fold",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--end-fold",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--sigma-epochs",
        type=int,
        default=100,
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
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "跳过已经完整生成结果的折；"
            "不执行epoch级断点续训"
        ),
    )

    return parser.parse_args()


def run_command(
    command: list[str],
) -> None:
    print(
        "\n执行命令：\n"
        + " ".join(command)
        + "\n"
    )

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )


def to_bool(
    series: pd.Series,
) -> pd.Series:
    if pd.api.types.is_bool_dtype(
        series
    ):
        return series.fillna(False)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
        .fillna(False)
        .astype(bool)
    )


def wilson_interval(
    hit_count: int,
    sample_count: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if sample_count <= 0:
        return float("nan"), float("nan")

    probability = (
        hit_count
        / sample_count
    )

    denominator = (
        1.0
        + z * z / sample_count
    )

    center = (
        probability
        + z * z
        / (2.0 * sample_count)
    ) / denominator

    margin = (
        z
        * math.sqrt(
            (
                probability
                * (
                    1.0
                    - probability
                )
                / sample_count
            )
            + (
                z * z
                / (
                    4.0
                    * sample_count
                    * sample_count
                )
            )
        )
        / denominator
    )

    return (
        max(
            0.0,
            center - margin,
        ),
        min(
            1.0,
            center + margin,
        ),
    )


def fold_paths(
    fold_number: int,
) -> dict[str, Path]:
    prefix = (
        f"v2_cv_f{fold_number:02d}"
    )

    sigma_name = (
        f"{prefix}_sigma_r3_v1"
    )

    pipeline_name = (
        f"{prefix}_pipeline"
    )

    sigma_dir = (
        EXPERIMENT_ROOT
        / sigma_name
    )

    pipeline_dir = (
        EXPERIMENT_ROOT
        / pipeline_name
    )

    return {
        "fold_split": (
            FOLD_SPLIT_DIR
            / f"fold_{fold_number:02d}.csv"
        ),
        "sigma_name": Path(
            sigma_name
        ),
        "pipeline_name": Path(
            pipeline_name
        ),
        "sigma_best": (
            sigma_dir
            / "checkpoints"
            / "best.pt"
        ),
        "pipeline_best_dual": (
            pipeline_dir
            / "checkpoints"
            / "best_dual.pt"
        ),
        "pipeline_best_refiner": (
            pipeline_dir
            / "checkpoints"
            / "best_refiner.pt"
        ),
        "pipeline_summary": (
            pipeline_dir
            / "tables"
            / "summary.json"
        ),
        "pipeline_details": (
            pipeline_dir
            / "tables"
            / "test_details.csv"
        ),
        "pipeline_failures": (
            pipeline_dir
            / "tables"
            / "relaxed_failures.csv"
        ),
    }


def pipeline_is_complete(
    paths: dict[str, Path],
) -> bool:
    required = [
        paths[
            "pipeline_best_dual"
        ],
        paths[
            "pipeline_best_refiner"
        ],
        paths[
            "pipeline_summary"
        ],
        paths[
            "pipeline_details"
        ],
    ]

    return all(
        path.exists()
        for path in required
    )


def print_fold_statistics(
    fold_number: int,
    fold_path: Path,
) -> None:
    dataframe = pd.read_csv(
        fold_path
    )

    if "split" not in dataframe.columns:
        raise RuntimeError(
            f"{fold_path}中没有split列。"
        )

    print(
        "\n"
        + "=" * 74
    )

    print(
        f"第{fold_number}折数据划分"
    )

    print(
        "=" * 74
    )

    for split_name in [
        "train",
        "val",
        "test",
    ]:
        subset = dataframe[
            dataframe["split"]
            == split_name
        ]

        if "session_id" in subset.columns:
            session_count = (
                subset[
                    "session_id"
                ]
                .astype(str)
                .nunique()
            )

            print(
                f"{split_name}: "
                f"{len(subset)}个样本，"
                f"{session_count}个session"
            )

        else:
            print(
                f"{split_name}: "
                f"{len(subset)}个样本"
            )


def train_fold(
    fold_number: int,
    arguments: argparse.Namespace,
) -> None:
    paths = fold_paths(
        fold_number
    )

    fold_split_path = paths[
        "fold_split"
    ]

    if not fold_split_path.exists():
        raise FileNotFoundError(
            f"找不到第{fold_number}折划分："
            f"{fold_split_path}"
        )

    print_fold_statistics(
        fold_number,
        fold_split_path,
    )

    SPLIT_FILE.write_bytes(
        fold_split_path.read_bytes()
    )

    sigma_name = str(
        paths["sigma_name"]
    )

    pipeline_name = str(
        paths["pipeline_name"]
    )

    # --------------------------------------------------------
    # 阶段一：当前折Sigma模型
    # --------------------------------------------------------

    if (
        arguments.resume
        and paths[
            "sigma_best"
        ].exists()
    ):
        print(
            f"\n第{fold_number}折Sigma模型"
            "已存在，跳过训练。"
        )

    else:
        run_command(
            [
                sys.executable,
                "scripts/train_sigma_experiment.py",
                "--name",
                sigma_name,
                "--range-sigma",
                "3.0",
                "--velocity-sigma",
                "1.0",
                "--epochs",
                str(
                    arguments.sigma_epochs
                ),
                "--batch-size",
                str(
                    arguments.batch_size
                ),
                "--num-workers",
                str(
                    arguments.num_workers
                ),
                "--prediction-count",
                "0",
            ]
        )

    if not paths[
        "sigma_best"
    ].exists():
        raise FileNotFoundError(
            "当前折Sigma训练结束后"
            "仍未找到最佳模型："
            f"{paths['sigma_best']}"
        )

    # --------------------------------------------------------
    # 阶段二：v2双分支 + v2局部精修
    # --------------------------------------------------------

    if (
        arguments.resume
        and pipeline_is_complete(
            paths
        )
    ):
        print(
            f"\n第{fold_number}折v2完整结果"
            "已存在，跳过训练。"
        )

    else:
        run_command(
            [
                sys.executable,
                "scripts/train_v2_pipeline.py",
                "--pretrained",
                str(
                    paths[
                        "sigma_best"
                    ]
                ),
                "--name",
                pipeline_name,
                "--seed",
                str(
                    arguments.seed
                ),
                "--dual-epochs",
                str(
                    arguments.dual_epochs
                ),
                "--refiner-epochs",
                str(
                    arguments.refiner_epochs
                ),
                "--batch-size",
                str(
                    arguments.batch_size
                ),
                "--refiner-batch-size",
                str(
                    arguments.refiner_batch_size
                ),
                "--dual-learning-rate",
                "3e-4",
                "--refiner-learning-rate",
                "1e-3",
                "--notch-sigma",
                "2.0",
                "--notch-floor",
                "0.05",
                "--shift-probability",
                "0.80",
                "--max-range-shift",
                "6",
                "--max-velocity-shift",
                "12",
                "--positive-range-radius",
                "4",
                "--positive-velocity-radius",
                "2",
                "--hard-negative-margin",
                "0.20",
                "--hard-negative-weight",
                "1.00",
                "--zero-ranking-weight",
                "0.50",
                "--zero-band-weight",
                "0.10",
                "--notch-aux-weight",
                "0.30",
                "--crop-size",
                "17",
                "--max-range-offset",
                "4",
                "--max-velocity-offset",
                "2",
                "--invalid-refiner-weight",
                "0",
                "--num-workers",
                str(
                    arguments.num_workers
                ),
            ]
        )

    if not pipeline_is_complete(
        paths
    ):
        raise RuntimeError(
            f"第{fold_number}折v2训练结果不完整。"
        )


def read_fold_result(
    fold_number: int,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
] | None:
    paths = fold_paths(
        fold_number
    )

    if not pipeline_is_complete(
        paths
    ):
        return None

    summary = json.loads(
        paths[
            "pipeline_summary"
        ].read_text(
            encoding="utf-8"
        )
    )

    details = pd.read_csv(
        paths[
            "pipeline_details"
        ]
    )

    if "cv_fold" in details.columns:
        details["cv_fold"] = (
            fold_number
        )
    else:
        details.insert(
            0,
            "cv_fold",
            fold_number,
        )

    try:
        coarse = summary[
            "test_refiner_metrics"
        ]["coarse"]

        refined = summary[
            "test_refiner_metrics"
        ]["refined"]

    except KeyError as error:
        raise RuntimeError(
            "v2 summary.json结构不符合预期："
            f"{paths['pipeline_summary']}"
        ) from error

    record = {
        "cv_fold":
            fold_number,

        "test_sample_count":
            int(len(details)),

        "best_dual_epoch":
            summary.get(
                "best_dual_epoch",
                np.nan,
            ),

        "best_refiner_epoch":
            summary.get(
                "best_refiner_epoch",
                np.nan,
            ),

        "coarse_mean_range_error":
            coarse[
                "mean_range_error"
            ],

        "coarse_mean_velocity_error":
            coarse[
                "mean_velocity_error"
            ],

        "coarse_strict_hit_rate":
            coarse[
                "strict_hit_rate"
            ],

        "coarse_relaxed_hit_rate":
            coarse[
                "relaxed_hit_rate"
            ],

        "coarse_application_hit_rate":
            coarse[
                "application_hit_rate"
            ],

        "coarse_zero_false_peak_count":
            coarse[
                "zero_false_peak_count"
            ],

        "refined_mean_range_error":
            refined[
                "mean_range_error"
            ],

        "refined_mean_velocity_error":
            refined[
                "mean_velocity_error"
            ],

        "refined_strict_hit_rate":
            refined[
                "strict_hit_rate"
            ],

        "refined_relaxed_hit_rate":
            refined[
                "relaxed_hit_rate"
            ],

        "refined_application_hit_rate":
            refined[
                "application_hit_rate"
            ],

        "refined_zero_false_peak_count":
            refined[
                "zero_false_peak_count"
            ],

        "invalid_refiner_sample_count":
            summary[
                "test_refiner_metrics"
            ].get(
                "invalid_sample_count",
                np.nan,
            ),
    }

    return record, details


def collect_completed_results() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    fold_records = []
    detail_frames = []

    for fold_number in range(
        1,
        FOLD_COUNT + 1,
    ):
        result = read_fold_result(
            fold_number
        )

        if result is None:
            continue

        record, details = result

        fold_records.append(
            record
        )

        detail_frames.append(
            details
        )

    fold_dataframe = pd.DataFrame(
        fold_records
    )

    if detail_frames:
        oof_dataframe = pd.concat(
            detail_frames,
            ignore_index=True,
        )
    else:
        oof_dataframe = (
            pd.DataFrame()
        )

    return (
        fold_dataframe,
        oof_dataframe,
    )


def build_oof_summary(
    fold_dataframe: pd.DataFrame,
    oof_dataframe: pd.DataFrame,
) -> dict[str, Any]:
    required_columns = [
        "sample_id",
        "refined_range_error",
        "refined_velocity_error",
        "refined_strict_hit",
        "refined_relaxed_hit",
        "refined_application_hit",
        "true_velocity_index",
        "refined_velocity_index",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column
        not in oof_dataframe.columns
    ]

    if missing_columns:
        raise RuntimeError(
            "OOF文件缺少必要列："
            f"{missing_columns}"
        )

    strict_hits = to_bool(
        oof_dataframe[
            "refined_strict_hit"
        ]
    )

    relaxed_hits = to_bool(
        oof_dataframe[
            "refined_relaxed_hit"
        ]
    )

    application_hits = to_bool(
        oof_dataframe[
            "refined_application_hit"
        ]
    )

    sample_count = int(
        len(oof_dataframe)
    )

    strict_count = int(
        strict_hits.sum()
    )

    relaxed_count = int(
        relaxed_hits.sum()
    )

    application_count = int(
        application_hits.sum()
    )

    strict_interval = wilson_interval(
        strict_count,
        sample_count,
    )

    relaxed_interval = wilson_interval(
        relaxed_count,
        sample_count,
    )

    application_interval = (
        wilson_interval(
            application_count,
            sample_count,
        )
    )

    true_away_from_zero = (
        (
            oof_dataframe[
                "true_velocity_index"
            ].astype(int)
            - 64
        ).abs()
        > 6
    )

    predicted_in_zero_band = (
        (
            oof_dataframe[
                "refined_velocity_index"
            ].astype(int)
            - 64
        ).abs()
        <= 3
    )

    zero_false_peaks = (
        true_away_from_zero
        & predicted_in_zero_band
    )

    range_errors = (
        oof_dataframe[
            "refined_range_error"
        ].astype(float)
    )

    velocity_errors = (
        oof_dataframe[
            "refined_velocity_error"
        ].astype(float)
    )

    summary = {
        "completed_fold_count":
            int(
                fold_dataframe[
                    "cv_fold"
                ].nunique()
            ),

        "oof_sample_count":
            sample_count,

        "oof_unique_sample_count":
            int(
                oof_dataframe[
                    "sample_id"
                ].astype(str)
                .nunique()
            ),

        "oof_duplicate_sample_count":
            int(
                oof_dataframe[
                    "sample_id"
                ].astype(str)
                .duplicated()
                .sum()
            ),

        "oof_mean_range_error":
            float(
                range_errors.mean()
            ),

        "oof_median_range_error":
            float(
                range_errors.median()
            ),

        "oof_range_error_p90":
            float(
                range_errors.quantile(
                    0.90
                )
            ),

        "oof_range_error_p95":
            float(
                range_errors.quantile(
                    0.95
                )
            ),

        "oof_range_error_p99":
            float(
                range_errors.quantile(
                    0.99
                )
            ),

        "oof_max_range_error":
            float(
                range_errors.max()
            ),

        "oof_mean_velocity_error":
            float(
                velocity_errors.mean()
            ),

        "oof_median_velocity_error":
            float(
                velocity_errors.median()
            ),

        "oof_velocity_error_p90":
            float(
                velocity_errors.quantile(
                    0.90
                )
            ),

        "oof_velocity_error_p95":
            float(
                velocity_errors.quantile(
                    0.95
                )
            ),

        "oof_velocity_error_p99":
            float(
                velocity_errors.quantile(
                    0.99
                )
            ),

        "oof_max_velocity_error":
            float(
                velocity_errors.max()
            ),

        "oof_strict_hit_count":
            strict_count,

        "oof_strict_hit_rate":
            strict_count
            / sample_count,

        "oof_strict_hit_rate_ci95":
            list(
                strict_interval
            ),

        "oof_relaxed_hit_count":
            relaxed_count,

        "oof_relaxed_hit_rate":
            relaxed_count
            / sample_count,

        "oof_relaxed_hit_rate_ci95":
            list(
                relaxed_interval
            ),

        "oof_application_hit_count":
            application_count,

        "oof_application_hit_rate":
            application_count
            / sample_count,

        "oof_application_hit_rate_ci95":
            list(
                application_interval
            ),

        "oof_relaxed_failure_count":
            int(
                (
                    ~relaxed_hits
                ).sum()
            ),

        "oof_zero_false_peak_count":
            int(
                zero_false_peaks.sum()
            ),

        "fold_refined_strict_mean":
            float(
                fold_dataframe[
                    "refined_strict_hit_rate"
                ].mean()
            ),

        "fold_refined_strict_std":
            float(
                fold_dataframe[
                    "refined_strict_hit_rate"
                ].std(
                    ddof=1
                )
            )
            if len(
                fold_dataframe
            ) > 1
            else 0.0,

        "fold_refined_relaxed_mean":
            float(
                fold_dataframe[
                    "refined_relaxed_hit_rate"
                ].mean()
            ),

        "fold_refined_relaxed_std":
            float(
                fold_dataframe[
                    "refined_relaxed_hit_rate"
                ].std(
                    ddof=1
                )
            )
            if len(
                fold_dataframe
            ) > 1
            else 0.0,
    }

    return summary


def print_summary(
    summary: dict[str, Any],
) -> None:
    print(
        "\n"
        + "=" * 78
    )

    print(
        "v2分组交叉验证当前汇总"
    )

    print(
        "=" * 78
    )

    print(
        "已完成折数："
        f"{summary['completed_fold_count']}/5"
    )

    print(
        "OOF样本数："
        f"{summary['oof_sample_count']}"
    )

    print(
        "唯一样本数："
        f"{summary['oof_unique_sample_count']}"
    )

    print(
        "平均距离误差："
        f"{summary['oof_mean_range_error']:.3f}门"
    )

    print(
        "距离误差中位数："
        f"{summary['oof_median_range_error']:.3f}门"
    )

    print(
        "距离误差95%分位："
        f"{summary['oof_range_error_p95']:.3f}门"
    )

    print(
        "平均速度误差："
        f"{summary['oof_mean_velocity_error']:.3f}单元"
    )

    print(
        "速度误差中位数："
        f"{summary['oof_median_velocity_error']:.3f}单元"
    )

    print(
        "速度误差95%分位："
        f"{summary['oof_velocity_error_p95']:.3f}单元"
    )

    strict_ci = summary[
        "oof_strict_hit_rate_ci95"
    ]

    relaxed_ci = summary[
        "oof_relaxed_hit_rate_ci95"
    ]

    print(
        "严格命中率："
        f"{summary['oof_strict_hit_rate']:.2%} "
        f"({summary['oof_strict_hit_count']}/"
        f"{summary['oof_sample_count']})，"
        "95% CI="
        f"[{strict_ci[0]:.2%}, "
        f"{strict_ci[1]:.2%}]"
    )

    print(
        "宽松命中率："
        f"{summary['oof_relaxed_hit_rate']:.2%} "
        f"({summary['oof_relaxed_hit_count']}/"
        f"{summary['oof_sample_count']})，"
        "95% CI="
        f"[{relaxed_ci[0]:.2%}, "
        f"{relaxed_ci[1]:.2%}]"
    )

    print(
        "宽松失败样本："
        f"{summary['oof_relaxed_failure_count']}"
    )

    print(
        "零多普勒误峰："
        f"{summary['oof_zero_false_peak_count']}"
    )


def print_v1_comparison(
    v2_summary: dict[str, Any],
) -> None:
    if not V1_SUMMARY_PATH.exists():
        return

    v1_summary = json.loads(
        V1_SUMMARY_PATH.read_text(
            encoding="utf-8"
        )
    )

    required_keys = [
        "oof_strict_hit_rate",
        "oof_relaxed_hit_rate",
        "oof_mean_range_error",
        "oof_mean_velocity_error",
        "oof_zero_false_peak_count",
    ]

    if not all(
        key in v1_summary
        for key in required_keys
    ):
        return

    print(
        "\n========== v1 与 v2 对比 =========="
    )

    print(
        "严格命中率："
        f"v1={v1_summary['oof_strict_hit_rate']:.2%}，"
        f"v2={v2_summary['oof_strict_hit_rate']:.2%}，"
        "变化="
        f"{v2_summary['oof_strict_hit_rate'] - v1_summary['oof_strict_hit_rate']:+.2%}"
    )

    print(
        "宽松命中率："
        f"v1={v1_summary['oof_relaxed_hit_rate']:.2%}，"
        f"v2={v2_summary['oof_relaxed_hit_rate']:.2%}，"
        "变化="
        f"{v2_summary['oof_relaxed_hit_rate'] - v1_summary['oof_relaxed_hit_rate']:+.2%}"
    )

    print(
        "平均距离误差："
        f"v1={v1_summary['oof_mean_range_error']:.3f}门，"
        f"v2={v2_summary['oof_mean_range_error']:.3f}门"
    )

    print(
        "平均速度误差："
        f"v1={v1_summary['oof_mean_velocity_error']:.3f}单元，"
        f"v2={v2_summary['oof_mean_velocity_error']:.3f}单元"
    )

    print(
        "零多普勒误峰："
        f"v1={v1_summary['oof_zero_false_peak_count']}，"
        f"v2={v2_summary['oof_zero_false_peak_count']}"
    )


def save_completed_results(
    arguments: argparse.Namespace,
) -> None:
    (
        fold_dataframe,
        oof_dataframe,
    ) = collect_completed_results()

    if fold_dataframe.empty:
        print(
            "\n当前尚无完整折结果。"
        )
        return

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_metrics_path = (
        OUTPUT_DIR
        / "fold_metrics.csv"
    )

    oof_path = (
        OUTPUT_DIR
        / "oof_predictions.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "summary.json"
    )

    failures_path = (
        OUTPUT_DIR
        / "relaxed_failures.csv"
    )

    fold_dataframe = (
        fold_dataframe
        .sort_values(
            "cv_fold"
        )
        .reset_index(
            drop=True
        )
    )

    oof_dataframe = (
        oof_dataframe
        .sort_values(
            [
                "cv_fold",
                "sample_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    fold_dataframe.to_csv(
        fold_metrics_path,
        index=False,
        encoding="utf-8-sig",
    )

    oof_dataframe.to_csv(
        oof_path,
        index=False,
        encoding="utf-8-sig",
    )

    relaxed_hits = to_bool(
        oof_dataframe[
            "refined_relaxed_hit"
        ]
    )

    relaxed_failures = (
        oof_dataframe[
            ~relaxed_hits
        ].copy()
    )

    relaxed_failures.to_csv(
        failures_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary = build_oof_summary(
        fold_dataframe,
        oof_dataframe,
    )

    summary["configuration"] = {
        "seed":
            arguments.seed,
        "sigma_epochs":
            arguments.sigma_epochs,
        "dual_epochs":
            arguments.dual_epochs,
        "refiner_epochs":
            arguments.refiner_epochs,
        "batch_size":
            arguments.batch_size,
        "refiner_batch_size":
            arguments.refiner_batch_size,
    }

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print_summary(
        summary
    )

    if (
        summary[
            "completed_fold_count"
        ]
        == FOLD_COUNT
    ):
        if (
            summary[
                "oof_duplicate_sample_count"
            ]
            > 0
        ):
            raise RuntimeError(
                "完整OOF结果中出现重复样本。"
            )

        print_v1_comparison(
            summary
        )

    print(
        "\n========== 输出文件 =========="
    )

    print(
        f"各折指标：{fold_metrics_path}"
    )

    print(
        f"OOF逐样本结果：{oof_path}"
    )

    print(
        f"宽松失败样本：{failures_path}"
    )

    print(
        f"总体汇总：{summary_path}"
    )


def main() -> None:
    arguments = parse_arguments()

    if not SPLIT_FILE.exists():
        raise FileNotFoundError(
            f"找不到当前划分文件：{SPLIT_FILE}"
        )

    if arguments.start_fold < 1:
        raise ValueError(
            "start-fold必须不小于1。"
        )

    if arguments.end_fold > FOLD_COUNT:
        raise ValueError(
            "end-fold不能大于5。"
        )

    if (
        arguments.start_fold
        > arguments.end_fold
    ):
        raise ValueError(
            "start-fold不能大于end-fold。"
        )

    for fold_number in range(
        1,
        FOLD_COUNT + 1,
    ):
        fold_path = (
            FOLD_SPLIT_DIR
            / f"fold_{fold_number:02d}.csv"
        )

        if not fold_path.exists():
            raise FileNotFoundError(
                f"找不到五折划分文件：{fold_path}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    original_split_bytes = (
        SPLIT_FILE.read_bytes()
    )

    try:
        for fold_number in range(
            arguments.start_fold,
            arguments.end_fold + 1,
        ):
            print(
                "\n"
                + "#" * 82
            )

            print(
                f"开始v2第{fold_number}折"
            )

            print(
                "#" * 82
            )

            train_fold(
                fold_number,
                arguments,
            )

            # 每完成一折立刻生成阶段性汇总。
            save_completed_results(
                arguments
            )

    finally:
        SPLIT_FILE.write_bytes(
            original_split_bytes
        )

        print(
            "\n已恢复原始划分文件："
            f"{SPLIT_FILE}"
        )

    save_completed_results(
        arguments
    )


if __name__ == "__main__":
    main()
