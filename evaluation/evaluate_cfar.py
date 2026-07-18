from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.io import loadmat


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from baselines.ca_cfar import (
    ca_cfar_2d,
    extract_peak_candidates,
)


# ============================================================
# 文件路径
# ============================================================

SAMPLES_FILE = Path(
    "data/metadata/samples.csv"
)

DETAIL_OUTPUT = Path(
    "results/tables/cfar_results.csv"
)

CANDIDATE_OUTPUT = Path(
    "results/tables/cfar_candidates.csv"
)

SUMMARY_OUTPUT = Path(
    "results/tables/cfar_summary.csv"
)


# ============================================================
# 雷达参数
# ============================================================

C = 3.0e8
FC = 9300.0e6
WAVELENGTH = C / FC

PRF = 2900.0 / 2.0

NUM_PULSES = 128
NUM_GATES = 100

RANGE_RESOLUTION_M = 30.0

DOPPLER_BINS = np.arange(
    -NUM_PULSES // 2,
    NUM_PULSES // 2,
)

VELOCITY_AXIS = (
    -DOPPLER_BINS
    * WAVELENGTH
    * PRF
    / (2.0 * NUM_PULSES)
)

RANGE_AXIS = (
    np.arange(1, NUM_GATES + 1)
    * RANGE_RESOLUTION_M
)


# ============================================================
# CA-CFAR初始参数
# ============================================================

TRAINING_DOPPLER = 8
TRAINING_RANGE = 6

GUARD_DOPPLER = 2
GUARD_RANGE = 2

FALSE_ALARM_PROBABILITY = 1.0e-4


# ============================================================
# 命中标准
# ============================================================

STRICT_RANGE_TOLERANCE = 1
STRICT_VELOCITY_TOLERANCE = 1

RELAXED_RANGE_TOLERANCE = 4
RELAXED_VELOCITY_TOLERANCE = 1


def calculate_rd_power(
    iq_data: np.ndarray,
) -> np.ndarray:
    """
    根据慢时间复数IQ计算RD线性功率图。
    """

    iq_data = np.asarray(iq_data)

    if iq_data.shape != (
        NUM_PULSES,
        NUM_GATES,
    ):
        raise ValueError(
            f"IQ数据尺寸错误：{iq_data.shape}"
        )

    window = np.hanning(
        NUM_PULSES
    )[:, None]

    rd_complex = np.fft.fftshift(
        np.fft.fft(
            iq_data * window,
            axis=0,
        ),
        axes=0,
    )

    return (
        np.abs(rd_complex) ** 2
    )


def candidate_is_hit(
    candidate: dict,
    label_velocity_index: int,
    label_range_index: int,
    range_tolerance: int,
    velocity_tolerance: int,
) -> bool:
    """
    判断候选点是否位于标签容差范围内。
    """

    range_offset = (
        int(candidate["range_index"])
        - label_range_index
    )

    velocity_offset = (
        int(candidate["velocity_index"])
        - label_velocity_index
    )

    return (
        abs(range_offset)
        <= range_tolerance
        and abs(velocity_offset)
        <= velocity_tolerance
    )


def select_nearest_candidate(
    candidates: list[dict],
    label_velocity_index: int,
    label_range_index: int,
) -> dict | None:
    """
    选择距离标签最近的候选点。

    距离方向按4门归一化，
    速度方向按1单元归一化。
    """

    if not candidates:
        return None

    def distance_score(
        candidate: dict,
    ) -> tuple:
        range_offset = abs(
            int(candidate["range_index"])
            - label_range_index
        )

        velocity_offset = abs(
            int(candidate["velocity_index"])
            - label_velocity_index
        )

        normalized_distance = (
            range_offset
            / RELAXED_RANGE_TOLERANCE
            + velocity_offset
            / RELAXED_VELOCITY_TOLERANCE
        )

        # 距离相同时，优先选择功率更大的候选点。
        return (
            normalized_distance,
            -float(candidate["peak_power"]),
        )

    return min(
        candidates,
        key=distance_score,
    )


def evaluate_one_sample(
    row: pd.Series,
) -> tuple[dict, list[dict]]:
    """
    对一个样本执行H/V联合功率CA-CFAR。
    """

    sample_id = str(
        row["sample_id"]
    )

    mat_path = Path(
        row["mat_path"]
    )

    if not mat_path.exists():
        raise FileNotFoundError(
            f"找不到MAT文件：{mat_path}"
        )

    mat_data = loadmat(mat_path)

    if (
        "local_data_H" not in mat_data
        or "local_data_V" not in mat_data
    ):
        raise KeyError(
            f"{sample_id}缺少H或V通道"
        )

    power_h = calculate_rd_power(
        mat_data["local_data_H"]
    )

    power_v = calculate_rd_power(
        mat_data["local_data_V"]
    )

    # 第一版基线使用H/V联合线性功率。
    combined_power = (
        power_h + power_v
    )

    cfar_result = ca_cfar_2d(
        combined_power,
        training_doppler=TRAINING_DOPPLER,
        training_range=TRAINING_RANGE,
        guard_doppler=GUARD_DOPPLER,
        guard_range=GUARD_RANGE,
        false_alarm_probability=(
            FALSE_ALARM_PROBABILITY
        ),
    )

    candidates = extract_peak_candidates(
        cfar_result.detection_map,
        combined_power,
    )

    label_velocity_index = int(
        row["velocity_index_0"]
    )

    label_range_index = int(
        row["range_index_0"]
    )

    strict_hit_candidates = [
        candidate
        for candidate in candidates
        if candidate_is_hit(
            candidate,
            label_velocity_index,
            label_range_index,
            STRICT_RANGE_TOLERANCE,
            STRICT_VELOCITY_TOLERANCE,
        )
    ]

    relaxed_hit_candidates = [
        candidate
        for candidate in candidates
        if candidate_is_hit(
            candidate,
            label_velocity_index,
            label_range_index,
            RELAXED_RANGE_TOLERANCE,
            RELAXED_VELOCITY_TOLERANCE,
        )
    ]

    false_alarm_candidates = [
        candidate
        for candidate in candidates
        if not candidate_is_hit(
            candidate,
            label_velocity_index,
            label_range_index,
            RELAXED_RANGE_TOLERANCE,
            RELAXED_VELOCITY_TOLERANCE,
        )
    ]

    nearest_candidate = (
        select_nearest_candidate(
            candidates,
            label_velocity_index,
            label_range_index,
        )
    )

    label_power = float(
        combined_power[
            label_velocity_index,
            label_range_index,
        ]
    )

    label_threshold = float(
        cfar_result.threshold_map[
            label_velocity_index,
            label_range_index,
        ]
    )

    if (
        np.isfinite(label_threshold)
        and label_threshold > 0
    ):
        label_margin_db = float(
            10.0
            * np.log10(
                (
                    label_power
                    + 1.0e-12
                )
                / label_threshold
            )
        )
    else:
        label_margin_db = np.nan

    result = {
        "sample_id": sample_id,
        "session_id": row["session_id"],
        "beam_layer": int(
            row["beam_layer"]
        ),
        "azimuth_deg": float(
            row["azimuth_deg"]
        ),
        "distance_m": float(
            row["distance_m"]
        ),
        "velocity_mps": float(
            row["velocity_mps"]
        ),
        "label_range_index": (
            label_range_index
        ),
        "label_velocity_index": (
            label_velocity_index
        ),
        "raw_detection_cell_count": int(
            cfar_result.detection_map.sum()
        ),
        "candidate_count": len(
            candidates
        ),
        "strict_hit": bool(
            strict_hit_candidates
        ),
        "relaxed_hit": bool(
            relaxed_hit_candidates
        ),
        "strict_hit_candidate_count": len(
            strict_hit_candidates
        ),
        "relaxed_hit_candidate_count": len(
            relaxed_hit_candidates
        ),
        "false_alarm_candidate_count": len(
            false_alarm_candidates
        ),
        "label_cell_detected": bool(
            cfar_result.detection_map[
                label_velocity_index,
                label_range_index,
            ]
        ),
        "label_power_db": float(
            10.0
            * np.log10(
                label_power + 1.0e-12
            )
        ),
        "label_threshold_db": (
            float(
                10.0
                * np.log10(
                    label_threshold
                    + 1.0e-12
                )
            )
            if np.isfinite(label_threshold)
            else np.nan
        ),
        "label_margin_over_threshold_db": (
            label_margin_db
        ),
        "nearest_candidate_exists": (
            nearest_candidate is not None
        ),
        "nearest_range_index": np.nan,
        "nearest_velocity_index": np.nan,
        "nearest_range_offset_gates": np.nan,
        "nearest_velocity_offset_bins": np.nan,
        "nearest_range_error_m": np.nan,
        "nearest_velocity_error_mps": np.nan,
        "nearest_peak_power_db": np.nan,
        "read_success": True,
        "error": "",
    }

    if nearest_candidate is not None:
        nearest_range_index = int(
            nearest_candidate[
                "range_index"
            ]
        )

        nearest_velocity_index = int(
            nearest_candidate[
                "velocity_index"
            ]
        )

        result.update(
            {
                "nearest_range_index":
                    nearest_range_index,

                "nearest_velocity_index":
                    nearest_velocity_index,

                "nearest_range_offset_gates":
                    (
                        nearest_range_index
                        - label_range_index
                    ),

                "nearest_velocity_offset_bins":
                    (
                        nearest_velocity_index
                        - label_velocity_index
                    ),

                "nearest_range_error_m":
                    (
                        RANGE_AXIS[
                            nearest_range_index
                        ]
                        - RANGE_AXIS[
                            label_range_index
                        ]
                    ),

                "nearest_velocity_error_mps":
                    (
                        VELOCITY_AXIS[
                            nearest_velocity_index
                        ]
                        - VELOCITY_AXIS[
                            label_velocity_index
                        ]
                    ),

                "nearest_peak_power_db":
                    float(
                        nearest_candidate[
                            "peak_power_db"
                        ]
                    ),
            }
        )

    candidate_records = []

    for rank, candidate in enumerate(
        candidates,
        start=1,
    ):
        candidate_range_index = int(
            candidate["range_index"]
        )

        candidate_velocity_index = int(
            candidate["velocity_index"]
        )

        range_offset = (
            candidate_range_index
            - label_range_index
        )

        velocity_offset = (
            candidate_velocity_index
            - label_velocity_index
        )

        candidate_records.append(
            {
                "sample_id": sample_id,
                "beam_layer": int(
                    row["beam_layer"]
                ),
                "candidate_rank": rank,
                "component_id": int(
                    candidate[
                        "component_id"
                    ]
                ),
                "component_area": int(
                    candidate[
                        "component_area"
                    ]
                ),
                "range_index": (
                    candidate_range_index
                ),
                "velocity_index": (
                    candidate_velocity_index
                ),
                "range_m": float(
                    RANGE_AXIS[
                        candidate_range_index
                    ]
                ),
                "velocity_mps": float(
                    VELOCITY_AXIS[
                        candidate_velocity_index
                    ]
                ),
                "range_offset_gates": (
                    range_offset
                ),
                "velocity_offset_bins": (
                    velocity_offset
                ),
                "peak_power_db": float(
                    candidate[
                        "peak_power_db"
                    ]
                ),
                "strict_hit": (
                    abs(range_offset)
                    <= STRICT_RANGE_TOLERANCE
                    and abs(
                        velocity_offset
                    )
                    <= STRICT_VELOCITY_TOLERANCE
                ),
                "relaxed_hit": (
                    abs(range_offset)
                    <= RELAXED_RANGE_TOLERANCE
                    and abs(
                        velocity_offset
                    )
                    <= RELAXED_VELOCITY_TOLERANCE
                ),
            }
        )

    return result, candidate_records


def build_summary(
    result_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    生成总体和各波束层统计摘要。
    """

    summary_records = []

    groups = [
        ("全部样本", result_df)
    ]

    for beam_layer, beam_df in (
        result_df.groupby("beam_layer")
    ):
        groups.append(
            (
                f"波束层{int(beam_layer)}",
                beam_df,
            )
        )

    for group_name, group_df in groups:
        relaxed_hit_df = group_df.loc[
            group_df["relaxed_hit"]
        ]

        summary_records.append(
            {
                "group": group_name,
                "sample_count": len(
                    group_df
                ),
                "strict_detection_rate": float(
                    group_df[
                        "strict_hit"
                    ].mean()
                ),
                "relaxed_detection_rate": float(
                    group_df[
                        "relaxed_hit"
                    ].mean()
                ),
                "label_cell_detection_rate": float(
                    group_df[
                        "label_cell_detected"
                    ].mean()
                ),
                "average_raw_detection_cells": float(
                    group_df[
                        "raw_detection_cell_count"
                    ].mean()
                ),
                "average_candidate_count": float(
                    group_df[
                        "candidate_count"
                    ].mean()
                ),
                "average_false_alarm_candidates": float(
                    group_df[
                        "false_alarm_candidate_count"
                    ].mean()
                ),
                "median_false_alarm_candidates": float(
                    group_df[
                        "false_alarm_candidate_count"
                    ].median()
                ),
                "mean_absolute_range_error_gates":
                    (
                        float(
                            relaxed_hit_df[
                                "nearest_range_offset_gates"
                            ]
                            .abs()
                            .mean()
                        )
                        if not relaxed_hit_df.empty
                        else np.nan
                    ),
                "mean_absolute_velocity_error_bins":
                    (
                        float(
                            relaxed_hit_df[
                                "nearest_velocity_offset_bins"
                            ]
                            .abs()
                            .mean()
                        )
                        if not relaxed_hit_df.empty
                        else np.nan
                    ),
                "mean_label_margin_db": float(
                    group_df[
                        "label_margin_over_threshold_db"
                    ].mean()
                ),
            }
        )

    return pd.DataFrame(
        summary_records
    )


def main() -> None:
    if not SAMPLES_FILE.exists():
        raise FileNotFoundError(
            f"找不到样本总表："
            f"{SAMPLES_FILE.resolve()}"
        )

    samples_df = pd.read_csv(
        SAMPLES_FILE
    )

    print(
        f"共发现 {len(samples_df)} 个样本。"
    )

    print(
        "\n========== CA-CFAR参数 =========="
    )

    print(
        f"速度训练单元：±{TRAINING_DOPPLER}"
    )

    print(
        f"距离训练单元：±{TRAINING_RANGE}"
    )

    print(
        f"速度保护单元：±{GUARD_DOPPLER}"
    )

    print(
        f"距离保护单元：±{GUARD_RANGE}"
    )

    print(
        "理论单元虚警概率："
        f"{FALSE_ALARM_PROBABILITY:.1e}"
    )

    result_records = []
    all_candidate_records = []

    for index, row in samples_df.iterrows():
        sample_id = str(
            row["sample_id"]
        )

        try:
            (
                result,
                candidate_records,
            ) = evaluate_one_sample(row)

            status = "完成"

        except Exception as exc:
            result = {
                "sample_id": sample_id,
                "read_success": False,
                "error": str(exc),
            }

            candidate_records = []
            status = "异常"

        result_records.append(result)

        all_candidate_records.extend(
            candidate_records
        )

        print(
            f"[{index + 1:03d}/"
            f"{len(samples_df):03d}] "
            f"{status}：{sample_id}"
        )

    result_df = pd.DataFrame(
        result_records
    )

    valid_df = result_df.loc[
        result_df["read_success"] == True
    ].copy()

    if valid_df.empty:
        raise RuntimeError(
            "没有任何样本成功完成CFAR评估"
        )

    candidate_df = pd.DataFrame(
        all_candidate_records
    )

    summary_df = build_summary(
        valid_df
    )

    DETAIL_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        DETAIL_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    candidate_df.to_csv(
        CANDIDATE_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    summary_df.to_csv(
        SUMMARY_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n========== CA-CFAR评估结果 =========="
    )

    print(
        f"样本总数：{len(result_df)}"
    )

    print(
        f"成功样本：{len(valid_df)}"
    )

    print(
        f"异常样本："
        f"{len(result_df) - len(valid_df)}"
    )

    print(
        "\n========== 总体指标 =========="
    )

    overall = summary_df.iloc[0]

    print(
        "严格命中率："
        f"{overall['strict_detection_rate']:.2%}"
    )

    print(
        "宽松命中率："
        f"{overall['relaxed_detection_rate']:.2%}"
    )

    print(
        "标签单元直接过门限比例："
        f"{overall['label_cell_detection_rate']:.2%}"
    )

    print(
        "平均原始检测单元数："
        f"{overall['average_raw_detection_cells']:.2f}"
    )

    print(
        "平均候选目标数："
        f"{overall['average_candidate_count']:.2f}"
    )

    print(
        "平均虚警候选数："
        f"{overall['average_false_alarm_candidates']:.2f}"
    )

    print(
        "命中样本平均绝对距离误差："
        f"{overall['mean_absolute_range_error_gates']:.3f} 门"
    )

    print(
        "命中样本平均绝对速度误差："
        f"{overall['mean_absolute_velocity_error_bins']:.3f} 单元"
    )

    print(
        "\n详细结果："
        f"{DETAIL_OUTPUT.resolve()}"
    )

    print(
        "候选点表："
        f"{CANDIDATE_OUTPUT.resolve()}"
    )

    print(
        "统计摘要："
        f"{SUMMARY_OUTPUT.resolve()}"
    )


if __name__ == "__main__":
    main()