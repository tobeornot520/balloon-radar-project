from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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

from utils.plot_config import (
    configure_chinese_font,
)


# ============================================================
# 输入和输出
# ============================================================

SAMPLES_FILE = Path(
    "data/metadata/samples_split.csv"
)

OUTPUT_DIR = Path(
    "results/figures/cfar_final_check"
)

OUTPUT_TABLE = Path(
    "results/tables/cfar_final_check_cases.csv"
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

VELOCITY_RESOLUTION_MPS = abs(
    VELOCITY_AXIS[1]
    - VELOCITY_AXIS[0]
)


# ============================================================
# 最终冻结前使用的CFAR参数
# ============================================================

TRAINING_DOPPLER = 8
TRAINING_RANGE = 6

GUARD_DOPPLER = 2
GUARD_RANGE = 2

PFA = 1.0e-9


# ============================================================
# 目标邻域标准
# ============================================================

STRICT_RANGE_RADIUS = 1
STRICT_VELOCITY_RADIUS = 1

RELAXED_RANGE_RADIUS = 4
RELAXED_VELOCITY_RADIUS = 1


def resolve_project_path(
    path_value: str,
) -> Path:
    path = Path(
        str(path_value)
    )

    if not path.is_absolute():
        path = (
            PROJECT_ROOT / path
        )

    return path


def calculate_rd_power(
    iq_data: np.ndarray,
) -> np.ndarray:
    """
    与前面处理保持一致：
    沿慢时间方向加汉宁窗并做FFT。
    """

    iq_data = np.asarray(
        iq_data
    )

    if iq_data.shape != (
        NUM_PULSES,
        NUM_GATES,
    ):
        raise ValueError(
            f"IQ尺寸错误：{iq_data.shape}"
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
    ).astype(
        np.float64
    )


def detection_exists_in_window(
    detection_map: np.ndarray,
    label_velocity_index: int,
    label_range_index: int,
    velocity_radius: int,
    range_radius: int,
) -> bool:
    velocity_start = max(
        0,
        label_velocity_index
        - velocity_radius,
    )

    velocity_end = min(
        detection_map.shape[0],
        label_velocity_index
        + velocity_radius
        + 1,
    )

    range_start = max(
        0,
        label_range_index
        - range_radius,
    )

    range_end = min(
        detection_map.shape[1],
        label_range_index
        + range_radius
        + 1,
    )

    return bool(
        detection_map[
            velocity_start:velocity_end,
            range_start:range_end,
        ].any()
    )


def candidate_is_target(
    candidate: dict,
    label_velocity_index: int,
    label_range_index: int,
) -> bool:
    range_offset = abs(
        int(candidate["range_index"])
        - label_range_index
    )

    velocity_offset = abs(
        int(candidate["velocity_index"])
        - label_velocity_index
    )

    return (
        range_offset
        <= RELAXED_RANGE_RADIUS
        and velocity_offset
        <= RELAXED_VELOCITY_RADIUS
    )


def find_nearest_candidate(
    candidates: list[dict],
    label_velocity_index: int,
    label_range_index: int,
) -> dict | None:
    if not candidates:
        return None

    def candidate_score(
        candidate: dict,
    ):
        range_offset = abs(
            int(candidate["range_index"])
            - label_range_index
        )

        velocity_offset = abs(
            int(candidate["velocity_index"])
            - label_velocity_index
        )

        return (
            velocity_offset,
            range_offset,
            -float(candidate["peak_power"]),
        )

    return min(
        candidates,
        key=candidate_score,
    )


def analyze_sample(
    row: pd.Series,
) -> dict:
    sample_id = str(
        row["sample_id"]
    )

    mat_path = resolve_project_path(
        row["mat_path"]
    )

    mat_data = loadmat(
        mat_path
    )

    power_h = calculate_rd_power(
        mat_data["local_data_H"]
    )

    power_v = calculate_rd_power(
        mat_data["local_data_V"]
    )

    combined_power = (
        power_h + power_v
    )

    cfar_result = ca_cfar_2d(
        combined_power,
        training_doppler=(
            TRAINING_DOPPLER
        ),
        training_range=(
            TRAINING_RANGE
        ),
        guard_doppler=(
            GUARD_DOPPLER
        ),
        guard_range=(
            GUARD_RANGE
        ),
        false_alarm_probability=PFA,
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

    label_cell_detected = bool(
        cfar_result.detection_map[
            label_velocity_index,
            label_range_index,
        ]
    )

    strict_window_detected = (
        detection_exists_in_window(
            cfar_result.detection_map,
            label_velocity_index,
            label_range_index,
            STRICT_VELOCITY_RADIUS,
            STRICT_RANGE_RADIUS,
        )
    )

    relaxed_window_detected = (
        detection_exists_in_window(
            cfar_result.detection_map,
            label_velocity_index,
            label_range_index,
            RELAXED_VELOCITY_RADIUS,
            RELAXED_RANGE_RADIUS,
        )
    )

    target_candidates = [
        candidate
        for candidate in candidates
        if candidate_is_target(
            candidate,
            label_velocity_index,
            label_range_index,
        )
    ]

    false_candidates = [
        candidate
        for candidate in candidates
        if not candidate_is_target(
            candidate,
            label_velocity_index,
            label_range_index,
        )
    ]

    nearest_candidate = (
        find_nearest_candidate(
            candidates,
            label_velocity_index,
            label_range_index,
        )
    )

    nearest_range_offset = np.nan
    nearest_velocity_offset = np.nan

    if nearest_candidate is not None:
        nearest_range_offset = (
            int(
                nearest_candidate[
                    "range_index"
                ]
            )
            - label_range_index
        )

        nearest_velocity_offset = (
            int(
                nearest_candidate[
                    "velocity_index"
                ]
            )
            - label_velocity_index
        )

    return {
        "sample_id": sample_id,
        "split": str(row["split"]),
        "beam_layer": int(
            row["beam_layer"]
        ),
        "mat_path": str(mat_path),
        "label_range_index": (
            label_range_index
        ),
        "label_velocity_index": (
            label_velocity_index
        ),
        "label_cell_detected": (
            label_cell_detected
        ),
        "strict_window_detected": (
            strict_window_detected
        ),
        "relaxed_window_detected": (
            relaxed_window_detected
        ),
        "raw_detection_cell_count": int(
            cfar_result.detection_map.sum()
        ),
        "candidate_count": len(
            candidates
        ),
        "target_candidate_count": len(
            target_candidates
        ),
        "false_candidate_count": len(
            false_candidates
        ),
        "nearest_range_offset_gates": (
            nearest_range_offset
        ),
        "nearest_velocity_offset_bins": (
            nearest_velocity_offset
        ),
        "combined_power": combined_power,
        "detection_map": (
            cfar_result.detection_map
        ),
        "threshold_map": (
            cfar_result.threshold_map
        ),
        "candidates": candidates,
    }


def select_three_cases(
    analysis_results: list[dict],
) -> list[dict]:
    """
    自动选择三类典型样本。

    1. 正常命中：
       标签单元直接检测，且虚警较少。

    2. 标签单元未命中但邻域命中：
       检查目标能量扩展情况。

    3. 虚警较多：
       选择目标仍被命中、但虚警候选最多的样本。
    """

    normal_candidates = [
        item
        for item in analysis_results
        if (
            item[
                "label_cell_detected"
            ]
            and item[
                "strict_window_detected"
            ]
        )
    ]

    if not normal_candidates:
        raise RuntimeError(
            "没有找到正常命中样本"
        )

    normal_case = min(
        normal_candidates,
        key=lambda item: (
            item[
                "false_candidate_count"
            ],
            item[
                "candidate_count"
            ],
        ),
    )

    neighbor_candidates = [
        item
        for item in analysis_results
        if (
            not item[
                "label_cell_detected"
            ]
            and item[
                "strict_window_detected"
            ]
        )
    ]

    # 如果严格窗口中没有，就退化为宽松窗口。
    if not neighbor_candidates:
        neighbor_candidates = [
            item
            for item in analysis_results
            if (
                not item[
                    "label_cell_detected"
                ]
                and item[
                    "relaxed_window_detected"
                ]
            )
        ]

    if not neighbor_candidates:
        raise RuntimeError(
            "没有找到标签单元未命中、"
            "但邻域命中的样本"
        )

    neighbor_case = min(
        neighbor_candidates,
        key=lambda item: (
            abs(
                item[
                    "nearest_velocity_offset_bins"
                ]
            ),
            abs(
                item[
                    "nearest_range_offset_gates"
                ]
            ),
        ),
    )

    false_alarm_candidates = [
        item
        for item in analysis_results
        if item[
            "relaxed_window_detected"
        ]
    ]

    if not false_alarm_candidates:
        raise RuntimeError(
            "没有找到目标邻域命中的样本"
        )

    false_alarm_case = max(
        false_alarm_candidates,
        key=lambda item: (
            item[
                "false_candidate_count"
            ],
            item[
                "raw_detection_cell_count"
            ],
        ),
    )

    normal_case = normal_case.copy()
    normal_case[
        "case_type"
    ] = "正常命中"

    neighbor_case = neighbor_case.copy()
    neighbor_case[
        "case_type"
    ] = "标签单元未命中但邻域命中"

    false_alarm_case = (
        false_alarm_case.copy()
    )

    false_alarm_case[
        "case_type"
    ] = "虚警较多"

    return [
        normal_case,
        neighbor_case,
        false_alarm_case,
    ]


def get_physical_extent():
    return [
        RANGE_AXIS[0]
        - RANGE_RESOLUTION_M / 2.0,

        RANGE_AXIS[-1]
        + RANGE_RESOLUTION_M / 2.0,

        VELOCITY_AXIS[-1]
        - VELOCITY_RESOLUTION_MPS / 2.0,

        VELOCITY_AXIS[0]
        + VELOCITY_RESOLUTION_MPS / 2.0,
    ]


def add_target_region(
    axis,
    label_range_index: int,
    label_velocity_index: int,
) -> None:
    """
    绘制宽松目标邻域：
    距离±4门、速度±1单元。
    """

    center_range = float(
        RANGE_AXIS[
            label_range_index
        ]
    )

    center_velocity = float(
        VELOCITY_AXIS[
            label_velocity_index
        ]
    )

    rectangle_width = (
        (
            RELAXED_RANGE_RADIUS
            * 2
            + 1
        )
        * RANGE_RESOLUTION_M
    )

    rectangle_height = (
        (
            RELAXED_VELOCITY_RADIUS
            * 2
            + 1
        )
        * VELOCITY_RESOLUTION_MPS
    )

    rectangle = Rectangle(
        (
            center_range
            - rectangle_width / 2.0,

            center_velocity
            - rectangle_height / 2.0,
        ),
        rectangle_width,
        rectangle_height,
        fill=False,
        edgecolor="lime",
        linewidth=1.8,
        linestyle="--",
        label="宽松目标邻域",
    )

    axis.add_patch(
        rectangle
    )


def draw_candidate_points(
    axis,
    candidates: list[dict],
    label_velocity_index: int,
    label_range_index: int,
) -> None:
    """
    绿色圆圈：目标邻域内候选点。
    白色叉号：目标邻域外候选点。
    """

    first_target = True
    first_false = True

    for candidate in candidates:
        candidate_range_index = int(
            candidate[
                "range_index"
            ]
        )

        candidate_velocity_index = int(
            candidate[
                "velocity_index"
            ]
        )

        candidate_range = float(
            RANGE_AXIS[
                candidate_range_index
            ]
        )

        candidate_velocity = float(
            VELOCITY_AXIS[
                candidate_velocity_index
            ]
        )

        if candidate_is_target(
            candidate,
            label_velocity_index,
            label_range_index,
        ):
            axis.plot(
                candidate_range,
                candidate_velocity,
                marker="o",
                markersize=8,
                markerfacecolor="none",
                markeredgecolor="lime",
                markeredgewidth=1.8,
                linestyle="None",
                label=(
                    "目标邻域候选点"
                    if first_target
                    else None
                ),
            )

            first_target = False

        else:
            axis.plot(
                candidate_range,
                candidate_velocity,
                marker="x",
                markersize=7,
                markeredgewidth=1.5,
                color="white",
                linestyle="None",
                label=(
                    "目标区域外候选点"
                    if first_false
                    else None
                ),
            )

            first_false = False


def draw_one_case(
    case: dict,
) -> Path:
    sample_id = case[
        "sample_id"
    ]

    power_map = case[
        "combined_power"
    ]

    detection_map = case[
        "detection_map"
    ]

    candidates = case[
        "candidates"
    ]

    label_range_index = int(
        case[
            "label_range_index"
        ]
    )

    label_velocity_index = int(
        case[
            "label_velocity_index"
        ]
    )

    label_range = float(
        RANGE_AXIS[
            label_range_index
        ]
    )

    label_velocity = float(
        VELOCITY_AXIS[
            label_velocity_index
        ]
    )

    power_db = (
        10.0
        * np.log10(
            power_map + 1.0e-12
        )
    )

    lower_limit = float(
        np.percentile(
            power_db,
            5.0,
        )
    )

    upper_limit = float(
        np.percentile(
            power_db,
            99.5,
        )
    )

    extent = get_physical_extent()

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5.5),
    )

    # --------------------------------------------------------
    # 图1：完整RD图和候选峰
    # --------------------------------------------------------

    image_1 = axes[0].imshow(
        power_db,
        origin="upper",
        aspect="auto",
        extent=extent,
        cmap="jet",
        vmin=lower_limit,
        vmax=upper_limit,
    )

    axes[0].plot(
        label_range,
        label_velocity,
        marker="+",
        markersize=16,
        markeredgewidth=2.5,
        color="red",
        linestyle="None",
        label="标签位置",
    )

    add_target_region(
        axes[0],
        label_range_index,
        label_velocity_index,
    )

    draw_candidate_points(
        axes[0],
        candidates,
        label_velocity_index,
        label_range_index,
    )

    axes[0].set_title(
        "联合RD功率图与候选峰"
    )

    axes[0].set_xlabel(
        "距离（米）"
    )

    axes[0].set_ylabel(
        "速度（米/秒）"
    )

    axes[0].legend(
        fontsize=8,
        loc="upper right",
    )

    figure.colorbar(
        image_1,
        ax=axes[0],
        label="功率（分贝）",
    )

    # --------------------------------------------------------
    # 图2：原始CFAR检测单元
    # --------------------------------------------------------

    axes[1].imshow(
        detection_map.astype(
            np.float32
        ),
        origin="upper",
        aspect="auto",
        extent=extent,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )

    axes[1].plot(
        label_range,
        label_velocity,
        marker="+",
        markersize=16,
        markeredgewidth=2.5,
        color="red",
        linestyle="None",
        label="标签位置",
    )

    add_target_region(
        axes[1],
        label_range_index,
        label_velocity_index,
    )

    axes[1].set_title(
        "CA-CFAR二值检测图"
    )

    axes[1].set_xlabel(
        "距离（米）"
    )

    axes[1].set_ylabel(
        "速度（米/秒）"
    )

    axes[1].legend(
        fontsize=8,
        loc="upper right",
    )

    # --------------------------------------------------------
    # 图3：标签附近局部放大
    # --------------------------------------------------------

    local_range_radius = 10
    local_velocity_radius = 7

    range_start = max(
        0,
        label_range_index
        - local_range_radius,
    )

    range_end = min(
        NUM_GATES,
        label_range_index
        + local_range_radius
        + 1,
    )

    velocity_start = max(
        0,
        label_velocity_index
        - local_velocity_radius,
    )

    velocity_end = min(
        NUM_PULSES,
        label_velocity_index
        + local_velocity_radius
        + 1,
    )

    local_power_db = power_db[
        velocity_start:velocity_end,
        range_start:range_end,
    ]

    local_extent = [
        RANGE_AXIS[range_start]
        - RANGE_RESOLUTION_M / 2.0,

        RANGE_AXIS[range_end - 1]
        + RANGE_RESOLUTION_M / 2.0,

        VELOCITY_AXIS[velocity_end - 1]
        - VELOCITY_RESOLUTION_MPS / 2.0,

        VELOCITY_AXIS[velocity_start]
        + VELOCITY_RESOLUTION_MPS / 2.0,
    ]

    image_3 = axes[2].imshow(
        local_power_db,
        origin="upper",
        aspect="auto",
        extent=local_extent,
        cmap="jet",
        vmin=lower_limit,
        vmax=upper_limit,
    )

    axes[2].plot(
        label_range,
        label_velocity,
        marker="+",
        markersize=18,
        markeredgewidth=2.8,
        color="red",
        linestyle="None",
        label="标签位置",
    )

    add_target_region(
        axes[2],
        label_range_index,
        label_velocity_index,
    )

    draw_candidate_points(
        axes[2],
        candidates,
        label_velocity_index,
        label_range_index,
    )

    axes[2].set_xlim(
        local_extent[0],
        local_extent[1],
    )

    axes[2].set_ylim(
        local_extent[2],
        local_extent[3],
    )

    axes[2].set_title(
        "标签附近局部放大"
    )

    axes[2].set_xlabel(
        "距离（米）"
    )

    axes[2].set_ylabel(
        "速度（米/秒）"
    )

    axes[2].legend(
        fontsize=8,
        loc="upper right",
    )

    figure.colorbar(
        image_3,
        ax=axes[2],
        label="功率（分贝）",
    )

    figure.suptitle(
        (
            f"{case['case_type']}｜"
            f"样本：{sample_id}｜"
            f"波束层：{case['beam_layer']}｜"
            f"标签单元检测："
            f"{'是' if case['label_cell_detected'] else '否'}｜"
            f"候选数：{case['candidate_count']}｜"
            f"虚警候选：{case['false_candidate_count']}"
        ),
        fontsize=13,
    )

    figure.tight_layout()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename_map = {
        "正常命中":
            "01_normal_hit.png",

        "标签单元未命中但邻域命中":
            "02_neighbor_hit.png",

        "虚警较多":
            "03_many_false_alarms.png",
    }

    output_path = (
        OUTPUT_DIR
        / filename_map[
            case["case_type"]
        ]
    )

    figure.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return output_path


def main() -> None:
    configure_chinese_font()

    if not SAMPLES_FILE.exists():
        raise FileNotFoundError(
            f"找不到数据集文件："
            f"{SAMPLES_FILE.resolve()}"
        )

    samples_df = pd.read_csv(
        SAMPLES_FILE
    )

    print(
        "========== CFAR收尾检查 =========="
    )

    print(
        f"样本数：{len(samples_df)}"
    )

    print(
        f"PFA：{PFA:.1e}"
    )

    print(
        "开始计算全部样本，"
        "自动筛选三类典型案例。"
    )

    analysis_results = []

    for index, row in (
        samples_df.iterrows()
    ):
        result = analyze_sample(
            row
        )

        analysis_results.append(
            result
        )

        print(
            f"[{index + 1:03d}/"
            f"{len(samples_df):03d}] "
            f"完成：{result['sample_id']}"
        )

    selected_cases = (
        select_three_cases(
            analysis_results
        )
    )

    table_records = []

    print(
        "\n========== 选中的典型案例 =========="
    )

    for case in selected_cases:
        output_path = draw_one_case(
            case
        )

        table_records.append(
            {
                "case_type":
                    case[
                        "case_type"
                    ],

                "sample_id":
                    case[
                        "sample_id"
                    ],

                "split":
                    case[
                        "split"
                    ],

                "beam_layer":
                    case[
                        "beam_layer"
                    ],

                "label_cell_detected":
                    case[
                        "label_cell_detected"
                    ],

                "strict_window_detected":
                    case[
                        "strict_window_detected"
                    ],

                "relaxed_window_detected":
                    case[
                        "relaxed_window_detected"
                    ],

                "raw_detection_cell_count":
                    case[
                        "raw_detection_cell_count"
                    ],

                "candidate_count":
                    case[
                        "candidate_count"
                    ],

                "target_candidate_count":
                    case[
                        "target_candidate_count"
                    ],

                "false_candidate_count":
                    case[
                        "false_candidate_count"
                    ],

                "nearest_range_offset_gates":
                    case[
                        "nearest_range_offset_gates"
                    ],

                "nearest_velocity_offset_bins":
                    case[
                        "nearest_velocity_offset_bins"
                    ],

                "figure_path":
                    str(
                        output_path
                    ),
            }
        )

        print(
            f"\n类型：{case['case_type']}"
        )

        print(
            f"样本：{case['sample_id']}"
        )

        print(
            "标签单元检测："
            f"{case['label_cell_detected']}"
        )

        print(
            "严格邻域检测："
            f"{case['strict_window_detected']}"
        )

        print(
            "宽松邻域检测："
            f"{case['relaxed_window_detected']}"
        )

        print(
            f"候选点数："
            f"{case['candidate_count']}"
        )

        print(
            f"虚警候选数："
            f"{case['false_candidate_count']}"
        )

        print(
            "最近候选距离偏移："
            f"{case['nearest_range_offset_gates']}门"
        )

        print(
            "最近候选速度偏移："
            f"{case['nearest_velocity_offset_bins']}单元"
        )

        print(
            f"图片：{output_path.resolve()}"
        )

    output_df = pd.DataFrame(
        table_records
    )

    OUTPUT_TABLE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_df.to_csv(
        OUTPUT_TABLE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n========== 输出完成 =========="
    )

    print(
        f"案例表：{OUTPUT_TABLE.resolve()}"
    )

    print(
        f"图片文件夹：{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()