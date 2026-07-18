from dataclasses import dataclass

import numpy as np
from scipy.ndimage import convolve
from scipy.ndimage import label as connected_component_label


@dataclass
class CFARResult:
    """二维CA-CFAR的输出结果。"""

    detection_map: np.ndarray
    threshold_map: np.ndarray
    noise_map: np.ndarray
    training_cell_count: int
    threshold_factor: float


def ca_cfar_2d(
    power_map: np.ndarray,
    training_doppler: int = 8,
    training_range: int = 6,
    guard_doppler: int = 2,
    guard_range: int = 2,
    false_alarm_probability: float = 1.0e-4,
) -> CFARResult:
    """
    对二维线性功率图执行CA-CFAR。

    参数
    ----------
    power_map:
        二维线性功率图，形状为：
        [速度单元数, 距离门数]

    training_doppler:
        速度方向单侧训练单元数。

    training_range:
        距离方向单侧训练单元数。

    guard_doppler:
        速度方向单侧保护单元数。

    guard_range:
        距离方向单侧保护单元数。

    false_alarm_probability:
        理论单元虚警概率。

    返回
    ----------
    CFARResult
        检测图、门限图、噪声估计图等。
    """

    power_map = np.asarray(
        power_map,
        dtype=np.float64,
    )

    if power_map.ndim != 2:
        raise ValueError(
            f"输入必须是二维数组，当前形状为：{power_map.shape}"
        )

    if not np.isfinite(power_map).all():
        raise ValueError(
            "输入功率图中存在NaN或无穷值"
        )

    if np.any(power_map < 0):
        raise ValueError(
            "CA-CFAR必须使用非负的线性功率图"
        )

    if not 0 < false_alarm_probability < 1:
        raise ValueError(
            "虚警概率必须位于0和1之间"
        )

    outer_doppler = (
        training_doppler
        + guard_doppler
    )

    outer_range = (
        training_range
        + guard_range
    )

    kernel_height = (
        2 * outer_doppler + 1
    )

    kernel_width = (
        2 * outer_range + 1
    )

    training_kernel = np.ones(
        (kernel_height, kernel_width),
        dtype=np.float64,
    )

    center_doppler = outer_doppler
    center_range = outer_range

    # 去掉保护单元和被检测单元。
    training_kernel[
        center_doppler - guard_doppler:
        center_doppler + guard_doppler + 1,

        center_range - guard_range:
        center_range + guard_range + 1,
    ] = 0.0

    training_cell_count = int(
        training_kernel.sum()
    )

    if training_cell_count <= 0:
        raise ValueError(
            "训练单元数量必须大于0"
        )

    training_sum = convolve(
        power_map,
        training_kernel,
        mode="constant",
        cval=0.0,
    )

    noise_map = (
        training_sum
        / training_cell_count
    )

    # 指数噪声假设下的CA-CFAR门限系数。
    threshold_factor = (
        training_cell_count
        * (
            false_alarm_probability
            ** (-1.0 / training_cell_count)
            - 1.0
        )
    )

    threshold_map = (
        noise_map
        * threshold_factor
    )

    valid_map = np.zeros_like(
        power_map,
        dtype=bool,
    )

    valid_map[
        outer_doppler:
        power_map.shape[0] - outer_doppler,

        outer_range:
        power_map.shape[1] - outer_range,
    ] = True

    detection_map = (
        (power_map > threshold_map)
        & valid_map
    )

    threshold_map = threshold_map.copy()
    noise_map = noise_map.copy()

    threshold_map[~valid_map] = np.nan
    noise_map[~valid_map] = np.nan

    return CFARResult(
        detection_map=detection_map,
        threshold_map=threshold_map,
        noise_map=noise_map,
        training_cell_count=training_cell_count,
        threshold_factor=float(
            threshold_factor
        ),
    )


def extract_peak_candidates(
    detection_map: np.ndarray,
    power_map: np.ndarray,
) -> list[dict]:
    """
    将相邻检测单元合并为连通域，
    每个连通域保留功率最大的一个候选点。
    """

    detection_map = np.asarray(
        detection_map,
        dtype=bool,
    )

    power_map = np.asarray(
        power_map,
        dtype=np.float64,
    )

    if (
        detection_map.shape
        != power_map.shape
    ):
        raise ValueError(
            "检测图和功率图尺寸不一致"
        )

    # 8邻域连通。
    structure = np.ones(
        (3, 3),
        dtype=np.int8,
    )

    component_map, component_count = (
        connected_component_label(
            detection_map,
            structure=structure,
        )
    )

    candidates = []

    for component_id in range(
        1,
        component_count + 1,
    ):
        component_indices = np.argwhere(
            component_map == component_id
        )

        if component_indices.size == 0:
            continue

        component_power = power_map[
            component_map == component_id
        ]

        local_maximum_index = int(
            np.argmax(component_power)
        )

        peak_velocity_index = int(
            component_indices[
                local_maximum_index,
                0,
            ]
        )

        peak_range_index = int(
            component_indices[
                local_maximum_index,
                1,
            ]
        )

        peak_power = float(
            power_map[
                peak_velocity_index,
                peak_range_index,
            ]
        )

        candidates.append(
            {
                "component_id": component_id,
                "component_area": int(
                    len(component_indices)
                ),
                "velocity_index": (
                    peak_velocity_index
                ),
                "range_index": (
                    peak_range_index
                ),
                "peak_power": peak_power,
                "peak_power_db": float(
                    10.0
                    * np.log10(
                        peak_power + 1.0e-12
                    )
                ),
            }
        )

    candidates.sort(
        key=lambda item: item["peak_power"],
        reverse=True,
    )

    return candidates