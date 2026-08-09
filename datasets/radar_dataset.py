from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA_FILE = (
    PROJECT_ROOT / "data" / "metadata" / "samples_split.csv"
)

NUM_PULSES = 128
NUM_GATES = 100
VALID_SPLITS = {"train", "val", "test"}
VALID_CHANNEL_MODES = {"H", "V", "HV"}


class RadarDataset(Dataset):
    """
    基于 H/V 双通道复数 IQ 数据生成距离-多普勒图的数据集。

    参数
    ----
    metadata_file:
        数据清单路径。相对路径以项目根目录为基准。
    split:
        train、val 或 test。
    channel_mode:
        "H"  -> 输入形状 [1, 128, 100]
        "V"  -> 输入形状 [1, 128, 100]
        "HV" -> 输入形状 [2, 128, 100]，默认模式。
    range_sigma:
        二维高斯标签在距离维的标准差。
    velocity_sigma:
        二维高斯标签在速度维的标准差。
    max_samples:
        可选，仅取当前 split 的前若干个样本，便于快速调试。

    返回字典
    --------
    input:
        float32 张量，形状由 channel_mode 决定。
    target:
        float32 二维高斯热力图，[1, 128, 100]。
    sample_id, beam_layer, distance_m, velocity_mps:
        样本元信息。
    range_index, velocity_index:
        0 基目标下标。
    """

    def __init__(
        self,
        metadata_file: Path | str = DEFAULT_METADATA_FILE,
        split: str = "train",
        channel_mode: str = "HV",
        range_sigma: float = 2.0,
        velocity_sigma: float = 1.0,
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.metadata_file = Path(metadata_file)
        if not self.metadata_file.is_absolute():
            self.metadata_file = PROJECT_ROOT / self.metadata_file
        self.metadata_file = self.metadata_file.resolve()

        if not self.metadata_file.exists():
            raise FileNotFoundError(
                f"找不到数据集文件：{self.metadata_file}"
            )

        if split not in VALID_SPLITS:
            raise ValueError(
                "split 必须为 train、val 或 test，"
                f"当前值为：{split!r}"
            )

        if channel_mode not in VALID_CHANNEL_MODES:
            raise ValueError(
                "channel_mode 必须为 H、V 或 HV，"
                f"当前值为：{channel_mode!r}"
            )

        if range_sigma <= 0:
            raise ValueError("range_sigma 必须大于 0")

        if velocity_sigma <= 0:
            raise ValueError("velocity_sigma 必须大于 0")

        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples 必须大于 0 或设为 None")

        metadata_df = pd.read_csv(self.metadata_file)

        required_columns = {
            "sample_id",
            "mat_path",
            "split",
            "range_index_0",
            "velocity_index_0",
            "beam_layer",
            "distance_m",
            "velocity_mps",
        }
        missing_columns = required_columns - set(metadata_df.columns)
        if missing_columns:
            raise KeyError(
                "数据清单缺少字段："
                + ", ".join(sorted(missing_columns))
            )

        samples_df = (
            metadata_df.loc[metadata_df["split"] == split]
            .copy()
            .reset_index(drop=True)
        )

        if max_samples is not None:
            samples_df = (
                samples_df.iloc[:max_samples]
                .copy()
                .reset_index(drop=True)
            )

        if samples_df.empty:
            raise ValueError(f"{split} 集合没有样本")

        self.samples_df = samples_df
        self.split = split
        self.channel_mode = channel_mode
        self.range_sigma = float(range_sigma)
        self.velocity_sigma = float(velocity_sigma)

    def __len__(self) -> int:
        return len(self.samples_df)

    @staticmethod
    def _resolve_mat_path(mat_path_value: object) -> Path:
        path = Path(str(mat_path_value))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

    @staticmethod
    def _calculate_rd_power(iq_data: np.ndarray) -> np.ndarray:
        """沿慢时间维（第 0 维）加 Hann 窗并执行多普勒 FFT。"""
        iq_array = np.asarray(iq_data)

        if iq_array.shape != (NUM_PULSES, NUM_GATES):
            raise ValueError(
                "IQ 尺寸错误："
                f"{iq_array.shape}，期望 {(NUM_PULSES, NUM_GATES)}"
            )

        if not np.iscomplexobj(iq_array):
            raise ValueError("IQ 数据必须为复数数组")

        if not np.isfinite(iq_array).all():
            raise ValueError("IQ 数据中存在 NaN 或 Inf")

        window = np.hanning(NUM_PULSES).astype(np.float32)[:, None]
        rd_complex = np.fft.fftshift(
            np.fft.fft(iq_array * window, axis=0),
            axes=0,
        )
        power = np.abs(rd_complex) ** 2

        return power.astype(np.float32, copy=False)

    @staticmethod
    def _power_to_normalized_db(power: np.ndarray) -> np.ndarray:
        """将功率转为 dB，并按 1%～99% 分位裁剪到 [0, 1]。"""
        power_array = np.asarray(power, dtype=np.float32)

        if power_array.shape != (NUM_PULSES, NUM_GATES):
            raise ValueError(
                "RD 功率图尺寸错误："
                f"{power_array.shape}，"
                f"期望 {(NUM_PULSES, NUM_GATES)}"
            )

        power_db = 10.0 * np.log10(power_array + 1e-12)
        low = float(np.percentile(power_db, 1))
        high = float(np.percentile(power_db, 99))

        if not np.isfinite(low) or not np.isfinite(high):
            raise ValueError("RD 功率图归一化分位数无效")

        if high <= low:
            return np.zeros_like(power_db, dtype=np.float32)

        normalized = (power_db - low) / (high - low)
        return np.clip(normalized, 0.0, 1.0).astype(
            np.float32,
            copy=False,
        )

    @staticmethod
    def _generate_heatmap(
        range_idx: int,
        velocity_idx: int,
        range_sigma: float,
        velocity_sigma: float,
    ) -> np.ndarray:
        """生成峰值位于标签坐标的二维高斯目标热力图。"""
        if not 0 <= range_idx < NUM_GATES:
            raise IndexError(
                f"距离下标越界：{range_idx}，有效范围为 0～{NUM_GATES - 1}"
            )

        if not 0 <= velocity_idx < NUM_PULSES:
            raise IndexError(
                "速度下标越界："
                f"{velocity_idx}，有效范围为 0～{NUM_PULSES - 1}"
            )

        range_axis = np.arange(NUM_GATES, dtype=np.float32)
        velocity_axis = np.arange(NUM_PULSES, dtype=np.float32)
        velocity_grid, range_grid = np.meshgrid(
            velocity_axis,
            range_axis,
            indexing="ij",
        )

        heatmap = np.exp(
            -(
                ((range_grid - range_idx) ** 2)
                / (2.0 * range_sigma**2)
                + ((velocity_grid - velocity_idx) ** 2)
                / (2.0 * velocity_sigma**2)
            )
        )

        return heatmap.astype(np.float32, copy=False)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.samples_df.iloc[index]
        mat_path = self._resolve_mat_path(row["mat_path"])

        if not mat_path.exists():
            raise FileNotFoundError(f"找不到 MAT 文件：{mat_path}")

        data = loadmat(mat_path)
        required_variables = {"local_data_H", "local_data_V"}
        missing_variables = required_variables - set(data.keys())
        if missing_variables:
            raise KeyError(
                f"{mat_path.name} 缺少变量："
                + ", ".join(sorted(missing_variables))
            )

        iq_h = data["local_data_H"]
        iq_v = data["local_data_V"]

        rd_h = self._power_to_normalized_db(
            self._calculate_rd_power(iq_h)
        )
        rd_v = self._power_to_normalized_db(
            self._calculate_rd_power(iq_v)
        )

        if self.channel_mode == "H":
            input_array = rd_h[None, :, :]
        elif self.channel_mode == "V":
            input_array = rd_v[None, :, :]
        else:
            input_array = np.stack([rd_h, rd_v], axis=0)

        range_idx = int(row["range_index_0"])
        velocity_idx = int(row["velocity_index_0"])

        target = self._generate_heatmap(
            range_idx=range_idx,
            velocity_idx=velocity_idx,
            range_sigma=self.range_sigma,
            velocity_sigma=self.velocity_sigma,
        )

        input_tensor = torch.from_numpy(
            np.ascontiguousarray(input_array)
        ).float()
        target_tensor = torch.from_numpy(
            np.ascontiguousarray(target[None, :, :])
        ).float()

        return {
            "input": input_tensor,
            "target": target_tensor,
            "sample_id": str(row["sample_id"]),
            "beam_layer": int(row["beam_layer"]),
            "distance_m": float(row["distance_m"]),
            "velocity_mps": float(row["velocity_mps"]),
            "range_index": range_idx,
            "velocity_index": velocity_idx,
        }
