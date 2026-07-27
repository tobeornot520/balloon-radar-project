from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


CLASS_NAMES: Tuple[str, ...] = (
    "uav",
    "balloon_line_array",
    "balloon_solar_panel",
    "balloon_box",
    "balloon_circuit_board",
)
CLASS_TO_LABEL = {name: idx for idx, name in enumerate(CLASS_NAMES)}


@dataclass(frozen=True)
class DetectionRadarConfig:
    """成都无人机检测数据的物理坐标设置。"""

    range_bins: int = 100
    doppler_bins: int = 128
    range_start_m: float = 30.0
    range_resolution_m: float = 30.0
    velocity_resolution_mps: float = 0.183
    zero_doppler_index: int = 64
    doppler_sign: int = -1
    gaussian_sigma: float = 5.0

    def range_to_index(self, distance_m: float) -> int:
        idx = int(round((distance_m - self.range_start_m) / self.range_resolution_m))
        return int(np.clip(idx, 0, self.range_bins - 1))

    def velocity_to_index(self, velocity_mps: float) -> int:
        idx = int(round(self.zero_doppler_index + self.doppler_sign * velocity_mps / self.velocity_resolution_mps))
        return int(np.clip(idx, 0, self.doppler_bins - 1))


def _non_private_keys(mat: Dict[str, Any]) -> List[str]:
    return [k for k in mat.keys() if not k.startswith("__")]


def _load_complex_pair(path: Path, h_key: str, v_key: str) -> Tuple[np.ndarray, np.ndarray]:
    mat = sio.loadmat(path)
    missing = [key for key in (h_key, v_key) if key not in mat]
    if missing:
        raise KeyError(f"{path} 缺少字段: {missing}; 实际字段: {_non_private_keys(mat)}")
    h = np.asarray(mat[h_key])
    v = np.asarray(mat[v_key])
    if h.shape != v.shape:
        raise ValueError(f"{path} 的 H/V 形状不一致: {h.shape} vs {v.shape}")
    if not np.iscomplexobj(h) or not np.iscomplexobj(v):
        raise TypeError(f"{path} 的 H/V 不是复数 IQ")
    if not np.isfinite(h).all() or not np.isfinite(v).all():
        raise ValueError(f"{path} 包含 NaN 或 Inf")
    return h, v


def _parse_label_file(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    values: Dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()

    required = ["Source_File", "Beam_Layer", "Azimuth(deg)", "Distance(m)", "Velocity(m/s)"]
    missing = [k for k in required if k not in values]
    if missing:
        raise ValueError(f"标签文件 {path} 缺少字段: {missing}")
    return {
        "source_file": values["Source_File"],
        "beam_layer": int(values["Beam_Layer"]),
        "azimuth_deg": float(values["Azimuth(deg)"]),
        "distance_m": float(values["Distance(m)"]),
        "velocity_mps": float(values["Velocity(m/s)"]),
    }


def _to_ri4(h: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.stack((h.real, h.imag, v.real, v.imag), axis=0).astype(np.float32, copy=False)


def _to_mag2(h: np.ndarray, v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return np.stack((20.0 * np.log10(np.abs(h) + eps), 20.0 * np.log10(np.abs(v) + eps)), axis=0).astype(np.float32)


def _local_mean_2d(x: np.ndarray, kernel: int = 3) -> np.ndarray:
    """不依赖额外图像库的二维局部均值。"""
    if x.ndim != 2:
        raise ValueError(f"_local_mean_2d 只接受二维数组，收到 {x.shape}")
    pad = kernel // 2
    padded = np.pad(x, ((pad, pad), (pad, pad)), mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    out = (
        integral[kernel:, kernel:]
        - integral[:-kernel, kernel:]
        - integral[kernel:, :-kernel]
        + integral[:-kernel, :-kernel]
    ) / float(kernel * kernel)
    return out


def _to_polar6(h: np.ndarray, v: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """检测任务六通道：[ReH, ImH, ReV, ImV, sZdr, sRhoCo]。"""
    ri4 = _to_ri4(h, v)
    power_h = np.abs(h) ** 2
    power_v = np.abs(v) ** 2
    szdr = 10.0 * np.log10((power_h + eps) / (power_v + eps))
    cross = _local_mean_2d(h * np.conj(v), 3)
    mean_h = _local_mean_2d(power_h, 3)
    mean_v = _local_mean_2d(power_v, 3)
    srho = np.abs(cross) / np.sqrt(mean_h * mean_v + eps)
    srho = np.clip(srho, 0.0, 1.0)
    return np.concatenate((ri4, szdr[None].astype(np.float32), srho[None].astype(np.float32)), axis=0)


def _gaussian_heatmap(rows: int, cols: int, row0: int, col0: int, sigma: float) -> np.ndarray:
    yy, xx = np.mgrid[0:rows, 0:cols]
    heat = np.exp(-((yy - row0) ** 2 + (xx - col0) ** 2) / (2.0 * sigma * sigma))
    return heat.astype(np.float32)[None, ...]


class DetectionRadarDataset(Dataset):
    """
    读取检测数据：background 与 UAV 正样本。

    规范目录：
      root/detection/{split}/background/iq/*.mat
      root/detection/{split}/uav/iq/*.mat
      root/detection/{split}/uav/labels/*.txt

    返回字典：
      x: [C, 128, 100] float32
      heatmap: [1, 128, 100] float32，背景为全零
      target_present: 0/1
      range_index, velocity_index: 背景为 -1
      metadata: 文件和物理标签
    """

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val", "test"] = "train",
        input_mode: Literal["ri4", "mag2", "polar6"] = "ri4",
        config: Optional[DetectionRadarConfig] = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.input_mode = input_mode
        self.config = config or DetectionRadarConfig()
        split_root = self.root / "detection" / split
        self.records: List[Dict[str, Any]] = []

        bg_dir = split_root / "background" / "iq"
        for path in sorted(bg_dir.glob("*.mat")) if bg_dir.exists() else []:
            self.records.append({"mat_path": path, "target_present": 0, "label_path": None})

        uav_iq_dir = split_root / "uav" / "iq"
        uav_label_dir = split_root / "uav" / "labels"
        for path in sorted(uav_iq_dir.glob("*.mat")) if uav_iq_dir.exists() else []:
            label_path = uav_label_dir / f"{path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"找不到 {path.name} 对应标签: {label_path}")
            self.records.append({"mat_path": path, "target_present": 1, "label_path": label_path})

        if not self.records:
            raise FileNotFoundError(f"{split_root} 中没有发现检测数据")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        rec = self.records[index]
        h, v = _load_complex_pair(rec["mat_path"], "local_data_H", "local_data_V")
        expected = (self.config.doppler_bins, self.config.range_bins)
        if h.shape != expected:
            raise ValueError(f"{rec['mat_path']} 形状应为 {expected}，实际为 {h.shape}")

        if self.input_mode == "ri4":
            x = _to_ri4(h, v)
        elif self.input_mode == "mag2":
            x = _to_mag2(h, v)
        elif self.input_mode == "polar6":
            x = _to_polar6(h, v)
        else:
            raise ValueError(f"未知 input_mode: {self.input_mode}")

        target_present = int(rec["target_present"])
        metadata: Dict[str, Any] = {
            "mat_path": str(rec["mat_path"]),
            "filename": rec["mat_path"].name,
            "split": self.split,
            "target_present": target_present,
        }
        if target_present:
            label = _parse_label_file(rec["label_path"])
            range_index = self.config.range_to_index(label["distance_m"])
            velocity_index = self.config.velocity_to_index(label["velocity_mps"])
            heatmap = _gaussian_heatmap(
                self.config.doppler_bins,
                self.config.range_bins,
                velocity_index,
                range_index,
                self.config.gaussian_sigma,
            )
            metadata.update(label)
            metadata["label_path"] = str(rec["label_path"])
        else:
            range_index = -1
            velocity_index = -1
            heatmap = np.zeros((1, self.config.doppler_bins, self.config.range_bins), dtype=np.float32)
            metadata.update({
                "source_file": None,
                "beam_layer": _extract_int(rec["mat_path"].stem, r"beam(\d+)"),
                "azimuth_deg": _extract_int(rec["mat_path"].stem, r"az(\d+)"),
                "distance_m": None,
                "velocity_mps": None,
            })

        return {
            "x": torch.from_numpy(np.ascontiguousarray(x)),
            "heatmap": torch.from_numpy(heatmap),
            "target_present": torch.tensor(target_present, dtype=torch.long),
            "range_index": torch.tensor(range_index, dtype=torch.long),
            "velocity_index": torch.tensor(velocity_index, dtype=torch.long),
            "metadata": metadata,
        }


def _extract_int(text: str, pattern: str) -> Optional[int]:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def polar5_from_iq(h: np.ndarray, v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """分类任务 Polar5，输出 [5, slow_time, range_gate]。"""
    ah = 20.0 * np.log10(np.abs(h) + eps)
    av = 20.0 * np.log10(np.abs(v) + eps)
    rlog = np.log((np.abs(h) + eps) / (np.abs(v) + eps))
    phase_diff = np.angle(h * np.conj(v))
    corr = np.real(h * np.conj(v)) / (np.abs(h) * np.abs(v) + eps)
    corr = np.clip(corr, -1.0, 1.0)
    return np.stack((ah, av, rlog, phase_diff, corr), axis=0).astype(np.float32)


@lru_cache(maxsize=8)
def _load_classification_file_cached(path_str: str) -> Tuple[np.ndarray, np.ndarray]:
    return _load_complex_pair(Path(path_str), "UAV_h", "UAV_v")


class ClassificationRadarDataset(Dataset):
    """
    读取五类分类数据。

    规范目录：
      root/classification/{split}/uav/*.mat
      root/classification/{split}/balloon_line_array/*.mat
      root/classification/{split}/balloon_solar_panel/*.mat
      root/classification/{split}/balloon_box/*.mat
      root/classification/{split}/balloon_circuit_board/*.mat

    每个 .mat 的第三维包含多个样本；本类会把每个 sample_index 展开成独立样本。
    """

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val", "test"] = "train",
        input_mode: Literal["ri4", "polar5"] = "polar5",
        include_angle: bool = True,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.input_mode = input_mode
        self.include_angle = include_angle
        self.records: List[Dict[str, Any]] = []
        split_root = self.root / "classification" / split

        for class_name in CLASS_NAMES:
            class_dir = split_root / class_name
            if not class_dir.exists():
                continue
            for mat_path in sorted(class_dir.glob("*.mat")):
                info = {name: shape for name, shape, _ in sio.whosmat(mat_path)}
                if "UAV_h" not in info or "UAV_v" not in info:
                    raise KeyError(f"{mat_path} 缺少 UAV_h/UAV_v")
                if info["UAV_h"] != info["UAV_v"]:
                    raise ValueError(f"{mat_path} 的 UAV_h/UAV_v 形状不一致")
                shape = info["UAV_h"]
                if len(shape) != 3 or shape[1] < 3:
                    raise ValueError(f"{mat_path} 形状应为 (slow_time, >=3, samples)，实际 {shape}")
                for sample_idx in range(shape[2]):
                    self.records.append({
                        "mat_path": mat_path,
                        "sample_index": sample_idx,
                        "class_name": class_name,
                        "label": CLASS_TO_LABEL[class_name],
                        "shape": shape,
                    })

        if not self.records:
            raise FileNotFoundError(f"{split_root} 中没有发现五类分类数据")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        rec = self.records[index]
        h_all, v_all = _load_classification_file_cached(str(rec["mat_path"]))
        i = int(rec["sample_index"])
        h_raw = h_all[:, :, i]
        v_raw = v_all[:, :, i]
        angle_h = h_raw[:, :2].real.astype(np.float32, copy=False).T
        angle_v = v_raw[:, :2].real.astype(np.float32, copy=False).T
        h = h_raw[:, 2:]
        v = v_raw[:, 2:]

        if self.input_mode == "ri4":
            x = _to_ri4(h, v)
        elif self.input_mode == "polar5":
            x = polar5_from_iq(h, v)
        else:
            raise ValueError(f"未知 input_mode: {self.input_mode}")

        result: Dict[str, Any] = {
            "x": torch.from_numpy(np.ascontiguousarray(x)),
            "label": torch.tensor(rec["label"], dtype=torch.long),
            "class_name": rec["class_name"],
            "sample_index": torch.tensor(i, dtype=torch.long),
            "metadata": {
                "mat_path": str(rec["mat_path"]),
                "filename": rec["mat_path"].name,
                "split": self.split,
                "sample_index": i,
                "class_name": rec["class_name"],
                "label": rec["label"],
                "slow_time": int(h.shape[0]),
                "range_gates": int(h.shape[1]),
            },
        }
        if self.include_angle:
            result["angle_h"] = torch.from_numpy(np.ascontiguousarray(angle_h))
            result["angle_v"] = torch.from_numpy(np.ascontiguousarray(angle_v))
        return result
