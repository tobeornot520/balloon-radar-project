from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "detection_dataset"
NUM_PULSES = 128
NUM_GATES = 100
VALID_SPLITS = {"train", "val", "test"}
VALID_CHANNEL_MODES = {"H", "V", "HV"}


@dataclass(frozen=True)
class DetectionGeometry:
    range_start_m: float = 30.0
    range_resolution_m: float = 30.0
    velocity_resolution_mps: float = 0.183
    zero_doppler_index: int = 64
    doppler_sign: int = -1

    def range_to_index(self, distance_m: float) -> int:
        idx = int(round((distance_m - self.range_start_m) / self.range_resolution_m))
        if not 0 <= idx < NUM_GATES:
            raise IndexError(f"距离 {distance_m} m 映射到下标 {idx}，超出 0~{NUM_GATES - 1}")
        return idx

    def velocity_to_index(self, velocity_mps: float) -> int:
        idx = int(round(self.zero_doppler_index + self.doppler_sign * velocity_mps / self.velocity_resolution_mps))
        if not 0 <= idx < NUM_PULSES:
            raise IndexError(f"速度 {velocity_mps} m/s 映射到下标 {idx}，超出 0~{NUM_PULSES - 1}")
        return idx


def _resolve_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def _parse_label(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    values: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()

    required = ("Source_File", "Beam_Layer", "Azimuth(deg)", "Distance(m)", "Velocity(m/s)")
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(f"标签 {path} 缺少字段: {missing}")

    return {
        "source_file": values["Source_File"],
        "beam_layer": int(values["Beam_Layer"]),
        "azimuth_deg": float(values["Azimuth(deg)"]),
        "distance_m": float(values["Distance(m)"]),
        "velocity_mps": float(values["Velocity(m/s)"]),
    }


def _extract_int(text: str, pattern: str) -> Optional[int]:
    m = re.search(pattern, text, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def _load_iq_pair(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = loadmat(path)
    missing = {"local_data_H", "local_data_V"} - set(data)
    if missing:
        raise KeyError(f"{path} 缺少字段: {sorted(missing)}")
    h = np.asarray(data["local_data_H"])
    v = np.asarray(data["local_data_V"])
    if h.shape != (NUM_PULSES, NUM_GATES) or v.shape != (NUM_PULSES, NUM_GATES):
        raise ValueError(f"{path.name} 的 H/V 形状应为 {(NUM_PULSES, NUM_GATES)}，实际 {h.shape}/{v.shape}")
    if not np.iscomplexobj(h) or not np.iscomplexobj(v):
        raise TypeError(f"{path.name} 不是复数 IQ")
    if not np.isfinite(h).all() or not np.isfinite(v).all():
        raise ValueError(f"{path.name} 包含 NaN 或 Inf")
    return h, v


def _calculate_rd_power(iq: np.ndarray) -> np.ndarray:
    window = np.hanning(NUM_PULSES).astype(np.float32)[:, None]
    rd = np.fft.fftshift(np.fft.fft(iq * window, axis=0), axes=0)
    return (np.abs(rd) ** 2).astype(np.float32, copy=False)


def _normalize_power_db(power: np.ndarray) -> np.ndarray:
    db = 10.0 * np.log10(np.asarray(power, dtype=np.float32) + 1e-12)
    low = float(np.percentile(db, 1.0))
    high = float(np.percentile(db, 99.0))
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("RD 图归一化分位数无效")
    if high <= low:
        return np.zeros_like(db, dtype=np.float32)
    return np.clip((db - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _generate_heatmap(
    range_index: int,
    velocity_index: int,
    range_sigma: float,
    velocity_sigma: float,
) -> np.ndarray:
    r = np.arange(NUM_GATES, dtype=np.float32)[None, :]
    v = np.arange(NUM_PULSES, dtype=np.float32)[:, None]
    heatmap = np.exp(
        -(
            (r - range_index) ** 2 / (2.0 * range_sigma**2)
            + (v - velocity_index) ** 2 / (2.0 * velocity_sigma**2)
        )
    )
    return heatmap.astype(np.float32, copy=False)[None]


class DetectionRadarDatasetV2(Dataset):
    """直接读取当前目录结构中的 Background 与 UAV 检测数据。

    目录结构::

        data/raw/detection_dataset/{split}/
        ├── Background_IQ/IQ_Data/*.mat
        └── UAV_IQ/
            ├── IQ_Data/*.mat
            └── Labels/*.txt

    返回字段与旧 RadarDataset 尽量保持一致：
      input [C,128,100]、target [1,128,100]、target_present、
      range_index、velocity_index、距离/速度/波束等元数据。
    """

    def __init__(
        self,
        data_root: str | Path = DEFAULT_DATA_ROOT,
        split: Literal["train", "val", "test"] = "train",
        channel_mode: Literal["H", "V", "HV"] = "HV",
        range_sigma: float = 2.0,
        velocity_sigma: float = 1.0,
        include_background: bool = True,
        include_uav: bool = True,
        max_samples: Optional[int] = None,
        geometry: Optional[DetectionGeometry] = None,
    ) -> None:
        super().__init__()
        if split not in VALID_SPLITS:
            raise ValueError(f"split 必须是 {sorted(VALID_SPLITS)}，当前为 {split!r}")
        if channel_mode not in VALID_CHANNEL_MODES:
            raise ValueError(f"channel_mode 必须是 {sorted(VALID_CHANNEL_MODES)}，当前为 {channel_mode!r}")
        if range_sigma <= 0 or velocity_sigma <= 0:
            raise ValueError("range_sigma 和 velocity_sigma 必须大于 0")
        if not include_background and not include_uav:
            raise ValueError("include_background 与 include_uav 不能同时为 False")

        self.data_root = _resolve_path(data_root)
        self.split = split
        self.channel_mode = channel_mode
        self.range_sigma = float(range_sigma)
        self.velocity_sigma = float(velocity_sigma)
        self.geometry = geometry or DetectionGeometry()
        split_root = self.data_root / split
        if not split_root.exists():
            raise FileNotFoundError(f"找不到数据划分目录: {split_root}")

        records: list[dict[str, Any]] = []
        if include_background:
            bg_dir = split_root / "Background_IQ" / "IQ_Data"
            for mat_path in sorted(bg_dir.glob("*.mat")):
                records.append({"mat_path": mat_path, "label_path": None, "target_present": 0})

        if include_uav:
            iq_dir = split_root / "UAV_IQ" / "IQ_Data"
            label_dir = split_root / "UAV_IQ" / "Labels"
            for mat_path in sorted(iq_dir.glob("*.mat")):
                label_path = label_dir / f"{mat_path.stem}.txt"
                if not label_path.exists():
                    raise FileNotFoundError(f"{mat_path.name} 缺少对应标签 {label_path}")
                records.append({"mat_path": mat_path, "label_path": label_path, "target_present": 1})

        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError("max_samples 必须大于 0")
            records = records[:max_samples]
        if not records:
            raise ValueError(f"{split_root} 中未发现可读取样本")
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rec = self.records[index]
        h, v = _load_iq_pair(rec["mat_path"])
        rd_h = _normalize_power_db(_calculate_rd_power(h))
        rd_v = _normalize_power_db(_calculate_rd_power(v))

        if self.channel_mode == "H":
            input_array = rd_h[None]
        elif self.channel_mode == "V":
            input_array = rd_v[None]
        else:
            input_array = np.stack((rd_h, rd_v), axis=0)

        target_present = int(rec["target_present"])
        if target_present:
            label = _parse_label(rec["label_path"])
            range_index = self.geometry.range_to_index(label["distance_m"])
            velocity_index = self.geometry.velocity_to_index(label["velocity_mps"])
            target = _generate_heatmap(
                range_index,
                velocity_index,
                self.range_sigma,
                self.velocity_sigma,
            )
            beam_layer = label["beam_layer"]
            azimuth_deg = label["azimuth_deg"]
            distance_m = label["distance_m"]
            velocity_mps = label["velocity_mps"]
            source_file = label["source_file"]
        else:
            target = np.zeros((1, NUM_PULSES, NUM_GATES), dtype=np.float32)
            range_index = -1
            velocity_index = -1
            beam_layer = _extract_int(rec["mat_path"].stem, r"beam(\d+)")
            azimuth_deg = _extract_int(rec["mat_path"].stem, r"az(\d+)")
            distance_m = float("nan")
            velocity_mps = float("nan")
            source_file = rec["mat_path"].stem.split("_beam", 1)[0]

        sample_id = rec["mat_path"].stem
        return {
            "input": torch.from_numpy(np.ascontiguousarray(input_array)).float(),
            "target": torch.from_numpy(np.ascontiguousarray(target)).float(),
            "target_present": torch.tensor(target_present, dtype=torch.long),
            "sample_id": sample_id,
            "beam_layer": -1 if beam_layer is None else int(beam_layer),
            "azimuth_deg": float("nan") if azimuth_deg is None else float(azimuth_deg),
            "distance_m": float(distance_m),
            "velocity_mps": float(velocity_mps),
            "range_index": int(range_index),
            "velocity_index": int(velocity_index),
            "source_file": source_file,
            "mat_path": str(rec["mat_path"]),
            "label_path": "" if rec["label_path"] is None else str(rec["label_path"]),
        }
