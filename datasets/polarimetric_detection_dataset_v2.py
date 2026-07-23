from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.detection_dataset_v2 import DetectionGeometry, _generate_heatmap, _load_iq_pair
from features.polarimetric_rd import (
    PolarimetricConfig,
    explicit_polarimetric_rd,
    make_power2,
    make_ri4,
)
from features.polarimetric_gated_rd import (
    PolarimetricGateConfig,
    make_polar6_gated,
    make_ri8_gated,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "results/data_audit/dataset_v4_multifold/fold_01_manifest.csv"
VALID_SPLITS = {"train", "val", "test"}
VALID_INPUT_MODES = {"power2", "ri4", "polar6_gated", "ri8_gated"}
NUM_PULSES = 128
NUM_GATES = 100
CANONICAL_CHANNELS = 8


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _recover_data_path(value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_file():
        return path.resolve()
    parts = path.parts
    if "data" in parts:
        local = PROJECT_ROOT.joinpath(*parts[parts.index("data"):])
        if local.is_file():
            return local.resolve()
    matches = list((PROJECT_ROOT / "data").rglob(path.name))
    if len(matches) == 1:
        return matches[0].resolve()
    if not matches:
        raise FileNotFoundError(f"Cannot locate data file: {value}")
    raise RuntimeError(f"Ambiguous basename {path.name}: {matches[:5]}")


def _to_float(value: str, default: float = float("nan")) -> float:
    text = str(value).strip()
    return float(text) if text else default


def _to_int(value: str, default: int = -1) -> int:
    text = str(value).strip()
    return int(float(text)) if text else default


def representation_channels(mode: str) -> tuple[str, ...]:
    mapping = {
        "power2": ("H_power", "V_power"),
        "ri4": ("Re_H", "Im_H", "Re_V", "Im_V"),
        "polar6_gated": (
            "H_power", "V_power", "gated_relative_ZDR_like", "gated_local_rho_HV",
            "gated_cos_relative_phase", "gated_sin_relative_phase",
        ),
        "ri8_gated": (
            "Re_H", "Im_H", "Re_V", "Im_V", "gated_relative_ZDR_like",
            "gated_local_rho_HV", "gated_cos_relative_phase", "gated_sin_relative_phase",
        ),
    }
    if mode not in mapping:
        raise ValueError(f"Unsupported input mode: {mode}")
    return mapping[mode]


def to_canonical8(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != (NUM_PULSES, NUM_GATES):
        raise ValueError(f"Expected [C,128,100], got {array.shape}")
    if array.shape[0] > CANONICAL_CHANNELS:
        raise ValueError(f"Too many channels: {array.shape[0]}")
    result = np.zeros((CANONICAL_CHANNELS, NUM_PULSES, NUM_GATES), dtype=np.float32)
    result[: array.shape[0]] = array
    return result


class PolarimetricDetectionDatasetV2(Dataset):
    """Manifest-based, sample-independent H/V representation benchmark dataset v2.

    Every representation is zero-padded to a canonical 8-channel tensor so all
    benchmark modes use exactly the same network architecture and parameter count.
    """

    def __init__(
        self,
        manifest_path: str | Path = DEFAULT_MANIFEST,
        split: Literal["train", "val", "test"] = "train",
        input_mode: Literal["power2", "ri4", "polar6_gated", "ri8_gated"] = "power2",
        range_sigma: float = 5.0,
        velocity_sigma: float = 5.0,
        max_samples: Optional[int] = None,
        geometry: Optional[DetectionGeometry] = None,
        velocity_window: int = 5,
        range_window: int = 3,
        zdr_clip_db: float = 20.0,
        gate_low_percentile: float = 50.0,
        gate_high_percentile: float = 99.0,
        gate_gamma: float = 1.5,
    ) -> None:
        super().__init__()
        if split not in VALID_SPLITS:
            raise ValueError(f"Invalid split: {split}")
        if input_mode not in VALID_INPUT_MODES:
            raise ValueError(f"Invalid input_mode: {input_mode}")
        if range_sigma <= 0 or velocity_sigma <= 0:
            raise ValueError("range_sigma and velocity_sigma must be positive")
        self.manifest_path = _resolve_path(manifest_path)
        self.split = split
        self.input_mode = input_mode
        self.range_sigma = float(range_sigma)
        self.velocity_sigma = float(velocity_sigma)
        self.geometry = geometry or DetectionGeometry()
        self.polar_config = PolarimetricConfig(
            velocity_window=velocity_window,
            range_window=range_window,
            zdr_clip_db=zdr_clip_db,
        )
        self.gate_config = PolarimetricGateConfig(
            low_percentile=gate_low_percentile,
            high_percentile=gate_high_percentile,
            gamma=gate_gamma,
        )
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        records: list[dict[str, Any]] = []
        with self.manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                "new_split", "class_name", "target_present", "sample_id", "source_file",
                "beam_layer", "azimuth_deg", "distance_m", "velocity_mps", "mat_path",
                "label_path",
            }
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Manifest missing columns: {sorted(missing)}")
            for row in reader:
                if row["new_split"] != split:
                    continue
                present = int(row["target_present"])
                if present not in (0, 1):
                    raise ValueError(f"Invalid target_present for {row['sample_id']}")
                mat_path = _recover_data_path(row["mat_path"])
                label_text = str(row["label_path"]).strip()
                label_path = _recover_data_path(label_text) if label_text else None
                if present and label_path is None:
                    raise FileNotFoundError(f"Missing label for {row['sample_id']}")
                records.append({
                    "class_name": row["class_name"],
                    "target_present": present,
                    "sample_id": row["sample_id"],
                    "source_file": row["source_file"],
                    "beam_layer": _to_int(row["beam_layer"]),
                    "azimuth_deg": _to_float(row["azimuth_deg"]),
                    "distance_m": _to_float(row["distance_m"]),
                    "velocity_mps": _to_float(row["velocity_mps"]),
                    "mat_path": mat_path,
                    "label_path": label_path,
                })
        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError("max_samples must be positive")
            records = records[:max_samples]
        if not records:
            raise ValueError(f"No records for split={split}")
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def _make_input(self, h: np.ndarray, v: np.ndarray) -> np.ndarray:
        features = explicit_polarimetric_rd(h, v, self.polar_config)
        if self.input_mode == "power2":
            array = make_power2(features)
        elif self.input_mode == "ri4":
            array = make_ri4(features)
        elif self.input_mode == "polar6_gated":
            array = make_polar6_gated(
                features,
                zdr_clip_db=self.polar_config.zdr_clip_db,
                gate_config=self.gate_config,
            )
        else:
            array = make_ri8_gated(
                features,
                zdr_clip_db=self.polar_config.zdr_clip_db,
                gate_config=self.gate_config,
            )
        return to_canonical8(array)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        h, v = _load_iq_pair(record["mat_path"])
        input_array = self._make_input(h, v)
        present = int(record["target_present"])
        if present:
            distance_m = float(record["distance_m"])
            velocity_mps = float(record["velocity_mps"])
            range_index = self.geometry.range_to_index(distance_m)
            velocity_index = self.geometry.velocity_to_index(velocity_mps)
            target = _generate_heatmap(
                range_index, velocity_index, self.range_sigma, self.velocity_sigma
            )
        else:
            distance_m = float("nan")
            velocity_mps = float("nan")
            range_index = -1
            velocity_index = -1
            target = np.zeros((1, NUM_PULSES, NUM_GATES), dtype=np.float32)
        return {
            "input": torch.from_numpy(np.ascontiguousarray(input_array)).float(),
            "target": torch.from_numpy(np.ascontiguousarray(target)).float(),
            "target_present": torch.tensor(present, dtype=torch.long),
            "sample_id": record["sample_id"],
            "class_name": record["class_name"],
            "source_file": record["source_file"],
            "beam_layer": record["beam_layer"],
            "azimuth_deg": record["azimuth_deg"],
            "distance_m": distance_m,
            "velocity_mps": velocity_mps,
            "range_index": range_index,
            "velocity_index": velocity_index,
            "mat_path": str(record["mat_path"]),
            "label_path": str(record["label_path"]) if record["label_path"] else "",
            "input_mode": self.input_mode,
            "active_channels": len(representation_channels(self.input_mode)),
        }
