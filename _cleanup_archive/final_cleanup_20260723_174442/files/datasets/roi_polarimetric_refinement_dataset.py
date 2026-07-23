from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.detection_dataset_v2 import _load_iq_pair
from datasets.polarimetric_detection_dataset_v2 import PolarimetricDetectionDatasetV2
from features.polarimetric_gated_rd import PolarimetricGateConfig
from features.polarimetric_rd import PolarimetricConfig, explicit_polarimetric_rd
from features.roi_polarimetric_refinement import (
    ROI_MODES,
    build_roi_source,
    canonical_roi_channels,
)


class ROIPolarimetricSourceDataset(Dataset):
    """Full-frame source dataset used only while building the frozen-candidate cache."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: Literal["train", "val", "test"],
        *,
        range_sigma: float = 5.0,
        velocity_sigma: float = 5.0,
        velocity_window: int = 5,
        range_window: int = 3,
        zdr_clip_db: float = 20.0,
        gate_low_percentile: float = 50.0,
        gate_high_percentile: float = 99.0,
        gate_gamma: float = 1.5,
        debug_per_class: int = 0,
    ) -> None:
        base = PolarimetricDetectionDatasetV2(
            manifest_path=manifest_path,
            split=split,
            input_mode="power2",
            range_sigma=range_sigma,
            velocity_sigma=velocity_sigma,
            velocity_window=velocity_window,
            range_window=range_window,
            zdr_clip_db=zdr_clip_db,
            gate_low_percentile=gate_low_percentile,
            gate_high_percentile=gate_high_percentile,
            gate_gamma=gate_gamma,
        )
        records = list(base.records)
        if debug_per_class:
            if debug_per_class <= 0:
                raise ValueError("debug_per_class must be non-negative")
            selected: list[dict[str, Any]] = []
            for present in (0, 1):
                selected.extend(
                    [row for row in records if int(row["target_present"]) == present][
                        :debug_per_class
                    ]
                )
            records = selected
        if not records:
            raise ValueError(f"No records for split={split}")
        self.records = records
        self.split = split
        self.geometry = base.geometry
        self.range_sigma = float(range_sigma)
        self.velocity_sigma = float(velocity_sigma)
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

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        h, v = _load_iq_pair(record["mat_path"])
        features = explicit_polarimetric_rd(h, v, self.polar_config)
        source, confidence = build_roi_source(
            features,
            zdr_clip_db=self.polar_config.zdr_clip_db,
            gate_config=self.gate_config,
        )
        return {
            "roi_source": torch.from_numpy(np.ascontiguousarray(source)).float(),
            "confidence_map": torch.from_numpy(np.ascontiguousarray(confidence)).float(),
            "target_present": torch.tensor(int(record["target_present"]), dtype=torch.long),
            "sample_id": record["sample_id"],
            "source_file": record["source_file"],
            "beam_layer": int(record["beam_layer"]),
            "azimuth_deg": float(record["azimuth_deg"]),
            "distance_m": float(record["distance_m"]),
            "velocity_mps": float(record["velocity_mps"]),
            "range_index": int(
                self.geometry.range_to_index(float(record["distance_m"]))
                if int(record["target_present"])
                else -1
            ),
            "velocity_index": int(
                self.geometry.velocity_to_index(float(record["velocity_mps"]))
                if int(record["target_present"])
                else -1
            ),
            "mat_path": str(record["mat_path"]),
        }


class ROICandidateCacheDataset(Dataset):
    """Read a Stage-4 cache and expose a common eight-channel ROI input."""

    def __init__(self, cache_path: str | Path, mode: str) -> None:
        if mode not in ROI_MODES:
            raise ValueError(f"Unsupported ROI mode: {mode}")
        self.cache_path = Path(cache_path).expanduser().resolve()
        if not self.cache_path.is_file():
            raise FileNotFoundError(self.cache_path)
        payload = torch.load(self.cache_path, map_location="cpu", weights_only=False)
        required = {
            "roi_source",
            "roi_valid_mask",
            "raw_score",
            "raw_logit",
            "target_present",
            "localization_ok",
            "pred_range_index",
            "pred_velocity_index",
            "true_range_index",
            "true_velocity_index",
            "roi_quality",
            "polarimetric_confidence",
            "metadata",
            "base_threshold",
        }
        missing = required - set(payload)
        if missing:
            raise ValueError(f"Cache missing fields: {sorted(missing)}")
        lengths = {
            len(payload["roi_source"]),
            len(payload["raw_score"]),
            len(payload["metadata"]),
        }
        if len(lengths) != 1:
            raise ValueError(f"Cache length mismatch: {lengths}")
        self.payload = payload
        self.mode = mode

    def __len__(self) -> int:
        return len(self.payload["raw_score"])

    @property
    def base_threshold(self) -> float:
        return float(self.payload["base_threshold"])

    def __getitem__(self, index: int) -> dict[str, Any]:
        source = self.payload["roi_source"][index].float()
        roi = canonical_roi_channels(source, self.mode)
        meta = self.payload["metadata"][index]
        return {
            "roi": roi,
            "roi_valid_mask": self.payload["roi_valid_mask"][index].float(),
            "raw_score": self.payload["raw_score"][index].float(),
            "raw_logit": self.payload["raw_logit"][index].float(),
            "target_present": self.payload["target_present"][index].long(),
            "localization_ok": self.payload["localization_ok"][index].bool(),
            "pred_range_index": self.payload["pred_range_index"][index].long(),
            "pred_velocity_index": self.payload["pred_velocity_index"][index].long(),
            "true_range_index": self.payload["true_range_index"][index].long(),
            "true_velocity_index": self.payload["true_velocity_index"][index].long(),
            "roi_quality": self.payload["roi_quality"][index].float(),
            "polarimetric_confidence": self.payload["polarimetric_confidence"][index].float(),
            "sample_id": str(meta["sample_id"]),
            "source_file": str(meta["source_file"]),
            "beam_layer": int(meta["beam_layer"]),
            "azimuth_deg": float(meta["azimuth_deg"]),
            "distance_m": float(meta["distance_m"]),
            "velocity_mps": float(meta["velocity_mps"]),
            "mat_path": str(meta["mat_path"]),
        }
