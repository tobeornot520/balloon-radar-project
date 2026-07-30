from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np

from datasets.detection_dataset_v2 import DetectionGeometry
from datasets.polarimetric_detection_dataset_v2 import (
    DEFAULT_MANIFEST,
    PolarimetricDetectionDatasetV2,
)
from features.polarimetric_rd import explicit_polarimetric_rd, make_thesis_tian6


class ThesisTian2024Dataset(PolarimetricDetectionDatasetV2):
    """Manifest dataset for the thesis Tian2024 six-channel adaptation."""

    def __init__(
        self,
        manifest_path: str | Path = DEFAULT_MANIFEST,
        split: Literal["train", "val", "test"] = "train",
        max_samples: Optional[int] = None,
        geometry: Optional[DetectionGeometry] = None,
        eps: float = 1e-8,
    ) -> None:
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.feature_eps = float(eps)
        super().__init__(
            manifest_path=manifest_path,
            split=split,
            input_mode="ri4",
            max_samples=max_samples,
            geometry=geometry,
            velocity_window=3,
            range_window=3,
        )

    def _make_input(self, h: np.ndarray, v: np.ndarray) -> np.ndarray:
        features = explicit_polarimetric_rd(h, v, self.polar_config)
        return make_thesis_tian6(features, eps=self.feature_eps)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        item["input_mode"] = "thesis_tian6"
        item["active_channels"] = 6
        return item
