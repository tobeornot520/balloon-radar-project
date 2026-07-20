from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.detection_dataset_v2 import (
    DetectionGeometry,
    _calculate_rd_power,
    _generate_heatmap,
    _load_iq_pair,
    _normalize_power_db,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "results"
    / "data_audit"
    / "dataset_v3_grouped"
    / "all_samples_manifest_v3.csv"
)

VALID_SPLITS = {"train", "val", "test"}
VALID_CHANNEL_MODES = {"H", "V", "HV"}

NUM_PULSES = 128
NUM_GATES = 100


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()

    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved

    return resolved.resolve()


def _to_float(
    value: str,
    default: float = float("nan"),
) -> float:
    text = str(value).strip()

    if not text:
        return default

    return float(text)


def _to_int(
    value: str,
    default: int = -1,
) -> int:
    text = str(value).strip()

    if not text:
        return default

    return int(float(text))


class DetectionRadarDatasetV3(Dataset):
    """
    Read detection samples from the leakage-controlled V3 manifest.

    The raw MAT and TXT files are not moved or modified.
    """

    def __init__(
        self,
        manifest_path: str | Path = DEFAULT_MANIFEST,
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
            raise ValueError(
                f"Invalid split: {split}. "
                f"Expected one of {sorted(VALID_SPLITS)}"
            )

        if channel_mode not in VALID_CHANNEL_MODES:
            raise ValueError(
                f"Invalid channel_mode: {channel_mode}. "
                f"Expected one of {sorted(VALID_CHANNEL_MODES)}"
            )

        if range_sigma <= 0 or velocity_sigma <= 0:
            raise ValueError(
                "range_sigma and velocity_sigma must be positive"
            )

        if not include_background and not include_uav:
            raise ValueError(
                "include_background and include_uav "
                "cannot both be False"
            )

        self.manifest_path = _resolve_path(manifest_path)
        self.split = split
        self.channel_mode = channel_mode
        self.range_sigma = float(range_sigma)
        self.velocity_sigma = float(velocity_sigma)
        self.geometry = geometry or DetectionGeometry()

        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"V3 manifest not found: {self.manifest_path}"
            )

        records: list[dict[str, Any]] = []

        with self.manifest_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            reader = csv.DictReader(handle)

            required_columns = {
                "new_split",
                "class_name",
                "target_present",
                "sample_id",
                "source_file",
                "beam_layer",
                "azimuth_deg",
                "distance_m",
                "velocity_mps",
                "mat_path",
                "label_path",
            }

            actual_columns = set(reader.fieldnames or [])
            missing_columns = required_columns - actual_columns

            if missing_columns:
                raise ValueError(
                    "Manifest is missing columns: "
                    f"{sorted(missing_columns)}"
                )

            for row in reader:
                if row["new_split"] != split:
                    continue

                target_present = int(row["target_present"])

                if target_present not in (0, 1):
                    raise ValueError(
                        "Invalid target_present value for "
                        f"{row['sample_id']}: {target_present}"
                    )

                if target_present == 0 and not include_background:
                    continue

                if target_present == 1 and not include_uav:
                    continue

                mat_path = Path(
                    row["mat_path"]
                ).expanduser().resolve()

                label_text = row["label_path"].strip()

                label_path = (
                    Path(label_text).expanduser().resolve()
                    if label_text
                    else None
                )

                if not mat_path.is_file():
                    raise FileNotFoundError(
                        f"Missing MAT file: {mat_path}"
                    )

                if target_present:
                    if label_path is None:
                        raise FileNotFoundError(
                            "Empty label path for "
                            f"{row['sample_id']}"
                        )

                    if not label_path.is_file():
                        raise FileNotFoundError(
                            f"Missing label file: {label_path}"
                        )

                records.append(
                    {
                        "class_name": row["class_name"],
                        "target_present": target_present,
                        "sample_id": row["sample_id"],
                        "source_file": row["source_file"],
                        "beam_layer": _to_int(
                            row["beam_layer"]
                        ),
                        "azimuth_deg": _to_float(
                            row["azimuth_deg"]
                        ),
                        "distance_m": _to_float(
                            row["distance_m"]
                        ),
                        "velocity_mps": _to_float(
                            row["velocity_mps"]
                        ),
                        "mat_path": mat_path,
                        "label_path": label_path,
                    }
                )

        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError(
                    "max_samples must be positive"
                )

            records = records[:max_samples]

        if not records:
            raise ValueError(
                f"No records found for split={split} "
                f"in {self.manifest_path}"
            )

        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        record = self.records[index]

        h, v = _load_iq_pair(record["mat_path"])

        rd_h = _normalize_power_db(
            _calculate_rd_power(h)
        )
        rd_v = _normalize_power_db(
            _calculate_rd_power(v)
        )

        if self.channel_mode == "H":
            input_array = rd_h[None]

        elif self.channel_mode == "V":
            input_array = rd_v[None]

        else:
            input_array = np.stack(
                (rd_h, rd_v),
                axis=0,
            )

        target_present = int(
            record["target_present"]
        )

        if target_present:
            distance_m = float(
                record["distance_m"]
            )
            velocity_mps = float(
                record["velocity_mps"]
            )

            range_index = self.geometry.range_to_index(
                distance_m
            )
            velocity_index = (
                self.geometry.velocity_to_index(
                    velocity_mps
                )
            )

            target = _generate_heatmap(
                range_index,
                velocity_index,
                self.range_sigma,
                self.velocity_sigma,
            )

        else:
            distance_m = float("nan")
            velocity_mps = float("nan")
            range_index = -1
            velocity_index = -1

            target = np.zeros(
                (1, NUM_PULSES, NUM_GATES),
                dtype=np.float32,
            )

        return {
            "input": torch.from_numpy(
                np.ascontiguousarray(input_array)
            ).float(),
            "target": torch.from_numpy(
                np.ascontiguousarray(target)
            ).float(),
            "target_present": torch.tensor(
                target_present,
                dtype=torch.long,
            ),
            "sample_id": record["sample_id"],
            "beam_layer": int(
                record["beam_layer"]
            ),
            "azimuth_deg": float(
                record["azimuth_deg"]
            ),
            "distance_m": distance_m,
            "velocity_mps": velocity_mps,
            "range_index": int(range_index),
            "velocity_index": int(velocity_index),
            "source_file": record["source_file"],
            "mat_path": str(
                record["mat_path"]
            ),
            "label_path": (
                ""
                if record["label_path"] is None
                else str(record["label_path"])
            ),
        }
