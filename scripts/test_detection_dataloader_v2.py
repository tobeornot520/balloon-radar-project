#!/usr/bin/env python3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path

from torch.utils.data import DataLoader

from datasets.detection_dataset_v2 import DetectionRadarDatasetV2


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    root = project / "data" / "raw" / "detection_dataset"
    for split in ("train", "val", "test"):
        if not (root / split).exists():
            continue
        ds = DetectionRadarDatasetV2(root, split=split, channel_mode="HV")
        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        positives = int(batch["target_present"].sum().item())
        print(
            f"{split}: dataset={len(ds)}, input={tuple(batch['input'].shape)}, "
            f"target={tuple(batch['target'].shape)}, batch_positive={positives}"
        )


if __name__ == "__main__":
    main()
