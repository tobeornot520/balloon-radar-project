#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.detection_dataset_v2 import DetectionRadarDatasetV2
from models.simple_fcn import SimpleRadarFCN
from scripts.train_detection_baseline_v2 import SampleWeightedHeatmapMSE


def main() -> None:
    dataset = DetectionRadarDatasetV2(
        data_root="data/raw/detection_dataset",
        split="train",
        channel_mode="HV",
        range_sigma=5.0,
        velocity_sigma=5.0,
    )
    background = next(i for i, r in enumerate(dataset.records) if int(r["target_present"]) == 0)
    positive = next(i for i, r in enumerate(dataset.records) if int(r["target_present"]) == 1)
    loader = DataLoader(Subset(dataset, [background, positive]), batch_size=2, shuffle=False)
    batch = next(iter(loader))
    model = SimpleRadarFCN(in_channels=2)
    logits = model(batch["input"])
    loss = SampleWeightedHeatmapMSE(10.0)(
        logits, batch["target"], batch["target_present"]
    )
    loss.backward()
    print(f"input={tuple(batch['input'].shape)}")
    print(f"target={tuple(batch['target'].shape)}")
    print(f"present={batch['target_present'].tolist()}")
    print(f"logits={tuple(logits.shape)}")
    print(f"loss={float(loss.item()):.8f}")
    print("完整检测基线前向与反向传播测试通过。")


if __name__ == "__main__":
    main()
