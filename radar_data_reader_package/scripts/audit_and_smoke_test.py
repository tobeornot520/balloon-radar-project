#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import scipy.io as sio
from torch.utils.data import DataLoader

from datasets.radar_tasks import CLASS_NAMES, ClassificationRadarDataset, DetectionRadarDataset


def audit_detection(root: Path, split: str) -> Dict[str, Any]:
    base = root / "detection" / split
    bg = sorted((base / "background" / "iq").glob("*.mat"))
    uav = sorted((base / "uav" / "iq").glob("*.mat"))
    labels = sorted((base / "uav" / "labels").glob("*.txt"))
    report: Dict[str, Any] = {
        "split": split,
        "background_mat_files": len(bg),
        "uav_mat_files": len(uav),
        "uav_label_files": len(labels),
        "missing_labels": sorted(p.name for p in uav if not (base / "uav" / "labels" / f"{p.stem}.txt").exists()),
        "schemas": Counter(),
    }
    for p in bg + uav:
        info = tuple((n, tuple(s), t) for n, s, t in sio.whosmat(p))
        report["schemas"][str(info)] += 1
    report["schemas"] = dict(report["schemas"])
    return report


def audit_classification(root: Path, split: str) -> Dict[str, Any]:
    base = root / "classification" / split
    classes: Dict[str, Any] = {}
    for class_name in CLASS_NAMES:
        files = sorted((base / class_name).glob("*.mat"))
        sample_count = 0
        schemas = Counter()
        for p in files:
            info = {n: (tuple(s), t) for n, s, t in sio.whosmat(p)}
            schemas[str(info)] += 1
            if "UAV_h" in info:
                sample_count += info["UAV_h"][0][2]
        classes[class_name] = {
            "mat_files": len(files),
            "samples": sample_count,
            "schemas": dict(schemas),
        }
    return {"split": split, "classes": classes, "total_samples": sum(v["samples"] for v in classes.values())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/data_audit"))
    args = parser.parse_args()
    root = args.data_root.resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {"data_root": str(root), "detection": {}, "classification": {}}
    for split in ("train", "val", "test"):
        if (root / "detection" / split).exists():
            report["detection"][split] = audit_detection(root, split)
        if (root / "classification" / split).exists():
            report["classification"][split] = audit_classification(root, split)

    (args.output / "audit_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 真实读取 smoke test
    for split in report["detection"]:
        ds = DetectionRadarDataset(root, split=split, input_mode="polar6")
        bg_idx = next((i for i, r in enumerate(ds.records) if r["target_present"] == 0), None)
        pos_idx = next((i for i, r in enumerate(ds.records) if r["target_present"] == 1), None)
        for name, idx in (("background", bg_idx), ("uav", pos_idx)):
            if idx is None:
                continue
            item = ds[idx]
            print(f"detection/{split}/{name}: x={tuple(item['x'].shape)}, heatmap={tuple(item['heatmap'].shape)}, present={item['target_present'].item()}, indices=({item['velocity_index'].item()}, {item['range_index'].item()})")

    for split in report["classification"]:
        try:
            ds = ClassificationRadarDataset(root, split=split, input_mode="polar5")
        except FileNotFoundError:
            continue
        item = ds[0]
        print(f"classification/{split}: n={len(ds)}, x={tuple(item['x'].shape)}, label={item['label'].item()}, class={item['class_name']}, angle={tuple(item['angle_h'].shape)}")

    print(f"审计报告已保存：{args.output / 'audit_summary.json'}")


if __name__ == "__main__":
    main()
