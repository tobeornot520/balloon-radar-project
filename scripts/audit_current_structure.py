#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat, whosmat

from datasets.classification_dataset_v2 import CLASS_SPECS, ClassificationRadarDatasetV2
from datasets.detection_dataset_v2 import DetectionRadarDatasetV2


def label_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def audit_detection(root: Path, split: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base = root / split
    bg = sorted((base / "Background_IQ" / "IQ_Data").glob("*.mat"))
    uav = sorted((base / "UAV_IQ" / "IQ_Data").glob("*.mat"))
    label_dir = base / "UAV_IQ" / "Labels"
    labels = sorted(label_dir.glob("*.txt"))
    missing = [p.name for p in uav if not (label_dir / f"{p.stem}.txt").exists()]
    orphan = [p.name for p in labels if not (base / "UAV_IQ" / "IQ_Data" / f"{p.stem}.mat").exists()]

    schemas = Counter()
    bad_shapes: list[str] = []
    for p in bg + uav:
        schema = tuple((name, tuple(shape), dtype) for name, shape, dtype in whosmat(p))
        schemas[str(schema)] += 1
        info = {name: tuple(shape) for name, shape, _ in whosmat(p)}
        if info.get("local_data_H") != (128, 100) or info.get("local_data_V") != (128, 100):
            bad_shapes.append(p.name)

    rows: list[dict[str, Any]] = []
    for p in bg:
        rows.append({
            "split": split, "target_present": 0, "sample_id": p.stem,
            "mat_path": str(p.resolve()), "label_path": "", "beam_layer": "",
            "azimuth_deg": "", "distance_m": "", "velocity_mps": "",
        })
    for p in uav:
        lp = label_dir / f"{p.stem}.txt"
        vals = label_values(lp) if lp.exists() else {}
        rows.append({
            "split": split, "target_present": 1, "sample_id": p.stem,
            "mat_path": str(p.resolve()), "label_path": str(lp.resolve()) if lp.exists() else "",
            "beam_layer": vals.get("Beam_Layer", ""), "azimuth_deg": vals.get("Azimuth(deg)", ""),
            "distance_m": vals.get("Distance(m)", ""), "velocity_mps": vals.get("Velocity(m/s)", ""),
        })

    return {
        "background_mat": len(bg), "uav_mat": len(uav), "uav_labels": len(labels),
        "total_samples": len(bg) + len(uav), "missing_labels": missing,
        "orphan_labels": orphan, "bad_shapes": bad_shapes, "schemas": dict(schemas),
    }, rows


def audit_classification(root: Path, split: str) -> dict[str, Any]:
    base = root / split
    result: dict[str, Any] = {"total_mat_files": 0, "total_samples": 0, "classes": {}}
    for label, class_name, aliases in CLASS_SPECS:
        class_dir = next((base / a for a in aliases if (base / a).is_dir()), None)
        files = sorted(class_dir.glob("*.mat")) if class_dir else []
        samples = 0
        schemas = Counter()
        for p in files:
            info = {name: tuple(shape) for name, shape, dtype in whosmat(p)}
            schemas[str(info)] += 1
            if "UAV_h" in info and len(info["UAV_h"]) == 3:
                samples += info["UAV_h"][2]
        result["classes"][class_name] = {
            "label": label, "directory": "" if class_dir is None else str(class_dir),
            "mat_files": len(files), "samples": samples, "schemas": dict(schemas),
        }
        result["total_mat_files"] += len(files)
        result["total_samples"] += samples
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="审计当前 data/raw 目录并测试读取")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("results/data_audit_current"))
    args = parser.parse_args()

    project = args.project_root.expanduser().resolve()
    det_root = project / "data" / "raw" / "detection_dataset"
    cls_root = project / "data" / "raw" / "classification_dataset"
    output = (project / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {"project_root": str(project), "detection": {}, "classification": {}}
    manifest_rows: list[dict[str, Any]] = []

    for split in ("train", "val", "test"):
        if (det_root / split).exists():
            report["detection"][split], rows = audit_detection(det_root, split)
            manifest_rows.extend(rows)
        if (cls_root / split).exists():
            report["classification"][split] = audit_classification(cls_root, split)

    (output / "audit_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata_dir = project / "data" / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    if manifest_rows:
        with (metadata_dir / "detection_manifest_v2.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0]))
            writer.writeheader()
            writer.writerows(manifest_rows)

    print("=== 检测数据审计 ===")
    for split, info in report["detection"].items():
        print(f"{split}: background={info['background_mat']}, uav={info['uav_mat']}, labels={info['uav_labels']}, total={info['total_samples']}")
        print(f"      missing_labels={len(info['missing_labels'])}, orphan_labels={len(info['orphan_labels'])}, bad_shapes={len(info['bad_shapes'])}")

    print("=== 分类数据审计 ===")
    for split, info in report["classification"].items():
        print(f"{split}: mat_files={info['total_mat_files']}, expanded_samples={info['total_samples']}")
        for name, c in info["classes"].items():
            if c["mat_files"]:
                print(f"      {name}: mat={c['mat_files']}, samples={c['samples']}")

    print("=== 真实读取 smoke test ===")
    for split in report["detection"]:
        ds = DetectionRadarDatasetV2(det_root, split=split, channel_mode="HV")
        bg_index = next((i for i, r in enumerate(ds.records) if r["target_present"] == 0), None)
        uav_index = next((i for i, r in enumerate(ds.records) if r["target_present"] == 1), None)
        for kind, idx in (("background", bg_index), ("uav", uav_index)):
            if idx is None:
                continue
            item = ds[idx]
            print(
                f"detection/{split}/{kind}: input={tuple(item['input'].shape)}, "
                f"target={tuple(item['target'].shape)}, present={item['target_present'].item()}, "
                f"indices=({item['velocity_index']},{item['range_index']})"
            )

    for split in report["classification"]:
        try:
            ds = ClassificationRadarDatasetV2(cls_root, split=split, input_mode="polar5")
        except ValueError:
            continue
        item = ds[0]
        print(f"classification/{split}: expanded_n={len(ds)}, input={tuple(item['input'].shape)}, class={item['class_name']}")

    print(f"审计报告: {output / 'audit_summary.json'}")
    print(f"检测清单: {metadata_dir / 'detection_manifest_v2.csv'}")


if __name__ == "__main__":
    main()
