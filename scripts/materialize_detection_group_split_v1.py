#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd


def resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def place_file(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if mode == "symlink":
        destination.symlink_to(source.resolve())
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:
        raise ValueError(mode)


def main() -> None:
    parser = argparse.ArgumentParser(description="根据已审阅split_plan建立新数据根目录")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--plan", default="results/data_splits/detection_grouped_v1/split_plan.csv")
    parser.add_argument("--target-root", default="data/raw/detection_dataset_grouped_v1")
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    plan_path = resolve_path(project_root, args.plan)
    target_root = resolve_path(project_root, args.target_root)
    frame = pd.read_csv(plan_path, encoding="utf-8-sig")

    if target_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"目标目录已存在：{target_root}；确认后使用--overwrite")
        backup = target_root.with_name(target_root.name + ".bak")
        if backup.exists():
            shutil.rmtree(backup)
        target_root.rename(backup)
        print("旧目标目录已备份：", backup)

    for row in frame.itertuples(index=False):
        split = str(row.new_split)
        class_name = str(row.class_name)
        mat_source = Path(str(row.mat_path))
        if not mat_source.exists():
            raise FileNotFoundError(mat_source)
        if class_name == "background":
            mat_dest = target_root / split / "Background_IQ" / "IQ_Data" / mat_source.name
            place_file(mat_source, mat_dest, args.mode)
        elif class_name == "uav":
            mat_dest = target_root / split / "UAV_IQ" / "IQ_Data" / mat_source.name
            place_file(mat_source, mat_dest, args.mode)
            label_value = str(row.label_path)
            if not label_value or label_value.lower() == "nan":
                raise FileNotFoundError(f"UAV样本缺少label_path：{row.sample_id}")
            label_source = Path(label_value)
            if not label_source.exists():
                raise FileNotFoundError(label_source)
            label_dest = target_root / split / "UAV_IQ" / "Labels" / label_source.name
            place_file(label_source, label_dest, args.mode)
        else:
            raise ValueError(f"未知class_name：{class_name}")

    counts = {}
    for split in ("train", "val", "test"):
        counts[split] = {
            "background": len(list((target_root / split / "Background_IQ" / "IQ_Data").glob("*.mat"))),
            "uav": len(list((target_root / split / "UAV_IQ" / "IQ_Data").glob("*.mat"))),
        }
    metadata = {
        "source_plan": str(plan_path),
        "target_root": str(target_root),
        "mode": args.mode,
        "counts": counts,
        "original_data_untouched": True,
    }
    (target_root / "GROUPED_SPLIT_METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("=" * 78)
    print("新分组数据根目录创建完成")
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    print("目标目录：", target_root)
    print("原始data/raw/detection_dataset未修改")
    print("=" * 78)


if __name__ == "__main__":
    main()
