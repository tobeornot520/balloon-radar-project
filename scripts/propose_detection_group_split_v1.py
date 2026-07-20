#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import pandas as pd

SPLITS = ("train", "val", "test")


def resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def score_assignment(counts: dict[str, int], total: int, ratios: dict[str, float]) -> float:
    score = 0.0
    for split in SPLITS:
        target = ratios[split] * total
        scale = max(target, 1.0)
        score += ((counts[split] - target) / scale) ** 2
    return score


def optimize_groups(group_counts: dict[str, int], ratios: dict[str, float], seed: int, trials: int) -> dict[str, str]:
    groups = list(group_counts)
    if len(groups) < 3:
        raise ValueError(f"只有{len(groups)}个组，无法分配到train/val/test")
    total = sum(group_counts.values())
    rng = random.Random(seed)
    best_score = math.inf
    best: dict[str, str] | None = None

    # Deterministic size-first candidate plus randomized candidates.
    candidates = [sorted(groups, key=lambda g: (-group_counts[g], g))]
    for _ in range(max(trials, 1)):
        order = groups[:]
        rng.shuffle(order)
        candidates.append(order)

    for order in candidates:
        assignment: dict[str, str] = {}
        counts = {split: 0 for split in SPLITS}
        group_num = {split: 0 for split in SPLITS}
        # Seed one group into each split to avoid empty partitions.
        seed_splits = ["train", "val", "test"]
        for group, split in zip(order[:3], seed_splits):
            assignment[group] = split
            counts[split] += group_counts[group]
            group_num[split] += 1
        for group in order[3:]:
            choices = []
            for split in SPLITS:
                trial_counts = counts.copy()
                trial_counts[split] += group_counts[group]
                value = score_assignment(trial_counts, total, ratios)
                # Mild penalty on group-count imbalance, only as a tie helper.
                trial_group_num = group_num.copy()
                trial_group_num[split] += 1
                value += 1e-4 * sum((trial_group_num[s] - ratios[s] * len(groups)) ** 2 for s in SPLITS)
                choices.append((value, split))
            _, chosen = min(choices, key=lambda item: (item[0], SPLITS.index(item[1])))
            assignment[group] = chosen
            counts[chosen] += group_counts[group]
            group_num[chosen] += 1
        value = score_assignment(counts, total, ratios)
        if value < best_score:
            best_score = value
            best = assignment
    assert best is not None
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="根据审计manifest生成无组重叠的新划分方案；不移动文件")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", default="results/data_audit/detection_group_leakage_v1/all_samples_manifest.csv")
    parser.add_argument("--output-dir", default="results/data_splits/detection_grouped_v1")
    parser.add_argument("--group-level", choices=["exact", "session"], default="exact")
    parser.add_argument("--session-gap", type=int, default=60)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--trials", type=int, default=20000)
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    manifest_path = resolve_path(project_root, args.manifest)
    output_dir = resolve_path(project_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ratios = {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio}
    if not math.isclose(sum(ratios.values()), 1.0, abs_tol=1e-9):
        raise ValueError("train/val/test比例之和必须为1")
    if min(ratios.values()) <= 0:
        raise ValueError("三个比例都必须大于0")

    frame = pd.read_csv(manifest_path, encoding="utf-8-sig")
    group_col = "exact_group" if args.group_level == "exact" else f"session_gap{args.session_gap}"
    if group_col not in frame.columns:
        raise KeyError(f"manifest没有{group_col}，请先使用包含该gap的审计脚本")

    frame["assignment_key"] = frame["class_name"].astype(str) + "::" + frame[group_col].astype(str)
    assignments: dict[str, str] = {}
    class_warnings: list[str] = []
    for class_name, part in frame.groupby("class_name", sort=True):
        counts = part.groupby("assignment_key").size().to_dict()
        if len(counts) < 3:
            class_warnings.append(
                f"{class_name}在{group_col}下只有{len(counts)}组，不能构建严格三划分"
            )
            continue
        assignments.update(optimize_groups(counts, ratios, args.seed + len(assignments), args.trials))

    if class_warnings:
        message = "；".join(class_warnings)
        (output_dir / "SPLIT_ABORTED.txt").write_text(message, encoding="utf-8")
        raise RuntimeError(message)

    frame["new_split"] = frame["assignment_key"].map(assignments)
    if frame["new_split"].isna().any():
        raise RuntimeError("部分样本未获得new_split")

    leakage = (
        frame.groupby(["class_name", group_col])["new_split"].nunique().max()
    )
    if int(leakage) != 1:
        raise RuntimeError("生成方案仍存在group跨split，程序拒绝输出")

    frame.to_csv(output_dir / "split_plan.csv", index=False, encoding="utf-8-sig")
    group_assignment = (
        frame.groupby(["class_name", group_col, "assignment_key", "new_split"])
        .size().rename("sample_count").reset_index()
    )
    group_assignment.to_csv(output_dir / "group_assignment.csv", index=False, encoding="utf-8-sig")
    summary = (
        frame.groupby(["new_split", "class_name"])
        .size().rename("sample_count").reset_index()
    )
    summary.to_csv(output_dir / "split_summary.csv", index=False, encoding="utf-8-sig")

    meta = {
        "selection_uses_model_results": False,
        "group_level": args.group_level,
        "group_column": group_col,
        "session_gap_seconds": args.session_gap if args.group_level == "session" else None,
        "ratios": ratios,
        "seed": args.seed,
        "trials": args.trials,
        "sample_count": len(frame),
        "summary": summary.to_dict(orient="records"),
    }
    (output_dir / "split_plan_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    readme = [
        "# 新分组划分方案", "",
        f"- 分组级别：`{args.group_level}`",
        f"- 分组字段：`{group_col}`",
        "- 划分只使用组ID与样本数量，不使用任何模型结果、难度或测试表现。",
        "- 此脚本只生成方案，没有移动、复制或删除原始数据。", "",
        "## 样本数", "", "```text", summary.to_string(index=False), "```", "",
        "确认方案后再运行materialize脚本。",
    ]
    (output_dir / "README_划分方案.md").write_text("\n".join(readme), encoding="utf-8")

    print("=" * 78)
    print("无组重叠划分方案已生成（尚未移动文件）")
    print(f"group_level={args.group_level}, group_column={group_col}")
    print(summary.to_string(index=False))
    print("方案目录：", output_dir)
    print("=" * 78)


if __name__ == "__main__":
    main()
