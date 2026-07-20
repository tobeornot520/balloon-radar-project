#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

SPLITS = ("train", "val", "test")
TIMESTAMP_RE = re.compile(r"(?P<date>\d{8})[_-]?(?P<time>\d{6})")


def resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def parse_label(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def timestamp_from_text(text: str) -> tuple[str, datetime | None]:
    match = TIMESTAMP_RE.search(text)
    if not match:
        return "", None
    stamp = f"{match.group('date')}_{match.group('time')}"
    try:
        dt = datetime.strptime(stamp, "%Y%m%d_%H%M%S")
    except ValueError:
        dt = None
    return stamp, dt


def normalize_source_group(raw: str, fallback: str) -> str:
    value = Path(str(raw).replace("\\", "/")).stem if raw else fallback
    value = re.sub(r"_beam\d+(?:_az\d+)?$", "", value, flags=re.IGNORECASE)
    stamp, _ = timestamp_from_text(value)
    return stamp or value or fallback


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def scan_dataset(data_root: Path, hash_files: bool) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        split_root = data_root / split
        if not split_root.exists():
            continue
        bg_dir = split_root / "Background_IQ" / "IQ_Data"
        for mat_path in sorted(bg_dir.glob("*.mat")):
            sample_id = mat_path.stem
            timestamp, dt = timestamp_from_text(sample_id)
            exact_group = timestamp or normalize_source_group("", sample_id)
            rows.append({
                "current_split": split,
                "class_name": "background",
                "target_present": 0,
                "sample_id": sample_id,
                "source_file_raw": exact_group,
                "exact_group": exact_group,
                "timestamp": timestamp,
                "timestamp_iso": "" if dt is None else dt.isoformat(),
                "mat_path": str(mat_path.resolve()),
                "label_path": "",
                "file_size_bytes": mat_path.stat().st_size,
                "sha256": sha256_file(mat_path) if hash_files else "",
            })

        iq_dir = split_root / "UAV_IQ" / "IQ_Data"
        label_dir = split_root / "UAV_IQ" / "Labels"
        for mat_path in sorted(iq_dir.glob("*.mat")):
            sample_id = mat_path.stem
            label_path = label_dir / f"{sample_id}.txt"
            values = parse_label(label_path) if label_path.exists() else {}
            source_raw = values.get("Source_File", "")
            fallback_stamp, fallback_dt = timestamp_from_text(sample_id)
            exact_group = normalize_source_group(source_raw, fallback_stamp or sample_id)
            timestamp, dt = timestamp_from_text(exact_group)
            if dt is None:
                timestamp, dt = fallback_stamp, fallback_dt
            rows.append({
                "current_split": split,
                "class_name": "uav",
                "target_present": 1,
                "sample_id": sample_id,
                "source_file_raw": source_raw,
                "exact_group": exact_group,
                "timestamp": timestamp,
                "timestamp_iso": "" if dt is None else dt.isoformat(),
                "mat_path": str(mat_path.resolve()),
                "label_path": str(label_path.resolve()) if label_path.exists() else "",
                "file_size_bytes": mat_path.stat().st_size,
                "sha256": sha256_file(mat_path) if hash_files else "",
            })
    if not rows:
        raise FileNotFoundError(f"未在 {data_root} 中发现检测数据")
    frame = pd.DataFrame(rows)
    frame["class_group_key"] = frame["class_name"] + "::" + frame["exact_group"].astype(str)
    return frame


def add_session_ids(frame: pd.DataFrame, gaps: Iterable[int]) -> pd.DataFrame:
    result = frame.copy()
    group_times = (
        result[["class_name", "exact_group", "timestamp_iso"]]
        .drop_duplicates()
        .copy()
    )
    group_times["dt"] = pd.to_datetime(group_times["timestamp_iso"], errors="coerce")
    for gap in gaps:
        mapping: dict[tuple[str, str], str] = {}
        for class_name, part in group_times.groupby("class_name", sort=True):
            known = part[part["dt"].notna()].sort_values(["dt", "exact_group"])
            unknown = part[part["dt"].isna()].sort_values("exact_group")
            session_index = 0
            previous = None
            for row in known.itertuples(index=False):
                if previous is None or (row.dt - previous).total_seconds() > gap:
                    session_index += 1
                mapping[(class_name, row.exact_group)] = f"{class_name}_session_gap{gap}_{session_index:03d}"
                previous = row.dt
            for row in unknown.itertuples(index=False):
                session_index += 1
                mapping[(class_name, row.exact_group)] = f"{class_name}_session_gap{gap}_{session_index:03d}"
        result[f"session_gap{gap}"] = [
            mapping[(c, g)] for c, g in zip(result["class_name"], result["exact_group"])
        ]
    return result


def overlap_rows(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(["class_name", group_col], dropna=False)
    for (class_name, group_id), part in grouped:
        splits = sorted(part["current_split"].unique())
        if len(splits) <= 1:
            continue
        counts = part.groupby("current_split").size().to_dict()
        rows.append({
            "class_name": class_name,
            "group_column": group_col,
            "group_id": group_id,
            "splits": ",".join(splits),
            "split_count": len(splits),
            "sample_count": len(part),
            **{f"count_{split}": int(counts.get(split, 0)) for split in SPLITS},
        })
    return pd.DataFrame(rows)


def group_summary(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    return (
        frame.groupby(["class_name", group_col, "current_split"], dropna=False)
        .size()
        .rename("sample_count")
        .reset_index()
        .sort_values(["class_name", group_col, "current_split"])
    )


def duplicate_summary(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    if key not in frame or (frame[key].astype(str) == "").all():
        return pd.DataFrame()
    rows = []
    for value, part in frame[frame[key].astype(str) != ""].groupby(key):
        if len(part) <= 1:
            continue
        rows.append({
            key: value,
            "sample_count": len(part),
            "splits": ",".join(sorted(part.current_split.unique())),
            "sample_ids": "|".join(part.sample_id.astype(str).head(20)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="检测数据scan/session级泄漏审计")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", default="data/raw/detection_dataset")
    parser.add_argument("--output-dir", default="results/data_audit/detection_group_leakage_v1")
    parser.add_argument("--session-gaps", type=int, nargs="+", default=[30, 60, 120, 300])
    parser.add_argument("--hash-files", action="store_true", help="计算全部MAT文件SHA256，速度较慢")
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    data_root = resolve_path(project_root, args.data_root)
    output_dir = resolve_path(project_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = scan_dataset(data_root, args.hash_files)
    frame = add_session_ids(frame, args.session_gaps)
    frame.to_csv(output_dir / "all_samples_manifest.csv", index=False, encoding="utf-8-sig")

    split_counts = (
        frame.groupby(["current_split", "class_name"]).size()
        .rename("sample_count").reset_index()
    )
    split_counts.to_csv(output_dir / "split_class_counts.csv", index=False, encoding="utf-8-sig")

    exact_summary = group_summary(frame, "exact_group")
    exact_summary.to_csv(output_dir / "exact_group_summary.csv", index=False, encoding="utf-8-sig")
    exact_overlap = overlap_rows(frame, "exact_group")
    exact_overlap.to_csv(output_dir / "exact_group_overlap.csv", index=False, encoding="utf-8-sig")

    session_report: dict[str, Any] = {}
    for gap in args.session_gaps:
        col = f"session_gap{gap}"
        summary = group_summary(frame, col)
        overlap = overlap_rows(frame, col)
        summary.to_csv(output_dir / f"session_gap{gap}_summary.csv", index=False, encoding="utf-8-sig")
        overlap.to_csv(output_dir / f"session_gap{gap}_overlap.csv", index=False, encoding="utf-8-sig")
        session_report[str(gap)] = {
            "groups_by_class": {
                cls: int(frame.loc[frame.class_name == cls, col].nunique())
                for cls in sorted(frame.class_name.unique())
            },
            "overlapping_groups": int(len(overlap)),
            "overlapping_samples": int(overlap["sample_count"].sum()) if len(overlap) else 0,
        }

    duplicate_ids = duplicate_summary(frame, "sample_id")
    duplicate_ids.to_csv(output_dir / "duplicate_sample_ids.csv", index=False, encoding="utf-8-sig")
    duplicate_hashes = duplicate_summary(frame, "sha256") if args.hash_files else pd.DataFrame()
    duplicate_hashes.to_csv(output_dir / "duplicate_hashes.csv", index=False, encoding="utf-8-sig")

    class_group_counts = {
        cls: int(frame.loc[frame.class_name == cls, "exact_group"].nunique())
        for cls in sorted(frame.class_name.unique())
    }
    exact_overlap_by_class = {
        cls: int((exact_overlap.class_name == cls).sum()) if len(exact_overlap) else 0
        for cls in sorted(frame.class_name.unique())
    }

    warnings: list[str] = []
    for cls, count in class_group_counts.items():
        if count < 3:
            warnings.append(f"{cls} exact_group仅{count}组，不足以构建train/val/test三组独立划分")
    for gap, report in session_report.items():
        for cls, count in report["groups_by_class"].items():
            if count < 3:
                warnings.append(
                    f"{cls}在gap={gap}s定义下仅{count}个session，无法进行严格session级三划分；需要补采独立时段数据"
                )

    summary = {
        "data_root": str(data_root),
        "sample_count": int(len(frame)),
        "split_class_counts": split_counts.to_dict(orient="records"),
        "exact_groups_by_class": class_group_counts,
        "exact_overlapping_groups_by_class": exact_overlap_by_class,
        "session_sensitivity": session_report,
        "duplicate_sample_id_groups": int(len(duplicate_ids)),
        "duplicate_hash_groups": int(len(duplicate_hashes)),
        "hash_files_enabled": bool(args.hash_files),
        "warnings": sorted(set(warnings)),
    }
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# 检测数据分组泄漏审计结论", "",
        f"- 数据根目录：`{data_root}`",
        f"- 样本总数：{len(frame)}", "",
        "## Exact scan/source group", "",
    ]
    for cls in sorted(class_group_counts):
        lines.append(
            f"- {cls}：{class_group_counts[cls]}个exact group，其中{exact_overlap_by_class[cls]}个跨split。"
        )
    lines += ["", "## Session敏感性", ""]
    for gap in args.session_gaps:
        report = session_report[str(gap)]
        groups = ", ".join(f"{k}={v}" for k, v in report["groups_by_class"].items())
        lines.append(
            f"- gap={gap}s：{groups}；跨split session={report['overlapping_groups']}。"
        )
    lines += ["", "## 自动警告", ""]
    if warnings:
        lines.extend(f"- {item}" for item in sorted(set(warnings)))
    else:
        lines.append("- 未发现组数不足警告。")
    lines += [
        "", "## 下一步", "",
        "1. 先审阅本目录中的exact/session overlap表。",
        "2. 根据真实采集流程确定UAV连续session时间间隔，不能仅凭模型结果选择。",
        "3. 若某一类别严格session少于3组，不能声称跨session独立测试，应补采数据。",
        "4. 确定分组级别后，再运行`propose_detection_group_split_v1.py`，不要直接移动原始数据。",
    ]
    (output_dir / "README_审计结论.md").write_text("\n".join(lines), encoding="utf-8")

    print("=" * 78)
    print("检测数据scan/session级泄漏审计完成")
    print(split_counts.to_string(index=False))
    print("-" * 78)
    print("exact group数量：", class_group_counts)
    print("exact跨split组数：", exact_overlap_by_class)
    for gap in args.session_gaps:
        print(f"gap={gap}s：{session_report[str(gap)]}")
    if warnings:
        print("-" * 78)
        print("警告：")
        for item in sorted(set(warnings)):
            print(" -", item)
    print("结果目录：", output_dir)
    print("=" * 78)


if __name__ == "__main__":
    main()
