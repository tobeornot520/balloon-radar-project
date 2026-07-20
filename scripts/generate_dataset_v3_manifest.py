#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "detection_dataset"
OUTPUT_DIR = PROJECT_ROOT / "results" / "data_audit" / "dataset_v3_grouped"

OLD_SPLITS = ("train", "val", "test")
NEW_SPLITS = ("train", "val", "test")

BACKGROUND_SPLIT = {
    "20260204_100739": "train",
    "20260204_100802": "train",
    "20260204_100826": "train",
    "20260204_100845": "train",
    "20260204_100908": "val",
    "20260204_100932": "test",
}

UAV_RANGES = {
    "train": ("20260202_144238", "20260202_144955"),
    "val": ("20260202_145004", "20260202_145216"),
    "test": ("20260202_145713", "20260202_145836"),
}


@dataclass(frozen=True)
class Record:
    new_split: str
    original_split: str
    class_name: str
    target_present: int
    sample_id: str
    source_file: str
    beam_layer: int
    azimuth_deg: float
    distance_m: float
    velocity_mps: float
    mat_path: str
    label_path: str


def parse_label(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}

    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()

    required = (
        "Source_File",
        "Beam_Layer",
        "Azimuth(deg)",
        "Distance(m)",
        "Velocity(m/s)",
    )

    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(
            f"Missing label fields in {path}: {missing}"
        )

    return values


def extract_int(text: str, pattern: str) -> int:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else -1


def assign_uav_split(source_file: str) -> str | None:
    for split, (start, end) in UAV_RANGES.items():
        if start <= source_file <= end:
            return split

    return None


def collect_uav_records() -> tuple[list[Record], list[str]]:
    records: list[Record] = []
    unassigned: set[str] = set()

    for old_split in OLD_SPLITS:
        iq_dir = DATA_ROOT / old_split / "UAV_IQ" / "IQ_Data"
        label_dir = DATA_ROOT / old_split / "UAV_IQ" / "Labels"

        for mat_path in sorted(iq_dir.glob("*.mat")):
            label_path = label_dir / f"{mat_path.stem}.txt"

            if not label_path.is_file():
                raise FileNotFoundError(
                    f"Missing label for {mat_path}: {label_path}"
                )

            label = parse_label(label_path)
            source_file = label["Source_File"]
            new_split = assign_uav_split(source_file)

            if new_split is None:
                unassigned.add(source_file)
                continue

            records.append(
                Record(
                    new_split=new_split,
                    original_split=old_split,
                    class_name="UAV",
                    target_present=1,
                    sample_id=mat_path.stem,
                    source_file=source_file,
                    beam_layer=int(label["Beam_Layer"]),
                    azimuth_deg=float(label["Azimuth(deg)"]),
                    distance_m=float(label["Distance(m)"]),
                    velocity_mps=float(label["Velocity(m/s)"]),
                    mat_path=str(mat_path.resolve()),
                    label_path=str(label_path.resolve()),
                )
            )

    return records, sorted(unassigned)


def collect_background_records() -> tuple[list[Record], list[str]]:
    records: list[Record] = []
    unassigned: set[str] = set()

    for old_split in OLD_SPLITS:
        iq_dir = (
            DATA_ROOT
            / old_split
            / "Background_IQ"
            / "IQ_Data"
        )

        for mat_path in sorted(iq_dir.glob("*.mat")):
            source_file = mat_path.stem.split("_beam", 1)[0]
            new_split = BACKGROUND_SPLIT.get(source_file)

            if new_split is None:
                unassigned.add(source_file)
                continue

            records.append(
                Record(
                    new_split=new_split,
                    original_split=old_split,
                    class_name="Background",
                    target_present=0,
                    sample_id=mat_path.stem,
                    source_file=source_file,
                    beam_layer=extract_int(
                        mat_path.stem,
                        r"beam(\d+)",
                    ),
                    azimuth_deg=float(
                        extract_int(
                            mat_path.stem,
                            r"az(\d+)",
                        )
                    ),
                    distance_m=float("nan"),
                    velocity_mps=float("nan"),
                    mat_path=str(mat_path.resolve()),
                    label_path="",
                )
            )

    return records, sorted(unassigned)


def write_csv(
    path: Path,
    rows: list[dict],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def check_duplicate_sample_ids(
    records: list[Record],
) -> list[str]:
    counter = Counter(
        record.sample_id
        for record in records
    )

    return sorted(
        sample_id
        for sample_id, count in counter.items()
        if count > 1
    )


def check_missing_paths(
    records: list[Record],
) -> list[str]:
    missing: list[str] = []

    for record in records:
        if not Path(record.mat_path).is_file():
            missing.append(record.mat_path)

        if record.target_present:
            if not record.label_path:
                missing.append(
                    f"Empty label path for {record.sample_id}"
                )
            elif not Path(record.label_path).is_file():
                missing.append(record.label_path)

    return sorted(set(missing))


def build_overlap_rows(
    records: list[Record],
) -> list[dict]:
    groups: dict[tuple[str, str], set[str]] = defaultdict(set)

    for record in records:
        groups[
            (record.new_split, record.class_name)
        ].add(record.source_file)

    rows: list[dict] = []

    for class_name in ("Background", "UAV"):
        for split_a, split_b in (
            ("train", "val"),
            ("train", "test"),
            ("val", "test"),
        ):
            overlap = sorted(
                groups[(split_a, class_name)]
                & groups[(split_b, class_name)]
            )

            rows.append(
                {
                    "class_name": class_name,
                    "split_a": split_a,
                    "split_b": split_b,
                    "overlap_group_count": len(overlap),
                    "overlap_groups": "|".join(overlap),
                }
            )

    return rows


def build_count_rows(
    records: list[Record],
) -> list[dict]:
    rows: list[dict] = []

    for split in NEW_SPLITS:
        split_records = [
            record
            for record in records
            if record.new_split == split
        ]

        total = len(split_records)

        for class_name in ("Background", "UAV"):
            class_records = [
                record
                for record in split_records
                if record.class_name == class_name
            ]

            count = len(class_records)
            group_count = len(
                {
                    record.source_file
                    for record in class_records
                }
            )

            rows.append(
                {
                    "split": split,
                    "class_name": class_name,
                    "sample_count": count,
                    "split_total": total,
                    "class_ratio": (
                        count / total
                        if total
                        else 0.0
                    ),
                    "group_count": group_count,
                }
            )

    return rows


def build_group_rows(
    records: list[Record],
) -> list[dict]:
    grouped: dict[
        tuple[str, str, str],
        list[Record],
    ] = defaultdict(list)

    for record in records:
        key = (
            record.class_name,
            record.source_file,
            record.new_split,
        )
        grouped[key].append(record)

    rows: list[dict] = []

    for key in sorted(grouped):
        class_name, source_file, new_split = key
        group_records = grouped[key]

        rows.append(
            {
                "class_name": class_name,
                "source_file": source_file,
                "new_split": new_split,
                "sample_count": len(group_records),
                "original_splits": "|".join(
                    sorted(
                        {
                            record.original_split
                            for record in group_records
                        }
                    )
                ),
                "first_sample_id": min(
                    record.sample_id
                    for record in group_records
                ),
                "last_sample_id": max(
                    record.sample_id
                    for record in group_records
                ),
            }
        )

    return rows


def main() -> None:
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(
            f"Detection dataset not found: {DATA_ROOT}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    uav_records, unassigned_uav = collect_uav_records()
    bg_records, unassigned_bg = collect_background_records()

    records = bg_records + uav_records

    records.sort(
        key=lambda record: (
            NEW_SPLITS.index(record.new_split),
            record.class_name,
            record.source_file,
            record.sample_id,
        )
    )

    if unassigned_uav:
        raise RuntimeError(
            "Unassigned UAV source files:\n"
            + "\n".join(unassigned_uav)
        )

    if unassigned_bg:
        raise RuntimeError(
            "Unassigned background scans:\n"
            + "\n".join(unassigned_bg)
        )

    duplicate_ids = check_duplicate_sample_ids(records)
    missing_paths = check_missing_paths(records)
    overlap_rows = build_overlap_rows(records)

    overlap_pair_count = sum(
        row["overlap_group_count"] > 0
        for row in overlap_rows
    )

    if duplicate_ids:
        raise RuntimeError(
            "Duplicate sample IDs found:\n"
            + "\n".join(duplicate_ids[:50])
        )

    if missing_paths:
        raise RuntimeError(
            "Missing files found:\n"
            + "\n".join(missing_paths[:50])
        )

    if overlap_pair_count:
        raise RuntimeError(
            "Cross-split group overlap found:\n"
            + json.dumps(
                overlap_rows,
                indent=2,
            )
        )

    manifest_rows = [
        asdict(record)
        for record in records
    ]
    count_rows = build_count_rows(records)
    group_rows = build_group_rows(records)

    write_csv(
        OUTPUT_DIR / "all_samples_manifest_v3.csv",
        manifest_rows,
        list(Record.__dataclass_fields__.keys()),
    )

    write_csv(
        OUTPUT_DIR / "split_class_counts.csv",
        count_rows,
        [
            "split",
            "class_name",
            "sample_count",
            "split_total",
            "class_ratio",
            "group_count",
        ],
    )

    write_csv(
        OUTPUT_DIR / "source_group_summary.csv",
        group_rows,
        [
            "class_name",
            "source_file",
            "new_split",
            "sample_count",
            "original_splits",
            "first_sample_id",
            "last_sample_id",
        ],
    )

    write_csv(
        OUTPUT_DIR / "group_overlap_check.csv",
        overlap_rows,
        [
            "class_name",
            "split_a",
            "split_b",
            "overlap_group_count",
            "overlap_groups",
        ],
    )

    per_split: dict[str, dict] = {}

    for split in NEW_SPLITS:
        split_records = [
            record
            for record in records
            if record.new_split == split
        ]

        bg_records_split = [
            record
            for record in split_records
            if record.class_name == "Background"
        ]

        uav_records_split = [
            record
            for record in split_records
            if record.class_name == "UAV"
        ]

        per_split[split] = {
            "total": len(split_records),
            "background": len(bg_records_split),
            "uav": len(uav_records_split),
            "background_groups": len(
                {
                    record.source_file
                    for record in bg_records_split
                }
            ),
            "uav_groups": len(
                {
                    record.source_file
                    for record in uav_records_split
                }
            ),
        }

    summary = {
        "project_root": str(PROJECT_ROOT),
        "data_root": str(DATA_ROOT),
        "output_dir": str(OUTPUT_DIR),
        "total_samples": len(records),
        "duplicate_sample_id_count": len(duplicate_ids),
        "missing_path_count": len(missing_paths),
        "overlap_pair_count": overlap_pair_count,
        "unassigned_uav_groups": unassigned_uav,
        "unassigned_background_groups": unassigned_bg,
        "per_split": per_split,
        "audit_passed": (
            not duplicate_ids
            and not missing_paths
            and overlap_pair_count == 0
            and not unassigned_uav
            and not unassigned_bg
        ),
    }

    with (
        OUTPUT_DIR / "audit_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print("=" * 78)
    print("Dataset V3 manifest created")
    print("=" * 78)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output dir:   {OUTPUT_DIR}")
    print()

    for split in NEW_SPLITS:
        item = per_split[split]
        total = item["total"]
        bg = item["background"]
        uav = item["uav"]

        print(
            f"{split:>5}: "
            f"total={total:4d}, "
            f"Background={bg:4d}, "
            f"UAV={uav:4d}, "
            f"BG groups={item['background_groups']}, "
            f"UAV groups={item['uav_groups']}"
        )

    print()
    print(
        "Duplicate sample IDs:",
        len(duplicate_ids),
    )
    print(
        "Missing paths:",
        len(missing_paths),
    )
    print(
        "Cross-split overlap pairs:",
        overlap_pair_count,
    )
    print(
        "Audit passed:",
        summary["audit_passed"],
    )
    print()
    print(
        "Manifest:",
        OUTPUT_DIR / "all_samples_manifest_v3.csv",
    )
    print()
    print(
        "No raw data files were moved, copied, or modified."
    )


if __name__ == "__main__":
    main()
