#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_MANIFEST = (
    PROJECT_ROOT
    / "results"
    / "data_audit"
    / "dataset_v3_grouped"
    / "all_samples_manifest_v3.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "data_audit"
    / "dataset_v4_multifold"
)

REQUIRED_COLUMNS = (
    "new_split",
    "original_split",
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
)

VALID_SPLITS = ("train", "val", "test")

# A gap larger than this must become a block boundary rather
# than remain inside one UAV temporal block.
MAX_INTERNAL_UAV_GAP_SECONDS = 120


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate six-fold grouped manifests from the "
            "Dataset V3 master manifest."
        )
    )
    parser.add_argument(
        "--input-manifest",
        default=str(DEFAULT_INPUT_MANIFEST),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--val-offset",
        type=int,
        default=1,
        help=(
            "Validation block offset relative to the test block. "
            "Default: test=i, val=i+1 modulo folds."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def parse_source_timestamp(source_file: str) -> datetime:
    try:
        return datetime.strptime(
            source_file,
            "%Y%m%d_%H%M%S",
        )
    except ValueError as exc:
        raise ValueError(
            f"Invalid source_file timestamp: {source_file}"
        ) from exc


def validate_input_manifest(frame: pd.DataFrame) -> None:
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in frame.columns
    ]
    if missing_columns:
        raise RuntimeError(
            f"Missing required columns: {missing_columns}"
        )

    duplicate_count = int(
        frame["sample_id"].duplicated().sum()
    )
    if duplicate_count:
        raise RuntimeError(
            f"Duplicate sample_id count: {duplicate_count}"
        )

    class_values = set(
        frame["class_name"].astype(str).unique()
    )
    if class_values != {"Background", "UAV"}:
        raise RuntimeError(
            f"Unexpected class_name values: {class_values}"
        )

    target_by_class = (
        frame.groupby("class_name")["target_present"]
        .unique()
        .to_dict()
    )

    if set(target_by_class["Background"]) != {0}:
        raise RuntimeError(
            "Background target_present must be 0."
        )

    if set(target_by_class["UAV"]) != {1}:
        raise RuntimeError(
            "UAV target_present must be 1."
        )

    source_class_counts = (
        frame.groupby("source_file")["class_name"]
        .nunique()
    )
    mixed_sources = source_class_counts[
        source_class_counts > 1
    ]
    if len(mixed_sources):
        raise RuntimeError(
            "Some source_file values contain multiple classes: "
            f"{mixed_sources.index.tolist()}"
        )


def contiguous_partition(
    groups: list[tuple[str, int]],
    block_count: int,
) -> list[list[tuple[str, int]]]:
    """
    Partition time-ordered groups into contiguous blocks.

    The objective minimizes the sum of squared deviations from
    the ideal sample count per block while keeping every group
    intact.
    """
    group_count = len(groups)
    if group_count < block_count:
        raise ValueError(
            f"Need at least {block_count} groups, "
            f"found {group_count}."
        )

    counts = [count for _, count in groups]
    prefix = [0]

    for count in counts:
        prefix.append(prefix[-1] + count)

    target = prefix[-1] / block_count

    infinity = float("inf")
    dp = [
        [infinity] * (group_count + 1)
        for _ in range(block_count + 1)
    ]
    previous: list[list[int | None]] = [
        [None] * (group_count + 1)
        for _ in range(block_count + 1)
    ]

    dp[0][0] = 0.0

    for block_index in range(1, block_count + 1):
        minimum_end = block_index
        maximum_end = (
            group_count
            - (block_count - block_index)
        )

        for end in range(
            minimum_end,
            maximum_end + 1,
        ):
            minimum_start = block_index - 1
            maximum_start = end - 1

            for start in range(
                minimum_start,
                maximum_start + 1,
            ):
                prior_cost = dp[
                    block_index - 1
                ][start]

                if math.isinf(prior_cost):
                    continue

                block_samples = (
                    prefix[end] - prefix[start]
                )
                block_cost = (
                    block_samples - target
                ) ** 2

                total_cost = prior_cost + block_cost

                if total_cost < dp[block_index][end]:
                    dp[block_index][end] = total_cost
                    previous[block_index][end] = start

    if previous[block_count][group_count] is None:
        raise RuntimeError(
            "Unable to create contiguous UAV blocks."
        )

    boundaries: list[tuple[int, int]] = []
    end = group_count

    for block_index in range(
        block_count,
        0,
        -1,
    ):
        start = previous[block_index][end]
        if start is None:
            raise RuntimeError(
                "Invalid dynamic-programming backtrack."
            )

        boundaries.append((start, end))
        end = start

    boundaries.reverse()

    return [
        groups[start:end]
        for start, end in boundaries
    ]


def source_gap_seconds(
    left_source: str,
    right_source: str,
) -> int:
    return int(
        (
            parse_source_timestamp(right_source)
            - parse_source_timestamp(left_source)
        ).total_seconds()
    )


def enforce_large_gap_boundaries(
    blocks: list[list[tuple[str, int]]],
    max_gap_seconds: int,
) -> tuple[
    list[list[tuple[str, int]]],
    list[dict[str, Any]],
]:
    """
    Ensure that a major acquisition pause is a boundary between
    UAV blocks rather than an internal gap inside one block.

    If a large gap lies inside a block, move the smaller side to
    an adjacent block while preserving temporal order and the
    original number of folds.
    """
    adjusted = [
        list(block)
        for block in blocks
    ]

    total_samples = sum(
        sample_count
        for block in adjusted
        for _, sample_count in block
    )
    target_samples = (
        total_samples / len(adjusted)
    )

    adjustments: list[dict[str, Any]] = []

    made_change = True

    while made_change:
        made_change = False

        for block_index, block in enumerate(
            adjusted
        ):
            large_positions = [
                position
                for position in range(1, len(block))
                if source_gap_seconds(
                    block[position - 1][0],
                    block[position][0],
                ) > max_gap_seconds
            ]

            if not large_positions:
                continue

            if len(large_positions) > 1:
                raise RuntimeError(
                    "A UAV block contains multiple large "
                    "internal gaps. Manual review is required: "
                    f"block={block_index + 1}, "
                    f"positions={large_positions}"
                )

            split_position = large_positions[0]
            left_part = block[:split_position]
            right_part = block[split_position:]

            left_samples = sum(
                count for _, count in left_part
            )
            right_samples = sum(
                count for _, count in right_part
            )

            gap_seconds = source_gap_seconds(
                left_part[-1][0],
                right_part[0][0],
            )

            options: list[
                tuple[
                    float,
                    str,
                    list[list[tuple[str, int]]],
                ]
            ] = []

            # Option A:
            # Move the prefix to the previous block.
            if block_index > 0:
                candidate = [
                    list(part)
                    for part in adjusted
                ]

                candidate[block_index - 1].extend(
                    left_part
                )
                candidate[block_index] = right_part

                previous_samples = sum(
                    count
                    for _, count
                    in candidate[block_index - 1]
                )
                current_samples = sum(
                    count
                    for _, count
                    in candidate[block_index]
                )

                cost = (
                    previous_samples
                    - target_samples
                ) ** 2 + (
                    current_samples
                    - target_samples
                ) ** 2

                options.append(
                    (
                        cost,
                        "move_prefix_to_previous",
                        candidate,
                    )
                )

            # Option B:
            # Move the suffix to the next block.
            if block_index < len(adjusted) - 1:
                candidate = [
                    list(part)
                    for part in adjusted
                ]

                candidate[block_index] = left_part
                candidate[block_index + 1] = (
                    right_part
                    + candidate[block_index + 1]
                )

                current_samples = sum(
                    count
                    for _, count
                    in candidate[block_index]
                )
                next_samples = sum(
                    count
                    for _, count
                    in candidate[block_index + 1]
                )

                cost = (
                    current_samples
                    - target_samples
                ) ** 2 + (
                    next_samples
                    - target_samples
                ) ** 2

                options.append(
                    (
                        cost,
                        "move_suffix_to_next",
                        candidate,
                    )
                )

            if not options:
                raise RuntimeError(
                    "Unable to move the large-gap boundary."
                )

            _, action, selected = min(
                options,
                key=lambda item: item[0],
            )

            adjustments.append(
                {
                    "original_block": block_index + 1,
                    "gap_left_source": left_part[-1][0],
                    "gap_right_source": right_part[0][0],
                    "gap_seconds": gap_seconds,
                    "left_sample_count": left_samples,
                    "right_sample_count": right_samples,
                    "action": action,
                }
            )

            adjusted = selected
            made_change = True
            break

    for block_index, block in enumerate(
        adjusted,
        start=1,
    ):
        if not block:
            raise RuntimeError(
                f"UAV block {block_index} became empty."
            )

        internal_gaps = [
            source_gap_seconds(
                block[position - 1][0],
                block[position][0],
            )
            for position in range(1, len(block))
        ]

        maximum_gap = max(
            internal_gaps,
            default=0,
        )

        if maximum_gap > max_gap_seconds:
            raise RuntimeError(
                f"UAV block {block_index} still contains "
                f"an internal gap of {maximum_gap} seconds."
            )

    return adjusted, adjustments


def path_exists(value: Any) -> bool:
    if pd.isna(value):
        return False

    text = str(value).strip()
    if not text:
        return False

    return Path(text).expanduser().is_file()


def build_fold_manifest(
    master: pd.DataFrame,
    fold_index: int,
    folds: int,
    val_offset: int,
    background_groups: list[str],
    uav_block_map: dict[str, int],
) -> pd.DataFrame:
    test_block = fold_index
    val_block = (
        fold_index + val_offset
    ) % folds

    background_test = background_groups[test_block]
    background_val = background_groups[val_block]

    def assign_split(row: pd.Series) -> str:
        class_name = str(row["class_name"])
        source_file = str(row["source_file"])

        if class_name == "Background":
            if source_file == background_test:
                return "test"
            if source_file == background_val:
                return "val"
            return "train"

        block_index = uav_block_map[source_file]

        if block_index == test_block:
            return "test"
        if block_index == val_block:
            return "val"
        return "train"

    fold = master.copy()
    fold["new_split"] = fold.apply(
        assign_split,
        axis=1,
    )

    split_order = pd.Categorical(
        fold["new_split"],
        categories=list(VALID_SPLITS),
        ordered=True,
    )

    fold = (
        fold.assign(_split_order=split_order)
        .sort_values(
            by=[
                "_split_order",
                "class_name",
                "source_file",
                "sample_id",
            ],
            kind="stable",
        )
        .drop(columns="_split_order")
        .reset_index(drop=True)
    )

    return fold


def summarize_fold(
    fold_number: int,
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for split in VALID_SPLITS:
        split_frame = frame[
            frame["new_split"] == split
        ]

        for class_name in ("Background", "UAV"):
            part = split_frame[
                split_frame["class_name"]
                == class_name
            ]

            sources = sorted(
                part["source_file"]
                .astype(str)
                .unique()
                .tolist()
            )

            rows.append(
                {
                    "fold": fold_number,
                    "split": split,
                    "class_name": class_name,
                    "sample_count": len(part),
                    "group_count": len(sources),
                    "source_files": "|".join(sources),
                }
            )

        rows.append(
            {
                "fold": fold_number,
                "split": split,
                "class_name": "ALL",
                "sample_count": len(split_frame),
                "group_count": int(
                    split_frame["source_file"].nunique()
                ),
                "source_files": "",
            }
        )

    return rows


def build_group_overlap_rows(
    fold_number: int,
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    grouped = frame.groupby(
        ["class_name", "source_file"],
        sort=True,
    )

    for (
        class_name,
        source_file,
    ), part in grouped:
        roles = sorted(
            part["new_split"]
            .astype(str)
            .unique()
            .tolist()
        )

        rows.append(
            {
                "fold": fold_number,
                "class_name": class_name,
                "source_file": source_file,
                "sample_count": len(part),
                "roles": "|".join(roles),
                "role_count": len(roles),
                "overlap_detected": len(roles) != 1,
            }
        )

    return rows


def main() -> None:
    args = parse_args()

    input_manifest = resolve_path(
        args.input_manifest
    )
    output_dir = resolve_path(
        args.output_dir
    )

    folds = int(args.folds)
    val_offset = int(args.val_offset)

    if folds < 3:
        raise ValueError("--folds must be at least 3.")

    if not 1 <= val_offset < folds:
        raise ValueError(
            "--val-offset must be between 1 and folds-1."
        )

    if not input_manifest.is_file():
        raise FileNotFoundError(
            f"Input manifest not found: {input_manifest}"
        )

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}\n"
                "Use --overwrite only if replacement is intended."
            )
        shutil.rmtree(output_dir)

    temporary_dir = output_dir.with_name(
        output_dir.name + ".tmp"
    )

    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)

    temporary_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    master = pd.read_csv(
        input_manifest,
        encoding="utf-8-sig",
    )

    validate_input_manifest(master)

    total_samples = len(master)
    total_sample_ids = set(
        master["sample_id"].astype(str)
    )

    background = master[
        master["class_name"] == "Background"
    ].copy()

    uav = master[
        master["class_name"] == "UAV"
    ].copy()

    background_group_counts = (
        background.groupby("source_file")
        .size()
        .sort_index()
    )

    background_groups = (
        background_group_counts.index
        .astype(str)
        .tolist()
    )

    if len(background_groups) != folds:
        raise RuntimeError(
            "The number of Background scans must equal "
            f"the fold count. Scans={len(background_groups)}, "
            f"folds={folds}."
        )

    for source_file in background_groups:
        parse_source_timestamp(source_file)

    uav_group_counts = (
        uav.groupby("source_file")
        .size()
        .sort_index()
    )

    uav_groups = [
        (str(source_file), int(sample_count))
        for source_file, sample_count
        in uav_group_counts.items()
    ]

    for source_file, _ in uav_groups:
        parse_source_timestamp(source_file)

    uav_blocks = contiguous_partition(
        uav_groups,
        folds,
    )

    (
        uav_blocks,
        large_gap_boundary_adjustments,
    ) = enforce_large_gap_boundaries(
        uav_blocks,
        MAX_INTERNAL_UAV_GAP_SECONDS,
    )

    uav_block_map: dict[str, int] = {}

    for block_index, block in enumerate(
        uav_blocks
    ):
        for source_file, _ in block:
            if source_file in uav_block_map:
                raise RuntimeError(
                    f"Repeated UAV group: {source_file}"
                )
            uav_block_map[source_file] = block_index

    if set(uav_block_map) != {
        source_file
        for source_file, _ in uav_groups
    }:
        raise RuntimeError(
            "UAV block assignment is incomplete."
        )

    uav_block_rows: list[dict[str, Any]] = []
    previous_end: datetime | None = None

    for block_index, block in enumerate(
        uav_blocks
    ):
        first_source = block[0][0]
        last_source = block[-1][0]
        first_time = parse_source_timestamp(
            first_source
        )
        last_time = parse_source_timestamp(
            last_source
        )

        preceding_gap_seconds = None
        if previous_end is not None:
            preceding_gap_seconds = int(
                (
                    first_time - previous_end
                ).total_seconds()
            )

        internal_gaps = [
            source_gap_seconds(
                block[position - 1][0],
                block[position][0],
            )
            for position in range(1, len(block))
        ]

        max_internal_gap_seconds = max(
            internal_gaps,
            default=0,
        )

        uav_block_rows.append(
            {
                "uav_block": block_index + 1,
                "first_source_file": first_source,
                "last_source_file": last_source,
                "group_count": len(block),
                "sample_count": sum(
                    count
                    for _, count in block
                ),
                "start_timestamp": first_time.isoformat(),
                "end_timestamp": last_time.isoformat(),
                "preceding_gap_seconds": (
                    preceding_gap_seconds
                ),
                "max_internal_gap_seconds": (
                    max_internal_gap_seconds
                ),
                "source_files": "|".join(
                    source_file
                    for source_file, _ in block
                ),
            }
        )

        previous_end = last_time

    fold_summary_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    fold_role_rows: list[dict[str, Any]] = []

    fold_audits: list[dict[str, Any]] = []
    source_role_counter: dict[
        tuple[str, str],
        Counter[str],
    ] = defaultdict(Counter)

    for fold_index in range(folds):
        fold_number = fold_index + 1
        val_index = (
            fold_index + val_offset
        ) % folds

        fold = build_fold_manifest(
            master=master,
            fold_index=fold_index,
            folds=folds,
            val_offset=val_offset,
            background_groups=background_groups,
            uav_block_map=uav_block_map,
        )

        fold_path = (
            temporary_dir
            / f"fold_{fold_number:02d}_manifest.csv"
        )

        fold.to_csv(
            fold_path,
            index=False,
            encoding="utf-8-sig",
        )

        fold_ids = set(
            fold["sample_id"].astype(str)
        )

        preserved_samples = (
            len(fold) == total_samples
            and fold_ids == total_sample_ids
        )

        invalid_splits = sorted(
            set(fold["new_split"].astype(str))
            - set(VALID_SPLITS)
        )

        group_role_counts = (
            fold.groupby(
                ["class_name", "source_file"]
            )["new_split"]
            .nunique()
        )

        overlapping_group_count = int(
            (group_role_counts != 1).sum()
        )

        fold_summary_rows.extend(
            summarize_fold(
                fold_number,
                fold,
            )
        )

        overlap_rows.extend(
            build_group_overlap_rows(
                fold_number,
                fold,
            )
        )

        for (
            class_name,
            source_file,
        ), part in fold.groupby(
            ["class_name", "source_file"],
            sort=True,
        ):
            role = str(
                part["new_split"].iloc[0]
            )

            source_role_counter[
                (class_name, source_file)
            ][role] += 1

            fold_role_rows.append(
                {
                    "fold": fold_number,
                    "class_name": class_name,
                    "source_file": source_file,
                    "role": role,
                    "sample_count": len(part),
                }
            )

        fold_counts = (
            fold.groupby(
                ["new_split", "class_name"]
            )
            .size()
            .to_dict()
        )

        fold_audits.append(
            {
                "fold": fold_number,
                "test_background_scan": (
                    background_groups[fold_index]
                ),
                "val_background_scan": (
                    background_groups[val_index]
                ),
                "test_uav_block": fold_index + 1,
                "val_uav_block": val_index + 1,
                "sample_count": len(fold),
                "preserved_all_samples": (
                    preserved_samples
                ),
                "invalid_splits": invalid_splits,
                "overlapping_group_count": (
                    overlapping_group_count
                ),
                "counts": {
                    split: {
                        class_name: int(
                            fold_counts.get(
                                (split, class_name),
                                0,
                            )
                        )
                        for class_name in (
                            "Background",
                            "UAV",
                        )
                    }
                    for split in VALID_SPLITS
                },
            }
        )

    role_frequency_rows: list[dict[str, Any]] = []
    role_frequency_errors: list[dict[str, Any]] = []

    expected_role_counts = {
        "train": folds - 2,
        "val": 1,
        "test": 1,
    }

    for (
        class_name,
        source_file,
    ), counter in sorted(
        source_role_counter.items()
    ):
        row = {
            "class_name": class_name,
            "source_file": source_file,
            "train_count": int(counter["train"]),
            "val_count": int(counter["val"]),
            "test_count": int(counter["test"]),
        }

        role_frequency_rows.append(row)

        if any(
            row[f"{role}_count"]
            != expected_count
            for role, expected_count
            in expected_role_counts.items()
        ):
            role_frequency_errors.append(row)

    missing_mat_paths = [
        str(value)
        for value in master["mat_path"]
        if not path_exists(value)
    ]

    positive_rows = master[
        master["target_present"] == 1
    ]

    missing_label_paths = [
        str(value)
        for value in positive_rows["label_path"]
        if not path_exists(value)
    ]

    duplicate_sample_id_count = int(
        master["sample_id"].duplicated().sum()
    )

    overlap_count = sum(
        int(row["overlap_detected"])
        for row in overlap_rows
    )

    audit_passed = (
        duplicate_sample_id_count == 0
        and len(missing_mat_paths) == 0
        and len(missing_label_paths) == 0
        and overlap_count == 0
        and len(role_frequency_errors) == 0
        and all(
            int(
                row["max_internal_gap_seconds"]
            )
            <= MAX_INTERNAL_UAV_GAP_SECONDS
            for row in uav_block_rows
        )
        and all(
            audit["preserved_all_samples"]
            and not audit["invalid_splits"]
            and audit["overlapping_group_count"] == 0
            for audit in fold_audits
        )
    )

    pd.DataFrame(
        uav_block_rows
    ).to_csv(
        temporary_dir / "uav_time_blocks.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        fold_summary_rows
    ).to_csv(
        temporary_dir / "fold_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        fold_role_rows
    ).to_csv(
        temporary_dir / "fold_group_roles.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        overlap_rows
    ).to_csv(
        temporary_dir / "group_overlap_check.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        role_frequency_rows
    ).to_csv(
        temporary_dir / "group_role_frequency.csv",
        index=False,
        encoding="utf-8-sig",
    )

    audit_summary = {
        "dataset_version": "V4",
        "design": (
            "six_fold_grouped_background_scan_"
            "and_contiguous_uav_time_block_rotation"
        ),
        "input_manifest": str(input_manifest),
        "fold_count": folds,
        "val_offset": val_offset,
        "total_samples": total_samples,
        "background_samples": len(background),
        "uav_samples": len(uav),
        "background_group_count": (
            len(background_groups)
        ),
        "uav_group_count": len(uav_groups),
        "uav_block_count": len(uav_blocks),
        "max_internal_uav_gap_seconds": (
            MAX_INTERNAL_UAV_GAP_SECONDS
        ),
        "large_gap_boundary_adjustments": (
            large_gap_boundary_adjustments
        ),
        "uav_block_internal_gap_violation_count": int(
            sum(
                int(
                    row["max_internal_gap_seconds"]
                )
                > MAX_INTERNAL_UAV_GAP_SECONDS
                for row in uav_block_rows
            )
        ),
        "duplicate_sample_id_count": (
            duplicate_sample_id_count
        ),
        "missing_mat_path_count": (
            len(missing_mat_paths)
        ),
        "missing_label_path_count": (
            len(missing_label_paths)
        ),
        "overlap_record_count": overlap_count,
        "role_frequency_error_count": (
            len(role_frequency_errors)
        ),
        "expected_role_frequency": (
            expected_role_counts
        ),
        "background_groups": (
            background_groups
        ),
        "uav_blocks": uav_block_rows,
        "folds": fold_audits,
        "role_frequency_errors": (
            role_frequency_errors
        ),
        "audit_passed": audit_passed,
        "interpretation_limit": (
            "All folds remain within one UAV flight session "
            "and one continuous Background session. Results "
            "must not be described as cross-session, "
            "cross-date, or cross-scene generalization."
        ),
    }

    (
        temporary_dir / "audit_summary.json"
    ).write_text(
        json.dumps(
            audit_summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if not audit_passed:
        raise RuntimeError(
            "Dataset V4 audit failed. "
            f"Inspect temporary output: {temporary_dir}"
        )

    temporary_dir.rename(output_dir)

    print("=" * 78)
    print("Dataset V4 multifold manifests generated.")
    print(f"Input:  {input_manifest}")
    print(f"Output: {output_dir}")
    print(
        f"Samples: total={total_samples}, "
        f"Background={len(background)}, "
        f"UAV={len(uav)}"
    )
    print(
        f"Groups: Background={len(background_groups)}, "
        f"UAV={len(uav_groups)}"
    )
    print()
    print("UAV contiguous blocks:")

    for row in uav_block_rows:
        print(
            f"  block {row['uav_block']:02d}: "
            f"{row['first_source_file']} -> "
            f"{row['last_source_file']}, "
            f"groups={row['group_count']}, "
            f"samples={row['sample_count']}, "
            f"preceding_gap_seconds="
            f"{row['preceding_gap_seconds']}, "
            f"max_internal_gap_seconds="
            f"{row['max_internal_gap_seconds']}"
        )

    print()
    print("Fold roles:")

    for audit in fold_audits:
        print(
            f"  fold {audit['fold']:02d}: "
            f"BG test={audit['test_background_scan']}, "
            f"BG val={audit['val_background_scan']}, "
            f"UAV test block={audit['test_uav_block']}, "
            f"UAV val block={audit['val_uav_block']}"
        )

    print()
    print(f"Audit passed: {audit_passed}")
    print("=" * 78)


if __name__ == "__main__":
    main()
