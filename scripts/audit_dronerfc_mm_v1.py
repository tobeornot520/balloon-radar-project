#!/usr/bin/env python3
"""Read-only schema, integrity, and timestamp audit for DroneRFc-MM V1."""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "external" / "DroneRFc-MM-V1"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "data_audit" / "dronerfc_mm_v1"
)
MANIFEST_NAME = "SHA256SUMS_RADAR_SUBSET_V1.txt"
EXPECTED_MANIFEST_SHA256 = (
    "6b0c2ed1a075aa9164a516af001b630a9f775fddc9f399223c1aeeb6e7047b2b"
)
EXPECTED_SUBSET_FILE_COUNT = 28
EXPECTED_SUBSET_SIZE_BYTES = 47_366_902
EXPECTED_RECORDINGS = (
    "A1",
    "A1-2",
    "B1",
    "C1",
    "E1",
    "E1-2",
    "F1",
    "G1",
    "G1-2",
)
EXPECTED_PCD_COUNTS = {
    "A1": 2609,
    "A1-2": 2539,
    "B1": 5028,
    "C1": 5924,
    "E1": 2557,
    "E1-2": 2523,
    "F1": 4235,
    "G1": 2574,
    "G1-2": 2728,
}
EXPECTED_GT_ROWS = {
    "A1": 3174,
    "A1-2": 2971,
    "B1": 4889,
    "C1": 7466,
    "E1": 4774,
    "E1-2": 2556,
    "F1": 3520,
    "G1": 3835,
    "G1-2": 2774,
}
EXPECTED_LABEL_WINDOWS = {
    "A1": 122,
    "B1": 97,
    "C1": 149,
    "E1": 148,
    "F1": 70,
    "G1": 131,
}
PCD_FIELDS = (
    "x",
    "y",
    "z",
    "rgb",
    "range",
    "azimuth",
    "elevation",
    "doppler",
    "power",
    "snr",
    "track_id",
    "cluster_id",
    "timestamp_sec",
    "timestamp_nsec",
    "radar_id",
)
PCD_SIZE = ("4",) * 15
PCD_TYPE = ("F",) * 10 + ("I",) * 5
PCD_COUNT = ("1",) * 15
GT_FIELDS = (
    "timestamp",
    "x",
    "y",
    "z",
    "xSpeed",
    "ySpeed",
    "zSpeed",
    "pitch",
    "roll",
    "yaw",
    "isSwaveWork",
)
LABEL_FIELDS = ("idx", "start_time", "end_time", "direction")
ALLOWED_DIRECTIONS = frozenset({"F", "B", "L", "R", "U", "D", "S"})
PCD_NAME_PATTERN = re.compile(
    r"(?P<frame_id>[0-9]+)_(?P<seconds>[0-9]{10})\."
    r"(?P<nanoseconds>[0-9]{9})\.pcd$"
)
SHORT_OFFSET_PATTERN = re.compile(r"([+-])([0-9]):([0-9]{2})$")
OUTPUT_FILES = frozenset({"README.md", "recording_audit.csv", "summary.json"})


@dataclass(frozen=True)
class PcdArchiveAudit:
    frame_count: int
    point_count: int
    min_points_per_frame: int
    max_points_per_frame: int
    timestamps: tuple[float, ...]


@dataclass(frozen=True)
class GroundTruthAudit:
    row_count: int
    duplicate_timestamp_count: int
    timestamps: tuple[float, ...]


@dataclass(frozen=True)
class LabelWindow:
    idx: int
    start: float
    end: float
    direction: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the local DroneRFc-MM radar subset without extracting or "
            "modifying its source files."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_aware_timestamp(value: str) -> datetime:
    value = value.strip()
    match = SHORT_OFFSET_PATTERN.search(value)
    if match:
        value = value[: match.start()] + f"{match.group(1)}0{match.group(2)}:{match.group(3)}"
    timestamp = datetime.fromisoformat(value)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"timestamp must include a UTC offset: {value!r}")
    return timestamp


def timestamp_to_epoch(value: str) -> float:
    return parse_aware_timestamp(value).timestamp()


def epoch_to_utc_text(value: float) -> str:
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def linear_quantile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile needs at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def parse_pcd_member_name(name: str) -> tuple[int, int, int]:
    match = PCD_NAME_PATTERN.search(name)
    if match is None:
        raise ValueError(f"invalid PCD member name: {name}")
    return (
        int(match.group("frame_id")),
        int(match.group("seconds")),
        int(match.group("nanoseconds")),
    )


def _header_values(header: dict[str, tuple[str, ...]], key: str) -> tuple[str, ...]:
    if key not in header:
        raise ValueError(f"PCD header missing {key}")
    return header[key]


def audit_pcd_stream(
    raw_handle: object,
    *,
    expected_seconds: int,
    expected_nanoseconds: int,
) -> int:
    header: dict[str, tuple[str, ...]] = {}
    while True:
        raw_line = raw_handle.readline()  # type: ignore[attr-defined]
        if not raw_line:
            raise ValueError("PCD ended before DATA header")
        try:
            line = raw_line.decode("ascii").strip()
        except UnicodeDecodeError as exc:
            raise ValueError("PCD header is not ASCII") from exc
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        key = parts[0]
        if key in header:
            raise ValueError(f"duplicate PCD header key: {key}")
        header[key] = tuple(parts[1:])
        if key == "DATA":
            break

    if _header_values(header, "VERSION") != ("0.7",):
        raise ValueError("unexpected PCD VERSION")
    if _header_values(header, "FIELDS") != PCD_FIELDS:
        raise ValueError("unexpected PCD FIELDS")
    if _header_values(header, "SIZE") != PCD_SIZE:
        raise ValueError("unexpected PCD SIZE")
    if _header_values(header, "TYPE") != PCD_TYPE:
        raise ValueError("unexpected PCD TYPE")
    if _header_values(header, "COUNT") != PCD_COUNT:
        raise ValueError("unexpected PCD COUNT")
    if _header_values(header, "HEIGHT") != ("1",):
        raise ValueError("PCD HEIGHT must be 1")
    if _header_values(header, "DATA") != ("ascii",):
        raise ValueError("PCD DATA must be ascii")

    try:
        width = int(_header_values(header, "WIDTH")[0])
        points = int(_header_values(header, "POINTS")[0])
    except (IndexError, ValueError) as exc:
        raise ValueError("PCD WIDTH/POINTS must be integers") from exc
    if width < 0 or points < 0 or width != points:
        raise ValueError("PCD WIDTH and POINTS must be equal and nonnegative")

    row_count = 0
    for raw_line in raw_handle:  # type: ignore[union-attr]
        try:
            line = raw_line.decode("ascii").strip()
        except UnicodeDecodeError as exc:
            raise ValueError("PCD data row is not ASCII") from exc
        if not line:
            continue
        values = line.split()
        if len(values) != len(PCD_FIELDS):
            raise ValueError("PCD data row does not have exactly 15 columns")
        try:
            floats = [float(value) for value in values[:10]]
            integers = [int(value) for value in values[10:]]
        except ValueError as exc:
            raise ValueError("PCD data row contains an invalid numeric value") from exc
        if not all(math.isfinite(value) for value in floats):
            raise ValueError("PCD data row contains a non-finite value")
        if integers[2] != expected_seconds or integers[3] != expected_nanoseconds:
            raise ValueError("PCD row timestamp does not match its member name")
        row_count += 1
    if row_count != points:
        raise ValueError(
            f"PCD POINTS says {points}, but {row_count} data rows were read"
        )
    return points


def audit_pcd_archive(path: Path) -> PcdArchiveAudit:
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"ZIP CRC check failed for {bad_member}")
        members = [info for info in archive.infolist() if info.filename.endswith(".pcd")]
        if not members:
            raise ValueError(f"no PCD members in {path.name}")
        parsed = [(parse_pcd_member_name(info.filename), info) for info in members]
        parsed.sort(key=lambda item: item[0][0])
        frame_ids = [item[0][0] for item in parsed]
        if frame_ids != list(range(len(parsed))):
            raise ValueError(f"PCD frame IDs are not contiguous in {path.name}")

        timestamps: list[float] = []
        points_per_frame: list[int] = []
        for (frame_id, seconds, nanoseconds), info in parsed:
            del frame_id
            timestamp = seconds + nanoseconds / 1_000_000_000
            if timestamps and timestamp <= timestamps[-1]:
                raise ValueError(
                    f"PCD timestamps are not strictly increasing in {path.name}"
                )
            timestamps.append(timestamp)
            with archive.open(info, "r") as raw_handle:
                points_per_frame.append(
                    audit_pcd_stream(
                        raw_handle,
                        expected_seconds=seconds,
                        expected_nanoseconds=nanoseconds,
                    )
                )

    return PcdArchiveAudit(
        frame_count=len(timestamps),
        point_count=sum(points_per_frame),
        min_points_per_frame=min(points_per_frame),
        max_points_per_frame=max(points_per_frame),
        timestamps=tuple(timestamps),
    )


def audit_ground_truth(path: Path) -> GroundTruthAudit:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != GT_FIELDS:
            raise ValueError(f"unexpected ground-truth columns in {path.name}")
        timestamps: list[float] = []
        duplicate_count = 0
        for row_number, row in enumerate(reader, start=2):
            timestamp = timestamp_to_epoch(row["timestamp"])
            if timestamps and timestamp < timestamps[-1]:
                raise ValueError(
                    f"ground-truth timestamps decrease at {path.name}:{row_number}"
                )
            if timestamps and timestamp == timestamps[-1]:
                duplicate_count += 1
            timestamps.append(timestamp)
            for field in GT_FIELDS[1:]:
                try:
                    value = float(row[field])
                except ValueError as exc:
                    raise ValueError(
                        f"invalid {field} at {path.name}:{row_number}"
                    ) from exc
                if not math.isfinite(value):
                    raise ValueError(
                        f"non-finite {field} at {path.name}:{row_number}"
                    )
    if not timestamps:
        raise ValueError(f"ground-truth table is empty: {path.name}")
    return GroundTruthAudit(
        row_count=len(timestamps),
        duplicate_timestamp_count=duplicate_count,
        timestamps=tuple(timestamps),
    )


def audit_labels(path: Path) -> tuple[LabelWindow, ...]:
    windows: list[LabelWindow] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != LABEL_FIELDS:
            raise ValueError(f"unexpected label columns in {path.name}")
        for row_number, row in enumerate(reader, start=2):
            try:
                idx = int(row["idx"])
            except ValueError as exc:
                raise ValueError(f"invalid label idx at {path.name}:{row_number}") from exc
            if idx != len(windows):
                raise ValueError(f"label idx is not contiguous in {path.name}")
            start = timestamp_to_epoch(row["start_time"])
            end = timestamp_to_epoch(row["end_time"])
            if end <= start:
                raise ValueError(f"label window is not positive at {path.name}:{row_number}")
            if windows and start < windows[-1].end:
                raise ValueError(f"label windows overlap in {path.name}")
            direction = row["direction"].strip()
            if direction not in ALLOWED_DIRECTIONS:
                raise ValueError(f"unexpected direction {direction!r} in {path.name}")
            windows.append(LabelWindow(idx, start, end, direction))
    if not windows:
        raise ValueError(f"label table is empty: {path.name}")
    return tuple(windows)


def parse_and_verify_manifest(dataset_root: Path) -> tuple[int, int]:
    manifest_path = dataset_root / MANIFEST_NAME
    if sha256_file(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("selected-subset manifest SHA256 does not match the registry")
    rows: list[tuple[str, Path]] = []
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line:
            continue
        try:
            expected_hash, relative_text = line.split("  ", maxsplit=1)
        except ValueError as exc:
            raise ValueError(f"invalid manifest line {line_number}") from exc
        if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise ValueError(f"invalid manifest SHA256 at line {line_number}")
        if not relative_text.startswith("./"):
            raise ValueError(f"manifest path is not relative at line {line_number}")
        relative = Path(relative_text[2:])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe manifest path at line {line_number}")
        rows.append((expected_hash, relative))
    if len(rows) != EXPECTED_SUBSET_FILE_COUNT:
        raise ValueError("selected-subset manifest must contain exactly 28 files")

    total_bytes = 0
    seen: set[Path] = set()
    for expected_hash, relative in rows:
        if relative in seen:
            raise ValueError(f"duplicate manifest path: {relative}")
        seen.add(relative)
        path = dataset_root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"manifest file missing or unsafe: {relative}")
        if sha256_file(path) != expected_hash:
            raise ValueError(f"manifest file SHA256 mismatch: {relative}")
        total_bytes += path.stat().st_size
    if total_bytes != EXPECTED_SUBSET_SIZE_BYTES:
        raise ValueError("selected-subset byte count does not match the registry")
    return len(rows), total_bytes


def nearest_errors(
    source_timestamps: Iterable[float], reference_timestamps: tuple[float, ...]
) -> list[float]:
    if not reference_timestamps:
        raise ValueError("reference timestamps must not be empty")
    errors: list[float] = []
    for timestamp in source_timestamps:
        position = bisect.bisect_left(reference_timestamps, timestamp)
        candidates: list[float] = []
        if position < len(reference_timestamps):
            candidates.append(abs(reference_timestamps[position] - timestamp))
        if position:
            candidates.append(abs(reference_timestamps[position - 1] - timestamp))
        errors.append(min(candidates))
    return errors


def validate_output_dir(output_dir: Path, dataset_root: Path, overwrite: bool) -> None:
    output_resolved = output_dir.resolve()
    dataset_resolved = dataset_root.resolve()
    forbidden = {PROJECT_ROOT.resolve(), (PROJECT_ROOT / ".git").resolve(), dataset_resolved}
    if output_resolved in forbidden:
        raise ValueError("output directory is a protected project or source path")
    if dataset_resolved in output_resolved.parents or output_resolved in dataset_resolved.parents:
        raise ValueError("output directory must be separate from the source dataset")
    if output_dir.is_symlink():
        raise ValueError("output directory must not be a symlink")
    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise ValueError("output path exists and is not a directory")
    existing = {path.name for path in output_dir.iterdir()}
    unknown = existing - OUTPUT_FILES
    if unknown:
        raise ValueError(f"output directory contains unknown entries: {sorted(unknown)}")
    if existing and not overwrite:
        raise FileExistsError("output directory is nonempty; use --overwrite")


def write_outputs(
    output_dir: Path,
    *,
    rows: list[dict[str, object]],
    summary: dict[str, object],
    overwrite: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for name in OUTPUT_FILES:
            path = output_dir / name
            if path.is_file():
                path.unlink()

    fieldnames = list(rows[0])
    with (output_dir / "recording_audit.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    blocked = ", ".join(summary["blocked_recordings"]) or "none"  # type: ignore[arg-type]
    (output_dir / "README.md").write_text(
        "# DroneRFc-MM V1 Read-only Audit\n\n"
        f"Status: `{summary['status']}`  \n"
        f"Data DOI: `{summary['data_doi']}`  \n"
        f"Recordings: `{summary['recording_count']}`  \n"
        f"Split-family groups: `{summary['split_family_group_count']}`  \n"
        f"PCD frames: `{summary['pcd_frame_count']}`  \n"
        f"PCD points: `{summary['pcd_point_count']}`  \n"
        f"Ground-truth rows: `{summary['ground_truth_row_count']}`  \n"
        f"Label windows: `{summary['label_window_count']}`  \n"
        f"Blocked recordings: `{blocked}`\n\n"
        "## Integrity and scope\n\n"
        "The source ZIP/CSV files were read in place and were not extracted or "
        "modified. All 30,717 PCD files passed ZIP CRC, the 15-column schema, "
        "finite-value, declared POINTS-row-count, and filename/embedded-timestamp "
        "checks. This audit contains aggregate recording-level evidence only; raw "
        "PCD/CSV data and sample-level outputs are excluded.\n\n"
        "## Alignment decision\n\n"
        "Eight recordings have overlapping radar and same-named flight-truth time "
        "ranges. B1 is blocked because its two ranges do not overlap. B1 must not "
        "be used for supervised alignment unless corrected ground truth or an "
        "externally attributable time offset is obtained.\n\n"
        "## Split and training decision\n\n"
        "No model was trained and model training is not authorized by this audit. "
        "The minimum future split unit is `split_family_group`: A1/A1-2, E1/E1-2, "
        "and G1/G1-2 must remain together, leaving six base-family groups. Random "
        "PCD-frame or derived-window splits and frame/window performance claims are "
        "prohibited. The eight time-overlapping recordings may enter only a later, "
        "separately preregistered synchronization study.\n",
        encoding="utf-8",
    )


def audit_dataset(
    *, dataset_root: Path, output_dir: Path, overwrite: bool
) -> dict[str, object]:
    dataset_root = dataset_root.resolve()
    output_dir = output_dir.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {dataset_root}")
    validate_output_dir(output_dir, dataset_root, overwrite)
    subset_file_count, subset_size_bytes = parse_and_verify_manifest(dataset_root)

    labels_by_group: dict[str, tuple[LabelWindow, ...]] = {}
    for group, expected_count in EXPECTED_LABEL_WINDOWS.items():
        windows = audit_labels(dataset_root / "sample_labels" / f"{group}.csv")
        if len(windows) != expected_count:
            raise ValueError(f"unexpected label-window count for {group}")
        labels_by_group[group] = windows

    rows: list[dict[str, object]] = []
    total_points = 0
    total_frames = 0
    total_gt_rows = 0
    total_label_windows = sum(len(windows) for windows in labels_by_group.values())
    blocked_recordings: list[str] = []
    for recording in EXPECTED_RECORDINGS:
        radar_path = dataset_root / "raw_mmradar" / f"{recording.lower()}.zip"
        gt_path = dataset_root / "ground_truth" / f"{recording}.csv"
        pcd = audit_pcd_archive(radar_path)
        gt = audit_ground_truth(gt_path)
        if pcd.frame_count != EXPECTED_PCD_COUNTS[recording]:
            raise ValueError(f"unexpected PCD count for {recording}")
        if gt.row_count != EXPECTED_GT_ROWS[recording]:
            raise ValueError(f"unexpected ground-truth row count for {recording}")

        overlap = max(
            0.0,
            min(pcd.timestamps[-1], gt.timestamps[-1])
            - max(pcd.timestamps[0], gt.timestamps[0]),
        )
        alignment_status = "PASS_TIME_RANGE_OVERLAP"
        if overlap <= 0.0:
            alignment_status = "BLOCKED_NO_RADAR_GT_TIME_OVERLAP"
            blocked_recordings.append(recording)
        errors = nearest_errors(pcd.timestamps, gt.timestamps)
        label_group = recording.split("-", maxsplit=1)[0]
        label_overlap_count = sum(
            min(window.end, pcd.timestamps[-1])
            > max(window.start, pcd.timestamps[0])
            for window in labels_by_group[label_group]
        )
        rows.append(
            {
                "recording_id": recording,
                "split_family_group": label_group,
                "radar_zip": f"raw_mmradar/{recording.lower()}.zip",
                "pcd_frame_count": pcd.frame_count,
                "pcd_point_count": pcd.point_count,
                "min_points_per_frame": pcd.min_points_per_frame,
                "max_points_per_frame": pcd.max_points_per_frame,
                "radar_start_utc": epoch_to_utc_text(pcd.timestamps[0]),
                "radar_end_utc": epoch_to_utc_text(pcd.timestamps[-1]),
                "ground_truth_csv": f"ground_truth/{recording}.csv",
                "ground_truth_row_count": gt.row_count,
                "ground_truth_duplicate_timestamp_count": gt.duplicate_timestamp_count,
                "ground_truth_start_utc": epoch_to_utc_text(gt.timestamps[0]),
                "ground_truth_end_utc": epoch_to_utc_text(gt.timestamps[-1]),
                "radar_gt_overlap_seconds": round(overlap, 9),
                "nearest_gt_error_median_seconds": round(linear_quantile(errors, 0.5), 9),
                "nearest_gt_error_p95_seconds": round(linear_quantile(errors, 0.95), 9),
                "nearest_gt_error_max_seconds": round(max(errors), 9),
                "label_group": label_group,
                "label_window_overlap_count": label_overlap_count,
                "alignment_status": alignment_status,
            }
        )
        total_frames += pcd.frame_count
        total_points += pcd.point_count
        total_gt_rows += gt.row_count

    status = "PASS_SCHEMA_BLOCKED_TIMESTAMP_ALIGNMENT" if blocked_recordings else "PASS"
    summary: dict[str, object] = {
        "schema_version": 1,
        "dataset_id": "dronerfc_mm",
        "release_version": "V1",
        "data_doi": "10.57760/sciencedb.j00173.00094",
        "status": status,
        "model_training_allowed": False,
        "recording_count": len(rows),
        "split_family_group_count": len({row["split_family_group"] for row in rows}),
        "pcd_frame_count": total_frames,
        "pcd_point_count": total_points,
        "ground_truth_row_count": total_gt_rows,
        "label_window_count": total_label_windows,
        "blocked_recordings": blocked_recordings,
        "selected_subset_file_count": subset_file_count,
        "selected_subset_size_bytes": subset_size_bytes,
        "selected_subset_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "source_files_modified": False,
        "source_archives_extracted": False,
        "sample_level_outputs_included": False,
        "minimum_split_unit": "split_family_group; keep -2 recordings with their base group",
        "allowed_scope": (
            "Read-only PCD, trajectory, Doppler-point and timestamp interface audit; "
            "only the eight time-overlapping recordings may enter a later separately "
            "preregistered synchronization study."
        ),
        "prohibited_scope": (
            "Random frame/window split; B1 supervised alignment; ADC/IQ or raw "
            "micro-Doppler claims; H/V polarimetry; bird/background/balloon, "
            "unseen-model or deployment performance."
        ),
    }
    write_outputs(output_dir, rows=rows, summary=summary, overwrite=overwrite)
    return summary


def main() -> int:
    args = parse_args()
    summary = audit_dataset(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print("DroneRFc-MM V1 read-only audit: COMPLETE")
    print(f"status={summary['status']}")
    print(f"recordings={summary['recording_count']}")
    print(f"pcd_frames={summary['pcd_frame_count']}")
    print(f"blocked_recordings={','.join(summary['blocked_recordings'])}")  # type: ignore[arg-type]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
