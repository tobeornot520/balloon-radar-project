#!/usr/bin/env python3
"""Read-only schema, route, and leakage audit for LSS-HSR-L ScienceDB V2."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import re
import stat
import tempfile
import unicodedata
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = (
    PROJECT_ROOT
    / "data/raw/external/LSS-HSR-L/ScienceDB_V2_dataset_and_instructions.zip"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/data_audit/lss_hsr_l_v2"

EXPECTED_ARCHIVE_SIZE_BYTES = 237_020_946
EXPECTED_ARCHIVE_SHA256 = (
    "fea8a21354110a96fb9644dc1c69649b6dc6d1a1b6da512498d9c2d74d839540"
)
JOURNAL_ARCHIVE_SIZE_BYTES = 209_569_478
JOURNAL_ARCHIVE_SHA256 = (
    "22112d4225636c5626845a9f0640abbf4503cc70763b592a609287760ab5f4a4"
)
AUDIT_STATUS = (
    "PASS_SCHEMA_BLOCKED_SOURCE_PROVENANCE_AND_PHYSICAL_AXIS"
)
SPLITS = ("train", "validation", "overflow")
CLASSES = (
    "UAV1",
    "UAV2",
    "UAV3",
    "UAV4",
    "Bird",
    "Birds",
    "Wing Bird",
    "Rotate",
    "Car",
)
WINDOW_SIZE = 10
FIRST_FRAME_REPEAT = 4
MAT_NAME_PATTERN = re.compile(r"^(?P<serial>[0-9]{4})_(?P<frames>[1-9][0-9]*)\.mat$")
ROUTE_ID_PATTERN = re.compile(r"^air_route_(?P<number>[1-9][0-9]*)$")
OUTPUT_FILES = {
    "REPORT.md",
    "summary.json",
    "split_summary.csv",
    "split_class_summary.csv",
    "feature_summary.csv",
    "route_audit.csv",
    "mat_audit.csv",
}

EXPECTED_ZIP_ENTRY_COUNT = 1_561
EXPECTED_ZIP_FILE_COUNT = 1_534
EXPECTED_ZIP_DIRECTORY_COUNT = 27
EXPECTED_MAT_FILE_COUNT = 1_530
EXPECTED_ROUTE_COUNT = 865
EXPECTED_STEP_LENGTH_COUNTS = {1: 802, 5: 63}
EXPECTED_SPLIT_MAT_COUNTS = {"train": 1269, "validation": 250, "overflow": 11}
EXPECTED_SPLIT_ROUTE_COUNTS = {"train": 723, "validation": 131, "overflow": 11}
EXPECTED_SPLIT_FRAME_COUNTS = {
    "train": 51_789,
    "validation": 10_655,
    "overflow": 704,
}
EXPECTED_SPLIT_WINDOW_COUNTS = {
    "train": 45_366,
    "validation": 9_336,
    "overflow": 529,
}
EXPECTED_SPLIT_CLASS_MAT_COUNTS = {
    ("train", "Bird"): 163,
    ("train", "Birds"): 114,
    ("train", "Car"): 276,
    ("train", "Rotate"): 122,
    ("train", "UAV1"): 116,
    ("train", "UAV2"): 120,
    ("train", "UAV3"): 117,
    ("train", "UAV4"): 122,
    ("train", "Wing Bird"): 119,
    ("validation", "Bird"): 27,
    ("validation", "Birds"): 24,
    ("validation", "Car"): 48,
    ("validation", "Rotate"): 28,
    ("validation", "UAV1"): 25,
    ("validation", "UAV2"): 25,
    ("validation", "UAV3"): 24,
    ("validation", "UAV4"): 25,
    ("validation", "Wing Bird"): 24,
    ("overflow", "Bird"): 2,
    ("overflow", "Car"): 3,
    ("overflow", "UAV3"): 1,
    ("overflow", "UAV4"): 5,
}
EXPECTED_SPLIT_CLASS_ROUTE_COUNTS = {
    ("train", "Bird"): 79,
    ("train", "Birds"): 78,
    ("train", "Car"): 271,
    ("train", "Rotate"): 48,
    ("train", "UAV1"): 78,
    ("train", "UAV2"): 29,
    ("train", "UAV3"): 57,
    ("train", "UAV4"): 40,
    ("train", "Wing Bird"): 43,
    ("validation", "Bird"): 12,
    ("validation", "Birds"): 17,
    ("validation", "Car"): 48,
    ("validation", "Rotate"): 4,
    ("validation", "UAV1"): 12,
    ("validation", "UAV2"): 9,
    ("validation", "UAV3"): 14,
    ("validation", "UAV4"): 6,
    ("validation", "Wing Bird"): 9,
    ("overflow", "Bird"): 2,
    ("overflow", "Car"): 3,
    ("overflow", "UAV3"): 1,
    ("overflow", "UAV4"): 5,
}
EXPECTED_SPLIT_CLASS_WINDOW_COUNTS = {
    ("train", "Bird"): 5019,
    ("train", "Birds"): 5035,
    ("train", "Car"): 5001,
    ("train", "Rotate"): 5004,
    ("train", "UAV1"): 5022,
    ("train", "UAV2"): 5066,
    ("train", "UAV3"): 5018,
    ("train", "UAV4"): 5133,
    ("train", "Wing Bird"): 5068,
    ("validation", "Bird"): 1025,
    ("validation", "Birds"): 1000,
    ("validation", "Car"): 1002,
    ("validation", "Rotate"): 1127,
    ("validation", "UAV1"): 1057,
    ("validation", "UAV2"): 1043,
    ("validation", "UAV3"): 1032,
    ("validation", "UAV4"): 1033,
    ("validation", "Wing Bird"): 1017,
    ("overflow", "Bird"): 131,
    ("overflow", "Car"): 21,
    ("overflow", "UAV3"): 21,
    ("overflow", "UAV4"): 356,
}
TRACK_FEATURES = (
    (1, "radial_velocity", "m/s", "positive means moving away from radar"),
    (2, "range", "km", "target-to-radar range"),
    (3, "azimuth", "degree", "target azimuth relative to radar"),
    (4, "height", "m", "target vertical height"),
    (5, "normalized_snr", "dB", "range-normalized signal-to-noise ratio"),
)


@dataclass(frozen=True)
class MatRecord:
    relative_path: str
    archive_path: str
    split: str
    category: str
    file_name: str
    declared_frame_count: int
    frame_count: int
    raw_sha256: str
    payload_sha256: str


@dataclass(frozen=True)
class RouteRecord:
    route_id: str
    route_number: int
    split: str
    category: str
    tracks: tuple[str, ...]
    step_length: int
    real_frame_count: int
    published_window_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit LSS-HSR-L ScienceDB V2 directly inside its ZIP archive."
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_known_journal_bundle(archive_path: Path) -> None:
    if archive_path.stat().st_size != JOURNAL_ARCHIVE_SIZE_BYTES:
        return
    if sha256_file(archive_path) == JOURNAL_ARCHIVE_SHA256:
        raise ValueError(
            "the historical journal bundle is not the ScienceDB V2 release and "
            "must remain isolated"
        )


def _array_payload_sha256(dpl: np.ndarray, track: np.ndarray) -> str:
    digest = hashlib.sha256()
    for label, value in (("dpl", dpl), ("track", track)):
        array = np.ascontiguousarray(value)
        digest.update(label.encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _json_without_duplicate_keys(payload: bytes, *, source: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{source}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        parsed = json.loads(payload.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{source}: invalid UTF-8 JSON") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"{source}: route JSON must contain an object")
    return parsed


def _safe_zip_inventory(
    archive: zipfile.ZipFile,
) -> tuple[list[zipfile.ZipInfo], list[zipfile.ZipInfo], str]:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise ValueError("ZIP contains duplicate member names")

    normalized_names: list[str] = []
    for info in infos:
        name = info.filename
        if "\\" in name or "\x00" in name:
            raise ValueError(f"unsafe ZIP member path: {name!r}")
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe ZIP member path: {name!r}")
        if path.parts and path.parts[0].endswith(":"):
            raise ValueError(f"unsafe ZIP member drive path: {name!r}")
        if info.flag_bits & 0x1:
            raise ValueError(f"encrypted ZIP member is forbidden: {name!r}")
        unix_mode = (info.external_attr >> 16) & 0xFFFF
        if stat.S_IFMT(unix_mode) == stat.S_IFLNK:
            raise ValueError(f"symlink ZIP member is forbidden: {name!r}")
        normalized_names.append(
            unicodedata.normalize("NFC", name).casefold().rstrip("/")
        )
    if len(normalized_names) != len(set(normalized_names)):
        raise ValueError("ZIP contains case-folded or Unicode-normalized path collisions")
    if archive.testzip() is not None:
        raise ValueError("ZIP CRC/integrity test failed")

    files = [info for info in infos if not info.is_dir()]
    route_files = [
        info
        for info in files
        if PurePosixPath(info.filename).parts[-2:] == ("LSS-HSR-L", "air_routes.json")
    ]
    if len(route_files) != 1:
        raise ValueError(f"expected one LSS-HSR-L/air_routes.json, found {len(route_files)}")
    dataset_prefix = route_files[0].filename[: -len("air_routes.json")]
    if not dataset_prefix.endswith("LSS-HSR-L/"):
        raise ValueError("cannot determine the canonical LSS-HSR-L ZIP prefix")

    dataset_py = [info for info in files if PurePosixPath(info.filename).name == "Dataset.py"]
    if len(dataset_py) != 1:
        raise ValueError(f"expected one Dataset.py, found {len(dataset_py)}")
    if sum(info.filename.lower().endswith(".docx") for info in files) != 1:
        raise ValueError("expected one DOCX user guide")
    if sum(info.filename.lower().endswith(".pdf") for info in files) != 1:
        raise ValueError("expected one PDF user guide")
    return infos, files, dataset_prefix


def _load_mat_payload(payload: bytes, *, relative_path: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        loaded = loadmat(BytesIO(payload), squeeze_me=False, struct_as_record=False)
    except Exception as error:
        raise ValueError(f"{relative_path}: unreadable MATLAB file") from error
    public = {key: value for key, value in loaded.items() if not key.startswith("__")}
    if set(public) != {"Trace_DPL_Data"}:
        raise ValueError(
            f"{relative_path}: expected only Trace_DPL_Data, found {sorted(public)}"
        )
    container = np.asarray(public["Trace_DPL_Data"])
    if container.shape != (1, 2) or container.dtype != object:
        raise ValueError(f"{relative_path}: invalid Trace_DPL_Data container")

    values: list[np.ndarray] = []
    for index, label in ((0, "DPL"), (1, "track")):
        cell = np.asarray(container[0, index])
        if cell.shape != (1, 1) or cell.dtype != object:
            raise ValueError(f"{relative_path}: invalid {label} cell container")
        values.append(np.asarray(cell[0, 0]))
    dpl, track = values
    if dpl.dtype != np.dtype("float64") or track.dtype != np.dtype("float64"):
        raise ValueError(f"{relative_path}: DPL and track storage dtype must be float64")
    if np.iscomplexobj(dpl) or np.iscomplexobj(track):
        raise ValueError(f"{relative_path}: processed DPL and track arrays must be real")
    if dpl.ndim != 2 or dpl.shape[1] != 512:
        raise ValueError(f"{relative_path}: invalid DPL shape {dpl.shape}")
    if track.ndim != 2 or track.shape[1] != 5:
        raise ValueError(f"{relative_path}: invalid track shape {track.shape}")
    if dpl.shape[0] != track.shape[0]:
        raise ValueError(
            f"{relative_path}: DPL/track length mismatch {dpl.shape[0]} != {track.shape[0]}"
        )
    if not np.isfinite(dpl).all() or not np.isfinite(track).all():
        raise ValueError(f"{relative_path}: DPL or track contains NaN/Inf")
    return dpl, track


def _parse_mat_members(
    archive: zipfile.ZipFile,
    files: list[zipfile.ZipInfo],
    dataset_prefix: str,
) -> tuple[
    dict[str, MatRecord],
    Counter[str],
    Counter[tuple[str, str]],
    int,
    int,
    int,
    int,
]:
    members = [
        info
        for info in files
        if info.filename.startswith(dataset_prefix) and info.filename.lower().endswith(".mat")
    ]
    other_mats = [
        info
        for info in files
        if info.filename.lower().endswith(".mat") and not info.filename.startswith(dataset_prefix)
    ]
    if other_mats:
        raise ValueError("MAT files outside the canonical LSS-HSR-L dataset root")

    records: dict[str, MatRecord] = {}
    raw_hashes: Counter[str] = Counter()
    payload_hashes: Counter[str] = Counter()
    split_frames: Counter[str] = Counter()
    split_class_frames: Counter[tuple[str, str]] = Counter()
    for info in sorted(members, key=lambda item: item.filename):
        relative = info.filename[len(dataset_prefix) :]
        parts = PurePosixPath(relative).parts
        if len(parts) != 3:
            raise ValueError(f"unexpected MAT relative path: {relative}")
        split, category, file_name = parts
        if split not in SPLITS or category not in CLASSES:
            raise ValueError(f"unexpected MAT split/class path: {relative}")
        match = MAT_NAME_PATTERN.fullmatch(file_name)
        if match is None:
            raise ValueError(f"unexpected MAT filename: {relative}")
        if relative in records:
            raise ValueError(f"duplicate MAT relative path: {relative}")

        payload = archive.read(info)
        dpl, track = _load_mat_payload(payload, relative_path=relative)
        declared_frames = int(match.group("frames"))
        if dpl.shape[0] != declared_frames:
            raise ValueError(
                f"{relative}: filename declares {declared_frames} frames, found {dpl.shape[0]}"
            )
        raw_sha = hashlib.sha256(payload).hexdigest()
        payload_sha = _array_payload_sha256(dpl, track)
        raw_hashes[raw_sha] += 1
        payload_hashes[payload_sha] += 1
        split_frames[split] += dpl.shape[0]
        split_class_frames[(split, category)] += dpl.shape[0]
        records[relative] = MatRecord(
            relative_path=relative,
            archive_path=info.filename,
            split=split,
            category=category,
            file_name=file_name,
            declared_frame_count=declared_frames,
            frame_count=int(dpl.shape[0]),
            raw_sha256=raw_sha,
            payload_sha256=payload_sha,
        )
    return (
        records,
        split_frames,
        split_class_frames,
        len(raw_hashes),
        len(payload_hashes),
        sum(count > 1 for count in raw_hashes.values()),
        sum(count > 1 for count in payload_hashes.values()),
    )


def _parse_routes(
    payload: bytes,
    mats: dict[str, MatRecord],
) -> tuple[list[RouteRecord], dict[str, str]]:
    parsed = _json_without_duplicate_keys(payload, source="air_routes.json")
    numbered: list[tuple[int, str, dict[str, Any]]] = []
    for route_id, value in parsed.items():
        match = ROUTE_ID_PATTERN.fullmatch(route_id)
        if match is None:
            raise ValueError(f"invalid route_id: {route_id!r}")
        if not isinstance(value, dict) or set(value) != {"tracks", "step_length"}:
            raise ValueError(f"{route_id}: expected tracks and step_length fields")
        numbered.append((int(match.group("number")), route_id, value))
    numbered.sort()
    if [number for number, _, _ in numbered] != list(range(1, len(numbered) + 1)):
        raise ValueError("route_id numbers must be unique and contiguous from one")

    owner: dict[str, str] = {}
    routes: list[RouteRecord] = []
    for number, route_id, value in numbered:
        tracks_value = value["tracks"]
        if not isinstance(tracks_value, list) or not tracks_value:
            raise ValueError(f"{route_id}: tracks must be a nonempty list")
        step = value["step_length"]
        if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
            raise ValueError(f"{route_id}: step_length must be a positive integer")

        tracks: list[str] = []
        splits: set[str] = set()
        categories: set[str] = set()
        frame_count = 0
        for track_value in tracks_value:
            if not isinstance(track_value, str) or "\\" in track_value:
                raise ValueError(f"{route_id}: invalid track path {track_value!r}")
            path = PurePosixPath(track_value)
            if path.is_absolute() or ".." in path.parts or len(path.parts) != 3:
                raise ValueError(f"{route_id}: unsafe track path {track_value!r}")
            track = path.as_posix()
            if track not in mats:
                raise ValueError(f"{route_id}: listed track does not exist: {track}")
            if track in owner:
                raise ValueError(
                    f"{track}: assigned to both {owner[track]} and {route_id}"
                )
            owner[track] = route_id
            record = mats[track]
            tracks.append(track)
            splits.add(record.split)
            categories.add(record.category)
            frame_count += record.frame_count
        if len(splits) != 1 or len(categories) != 1:
            raise ValueError(f"{route_id}: route mixes split or class")
        virtual_length = frame_count + FIRST_FRAME_REPEAT
        window_count = max(0, (virtual_length - WINDOW_SIZE) // step + 1)
        routes.append(
            RouteRecord(
                route_id=route_id,
                route_number=number,
                split=next(iter(splits)),
                category=next(iter(categories)),
                tracks=tuple(tracks),
                step_length=step,
                real_frame_count=frame_count,
                published_window_count=window_count,
            )
        )
    unlisted = sorted(set(mats) - set(owner))
    if unlisted:
        raise ValueError(f"MAT files missing from air_routes.json: {unlisted[:5]}")
    return routes, owner


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _feature_rows(*, strict_release: bool) -> list[dict[str, Any]]:
    if not strict_release:
        rows = [
            {
                "representation": "dpl",
                "feature_index": "all",
                "feature_name": "not_evaluated",
                "width": 512,
                "unit": "not_evaluated",
                "unit_source": "not evaluated in schema-fixture mode",
                "physical_axis_available": "NOT_EVALUATED",
            }
        ]
        rows.extend(
            {
                "representation": "track",
                "feature_index": index,
                "feature_name": "not_evaluated",
                "width": 1,
                "unit": "not_evaluated",
                "unit_source": "not evaluated in schema-fixture mode",
                "physical_axis_available": "NOT_EVALUATED",
            }
            for index, _, _, _ in TRACK_FEATURES
        )
        return rows

    rows = [
        {
            "representation": "dpl",
            "feature_index": "all",
            "feature_name": "processed_doppler_waterfall",
            "width": 512,
            "unit": "unverified",
            "unit_source": "not machine-readable in release",
            "physical_axis_available": False,
        }
    ]
    rows.extend(
        {
            "representation": "track",
            "feature_index": index,
            "feature_name": name,
            "width": 1,
            "unit": unit,
            "unit_source": f"official V2 user guide: {description}",
            "physical_axis_available": True,
        }
        for index, name, unit, description in TRACK_FEATURES
    )
    return rows


def _report(summary: dict[str, Any]) -> str:
    if summary["release_identity_verified"]:
        identity_result = (
            "- The canonical ScienceDB V2 ZIP passed identity, path-safety, CRC, "
            "inventory, and read-only source checks."
        )
        route_result = (
            "- Every MAT belongs to exactly one authoritative `route_id`; no route "
            "crosses a published split or class."
        )
        window_result = (
            "- The released `window_size=10` and `first_frame_repeat=4` rules "
            "reproduce "
            f"{summary['split_published_window_counts']['train']} train and "
            f"{summary['split_published_window_counts']['validation']} validation "
            "windows."
        )
        release_boundary = f"""- The undocumented `overflow` directory contributes
  {summary['split_published_window_counts']['overflow']} additional windows and must
  not be merged into train or validation.
- `route_id` is authoritative, but no acquisition-day, flight, scene, or source-session
  key is available. Published split membership is not proof of independent deployment
  generalization.
- The 512-bin DPL is a processed real waterfall, not raw ADC/complex IQ. The release has
  no machine-readable CPI duration, PRF, Doppler-bin-to-Hz/velocity mapping, or verified
  DPL amplitude unit. Track column units are documented separately and do not recover
  the DPL physical axes."""
    else:
        identity_result = (
            "- Release identity was not evaluated in schema-fixture mode; ZIP "
            "path-safety, CRC, and read-only source checks passed."
        )
        route_result = (
            "- The fixture route map assigns every MAT exactly once and no fixture "
            "route crosses a split or class."
        )
        window_result = (
            "- The configured `window_size=10` and `first_frame_repeat=4` arithmetic "
            "was reproduced for the fixture."
        )
        release_boundary = """- Release provenance, official track-column semantics and
  units, raw ADC/IQ and H/V availability, overflow publication role, and DPL physical
  axes were not evaluated in schema-fixture mode.
- Fixture route grouping is not evidence of acquisition-session independence.
- Fixture outputs are schema-only test evidence and cannot support dataset or model
  claims."""
    return f"""# LSS-HSR-L ScienceDB V2 Read-Only Audit

Status: `{summary['status']}`
Archive SHA256: `{summary['archive_sha256']}`
MAT files: `{summary['mat_file_count']}`
Routes: `{summary['route_count']}`

## Passed

{identity_result}
- All MAT files contain real finite float64 `Trace_DPL_Data` payloads with aligned
  `[T, 512]` and `[T, 5]` numeric arrays. Filename frame counts
  agree with `T`.
{route_result}
{window_result}

## Blockers and claim boundary

{release_boundary}
- Random MAT/frame/window splitting, physical-Hz micro-Doppler claims, H/V polarimetry,
  and model training are not authorized by this audit. A future published-split baseline
  requires a separate preregistered protocol.

The bundled `Dataset.py` was not executed and the source ZIP was not extracted or modified.
"""


def _freeze_strict_release(summary: dict[str, Any]) -> None:
    expected = {
        "archive_size_bytes": EXPECTED_ARCHIVE_SIZE_BYTES,
        "archive_sha256": EXPECTED_ARCHIVE_SHA256,
        "zip_entry_count": EXPECTED_ZIP_ENTRY_COUNT,
        "zip_file_count": EXPECTED_ZIP_FILE_COUNT,
        "zip_directory_count": EXPECTED_ZIP_DIRECTORY_COUNT,
        "mat_file_count": EXPECTED_MAT_FILE_COUNT,
        "route_count": EXPECTED_ROUTE_COUNT,
        "route_track_reference_count": EXPECTED_MAT_FILE_COUNT,
        "unique_raw_mat_count": EXPECTED_MAT_FILE_COUNT,
        "unique_numeric_payload_count": EXPECTED_MAT_FILE_COUNT,
        "total_frame_count": 63_148,
        "published_window_count": 55_231,
    }
    for field, value in expected.items():
        if summary.get(field) != value:
            raise ValueError(
                f"unexpected frozen HSR statistic {field}: {summary.get(field)!r} != {value!r}"
            )
    frozen_maps = {
        "step_length_counts": {
            str(key): value for key, value in EXPECTED_STEP_LENGTH_COUNTS.items()
        },
        "split_mat_counts": EXPECTED_SPLIT_MAT_COUNTS,
        "split_route_counts": EXPECTED_SPLIT_ROUTE_COUNTS,
        "split_frame_counts": EXPECTED_SPLIT_FRAME_COUNTS,
        "split_published_window_counts": EXPECTED_SPLIT_WINDOW_COUNTS,
    }
    for field, value in frozen_maps.items():
        if summary.get(field) != value:
            raise ValueError(f"unexpected frozen HSR mapping {field}")


def _audit_archive_to_directory(
    *, archive_path: Path, output_dir: Path, strict_release: bool
) -> dict[str, Any]:
    if not archive_path.is_file():
        raise FileNotFoundError(f"archive not found: {archive_path}")
    if not output_dir.is_dir() or output_dir.is_symlink() or any(output_dir.iterdir()):
        raise ValueError("staging output directory must be an empty regular directory")

    initial_stat = archive_path.stat()
    if strict_release and initial_stat.st_size != EXPECTED_ARCHIVE_SIZE_BYTES:
        raise ValueError(
            "unexpected V2 archive size: "
            f"{initial_stat.st_size} != {EXPECTED_ARCHIVE_SIZE_BYTES}"
        )
    archive_sha = sha256_file(archive_path)
    if strict_release:
        if archive_sha != EXPECTED_ARCHIVE_SHA256:
            raise ValueError(f"unexpected V2 archive SHA256: {archive_sha}")

    with zipfile.ZipFile(archive_path, "r") as archive:
        infos, files, dataset_prefix = _safe_zip_inventory(archive)
        route_info = next(
            info for info in files if info.filename == dataset_prefix + "air_routes.json"
        )
        (
            mats,
            split_frames,
            split_class_frames,
            unique_raw_count,
            unique_payload_count,
            raw_duplicate_groups,
            payload_duplicate_groups,
        ) = _parse_mat_members(archive, files, dataset_prefix)
        routes, owner = _parse_routes(archive.read(route_info), mats)

    final_stat = archive_path.stat()
    if (
        initial_stat.st_size != final_stat.st_size
        or initial_stat.st_mtime_ns != final_stat.st_mtime_ns
        or sha256_file(archive_path) != archive_sha
    ):
        raise RuntimeError("source ZIP changed while the read-only audit was running")

    split_mat_counts = Counter(record.split for record in mats.values())
    split_class_mat_counts = Counter(
        (record.split, record.category) for record in mats.values()
    )
    split_route_counts = Counter(route.split for route in routes)
    split_class_route_counts = Counter((route.split, route.category) for route in routes)
    split_windows = Counter()
    split_class_windows = Counter()
    step_counts = Counter()
    for route in routes:
        split_windows[route.split] += route.published_window_count
        split_class_windows[(route.split, route.category)] += route.published_window_count
        step_counts[route.step_length] += 1

    mat_rows = [
        {
            "relative_path": record.relative_path,
            "route_id": owner[record.relative_path],
            "split": record.split,
            "category": record.category,
            "file_name": record.file_name,
            "declared_frame_count": record.declared_frame_count,
            "frame_count": record.frame_count,
            "dpl_shape": f"{record.frame_count}x512",
            "track_shape": f"{record.frame_count}x5",
            "storage_dtype": "float64",
            "real_and_finite": True,
            "raw_sha256": record.raw_sha256,
            "numeric_payload_sha256": record.payload_sha256,
        }
        for record in sorted(mats.values(), key=lambda item: item.relative_path)
    ]
    route_rows = [
        {
            "route_id": route.route_id,
            "route_number": route.route_number,
            "split": route.split,
            "category": route.category,
            "track_count": len(route.tracks),
            "real_frame_count": route.real_frame_count,
            "step_length": route.step_length,
            "window_size": WINDOW_SIZE,
            "first_frame_repeat": FIRST_FRAME_REPEAT,
            "published_window_count": route.published_window_count,
            "single_split_and_class": True,
        }
        for route in routes
    ]
    split_rows = [
        {
            "split": split,
            "mat_file_count": split_mat_counts[split],
            "route_count": split_route_counts[split],
            "frame_count": split_frames[split],
            "published_window_count": split_windows[split],
            "class_count": len(
                {record.category for record in mats.values() if record.split == split}
            ),
            "role": (
                (
                    "published_model_split"
                    if split in {"train", "validation"}
                    else "quarantined_undocumented_overflow"
                )
                if strict_release
                else "NOT_EVALUATED_TEST_FIXTURE"
            ),
        }
        for split in SPLITS
    ]
    split_class_rows = [
        {
            "split": split,
            "category": category,
            "mat_file_count": split_class_mat_counts[(split, category)],
            "route_count": split_class_route_counts[(split, category)],
            "frame_count": split_class_frames[(split, category)],
            "published_window_count": split_class_windows[(split, category)],
        }
        for split in SPLITS
        for category in CLASSES
        if split_class_mat_counts[(split, category)] > 0
    ]

    if strict_release:
        blockers = [
            "overflow is present in V2 but omitted from the published "
            "train/validation statistics and Dataset.py main loop",
            "route_id exists but acquisition day, flight, scene, and "
            "source-session identities are unavailable",
            "the processed DPL has no machine-readable CPI duration, PRF, or "
            "bin-to-Hz/velocity mapping",
            "no preregistered modeling protocol has been approved for this release",
        ]
        allowed_use = (
            "read-only loader/schema verification, route-grouped method design, and "
            "exact reproduction of published window counts"
        )
    else:
        blockers = [
            "release identity and provenance were not evaluated",
            "official track semantics and physical units were not evaluated",
            "raw IQ, H/V polarimetry, overflow publication role, and DPL physical "
            "axes were not evaluated",
            "schema-fixture results are not dataset or model evidence",
        ]
        allowed_use = "schema-fixture test execution only"

    summary: dict[str, Any] = {
        "schema_version": 1,
        "dataset_id": "lss_hsr_l",
        "target_release_version": "V2",
        "release_version": "V2" if strict_release else None,
        "audit_mode": "strict_release" if strict_release else "schema_fixture",
        "release_identity_verified": strict_release,
        "target_data_doi": "10.57760/sciencedb.radars.00063",
        "data_doi": "10.57760/sciencedb.radars.00063" if strict_release else None,
        "status": AUDIT_STATUS,
        "gates": {
            "release_identity": (
                "PASS" if strict_release else "NOT_EVALUATED_TEST_FIXTURE"
            ),
            "zip_safety_and_integrity": "PASS",
            "mat_schema_and_finite": "PASS",
            "route_mapping": "PASS",
            "published_train_validation_route_isolation": (
                "PASS" if strict_release else "NOT_EVALUATED_TEST_FIXTURE"
            ),
            "overflow_role": (
                "BLOCKED_UNDOCUMENTED"
                if strict_release
                else "NOT_EVALUATED_TEST_FIXTURE"
            ),
            "acquisition_session_identity": (
                "BLOCKED_NOT_AVAILABLE"
                if strict_release
                else "NOT_EVALUATED_TEST_FIXTURE"
            ),
            "dpl_physical_time_axis": (
                "BLOCKED_NOT_AVAILABLE"
                if strict_release
                else "NOT_EVALUATED_TEST_FIXTURE"
            ),
            "dpl_physical_doppler_axis": (
                "BLOCKED_NOT_AVAILABLE"
                if strict_release
                else "NOT_EVALUATED_TEST_FIXTURE"
            ),
            "model_training": "BLOCKED",
        },
        "archive_size_bytes": initial_stat.st_size,
        "archive_sha256": archive_sha,
        "zip_entry_count": len(infos),
        "zip_file_count": len(files),
        "zip_directory_count": len(infos) - len(files),
        "zip_integrity_passed": True,
        "zip_paths_safe": True,
        "mat_file_count": len(mats),
        "route_count": len(routes),
        "route_track_reference_count": sum(len(route.tracks) for route in routes),
        "unique_raw_mat_count": unique_raw_count,
        "unique_numeric_payload_count": unique_payload_count,
        "exact_raw_mat_duplicate_group_count": raw_duplicate_groups,
        "exact_numeric_payload_duplicate_group_count": payload_duplicate_groups,
        "total_frame_count": sum(record.frame_count for record in mats.values()),
        "doppler_value_count": sum(record.frame_count * 512 for record in mats.values()),
        "track_value_count": sum(record.frame_count * 5 for record in mats.values()),
        "published_window_count": sum(route.published_window_count for route in routes),
        "window_size": WINDOW_SIZE,
        "first_frame_repeat": FIRST_FRAME_REPEAT,
        "step_length_counts": {str(key): value for key, value in sorted(step_counts.items())},
        "split_mat_counts": {split: split_mat_counts[split] for split in SPLITS},
        "split_route_counts": {split: split_route_counts[split] for split in SPLITS},
        "split_frame_counts": {split: split_frames[split] for split in SPLITS},
        "split_published_window_counts": {split: split_windows[split] for split in SPLITS},
        "mat_schema": {
            "public_fields": ["Trace_DPL_Data"],
            "dpl_shape": "[T, 512]",
            "track_shape": "[T, 5]",
            "storage_dtype": "float64",
            "real_values": True,
            "all_values_finite": True,
        },
        "track_features": (
            [
                {"index": index, "name": name, "unit": unit}
                for index, name, unit, _ in TRACK_FEATURES
            ]
            if strict_release
            else [
                {"index": index, "name": None, "unit": None}
                for index, _, _, _ in TRACK_FEATURES
            ]
        ),
        "route_id_present": True,
        "authoritative_route_id_available": True if strict_release else None,
        "every_mat_assigned_to_exactly_one_route": True,
        "train_validation_route_disjoint": True,
        "published_train_validation_route_disjoint": (
            True if strict_release else None
        ),
        "published_train_validation_route_overlap_count": (
            0 if strict_release else None
        ),
        "published_train_validation_session_disjoint_verified": False,
        "published_split_preservation_required": True if strict_release else None,
        "overflow_merge_allowed": False,
        "overflow_isolated": True,
        "overflow_role_documented_in_published_statistics": (
            False if strict_release else None
        ),
        "session_identity_field_present": False,
        "acquisition_session_key_available": False if strict_release else None,
        "random_mat_split_allowed": False,
        "random_frame_or_window_split_allowed": False,
        "lowest_available_grouping_key": "route_id",
        "minimum_split_unit": "UNRESOLVED_SOURCE_SESSION_IDENTITY_UNAVAILABLE",
        "route_id_sufficient_for_independent_evaluation": False,
        "dpl_representation": (
            "processed real 512-bin Doppler waterfall" if strict_release else None
        ),
        "dpl_amplitude_unit_verified": False if strict_release else None,
        "dpl_physical_time_axis_available": False if strict_release else None,
        "dpl_physical_doppler_hz_axis_available": (
            False if strict_release else None
        ),
        "dpl_physical_velocity_axis_available": False if strict_release else None,
        "track_physical_units_available": strict_release,
        "track_physical_units_verified": strict_release,
        "raw_adc_or_iq_available": False if strict_release else None,
        "h_v_polarimetry_available": False if strict_release else None,
        "physical_micro_doppler_hz_allowed": False,
        "model_training_allowed": False,
        "model_training_performed": False,
        "official_dataset_py_executed": False,
        "source_archive_extracted": False,
        "source_archive_modified": False,
        "raw_data_included_in_outputs": False,
        "local_route_and_mat_audit_tables_included": True,
        "journal_bundle_mixing_allowed": False,
        "blockers": blockers,
        "allowed_use": allowed_use,
        "prohibited_claims": [
            "random MAT/frame/window split performance",
            "overflow as train, validation, test, or independent evidence",
            "independent-session or deployment generalization",
            "physical-Hz or physical-time micro-Doppler features",
            "raw IQ, H/V polarimetry, or balloon recognition",
            "model performance before a separate preregistered protocol",
        ],
    }

    if strict_release:
        _freeze_strict_release(summary)
        if split_class_mat_counts != Counter(EXPECTED_SPLIT_CLASS_MAT_COUNTS):
            raise ValueError("unexpected frozen HSR split/class MAT counts")
        if split_class_route_counts != Counter(EXPECTED_SPLIT_CLASS_ROUTE_COUNTS):
            raise ValueError("unexpected frozen HSR split/class route counts")
        if split_class_windows != Counter(EXPECTED_SPLIT_CLASS_WINDOW_COUNTS):
            raise ValueError("unexpected frozen HSR split/class window counts")
        expected_route_ranges = {
            "train": list(range(1, 724)),
            "validation": list(range(724, 855)),
            "overflow": list(range(855, 866)),
        }
        for split, expected_numbers in expected_route_ranges.items():
            actual_numbers = [route.route_number for route in routes if route.split == split]
            if actual_numbers != expected_numbers:
                raise ValueError(f"unexpected frozen HSR route-number range for {split}")

    _write_csv(output_dir / "mat_audit.csv", mat_rows, list(mat_rows[0]))
    _write_csv(output_dir / "route_audit.csv", route_rows, list(route_rows[0]))
    _write_csv(output_dir / "split_summary.csv", split_rows, list(split_rows[0]))
    _write_csv(
        output_dir / "split_class_summary.csv",
        split_class_rows,
        list(split_class_rows[0]),
    )
    feature_rows = _feature_rows(strict_release=strict_release)
    _write_csv(output_dir / "feature_summary.csv", feature_rows, list(feature_rows[0]))
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "REPORT.md").write_text(_report(summary), encoding="utf-8")
    _validate_complete_output(output_dir)
    return summary


def _managed_output_entries(output_dir: Path) -> list[Path]:
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise ValueError("output path must be a regular directory, not a symlink")
    entries = list(output_dir.iterdir())
    unknown = [entry.name for entry in entries if entry.name not in OUTPUT_FILES]
    if unknown:
        raise ValueError(f"unknown output entries: {sorted(unknown)}")
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            raise ValueError(f"non-regular output entry: {entry.name}")
    return entries


def _validate_output_dir(output_dir: Path, archive_path: Path, overwrite: bool) -> None:
    source_root = archive_path.parent
    if output_dir in {PROJECT_ROOT, source_root, archive_path}:
        raise ValueError("output directory must be separate from project and source roots")
    try:
        output_dir.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise ValueError("output directory must not be inside the protected source root")
    if output_dir.is_symlink():
        raise ValueError("output directory must not be a symlink")
    if output_dir.exists():
        entries = _managed_output_entries(output_dir)
        if entries and not overwrite:
            raise FileExistsError(f"output directory is nonempty: {output_dir}")


def _output_snapshot(output_dir: Path) -> tuple[tuple[str, str], ...] | None:
    if not output_dir.exists():
        return None
    return tuple(
        sorted((entry.name, sha256_file(entry)) for entry in _managed_output_entries(output_dir))
    )


def _validate_complete_output(output_dir: Path) -> None:
    actual = {entry.name for entry in _managed_output_entries(output_dir)}
    if actual != OUTPUT_FILES:
        raise RuntimeError(f"unexpected audit outputs: {sorted(actual)}")
    try:
        summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("generated summary.json cannot be parsed") from error
    if not isinstance(summary, dict) or summary.get("status") != AUDIT_STATUS:
        raise RuntimeError("generated summary.json has an invalid audit status")


def _remove_generated_output_dir(output_dir: Path) -> None:
    for entry in _managed_output_entries(output_dir):
        entry.unlink()
    output_dir.rmdir()


def _unfinished_transactions(output_dir: Path) -> list[Path]:
    prefix = f".{output_dir.name}.transaction-"
    return sorted(
        path for path in output_dir.parent.iterdir() if path.name.startswith(prefix)
    )


def _audit_archive_transaction(
    *, archive_path: Path, output_dir: Path, overwrite: bool, strict_release: bool
) -> dict[str, Any]:
    _validate_output_dir(output_dir, archive_path, overwrite)
    original_snapshot = _output_snapshot(output_dir)
    transaction_root = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.transaction-", dir=output_dir.parent
        )
    )
    new_output = transaction_root / "new"
    old_output = transaction_root / "old"
    new_output.mkdir()
    preserve_transaction = False

    try:
        summary = _audit_archive_to_directory(
            archive_path=archive_path,
            output_dir=new_output,
            strict_release=strict_release,
        )
        _validate_complete_output(new_output)
        _validate_output_dir(output_dir, archive_path, overwrite)
        if _output_snapshot(output_dir) != original_snapshot:
            raise RuntimeError("output directory changed while the audit was running")

        had_previous_output = output_dir.exists()
        if had_previous_output:
            output_dir.rename(old_output)
        try:
            new_output.rename(output_dir)
        except BaseException as promotion_error:
            if had_previous_output:
                try:
                    old_output.rename(output_dir)
                except BaseException as rollback_error:
                    preserve_transaction = True
                    promotion_error.add_note(
                        "automatic rollback failed; the previous output remains at "
                        f"{old_output}: {rollback_error}"
                    )
            raise
    except BaseException as error:
        if not preserve_transaction:
            try:
                if new_output.exists():
                    _remove_generated_output_dir(new_output)
                if old_output.exists():
                    _remove_generated_output_dir(old_output)
                transaction_root.rmdir()
            except BaseException as cleanup_error:
                error.add_note(
                    f"transaction cleanup failed; inspect {transaction_root}: {cleanup_error}"
                )
        raise

    if old_output.exists():
        try:
            _remove_generated_output_dir(old_output)
        except BaseException as cleanup_error:
            raise RuntimeError(
                "new audit output was committed, but the previous-output backup could "
                f"not be removed; inspect {old_output}"
            ) from cleanup_error
    transaction_root.rmdir()
    return summary


def audit_dataset(
    *,
    archive_path: Path,
    output_dir: Path,
    overwrite: bool = False,
    strict_release: bool = True,
) -> dict[str, Any]:
    archive_input = archive_path.expanduser()
    if not archive_input.is_absolute():
        archive_input = PROJECT_ROOT / archive_input
    if archive_input.is_symlink():
        raise ValueError("source archive must not be a symlink")
    archive_path = archive_input.resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"archive not found: {archive_path}")
    _reject_known_journal_bundle(archive_path)

    output_input = output_dir.expanduser()
    if not output_input.is_absolute():
        output_input = PROJECT_ROOT / output_input
    if output_input.is_symlink():
        raise ValueError("output directory must not be a symlink")
    output_dir = output_input.resolve()

    _validate_output_dir(output_dir, archive_path, overwrite)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(output_dir.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        unfinished = _unfinished_transactions(output_dir)
        if unfinished:
            paths = ", ".join(str(path) for path in unfinished)
            raise RuntimeError(
                "unfinished audit transaction detected; inspect and recover or remove "
                f"it before rerunning: {paths}"
            )
        return _audit_archive_transaction(
            archive_path=archive_path,
            output_dir=output_dir,
            overwrite=overwrite,
            strict_release=strict_release,
        )
    finally:
        os.close(lock_fd)


def main() -> int:
    args = parse_args()
    summary = audit_dataset(
        archive_path=args.archive,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
