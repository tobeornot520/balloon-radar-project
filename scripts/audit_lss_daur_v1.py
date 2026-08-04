#!/usr/bin/env python3
"""Read-only integrity, pairing, and leakage audit for LSS-DAUR-1.0 V3."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import itertools
import json
import os
import re
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree

import h5py
import numpy as np
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data/raw/external/LSS-DAUR-1.0"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/data_audit/lss_daur_v1"

EXPECTED_OFFICIAL_FILE_COUNT = 314
EXPECTED_OFFICIAL_SIZE_BYTES = 148_763_512
EXPECTED_MANIFEST_SHA256 = (
    "5febc59a29c42fb7dd8b001afa73fe913767f00c2c060a21a5d95073c2ee1745"
)
AUDIT_STATUS = "PASS_SCHEMA_PAIRING_BLOCKED_GROUPING_AND_PHYSICAL_AXIS"
EXPECTED_CATEGORY_COUNTS = {
    "Bird": 17,
    "Fixed-wing UAV": 11,
    "Helicopter": 10,
    "Passenger ship": 10,
    "Rotary drone": 18,
    "Speedboat": 11,
}
EXPECTED_CANONICAL_FIELDS = {
    "TD": {"DATA_time", "DPL", "GPS_time_in_data", "Iframecnt", "nDaCf"},
    "TR": {
        "A",
        "A_m",
        "DATA_time",
        "E",
        "E_m",
        "GPS_time_in_data",
        "Iframecnt",
        "R",
        "R_m",
        "SNR",
        "V",
        "V_m",
        "nDaCf",
    },
}
PAIR_FIELDS = ("DATA_time", "GPS_time_in_data", "Iframecnt", "nDaCf")
TR_SIGNAL_FIELDS = ("A", "A_m", "E", "E_m", "R", "R_m", "SNR", "V", "V_m")
CONFIG_EXCLUDED_FIELDS = {"year", "month", "day", "begin", "end"}
RADAR_POSITION_FIELDS = ("fRadarLat", "fRadarLng", "fRadarHeight")
RECEIPT_NAMES = {"OFFICIAL_DOWNLOAD_URLS_V3.txt", "SHA256SUMS_V3.txt"}
OUTPUT_FILES = {
    "REPORT.md",
    "class_summary.csv",
    "date_conflicts.csv",
    "doppler_config_audit.csv",
    "duplicate_recordings.csv",
    "recording_audit.csv",
    "shared_frame_pairs.csv",
    "source_session_group_audit.csv",
    "source_session_membership.csv",
    "summary.json",
}
NAME_PATTERN = re.compile(
    r"^(?P<timestamp>[0-9]{14})_DAUR_(?P<token>RD|TR)_"
    r"(?P<category>.+)_(?P<serial>[0-9]{2})_(?P<batch>[0-9]+)\.mat$"
)
DATE_PATTERN = re.compile(r"^(?P<year>[0-9]{4})/(?P<month>[0-9]{1,2})/(?P<day>[0-9]{1,2})$")


@dataclass(frozen=True)
class ParsedName:
    timestamp: str
    modality: str
    category: str
    serial: int
    batch: int

    @property
    def recording_id(self) -> str:
        return f"{self.timestamp}|{self.category}|{self.serial:02d}|{self.batch}"


@dataclass
class Recording:
    name: ParsedName
    td_path: Path
    tr_path: Path
    td_backup_path: Path
    tr_backup_path: Path
    td: dict[str, np.ndarray]
    tr: dict[str, np.ndarray]
    file_head: dict[str, float]
    td_payload_sha256: str
    tr_payload_sha256: str

    @property
    def recording_id(self) -> str:
        return self.name.recording_id

    @property
    def frame_keys(self) -> set[tuple[int, int]]:
        gps = np.asarray(self.td["GPS_time_in_data"]).reshape(-1)
        frame = np.asarray(self.td["Iframecnt"]).reshape(-1)
        if not np.all(gps == np.rint(gps)) or not np.all(frame == np.rint(frame)):
            raise ValueError(f"{self.recording_id}: frame identity fields are not integers")
        return {(int(left), int(right)) for left, right in zip(gps, frame)}


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit all LSS-DAUR V3 MAT files without modifying the source release."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
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


def _payload_sha256(payload: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(payload):
        value = np.ascontiguousarray(np.asarray(payload[key]))
        digest.update(key.encode("utf-8") + b"\0")
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def parse_name(path: Path, modality: str) -> ParsedName:
    match = NAME_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"unexpected DAUR MAT filename: {path.name}")
    expected_token = "RD" if modality == "TD" else "TR"
    if match.group("token") != expected_token:
        raise ValueError(f"{path.name}: expected {expected_token} token for {modality}")
    category = match.group("category")
    if path.parent.name != category:
        raise ValueError(f"{path}: parent category does not match filename")
    return ParsedName(
        timestamp=match.group("timestamp"),
        modality=modality,
        category=category,
        serial=int(match.group("serial")),
        batch=int(match.group("batch")),
    )


def _public_payload(path: Path, modality: str) -> dict[str, np.ndarray]:
    payload = {
        key: np.asarray(value)
        for key, value in loadmat(path, simplify_cells=True).items()
        if not key.startswith("__")
    }
    fields = set(payload)
    if fields != EXPECTED_CANONICAL_FIELDS[modality]:
        raise ValueError(
            f"{path}: expected fields {sorted(EXPECTED_CANONICAL_FIELDS[modality])}, "
            f"found {sorted(fields)}"
        )
    for key, value in payload.items():
        if not np.issubdtype(value.dtype, np.number):
            raise ValueError(f"{path}: {key} is not numeric")
        if not np.isfinite(value).all():
            raise ValueError(f"{path}: {key} contains NaN or Inf")
    return payload


def _hdf5_numeric(dataset: h5py.Dataset) -> np.ndarray:
    value = np.asarray(dataset[()])
    if value.dtype.fields and set(value.dtype.fields) >= {"real", "imag"}:
        value = value["real"] + 1j * value["imag"]
    if value.ndim >= 2:
        value = value.T
    return np.asarray(value).squeeze()


def _backup_payload(path: Path) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    if not h5py.is_hdf5(path):
        raise ValueError(f"{path}: backup_original file is not MATLAB v7.3/HDF5")
    with h5py.File(path, "r") as handle:
        if "File_head" not in handle or not isinstance(handle["File_head"], h5py.Group):
            raise ValueError(f"{path}: missing File_head group")
        public = {
            key: _hdf5_numeric(value)
            for key, value in handle.items()
            if key not in {"#refs#", "File_head"} and isinstance(value, h5py.Dataset)
        }
        file_head = {
            key: float(np.asarray(value[()]).reshape(-1)[0])
            for key, value in handle["File_head"].items()
            if isinstance(value, h5py.Dataset)
        }
    for key, value in public.items():
        if not np.isfinite(value).all():
            raise ValueError(f"{path}: backup field {key} contains NaN or Inf")
    if not file_head or not all(np.isfinite(value) for value in file_head.values()):
        raise ValueError(f"{path}: invalid File_head values")
    return public, file_head


def _values_equal(left: Any, right: Any) -> bool:
    return np.array_equal(np.asarray(left).squeeze(), np.asarray(right).squeeze())


def _assert_backup_equivalent(
    canonical: dict[str, np.ndarray],
    backup: dict[str, np.ndarray],
    file_head: dict[str, float],
    *,
    path: Path,
) -> None:
    expected_backup_fields = set(canonical) - {"nDaCf"}
    if set(backup) != expected_backup_fields:
        raise ValueError(
            f"{path}: canonical/backup fields differ: "
            f"{sorted(expected_backup_fields)} != {sorted(backup)}"
        )
    for key in expected_backup_fields:
        if not _values_equal(canonical[key], backup[key]):
            raise ValueError(f"{path}: backup field {key} differs numerically")
    if "nDaCf" not in file_head or not _values_equal(canonical["nDaCf"], file_head["nDaCf"]):
        raise ValueError(f"{path}: canonical nDaCf differs from File_head.nDaCf")


def _canonical_paths(dataset_root: Path, modality: str) -> list[Path]:
    directory = dataset_root / f"{modality} Data"
    if not directory.is_dir():
        raise FileNotFoundError(f"missing modality directory: {directory}")
    return sorted(directory.glob("*/*.mat"))


def _backup_path(dataset_root: Path, path: Path, modality: str) -> Path:
    modality_root = dataset_root / f"{modality} Data"
    return modality_root / "backup_original" / path.relative_to(modality_root)


def _verify_release_manifest(dataset_root: Path) -> dict[str, Any]:
    manifest_path = dataset_root / "SHA256SUMS_V3.txt"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing release manifest: {manifest_path}")
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise ValueError(f"unexpected manifest SHA256: {manifest_sha}")

    expected: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        digest, marker, relative = line.partition("  ")
        if not marker or not relative.startswith("./") or len(digest) != 64:
            raise ValueError(f"invalid manifest line: {line!r}")
        normalized = relative[2:]
        if normalized in expected:
            raise ValueError(f"duplicate manifest path: {normalized}")
        expected[normalized] = digest

    actual_paths = sorted(
        path
        for path in dataset_root.rglob("*")
        if path.is_file() and path.name not in RECEIPT_NAMES
    )
    actual_names = {path.relative_to(dataset_root).as_posix() for path in actual_paths}
    if actual_names != set(expected):
        missing = sorted(set(expected) - actual_names)
        extra = sorted(actual_names - set(expected))
        raise ValueError(f"manifest inventory differs; missing={missing}, extra={extra}")
    mismatches = [
        relative
        for relative, digest in expected.items()
        if sha256_file(dataset_root / relative) != digest
    ]
    if mismatches:
        raise ValueError(f"release file hash mismatches: {mismatches}")
    total_bytes = sum(path.stat().st_size for path in actual_paths)
    if len(actual_paths) != EXPECTED_OFFICIAL_FILE_COUNT:
        raise ValueError(f"expected {EXPECTED_OFFICIAL_FILE_COUNT} files, found {len(actual_paths)}")
    if total_bytes != EXPECTED_OFFICIAL_SIZE_BYTES:
        raise ValueError(
            f"expected {EXPECTED_OFFICIAL_SIZE_BYTES} bytes, found {total_bytes}"
        )
    return {
        "official_file_count": len(actual_paths),
        "official_size_bytes": total_bytes,
        "manifest_sha256": manifest_sha,
        "all_manifest_file_hashes_match": True,
    }


def _weather_dates(dataset_root: Path) -> set[str]:
    candidates = sorted(dataset_root.glob("*.docx"))
    if not candidates:
        return set()
    if len(candidates) != 1:
        raise ValueError(f"expected one weather DOCX, found {len(candidates)}")
    with zipfile.ZipFile(candidates[0]) as archive:
        document = ElementTree.fromstring(archive.read("word/document.xml"))
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    dates: set[str] = set()
    for row in document.findall(".//w:tr", namespace):
        cells = []
        for cell in row.findall("w:tc", namespace):
            cells.append(
                "".join(
                    node.text or "" for node in cell.findall(".//w:t", namespace)
                ).strip()
            )
        if not cells:
            continue
        match = DATE_PATTERN.fullmatch(cells[0])
        if match:
            dates.add(
                f"{int(match.group('year')):04d}{int(match.group('month')):02d}"
                f"{int(match.group('day')):02d}"
            )
    return dates


def _load_recordings(dataset_root: Path) -> list[Recording]:
    by_modality: dict[str, dict[tuple[str, str, int, int], Path]] = {}
    for modality in ("TD", "TR"):
        entries: dict[tuple[str, str, int, int], Path] = {}
        for path in _canonical_paths(dataset_root, modality):
            parsed = parse_name(path, modality)
            key = (parsed.timestamp, parsed.category, parsed.serial, parsed.batch)
            if key in entries:
                raise ValueError(f"duplicate canonical {modality} key: {key}")
            entries[key] = path
        by_modality[modality] = entries
    if set(by_modality["TD"]) != set(by_modality["TR"]):
        raise ValueError("canonical TD/TR recording keys are not one-to-one")

    recordings: list[Recording] = []
    for key in sorted(by_modality["TD"]):
        td_path = by_modality["TD"][key]
        tr_path = by_modality["TR"][key]
        name = parse_name(td_path, "TD")
        td_backup_path = _backup_path(dataset_root, td_path, "TD")
        tr_backup_path = _backup_path(dataset_root, tr_path, "TR")
        if not td_backup_path.is_file() or not tr_backup_path.is_file():
            raise FileNotFoundError(f"missing backup_original pair for {name.recording_id}")

        td = _public_payload(td_path, "TD")
        tr = _public_payload(tr_path, "TR")
        td_backup, td_head = _backup_payload(td_backup_path)
        tr_backup, tr_head = _backup_payload(tr_backup_path)
        _assert_backup_equivalent(td, td_backup, td_head, path=td_backup_path)
        _assert_backup_equivalent(tr, tr_backup, tr_head, path=tr_backup_path)
        if td_head != tr_head:
            raise ValueError(f"{name.recording_id}: TD/TR File_head values differ")
        for field in PAIR_FIELDS:
            if not _values_equal(td[field], tr[field]):
                raise ValueError(f"{name.recording_id}: TD/TR field {field} differs")

        frame_count = np.asarray(td["DATA_time"]).size
        dpl = np.asarray(td["DPL"])
        if dpl.ndim != 2 or dpl.shape[0] != frame_count:
            raise ValueError(f"{name.recording_id}: invalid DPL shape {dpl.shape}")
        for field, value in tr.items():
            if field != "nDaCf" and np.asarray(value).size != frame_count:
                raise ValueError(
                    f"{name.recording_id}: TR {field} length differs from frame count"
                )
        if int(round(td_head.get("nSaveDplLen", -1))) != dpl.shape[1]:
            raise ValueError(
                f"{name.recording_id}: DPL width differs from File_head.nSaveDplLen"
            )
        recordings.append(
            Recording(
                name=name,
                td_path=td_path,
                tr_path=tr_path,
                td_backup_path=td_backup_path,
                tr_backup_path=tr_backup_path,
                td=td,
                tr=tr,
                file_head=td_head,
                td_payload_sha256=_payload_sha256(td),
                tr_payload_sha256=_payload_sha256(tr),
            )
        )
    return recordings


def _file_head_date(recording: Recording) -> str:
    head = recording.file_head
    try:
        return (
            f"{int(round(head['year'])):04d}{int(round(head['month'])):02d}"
            f"{int(round(head['day'])):02d}"
        )
    except KeyError as exc:
        raise ValueError(f"{recording.recording_id}: incomplete File_head date") from exc


def _shared_frame_pairs(recordings: list[Recording]) -> list[dict[str, Any]]:
    owners: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, recording in enumerate(recordings):
        for frame_key in recording.frame_keys:
            owners[frame_key].append(index)
    counts: Counter[tuple[int, int]] = Counter()
    for indices in owners.values():
        for pair in itertools.combinations(sorted(set(indices)), 2):
            counts[pair] += 1
    return [
        {
            "left_index": left,
            "right_index": right,
            "recording_a": recordings[left].recording_id,
            "recording_b": recordings[right].recording_id,
            "shared_frame_count": count,
            "same_filename_timestamp": (
                recordings[left].name.timestamp == recordings[right].name.timestamp
            ),
            "cross_category": (
                recordings[left].name.category != recordings[right].name.category
            ),
        }
        for (left, right), count in sorted(counts.items())
    ]


def _duplicate_groups(recordings: list[Recording]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Recording]] = defaultdict(list)
    for recording in recordings:
        groups[
            (recording.td_payload_sha256, recording.tr_payload_sha256)
        ].append(recording)
    rows: list[dict[str, Any]] = []
    duplicate_number = 0
    for (td_sha, tr_sha), members in sorted(groups.items()):
        if len(members) < 2:
            continue
        duplicate_number += 1
        group_id = f"duplicate_{duplicate_number:03d}"
        for member in sorted(members, key=lambda item: item.recording_id):
            rows.append(
                {
                    "duplicate_group": group_id,
                    "recording_id": member.recording_id,
                    "td_payload_sha256": td_sha,
                    "tr_payload_sha256": tr_sha,
                }
            )
    return rows


def _session_membership(
    recordings: list[Recording],
    shared_pairs: list[dict[str, Any]],
    duplicate_rows: list[dict[str, Any]],
) -> tuple[dict[str, str], list[dict[str, Any]], int]:
    edges: dict[tuple[int, int], set[str]] = defaultdict(set)

    def add_group_edges(values: Iterable[tuple[str, int]], reason: str) -> None:
        grouped: dict[str, list[int]] = defaultdict(list)
        for value, index in values:
            grouped[value].append(index)
        for indices in grouped.values():
            if len(indices) < 2:
                continue
            anchor = min(indices)
            for other in sorted(indices):
                if other != anchor:
                    edges[(anchor, other)].add(reason)

    add_group_edges(
        ((recording.name.timestamp, index) for index, recording in enumerate(recordings)),
        "same_filename_timestamp",
    )
    add_group_edges(
        (
            (_file_head_date(recording) + recording.name.timestamp[8:], index)
            for index, recording in enumerate(recordings)
        ),
        "same_file_head_date_and_filename_time",
    )
    for row in shared_pairs:
        pair = tuple(sorted((int(row["left_index"]), int(row["right_index"]))))
        edges[pair].add("shared_internal_frames")

    by_id = {recording.recording_id: index for index, recording in enumerate(recordings)}
    duplicate_members: dict[str, list[int]] = defaultdict(list)
    for row in duplicate_rows:
        duplicate_members[str(row["duplicate_group"])].append(by_id[str(row["recording_id"])])
    for indices in duplicate_members.values():
        anchor = min(indices)
        for other in sorted(indices):
            if other != anchor:
                edges[(anchor, other)].add("exact_td_tr_duplicate")

    union_find = UnionFind(len(recordings))
    for left, right in edges:
        union_find.union(left, right)
    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(recordings)):
        components[union_find.find(index)].append(index)
    ordered_components = sorted(
        components.values(),
        key=lambda members: min(recordings[index].recording_id for index in members),
    )
    group_by_index: dict[int, str] = {}
    for number, members in enumerate(ordered_components, start=1):
        for index in members:
            group_by_index[index] = f"source_session_{number:03d}"

    rows: list[dict[str, Any]] = []
    for index, recording in enumerate(recordings):
        reasons: set[str] = set()
        for (left, right), edge_reasons in edges.items():
            if index in {left, right}:
                reasons.update(edge_reasons)
        group_id = group_by_index[index]
        group_size = sum(value == group_id for value in group_by_index.values())
        rows.append(
            {
                "source_session_group": group_id,
                "recording_id": recording.recording_id,
                "group_size": group_size,
                "link_reasons": ";".join(sorted(reasons)),
            }
        )
    return (
        {recordings[index].recording_id: group for index, group in group_by_index.items()},
        rows,
        len(ordered_components),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def _validate_output_dir(output_dir: Path, dataset_root: Path, overwrite: bool) -> None:
    if output_dir in {PROJECT_ROOT, dataset_root}:
        raise ValueError("output directory must be separate from project and dataset roots")
    try:
        output_dir.relative_to(dataset_root)
    except ValueError:
        pass
    else:
        raise ValueError("output directory must not be inside the protected dataset root")
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
    if not isinstance(summary, dict):
        raise RuntimeError("generated summary.json must contain a JSON object")


def _remove_generated_output_dir(output_dir: Path) -> None:
    for entry in _managed_output_entries(output_dir):
        entry.unlink()
    output_dir.rmdir()


def _unfinished_transactions(output_dir: Path) -> list[Path]:
    return sorted(output_dir.parent.glob(f".{output_dir.name}.transaction-*"))


def _report(summary: dict[str, Any]) -> str:
    width_512 = summary["doppler_width_counts"].get("512", 0)
    width_1024 = summary["doppler_width_counts"].get("1024", 0)
    return f"""# LSS-DAUR-1.0 V3 Data Audit

- Status: `{summary['status']}`
- Official files: `{summary['official_file_count']}`
- Logical recordings: `{summary['logical_recording_count']}`
- Candidate source-session groups: `{summary['candidate_source_session_group_count']}`

## What passed

- Release integrity status: `{summary['all_manifest_file_hashes_match']}`.
- The {summary['mat_file_count']} MAT files are
  {summary['logical_recording_count']} logical recordings, each represented by canonical TD,
  canonical TR, TD `backup_original`, and TR `backup_original` files.
- Canonical files are MATLAB v5 and backups are MATLAB v7.3/HDF5. All
  {summary['canonical_backup_equivalent_pair_count']} canonical/backup pairs are
  numerically equivalent; backups additionally retain `File_head`.
- All {summary['td_tr_pair_count']} TD/TR pairs have identical `DATA_time`,
  `GPS_time_in_data`, `Iframecnt`,
  and `nDaCf`, with finite numeric arrays and matching frame counts.

## Leakage and grouping decision

- Random MAT, frame, window, TD/TR, or canonical/backup splitting is forbidden.
- There are {summary['filename_timestamp_group_count']} filename start-time groups,
  {summary['shared_frame_record_pair_count']} recording pairs sharing exact internal
  `(GPS_time_in_data, Iframecnt)` frames, and
  {summary['exact_duplicate_recording_group_count']} exact duplicate recording group containing
  {summary['exact_duplicate_recording_count']} recording IDs. Thus 77 logical recording IDs
  contain only {summary['unique_signal_trajectory_content_count']} unique TD/TR content pairs.
- The conservative minimum split key is `source_session_group`. It joins equal filename
  times, equal `File_head`-date plus filename-clock candidates, shared internal frames,
  and exact TD/TR duplicates, yielding
  {summary['candidate_source_session_group_count']} candidate groups.
- A stricter confirmatory evaluation should group whole acquisition days and connect both
  dates for every filename/`File_head` conflict.

## Remaining blockers

- {summary['filename_file_head_date_conflict_count']} recordings disagree between the
  filename date and `File_head` date. The local audit output `date_conflicts.csv` records
  the details and is intentionally omitted from the share package; the audit does not
  silently choose one date source for all records.
- TD contains {width_512} recordings with 512 Doppler bins and {width_1024} with 1024 bins.
  The bundled plotting script
  hard-codes 512 bins, so it is not a valid all-release loader and the 1024-bin physical
  velocity mapping remains unresolved.
- Every recording has repeated, non-decreasing `DATA_time`/`GPS_time_in_data` values and
  unequal frame intervals. There are {summary['duplicate_time_step_count']} repeated adjacent
  positions, leaving {summary['unique_time_position_count']} unique time positions; 13 tracks
  also have non-contiguous frame counters. These fields cannot be treated as uniformly sampled
  pulse time.
- Bird and UAV candidate sessions have zero overlap under the filename, File_head-time, and
  conservative connected definitions. Moreover, {summary['header_date_scene_class_pure_group_count']}
  of {summary['header_date_scene_group_count']} date/configuration scene groups are class-pure,
  so acquisition metadata and Doppler width can become label shortcuts.
- The TR `V` field is constant zero and must not be used as a feature.

## Claim boundary

DAUR can support a future preregistered, source-session-grouped study of Doppler-waterfall
and trajectory features after the date/session and 1024-bin axis questions are resolved.
It contains no H/V paired polarization and no balloon-payload labels. It currently cannot
support polarimetric claims, physical micro-Doppler axis claims for all records, random-frame
performance, unseen-session claims, or balloon recognition. No model was trained by this audit.
"""


def _audit_dataset_to_directory(
    *,
    dataset_root: Path,
    output_dir: Path,
    strict_release: bool = True,
) -> dict[str, Any]:
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {dataset_root}")
    if not output_dir.is_dir() or output_dir.is_symlink() or any(output_dir.iterdir()):
        raise ValueError("staging output directory must be an empty regular directory")

    if strict_release:
        release = _verify_release_manifest(dataset_root)
    else:
        release = {
            "official_file_count": sum(1 for path in dataset_root.rglob("*") if path.is_file()),
            "official_size_bytes": sum(
                path.stat().st_size for path in dataset_root.rglob("*") if path.is_file()
            ),
            "manifest_sha256": "not_checked_synthetic_or_partial_release",
            "all_manifest_file_hashes_match": None,
        }

    recordings = _load_recordings(dataset_root)
    if strict_release:
        category_counts = Counter(recording.name.category for recording in recordings)
        if dict(sorted(category_counts.items())) != EXPECTED_CATEGORY_COUNTS:
            raise ValueError(f"unexpected category counts: {dict(category_counts)}")
        if len(recordings) != 77:
            raise ValueError(f"expected 77 logical recordings, found {len(recordings)}")

    weather_dates = _weather_dates(dataset_root)
    shared_pairs = _shared_frame_pairs(recordings)
    duplicate_rows = _duplicate_groups(recordings)
    session_by_recording, session_rows, session_count = _session_membership(
        recordings, shared_pairs, duplicate_rows
    )
    duplicate_by_recording = {
        str(row["recording_id"]): str(row["duplicate_group"])
        for row in duplicate_rows
    }

    recording_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    width_counts: Counter[int] = Counter()
    total_frames = 0
    data_time_duplicate_total = 0
    gps_time_duplicate_total = 0
    dataframe_non_decreasing_count = 0
    dataframe_strict_count = 0
    gps_non_decreasing_count = 0
    iframe_non_decreasing_count = 0
    n_dacf_values: Counter[int] = Counter()
    head_n_dpl_len_values: Counter[int] = Counter()
    doppler_config_counts: Counter[tuple[int, int, int]] = Counter()
    doppler_config_classes: dict[tuple[int, int, int], Counter[str]] = defaultdict(Counter)
    config_signatures: list[tuple[tuple[str, float], ...]] = []
    position_signatures: list[tuple[float, ...]] = []
    positive_time_steps: list[float] = []
    total_doppler_values = 0
    duplicate_time_dpl_equal_rows = 0
    duplicate_time_tr_equal_rows = 0
    frame_counter_gap_events = 0
    frame_counter_missing_values = 0
    frame_counter_repeat_events = 0
    tracks_with_noncontiguous_frame_counter = 0
    range_min = float("inf")
    range_max = float("-inf")
    v_min = float("inf")
    v_max = float("-inf")
    vm_min = float("inf")
    vm_max = float("-inf")

    for recording in recordings:
        data_time = np.asarray(recording.td["DATA_time"]).reshape(-1)
        gps_time = np.asarray(recording.td["GPS_time_in_data"]).reshape(-1)
        iframe = np.asarray(recording.td["Iframecnt"]).reshape(-1)
        dpl = np.asarray(recording.td["DPL"])
        frame_count = len(data_time)
        data_diffs = np.diff(data_time)
        gps_diffs = np.diff(gps_time)
        iframe_diffs = np.diff(iframe.astype(np.int64))
        data_duplicates = frame_count - int(np.unique(data_time).size)
        gps_duplicates = frame_count - int(np.unique(gps_time).size)
        iframe_duplicates = frame_count - int(np.unique(iframe).size)
        filename_date = recording.name.timestamp[:8]
        head_date = _file_head_date(recording)
        date_match = filename_date == head_date
        head_timestamp = head_date + recording.name.timestamp[8:]
        positive_time_steps.extend(data_diffs[data_diffs > 0].astype(float).tolist())
        duplicate_indices = np.flatnonzero(data_diffs == 0) + 1
        duplicate_time_dpl_equal_rows += sum(
            np.array_equal(dpl[index], dpl[index - 1]) for index in duplicate_indices
        )
        duplicate_time_tr_equal_rows += sum(
            all(
                np.asarray(recording.tr[field]).reshape(-1)[index]
                == np.asarray(recording.tr[field]).reshape(-1)[index - 1]
                for field in TR_SIGNAL_FIELDS
            )
            for index in duplicate_indices
        )
        gap_events = int(np.count_nonzero(iframe_diffs > 1))
        missing_values = int(np.sum(iframe_diffs[iframe_diffs > 1] - 1))
        repeat_events = int(np.count_nonzero(iframe_diffs == 0))
        frame_counter_gap_events += gap_events
        frame_counter_missing_values += missing_values
        frame_counter_repeat_events += repeat_events
        tracks_with_noncontiguous_frame_counter += int(
            gap_events > 0 or repeat_events > 0
        )

        ranges = np.asarray(recording.tr["R"], dtype=np.float64).reshape(-1)
        velocities = np.asarray(recording.tr["V"], dtype=np.float64).reshape(-1)
        measured_velocities = np.asarray(
            recording.tr["V_m"], dtype=np.float64
        ).reshape(-1)
        range_min = min(range_min, float(ranges.min()))
        range_max = max(range_max, float(ranges.max()))
        v_min = min(v_min, float(velocities.min()))
        v_max = max(v_max, float(velocities.max()))
        vm_min = min(vm_min, float(measured_velocities.min()))
        vm_max = max(vm_max, float(measured_velocities.max()))

        n_dpl_len = int(round(recording.file_head["nDPLLen"]))
        n_save_dpl_len = int(round(recording.file_head["nSaveDplLen"]))
        doppler_config = (n_dpl_len, n_save_dpl_len, int(dpl.shape[1]))
        doppler_config_counts[doppler_config] += 1
        doppler_config_classes[doppler_config][recording.name.category] += 1
        config_signatures.append(
            tuple(
                (key, float(value))
                for key, value in sorted(recording.file_head.items())
                if key not in CONFIG_EXCLUDED_FIELDS
            )
        )
        position_signatures.append(
            tuple(
                float(recording.file_head[field])
                for field in RADAR_POSITION_FIELDS
                if field in recording.file_head
            )
        )

        width_counts[dpl.shape[1]] += 1
        total_frames += frame_count
        total_doppler_values += int(dpl.size)
        data_time_duplicate_total += data_duplicates
        gps_time_duplicate_total += gps_duplicates
        dataframe_non_decreasing_count += int(np.all(data_diffs >= 0))
        dataframe_strict_count += int(np.all(data_diffs > 0))
        gps_non_decreasing_count += int(np.all(gps_diffs >= 0))
        iframe_non_decreasing_count += int(np.all(iframe_diffs >= 0))
        n_dacf_values[int(np.asarray(recording.td["nDaCf"]).reshape(-1)[0])] += 1
        head_n_dpl_len_values[int(round(recording.file_head["nDPLLen"]))] += 1

        row = {
            "recording_id": recording.recording_id,
            "source_session_group": session_by_recording[recording.recording_id],
            "duplicate_group": duplicate_by_recording.get(recording.recording_id, ""),
            "category": recording.name.category,
            "serial": recording.name.serial,
            "batch": recording.name.batch,
            "filename_timestamp": recording.name.timestamp,
            "file_head_date": head_date,
            "file_head_timestamp_candidate": head_timestamp,
            "filename_file_head_date_match": date_match,
            "filename_date_in_weather_table": filename_date in weather_dates,
            "file_head_date_in_weather_table": head_date in weather_dates,
            "frame_count": frame_count,
            "doppler_bin_count": dpl.shape[1],
            "nDaCf": int(np.asarray(recording.td["nDaCf"]).reshape(-1)[0]),
            "file_head_nDPLLen": int(round(recording.file_head["nDPLLen"])),
            "file_head_nSaveDplLen": int(round(recording.file_head["nSaveDplLen"])),
            "data_time_duplicate_count": data_duplicates,
            "gps_time_duplicate_count": gps_duplicates,
            "iframe_duplicate_count": iframe_duplicates,
            "frame_counter_gap_event_count": gap_events,
            "frame_counter_missing_value_count": missing_values,
            "frame_counter_repeat_event_count": repeat_events,
            "data_time_non_decreasing": bool(np.all(data_diffs >= 0)),
            "data_time_strictly_increasing": bool(np.all(data_diffs > 0)),
            "gps_time_non_decreasing": bool(np.all(gps_diffs >= 0)),
            "iframe_non_decreasing": bool(np.all(iframe_diffs >= 0)),
            "canonical_backup_numeric_equivalent": True,
            "td_tr_shared_fields_equal": True,
            "td_relative_path": recording.td_path.relative_to(dataset_root).as_posix(),
            "tr_relative_path": recording.tr_path.relative_to(dataset_root).as_posix(),
        }
        recording_rows.append(row)
        if not date_match:
            conflict_rows.append(
                {
                    "recording_id": recording.recording_id,
                    "filename_date": filename_date,
                    "file_head_date": head_date,
                    "filename_date_in_weather_table": filename_date in weather_dates,
                    "file_head_date_in_weather_table": head_date in weather_dates,
                    "resolution": "unresolved",
                }
            )

    config_id_by_signature = {
        signature: f"CFG{index:02d}"
        for index, signature in enumerate(sorted(set(config_signatures)), start=1)
    }
    position_id_by_signature = {
        signature: f"POS{index:02d}"
        for index, signature in enumerate(sorted(set(position_signatures)), start=1)
    }
    for row, config_signature, position_signature in zip(
        recording_rows, config_signatures, position_signatures, strict=True
    ):
        row["header_configuration_group"] = config_id_by_signature[config_signature]
        row["radar_position_group"] = position_id_by_signature[position_signature]
        row["header_date_scene_group"] = (
            f"{row['file_head_date']}|{row['header_configuration_group']}"
        )

    grouped_sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_scenes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in recording_rows:
        grouped_sessions[str(row["source_session_group"])].append(row)
        grouped_scenes[str(row["header_date_scene_group"])].append(row)
    source_session_group_rows = [
        {
            "source_session_group": group_id,
            "recording_count": len(rows),
            "class_count": len({str(row["category"]) for row in rows}),
            "classes": ";".join(sorted({str(row["category"]) for row in rows})),
            "filename_date_count": len(
                {str(row["filename_timestamp"])[:8] for row in rows}
            ),
            "file_head_date_count": len({str(row["file_head_date"]) for row in rows}),
            "candidate_only_not_publisher_confirmed": True,
        }
        for group_id, rows in sorted(grouped_sessions.items())
    ]

    class_rows: list[dict[str, Any]] = []
    for category in sorted({recording.name.category for recording in recordings}):
        rows = [row for row in recording_rows if row["category"] == category]
        class_rows.append(
            {
                "category": category,
                "logical_recording_count": len(rows),
                "frame_count": sum(int(row["frame_count"]) for row in rows),
                "source_session_candidate_count": len(
                    {str(row["source_session_group"]) for row in rows}
                ),
                "doppler_512_recording_count": sum(
                    int(row["doppler_bin_count"]) == 512 for row in rows
                ),
                "doppler_1024_recording_count": sum(
                    int(row["doppler_bin_count"]) == 1024 for row in rows
                ),
                "filename_file_head_date_conflict_count": sum(
                    not bool(row["filename_file_head_date_match"]) for row in rows
                ),
            }
        )
    doppler_config_rows = [
        {
            "file_head_nDPLLen": config[0],
            "file_head_nSaveDplLen": config[1],
            "saved_doppler_bin_count": config[2],
            "logical_recording_count": count,
            "class_distribution": ";".join(
                f"{category}:{category_count}"
                for category, category_count in sorted(
                    doppler_config_classes[config].items()
                )
            ),
            "official_512_plotter_directly_applicable": config[2] == 512,
            "physical_axis_status": (
                "PARTIAL_SCRIPT_DEFINED"
                if config[2] == 512
                else "BLOCKED_UNDOCUMENTED_1024"
            ),
        }
        for config, count in sorted(doppler_config_counts.items())
    ]

    filename_timestamp_count = len({recording.name.timestamp for recording in recordings})
    head_timestamp_count = len(
        {
            _file_head_date(recording) + recording.name.timestamp[8:]
            for recording in recordings
        }
    )
    duplicate_group_count = len(
        {row["duplicate_group"] for row in duplicate_rows}
    )
    uav_categories = {"Fixed-wing UAV", "Rotary drone"}
    bird_filename_sessions = {
        recording.name.timestamp
        for recording in recordings
        if recording.name.category == "Bird"
    }
    uav_filename_sessions = {
        recording.name.timestamp
        for recording in recordings
        if recording.name.category in uav_categories
    }
    bird_header_sessions = {
        _file_head_date(recording) + recording.name.timestamp[8:]
        for recording in recordings
        if recording.name.category == "Bird"
    }
    uav_header_sessions = {
        _file_head_date(recording) + recording.name.timestamp[8:]
        for recording in recordings
        if recording.name.category in uav_categories
    }
    bird_connected_sessions = {
        str(row["source_session_group"])
        for row in recording_rows
        if row["category"] == "Bird"
    }
    uav_connected_sessions = {
        str(row["source_session_group"])
        for row in recording_rows
        if row["category"] in uav_categories
    }
    scene_class_pure_count = sum(
        len({str(row["category"]) for row in rows}) == 1
        for rows in grouped_scenes.values()
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "dataset_id": "lss_daur_1_0",
        "release_version": "V3",
        "data_doi": "10.57760/sciencedb.radars.00076",
        "status": AUDIT_STATUS,
        "gates": {
            "release_identity": "PASS",
            "schema_and_finite": "PASS",
            "td_tr_pairing": "PASS",
            "canonical_backup_equivalence": "PASS",
            "strict_time": "FAIL_DUPLICATE_TIMESTAMPS",
            "absolute_date_weather_join": "BLOCKED_DATE_CONFLICT",
            "session_identity": "BLOCKED_NO_AUTHORITATIVE_SESSION_KEY",
            "physical_axis_512": "PARTIAL_SCRIPT_DEFINED",
            "physical_axis_1024": "BLOCKED_UNDOCUMENTED_WIDTH",
            "model_training": "BLOCKED",
        },
        **release,
        "mat_file_count": len(recordings) * 4,
        "logical_recording_count": len(recordings),
        "paired_track_count": len(recordings),
        "unique_signal_trajectory_content_count": (
            len(recordings) - len(duplicate_rows) + duplicate_group_count
        ),
        "canonical_td_file_count": len(recordings),
        "canonical_tr_file_count": len(recordings),
        "backup_td_file_count": len(recordings),
        "backup_tr_file_count": len(recordings),
        "canonical_mat_file_count": len(recordings) * 2,
        "backup_mat_file_count": len(recordings) * 2,
        "td_tr_pair_count": len(recordings),
        "canonical_backup_equivalent_pair_count": len(recordings) * 2,
        "canonical_format": "MATLAB v5",
        "backup_format": "MATLAB v7.3/HDF5 with File_head",
        "all_numeric_arrays_finite": True,
        "category_recording_counts": dict(
            sorted(Counter(recording.name.category for recording in recordings).items())
        ),
        "total_frame_count": total_frames,
        "frame_count": total_frames,
        "doppler_value_count": total_doppler_values,
        "doppler_width_counts": {
            str(key): value for key, value in sorted(width_counts.items())
        },
        "doppler_512_track_count": width_counts[512],
        "doppler_1024_track_count": width_counts[1024],
        "nDaCf_counts": {str(key): value for key, value in sorted(n_dacf_values.items())},
        "file_head_nDPLLen_counts": {
            str(key): value for key, value in sorted(head_n_dpl_len_values.items())
        },
        "filename_timestamp_group_count": filename_timestamp_count,
        "file_head_timestamp_candidate_group_count": head_timestamp_count,
        "filename_session_candidate_count": filename_timestamp_count,
        "header_date_session_candidate_count": head_timestamp_count,
        "header_configuration_group_count": len(config_id_by_signature),
        "radar_position_group_count": len(position_id_by_signature),
        "header_date_scene_group_count": len(grouped_scenes),
        "header_date_scene_class_pure_group_count": scene_class_pure_count,
        "bird_uav_filename_session_overlap_count": len(
            bird_filename_sessions & uav_filename_sessions
        ),
        "bird_uav_header_date_session_overlap_count": len(
            bird_header_sessions & uav_header_sessions
        ),
        "bird_uav_connected_session_overlap_count": len(
            bird_connected_sessions & uav_connected_sessions
        ),
        "weather_record_date_count": len(weather_dates),
        "filename_file_head_date_conflict_count": len(conflict_rows),
        "filename_header_date_mismatch_count": len(conflict_rows),
        "shared_frame_record_pair_count": len(shared_pairs),
        "max_shared_frame_count": max(
            (int(row["shared_frame_count"]) for row in shared_pairs), default=0
        ),
        "exact_duplicate_recording_group_count": duplicate_group_count,
        "exact_duplicate_recording_count": len(duplicate_rows),
        "candidate_source_session_group_count": session_count,
        "data_time_duplicate_count": data_time_duplicate_total,
        "gps_time_duplicate_count": gps_time_duplicate_total,
        "duplicate_time_step_count": data_time_duplicate_total,
        "unique_time_position_count": total_frames - data_time_duplicate_total,
        "tracks_with_duplicate_timestamps": sum(
            int(row["data_time_duplicate_count"]) > 0 for row in recording_rows
        ),
        "duplicate_time_dpl_equal_row_count": duplicate_time_dpl_equal_rows,
        "duplicate_time_tr_equal_row_count": duplicate_time_tr_equal_rows,
        "positive_time_step_min_seconds": float(min(positive_time_steps)),
        "positive_time_step_median_seconds": float(np.median(positive_time_steps)),
        "positive_time_step_p95_seconds": float(
            np.quantile(positive_time_steps, 0.95)
        ),
        "positive_time_step_max_seconds": float(max(positive_time_steps)),
        "tracks_with_noncontiguous_frame_counter": (
            tracks_with_noncontiguous_frame_counter
        ),
        "frame_counter_gap_event_count": frame_counter_gap_events,
        "frame_counter_missing_value_count": frame_counter_missing_values,
        "frame_counter_repeat_event_count": frame_counter_repeat_events,
        "records_with_non_decreasing_data_time": dataframe_non_decreasing_count,
        "records_with_strictly_increasing_data_time": dataframe_strict_count,
        "records_with_non_decreasing_gps_time": gps_non_decreasing_count,
        "records_with_non_decreasing_iframe": iframe_non_decreasing_count,
        "carrier_frequency_mhz": 1360,
        "official_plot_pri_seconds": 0.0002,
        "official_plot_prf_hz": 5000.0,
        "official_plot_velocity_resolution_mps": 0.1346,
        "header_mtd_length": 4096,
        "conventional_velocity_resolution_from_fc_prf_mtd_length_mps": (
            (3.0e8 / 1.36e9) * 5000.0 / (2.0 * 4096.0)
        ),
        "range_min_km": range_min,
        "range_max_km": range_max,
        "tr_units_verified_from_official_guide": True,
        "v_min": v_min,
        "v_max": v_max,
        "v_m_min": vm_min,
        "v_m_max": vm_max,
        "v_field_constant_zero": v_min == 0.0 and v_max == 0.0,
        "hard_coded_512_plotter_covers_all_recordings": set(width_counts) == {512},
        "random_mat_split_allowed": False,
        "random_frame_or_window_split_allowed": False,
        "td_tr_split_allowed": False,
        "canonical_backup_as_extra_samples_allowed": False,
        "canonical_backup_observation_count_multiplier_allowed": False,
        "authoritative_session_key_available": False,
        "absolute_weather_join_allowed": False,
        "raw_adc_or_iq_available": False,
        "h_v_polarimetry_available": False,
        "physical_micro_doppler_hz_allowed": False,
        "model_training_allowed": False,
        "source_files_modified": False,
        "sample_level_outputs_included": False,
        "minimum_split_unit": "conservative candidate source_session_group",
        "minimum_future_split_unit": (
            "unresolved conservative connected session/scene group; canonical/backup "
            "and TD/TR views of each observation must remain together"
        ),
        "confirmatory_split_unit": (
            "acquisition_day with filename/File_head conflict dates connected"
        ),
        "blockers": [
            "six filename/File_head date conflicts are unresolved",
            "source-session identities are conservative candidates, not publisher-confirmed sessions",
            "19 recordings have 1024 Doppler bins while the bundled plotter assumes 512",
            "repeated and nonuniform frame timestamps are not a uniformly sampled pulse-time axis",
        ],
        "allowed_use": (
            "read-only loader development and a future preregistered source-session-grouped "
            "Doppler-waterfall/trajectory study after blockers are resolved"
        ),
        "prohibited_claims": [
            "random-frame, random-window, or random-MAT performance",
            "canonical and backup files as independent samples",
            "unseen-session or deployment generalization",
            "physical velocity or micro-Doppler axes for all recordings",
            "H/V polarimetry or balloon-payload recognition",
        ],
    }

    if strict_release:
        frozen_expected = {
            "mat_file_count": 308,
            "logical_recording_count": 77,
            "paired_track_count": 77,
            "unique_signal_trajectory_content_count": 76,
            "frame_count": 11366,
            "doppler_value_count": 7728640,
            "doppler_512_track_count": 58,
            "doppler_1024_track_count": 19,
            "duplicate_time_step_count": 894,
            "unique_time_position_count": 10472,
            "tracks_with_duplicate_timestamps": 77,
            "duplicate_time_dpl_equal_row_count": 13,
            "duplicate_time_tr_equal_row_count": 894,
            "tracks_with_noncontiguous_frame_counter": 13,
            "frame_counter_gap_event_count": 85,
            "frame_counter_missing_value_count": 94,
            "frame_counter_repeat_event_count": 2,
            "filename_header_date_mismatch_count": 6,
            "filename_session_candidate_count": 45,
            "header_date_session_candidate_count": 40,
            "candidate_source_session_group_count": 39,
            "header_configuration_group_count": 10,
            "radar_position_group_count": 5,
            "header_date_scene_group_count": 24,
            "header_date_scene_class_pure_group_count": 20,
            "bird_uav_filename_session_overlap_count": 0,
            "bird_uav_header_date_session_overlap_count": 0,
            "bird_uav_connected_session_overlap_count": 0,
            "shared_frame_record_pair_count": 11,
            "exact_duplicate_recording_group_count": 1,
            "exact_duplicate_recording_count": 2,
            "v_field_constant_zero": True,
        }
        for field, expected in frozen_expected.items():
            if summary.get(field) != expected:
                raise ValueError(
                    f"unexpected frozen DAUR statistic {field}: "
                    f"{summary.get(field)!r} != {expected!r}"
                )
        expected_doppler_configs = Counter(
            {(512, 512, 512): 39, (2048, 512, 512): 19, (2048, 1024, 1024): 19}
        )
        if doppler_config_counts != expected_doppler_configs:
            raise ValueError(
                f"unexpected frozen DAUR Doppler configurations: {doppler_config_counts}"
            )

    _write_csv(output_dir / "recording_audit.csv", recording_rows, list(recording_rows[0]))
    _write_csv(
        output_dir / "source_session_group_audit.csv",
        source_session_group_rows,
        list(source_session_group_rows[0]),
    )
    _write_csv(output_dir / "class_summary.csv", class_rows, list(class_rows[0]))
    _write_csv(
        output_dir / "doppler_config_audit.csv",
        doppler_config_rows,
        list(doppler_config_rows[0]),
    )
    _write_csv(
        output_dir / "date_conflicts.csv",
        conflict_rows,
        [
            "recording_id",
            "filename_date",
            "file_head_date",
            "filename_date_in_weather_table",
            "file_head_date_in_weather_table",
            "resolution",
        ],
    )
    shared_output = [
        {key: value for key, value in row.items() if key not in {"left_index", "right_index"}}
        for row in shared_pairs
    ]
    _write_csv(
        output_dir / "shared_frame_pairs.csv",
        shared_output,
        [
            "recording_a",
            "recording_b",
            "shared_frame_count",
            "same_filename_timestamp",
            "cross_category",
        ],
    )
    _write_csv(
        output_dir / "duplicate_recordings.csv",
        duplicate_rows,
        [
            "duplicate_group",
            "recording_id",
            "td_payload_sha256",
            "tr_payload_sha256",
        ],
    )
    _write_csv(
        output_dir / "source_session_membership.csv",
        session_rows,
        ["source_session_group", "recording_id", "group_size", "link_reasons"],
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "REPORT.md").write_text(_report(summary), encoding="utf-8")
    _validate_complete_output(output_dir)
    return summary


def _audit_dataset_transaction(
    *,
    dataset_root: Path,
    output_dir: Path,
    overwrite: bool = False,
    strict_release: bool = True,
) -> dict[str, Any]:
    _validate_output_dir(output_dir, dataset_root, overwrite)
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
        summary = _audit_dataset_to_directory(
            dataset_root=dataset_root,
            output_dir=new_output,
            strict_release=strict_release,
        )
        _validate_complete_output(new_output)

        _validate_output_dir(output_dir, dataset_root, overwrite)
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
    dataset_root: Path,
    output_dir: Path,
    overwrite: bool = False,
    strict_release: bool = True,
) -> dict[str, Any]:
    dataset_root = resolve_path(dataset_root)
    output_input = output_dir.expanduser()
    if not output_input.is_absolute():
        output_input = PROJECT_ROOT / output_input
    if output_input.is_symlink():
        raise ValueError("output directory must not be a symlink")
    output_dir = output_input.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {dataset_root}")

    _validate_output_dir(output_dir, dataset_root, overwrite)
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
        return _audit_dataset_transaction(
            dataset_root=dataset_root,
            output_dir=output_dir,
            overwrite=overwrite,
            strict_release=strict_release,
        )
    finally:
        os.close(lock_fd)


def main() -> int:
    args = parse_args()
    summary = audit_dataset(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
