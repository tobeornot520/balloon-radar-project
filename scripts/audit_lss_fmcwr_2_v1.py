#!/usr/bin/env python3
"""Read-only archive and MAT-schema audit for LSS-FMCWR-2.0 ScienceDB V4."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import numpy as np
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data/raw/external/LSS-FMCWR-2.0"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/data_audit/lss_fmcwr_2_v1"
AUDIT_STATUS = (
    "PASS_ARCHIVE_SCHEMA_BLOCKED_GROUPING_PROVENANCE_AND_PHYSICAL_AXIS"
)


@dataclass(frozen=True)
class ExpectedArchive:
    size: int
    sha256: str
    target_id: str
    target_label: str
    mat_count: int
    directory_count: int
    uncompressed_size: int
    packed_size: int
    k_count: int
    l_count: int
    maximum_mat_size: int


EXPECTED_ARCHIVES = {
    "大疆M350（01）.rar": ExpectedArchive(
        112_631_521,
        "bb9b29768946b58ae2e8db0fa6551431070c3cd387ee66c651accec7e4624c14",
        "m350",
        "DJI M350",
        18,
        6,
        115_016_401,
        112_627_436,
        13,
        5,
        12_274_580,
    ),
    "大疆悟2（02）.rar": ExpectedArchive(
        113_324_845,
        "ba4c425ad24debc3ac0bcf39a0f0cbf3a650f53296828f21d8aa8288b435259f",
        "inspire2",
        "DJI Inspire 2",
        18,
        6,
        115_758_443,
        113_320_758,
        13,
        5,
        13_489_517,
    ),
    "大疆御2（03）.rar": ExpectedArchive(
        116_063_219,
        "442ab97e64b546656e9d52fb490fde76707bbcb4f4b34cf54a70c8bbec2d26b7",
        "mavic2",
        "DJI Mavic 2",
        18,
        6,
        118_646_630,
        116_059_180,
        13,
        5,
        14_478_156,
    ),
    "六旋翼无人机（04）.rar": ExpectedArchive(
        117_851_524,
        "a8f3e6c84c4c7b14913cf7e604933751d21963d5c2a549186a22ccfbe999cc4e",
        "hexacopter",
        "Hexacopter",
        18,
        6,
        120_032_063,
        117_847_141,
        13,
        5,
        14_702_015,
    ),
    "仿真飞鸟（05）.rar": ExpectedArchive(
        354_700_802,
        "d182e7a12b2e49f311c1b8f67c3a24c72ed4453b73673e80f2dec443d2114bf6",
        "simulated_bird",
        "Simulated bird",
        8,
        1,
        362_275_641,
        354_699_028,
        8,
        0,
        218_499_208,
    ),
    "直升机AC311（06）.rar": ExpectedArchive(
        198_884_718,
        "ae07dc4a7131f6483fce1ce1e9404e7444e1556a7223087a9c716d52488d27b7",
        "ac311",
        "AC311 helicopter",
        10,
        1,
        209_577_963,
        198_882_574,
        4,
        6,
        52_730_788,
    ),
}
EXPECTED_RAR_TOTAL_SIZE = 1_013_456_629
EXPECTED_ENTRY_COUNT = 116
EXPECTED_MAT_COUNT = 90
EXPECTED_DIRECTORY_COUNT = 26
EXPECTED_UNCOMPRESSED_MAT_SIZE = 1_041_307_141
EXPECTED_PACKED_MAT_SIZE = 1_013_436_117
EXPECTED_BAND_COUNTS = {"K": 64, "L": 26}
EXPECTED_RECORDING_STEM_COUNT = 66
EXPECTED_CANDIDATE_GROUP_COUNT = 48
EXPECTED_PATH_FILENAME_ANGLE_CONFLICT_COUNT = 1
EXPECTED_UNIQUE_RAW_MAT_COUNT = 71
EXPECTED_UNIQUE_NUMERIC_PAYLOAD_COUNT = 71
EXPECTED_DUPLICATE_GROUP_COUNT = 11
EXPECTED_DUPLICATE_MEMBER_COUNT = 30
EXPECTED_TARGET_RECORDING_STEM_COUNTS = {
    "m350": 15,
    "inspire2": 15,
    "mavic2": 15,
    "hexacopter": 15,
    "simulated_bird": 2,
    "ac311": 4,
}
EXPECTED_TARGET_CANDIDATE_GROUP_COUNTS = {
    "m350": 10,
    "inspire2": 12,
    "mavic2": 10,
    "hexacopter": 10,
    "simulated_bird": 2,
    "ac311": 4,
}

OUTPUT_FILES = {
    "REPORT.md",
    "summary.json",
    "archive_audit.csv",
    "mat_audit.csv",
    "group_summary.csv",
    "duplicate_groups.csv",
}
ORDINAL_PATTERN = re.compile(r"\((?P<ordinal>[1-9][0-9]*)\)")
NUMBER_PATTERN = re.compile(r"[0-9]+(?:\.[0-9]+)?")
BAND_SUFFIX_PATTERN = re.compile(r"-(?P<band>[KL])$")


@dataclass(frozen=True)
class RarEntry:
    name: str
    kind: str
    size: int | None
    packed_size: int | None
    crc32: str | None


@dataclass(frozen=True)
class ParsedMemberName:
    target_label: str
    band: str
    ordinal: int
    recording_stem: str
    path_angle: int | None
    filename_angle: int | None
    collection_angle: int
    angle_consistent: bool | None
    duration_token: str


@dataclass(frozen=True)
class MatRecord:
    archive_name: str
    target_id: str
    target_label: str
    member_path: str
    band: str
    ordinal: int
    recording_stem: str
    path_angle: int | None
    filename_angle: int | None
    collection_angle: int
    angle_consistent: bool | None
    duration_token: str
    member_size: int
    packed_size: int
    crc32: str
    raw_sha256: str
    numeric_payload_sha256: str
    channel_a_shape: tuple[int, int]
    channel_a_dtype: str
    channel_a_complex: bool
    channel_b_shape: tuple[int, int]
    channel_b_dtype: str
    channel_b_complex: bool


class Runner(Protocol):
    def __call__(self, args: list[str]) -> subprocess.CompletedProcess[bytes]: ...


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit LSS-FMCWR-2.0 V4 RAR files without extracting them."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--unrar")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_unrar(explicit: str | Path | None = None) -> str:
    if explicit is not None:
        value = str(explicit)
        if os.sep in value or (os.altsep is not None and os.altsep in value):
            candidate = Path(value).expanduser().resolve()
            if not candidate.is_file() or not os.access(candidate, os.X_OK):
                raise FileNotFoundError(f"unrar executable is unavailable: {candidate}")
            return str(candidate)
        found = shutil.which(value)
        if found is None:
            raise FileNotFoundError(f"unrar executable is unavailable: {value}")
        return found
    for name in ("unrar", "unrar-nonfree"):
        found = shutil.which(name)
        if found is not None:
            return found
    raise FileNotFoundError("unrar or unrar-nonfree was not found on PATH; use --unrar")


def _default_runner(args: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        shell=False,
    )


def _run(
    runner: Runner,
    args: list[str],
    *,
    purpose: str,
    stdout_must_be_empty: bool = False,
) -> bytes:
    try:
        result = runner(args)
    except OSError as error:
        raise RuntimeError(f"{purpose}: failed to start unrar") from error
    if not isinstance(result.stdout, bytes) or not isinstance(result.stderr, bytes):
        raise TypeError(f"{purpose}: runner stdout/stderr must be bytes")
    if result.returncode != 0:
        raise RuntimeError(
            f"{purpose}: unrar exited with status {result.returncode}; "
            f"stderr={result.stderr.decode('utf-8', errors='replace')!r}"
        )
    if result.stderr:
        raise RuntimeError(f"{purpose}: unexpected unrar stderr output")
    if stdout_must_be_empty and result.stdout:
        raise RuntimeError(f"{purpose}: unexpected unrar stdout output")
    return result.stdout


def parse_unrar_lt(payload: bytes, *, archive_argument: str) -> list[RarEntry]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("unrar lt output is not valid UTF-8") from error
    if "\x00" in text or "\r" in text:
        raise ValueError("unrar lt output contains forbidden control characters")
    blocks = re.split(r"\n[ \t]*\n", text.strip())
    if len(blocks) < 3:
        raise ValueError("unrar lt output is incomplete")
    if not re.fullmatch(r"UNRAR [0-9.]+ freeware.*", blocks[0]):
        raise ValueError("unrar lt output has an unexpected banner")
    header = blocks[1].splitlines()
    if header != [f"Archive: {archive_argument}", "Details: RAR 5"]:
        raise ValueError("RAR must be a non-solid, non-volume, unencrypted RAR 5 archive")

    entries: list[RarEntry] = []
    for block in blocks[2:]:
        fields: dict[str, str] = {}
        for raw_line in block.splitlines():
            line = raw_line.lstrip(" ")
            key, separator, value = line.partition(": ")
            if not separator or not key or value != value.strip():
                raise ValueError("unrar lt entry contains an unparseable field")
            if key in fields:
                raise ValueError(f"unrar lt entry repeats field {key!r}")
            fields[key] = value
        kind = fields.get("Type")
        if kind == "File":
            expected_fields = {
                "Name",
                "Type",
                "Size",
                "Packed size",
                "Ratio",
                "mtime",
                "Attributes",
                "CRC32",
                "Host OS",
                "Compression",
            }
            if set(fields) != expected_fields:
                raise ValueError("RAR file entry has unknown or missing technical fields")
            if not fields["Size"].isdigit() or not fields["Packed size"].isdigit():
                raise ValueError("RAR file entry has an invalid size")
            if re.fullmatch(r"[0-9A-F]{8}", fields["CRC32"]) is None:
                raise ValueError("RAR file entry has an invalid CRC32")
            if not fields["Compression"].startswith("RAR 5.0("):
                raise ValueError("RAR file entry does not use RAR 5 compression")
            entries.append(
                RarEntry(
                    name=fields["Name"],
                    kind="file",
                    size=int(fields["Size"]),
                    packed_size=int(fields["Packed size"]),
                    crc32=fields["CRC32"],
                )
            )
        elif kind == "Directory":
            expected_fields = {
                "Name",
                "Type",
                "mtime",
                "Attributes",
                "Host OS",
                "Compression",
            }
            if set(fields) != expected_fields:
                raise ValueError(
                    "RAR directory entry has unknown or missing technical fields"
                )
            entries.append(RarEntry(fields["Name"], "directory", None, None, None))
        else:
            raise ValueError(f"RAR entry type is forbidden: {kind!r}")
    if not entries:
        raise ValueError("RAR archive contains no entries")
    return entries


def _validate_inventory(entries: list[RarEntry], *, archive_name: str) -> str:
    names = [entry.name for entry in entries]
    if len(names) != len(set(names)):
        raise ValueError(f"{archive_name}: duplicate RAR member path")
    normalized: set[str] = set()
    roots: set[str] = set()
    for entry in entries:
        name = entry.name
        if not name or "\\" in name or "\x00" in name:
            raise ValueError(f"{archive_name}: unsafe RAR member path {name!r}")
        path = PurePosixPath(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != name
            or not path.parts
            or any(":" in part for part in path.parts)
            or any(part.startswith("-") for part in path.parts)
            or any(ord(character) < 32 for character in name)
        ):
            raise ValueError(f"{archive_name}: unsafe RAR member path {name!r}")
        folded = unicodedata.normalize("NFC", name).casefold()
        if folded in normalized:
            raise ValueError(f"{archive_name}: Unicode/case-folded path collision")
        normalized.add(folded)
        roots.add(path.parts[0])
        if entry.kind == "file" and path.suffix.lower() != ".mat":
            raise ValueError(f"{archive_name}: non-MAT file is forbidden: {name}")
    if len(roots) != 1:
        raise ValueError(f"{archive_name}: entries must have exactly one target root")
    root = next(iter(roots))
    if root != Path(archive_name).stem:
        raise ValueError(f"{archive_name}: member root does not match archive name")
    return root


def parse_member_name(member_path: str) -> ParsedMemberName:
    path = PurePosixPath(member_path)
    if len(path.parts) not in {2, 3} or path.suffix != ".mat":
        raise ValueError(f"unexpected FMCWR MAT path: {member_path}")
    path_angle: int | None = None
    if len(path.parts) == 3:
        if re.fullmatch(r"[0-9]+", path.parts[1]) is None:
            raise ValueError(f"invalid collection-angle directory: {member_path}")
        path_angle = int(path.parts[1])

    ordinals = list(ORDINAL_PATTERN.finditer(path.stem))
    if len(ordinals) != 1:
        raise ValueError(f"expected one recording ordinal in {member_path}")
    ordinal = int(ordinals[0].group("ordinal"))
    normalized_stem = ORDINAL_PATTERN.sub("", path.stem).strip()
    normalized_stem = re.sub(r"[ \t]+", " ", normalized_stem)
    normalized_stem = re.sub(r"[ \t]*-[ \t]*", "-", normalized_stem)
    band_match = BAND_SUFFIX_PATTERN.search(normalized_stem)
    if band_match is None:
        raise ValueError(f"cannot parse terminal K/L band in {member_path}")
    band = band_match.group("band")
    body = normalized_stem[: band_match.start()]
    tokens = body.split("-")
    if len(tokens) < 4 or any(token == "" for token in tokens):
        raise ValueError(f"cannot parse suffix tokens in {member_path}")
    duration_token = tokens[-3]
    if NUMBER_PATTERN.fullmatch(duration_token) is None:
        raise ValueError(f"invalid duration token in {member_path}")
    if any(NUMBER_PATTERN.fullmatch(token) is None for token in tokens[-3:]):
        raise ValueError(f"invalid numeric suffix tokens in {member_path}")
    filename_angle = (
        int(tokens[-4]) if re.fullmatch(r"[0-9]+", tokens[-4]) is not None else None
    )
    collection_angle = path_angle if path_angle is not None else filename_angle
    if collection_angle is None:
        raise ValueError(f"collection angle is unavailable in {member_path}")
    angle_consistent = (
        None
        if path_angle is None or filename_angle is None
        else path_angle == filename_angle
    )
    normalized_file = normalized_stem + path.suffix
    recording_stem = str(path.with_name(normalized_file))
    return ParsedMemberName(
        target_label=path.parts[0],
        band=band,
        ordinal=ordinal,
        recording_stem=recording_stem,
        path_angle=path_angle,
        filename_angle=filename_angle,
        collection_angle=collection_angle,
        angle_consistent=angle_consistent,
        duration_token=duration_token,
    )


def _numeric_payload_sha256(arrays: list[tuple[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for label, value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(label.encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def load_echoes(payload: bytes, *, member_path: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        loaded = loadmat(BytesIO(payload), squeeze_me=False, struct_as_record=True)
    except Exception as error:
        raise ValueError(f"{member_path}: unreadable MATLAB file") from error
    public = {key: value for key, value in loaded.items() if not key.startswith("__")}
    if set(public) != {"echoes"}:
        raise ValueError(f"{member_path}: expected only public field echoes")
    echoes = np.asarray(public["echoes"])
    if echoes.shape != (1, 1) or echoes.dtype.names != ("channelA", "channelB"):
        raise ValueError(f"{member_path}: echoes must be a channelA/channelB scalar struct")
    channels: list[np.ndarray] = []
    for field in ("channelA", "channelB"):
        channel = np.asarray(echoes[field][0, 0])
        if channel.ndim != 2 or not np.issubdtype(channel.dtype, np.number):
            raise ValueError(f"{member_path}: {field} must be a numeric 2-D array")
        if not np.isfinite(channel).all():
            raise ValueError(f"{member_path}: {field} contains NaN/Inf")
        channels.append(channel)
    if channels[0].size == 0:
        raise ValueError(f"{member_path}: channelA must not be empty")
    return channels[0], channels[1]


def _discover_archives(dataset_root: Path) -> list[Path]:
    if dataset_root.is_symlink() or not dataset_root.is_dir():
        raise ValueError("dataset root must be a regular directory, not a symlink")
    archives = sorted(
        (path for path in dataset_root.iterdir() if path.suffix.lower() == ".rar"),
        key=lambda path: path.name,
    )
    if not archives:
        raise ValueError("dataset root contains no RAR archives")
    for archive in archives:
        if archive.is_symlink() or not archive.is_file():
            raise ValueError(f"RAR source must be a regular file: {archive}")
    return archives


def _archive_snapshots(
    archives: list[Path], *, strict_release: bool
) -> dict[str, tuple[int, int, str]]:
    if strict_release and {path.name for path in archives} != set(EXPECTED_ARCHIVES):
        raise ValueError("strict V4 audit requires exactly the six frozen RAR files")
    if strict_release:
        for archive in archives:
            expected = EXPECTED_ARCHIVES[archive.name]
            if archive.stat().st_size != expected.size:
                raise ValueError(f"unexpected strict V4 size for {archive.name}")
        if sum(path.stat().st_size for path in archives) != EXPECTED_RAR_TOTAL_SIZE:
            raise ValueError("unexpected strict V4 total RAR size")

    snapshots: dict[str, tuple[int, int, str]] = {}
    for archive in archives:
        stat_result = archive.stat()
        sha256 = sha256_file(archive)
        if strict_release and sha256 != EXPECTED_ARCHIVES[archive.name].sha256:
            raise ValueError(f"unexpected strict V4 SHA256 for {archive.name}")
        snapshots[archive.name] = (
            stat_result.st_size,
            stat_result.st_mtime_ns,
            sha256,
        )
    return snapshots


def _verify_sources_unchanged(
    archives: list[Path], snapshots: dict[str, tuple[int, int, str]]
) -> None:
    for archive in archives:
        expected_size, expected_mtime_ns, expected_sha256 = snapshots[archive.name]
        current = archive.stat()
        if (
            current.st_size != expected_size
            or current.st_mtime_ns != expected_mtime_ns
            or sha256_file(archive) != expected_sha256
        ):
            raise RuntimeError(f"source RAR changed during audit: {archive.name}")


def _strict_inventory_check(archive_name: str, entries: list[RarEntry]) -> None:
    expected = EXPECTED_ARCHIVES[archive_name]
    files = [entry for entry in entries if entry.kind == "file"]
    directories = [entry for entry in entries if entry.kind == "directory"]
    parsed = [parse_member_name(entry.name) for entry in files]
    actual = {
        "mat_count": len(files),
        "directory_count": len(directories),
        "uncompressed_size": sum(entry.size or 0 for entry in files),
        "packed_size": sum(entry.packed_size or 0 for entry in files),
        "k_count": sum(item.band == "K" for item in parsed),
        "l_count": sum(item.band == "L" for item in parsed),
        "maximum_mat_size": max(entry.size or 0 for entry in files),
    }
    frozen = {
        "mat_count": expected.mat_count,
        "directory_count": expected.directory_count,
        "uncompressed_size": expected.uncompressed_size,
        "packed_size": expected.packed_size,
        "k_count": expected.k_count,
        "l_count": expected.l_count,
        "maximum_mat_size": expected.maximum_mat_size,
    }
    if actual != frozen:
        raise ValueError(f"unexpected frozen RAR inventory for {archive_name}")


def _read_inventories(
    *,
    archives: list[Path],
    executable: str,
    runner: Runner,
    strict_release: bool,
) -> tuple[dict[str, list[RarEntry]], dict[str, str]]:
    inventories: dict[str, list[RarEntry]] = {}
    target_roots: dict[str, str] = {}
    for archive in archives:
        archive_argument = str(archive)
        payload = _run(
            runner,
            [executable, "lt", "-c-", "-p-", archive_argument],
            purpose=f"list {archive.name}",
        )
        entries = parse_unrar_lt(payload, archive_argument=archive_argument)
        target_root = _validate_inventory(entries, archive_name=archive.name)
        if strict_release:
            _strict_inventory_check(archive.name, entries)
        inventories[archive.name] = entries
        target_roots[archive.name] = target_root

    if strict_release:
        all_entries = [entry for entries in inventories.values() for entry in entries]
        all_files = [entry for entry in all_entries if entry.kind == "file"]
        if len(all_entries) != EXPECTED_ENTRY_COUNT:
            raise ValueError("unexpected strict V4 RAR entry count")
        if len(all_files) != EXPECTED_MAT_COUNT:
            raise ValueError("unexpected strict V4 MAT count")
        if len(all_entries) - len(all_files) != EXPECTED_DIRECTORY_COUNT:
            raise ValueError("unexpected strict V4 directory count")
    return inventories, target_roots


def _test_archives(
    *, archives: list[Path], executable: str, runner: Runner
) -> None:
    for archive in archives:
        _run(
            runner,
            [executable, "t", "-idq", "-p-", str(archive)],
            purpose=f"integrity test {archive.name}",
            stdout_must_be_empty=True,
        )


def _extract_mat_records(
    *,
    archives: list[Path],
    inventories: dict[str, list[RarEntry]],
    target_roots: dict[str, str],
    executable: str,
    runner: Runner,
    strict_release: bool,
) -> list[MatRecord]:
    records: list[MatRecord] = []
    for archive in archives:
        expected = EXPECTED_ARCHIVES.get(archive.name)
        target_id = (
            expected.target_id
            if strict_release and expected
            else target_roots[archive.name]
        )
        target_label = (
            expected.target_label if strict_release and expected else target_roots[archive.name]
        )
        files = sorted(
            (
                entry
                for entry in inventories[archive.name]
                if entry.kind == "file"
            ),
            key=lambda entry: entry.name,
        )
        for entry in files:
            if entry.size is None or entry.packed_size is None or entry.crc32 is None:
                raise RuntimeError("file inventory entry is incomplete")
            parsed = parse_member_name(entry.name)
            if parsed.target_label != target_roots[archive.name]:
                raise ValueError(f"target root mismatch in {entry.name}")
            payload = _run(
                runner,
                [
                    executable,
                    "p",
                    "-inul",
                    "-p-",
                    str(archive),
                    entry.name,
                ],
                purpose=f"stream {archive.name}:{entry.name}",
            )
            if len(payload) != entry.size:
                raise ValueError(
                    f"{archive.name}:{entry.name}: streamed size differs from inventory"
                )
            channel_a, channel_b = load_echoes(payload, member_path=entry.name)
            if strict_release:
                if parsed.band == "K" and not np.iscomplexobj(channel_a):
                    raise ValueError(f"{entry.name}: strict K channelA must be complex")
                if parsed.band == "L" and np.iscomplexobj(channel_a):
                    raise ValueError(f"{entry.name}: strict L channelA must be real")
            records.append(
                MatRecord(
                    archive_name=archive.name,
                    target_id=target_id,
                    target_label=target_label,
                    member_path=entry.name,
                    band=parsed.band,
                    ordinal=parsed.ordinal,
                    recording_stem=parsed.recording_stem,
                    path_angle=parsed.path_angle,
                    filename_angle=parsed.filename_angle,
                    collection_angle=parsed.collection_angle,
                    angle_consistent=parsed.angle_consistent,
                    duration_token=parsed.duration_token,
                    member_size=entry.size,
                    packed_size=entry.packed_size,
                    crc32=entry.crc32,
                    raw_sha256=hashlib.sha256(payload).hexdigest(),
                    numeric_payload_sha256=_numeric_payload_sha256(
                        [("channelA", channel_a), ("channelB", channel_b)]
                    ),
                    channel_a_shape=(int(channel_a.shape[0]), int(channel_a.shape[1])),
                    channel_a_dtype=channel_a.dtype.str,
                    channel_a_complex=bool(np.iscomplexobj(channel_a)),
                    channel_b_shape=(int(channel_b.shape[0]), int(channel_b.shape[1])),
                    channel_b_dtype=channel_b.dtype.str,
                    channel_b_complex=bool(np.iscomplexobj(channel_b)),
                )
            )
    return records


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _candidate_groups(records: list[MatRecord]) -> tuple[dict[int, str], list[list[int]]]:
    disjoint = _DisjointSet(len(records))
    by_stem: dict[tuple[str, str], list[int]] = defaultdict(list)
    by_payload: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_stem[(record.archive_name, record.recording_stem)].append(index)
        by_payload[record.numeric_payload_sha256].append(index)
    for groups in (by_stem.values(), by_payload.values()):
        for members in groups:
            for index in members[1:]:
                disjoint.union(members[0], index)
    components_by_root: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        components_by_root[disjoint.find(index)].append(index)
    components = sorted(
        components_by_root.values(),
        key=lambda members: min(
            (records[index].archive_name, records[index].member_path)
            for index in members
        ),
    )
    identifiers: dict[int, str] = {}
    for number, members in enumerate(components, start=1):
        identifier = f"candidate_group_{number:04d}"
        for index in members:
            identifiers[index] = identifier
    return identifiers, components


def _duplicate_rows(records: list[MatRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for hash_type, field in (
        ("raw_mat_sha256", "raw_sha256"),
        ("numeric_payload_sha256", "numeric_payload_sha256"),
    ):
        groups: dict[str, list[MatRecord]] = defaultdict(list)
        for record in records:
            groups[getattr(record, field)].append(record)
        for digest, members in sorted(groups.items()):
            if len(members) < 2:
                continue
            rows.append(
                {
                    "hash_type": hash_type,
                    "sha256": digest,
                    "member_count": len(members),
                    "target_count": len({item.target_id for item in members}),
                    "archive_count": len({item.archive_name for item in members}),
                    "bands": "|".join(sorted({item.band for item in members})),
                    "members_json": json.dumps(
                        [
                            {
                                "archive": item.archive_name,
                                "member": item.member_path,
                            }
                            for item in sorted(
                                members,
                                key=lambda item: (item.archive_name, item.member_path),
                            )
                        ],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
            )
    return rows


def _shape_text(shape: tuple[int, int]) -> str:
    return f"{shape[0]}x{shape[1]}"


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _hash_group_statistics(
    records: list[MatRecord], field: str
) -> tuple[int, int, int, int]:
    groups: dict[str, list[MatRecord]] = defaultdict(list)
    for record in records:
        groups[getattr(record, field)].append(record)
    duplicates = [members for members in groups.values() if len(members) > 1]
    cross_target = sum(
        len({record.target_id for record in members}) > 1 for members in duplicates
    )
    return (
        len(groups),
        len(duplicates),
        sum(len(members) for members in duplicates),
        cross_target,
    )


def _strict_summary_check(summary: dict[str, Any]) -> None:
    expected_scalars = {
        "archive_count": 6,
        "archive_total_size_bytes": EXPECTED_RAR_TOTAL_SIZE,
        "rar_entry_count": EXPECTED_ENTRY_COUNT,
        "mat_file_count": EXPECTED_MAT_COUNT,
        "directory_count": EXPECTED_DIRECTORY_COUNT,
        "uncompressed_mat_size_bytes": EXPECTED_UNCOMPRESSED_MAT_SIZE,
        "packed_mat_size_bytes": EXPECTED_PACKED_MAT_SIZE,
        "recording_stem_count": EXPECTED_RECORDING_STEM_COUNT,
        "candidate_group_count": EXPECTED_CANDIDATE_GROUP_COUNT,
        "path_filename_angle_conflict_count": (
            EXPECTED_PATH_FILENAME_ANGLE_CONFLICT_COUNT
        ),
        "unique_raw_mat_count": EXPECTED_UNIQUE_RAW_MAT_COUNT,
        "unique_numeric_payload_count": EXPECTED_UNIQUE_NUMERIC_PAYLOAD_COUNT,
        "raw_duplicate_group_count": EXPECTED_DUPLICATE_GROUP_COUNT,
        "raw_duplicate_member_count": EXPECTED_DUPLICATE_MEMBER_COUNT,
        "numeric_duplicate_group_count": EXPECTED_DUPLICATE_GROUP_COUNT,
        "numeric_duplicate_member_count": EXPECTED_DUPLICATE_MEMBER_COUNT,
        "raw_cross_target_duplicate_group_count": 0,
        "numeric_cross_target_duplicate_group_count": 0,
    }
    for field, expected in expected_scalars.items():
        if summary.get(field) != expected:
            raise ValueError(
                f"unexpected strict V4 statistic {field}: "
                f"{summary.get(field)!r} != {expected!r}"
            )
    expected_maps = {
        "band_counts": EXPECTED_BAND_COUNTS,
        "target_recording_stem_counts": EXPECTED_TARGET_RECORDING_STEM_COUNTS,
        "target_candidate_group_counts": EXPECTED_TARGET_CANDIDATE_GROUP_COUNTS,
    }
    for field, expected in expected_maps.items():
        if summary.get(field) != expected:
            raise ValueError(f"unexpected strict V4 mapping {field}")


def _report(summary: dict[str, Any]) -> str:
    if summary["release_identity_verified"]:
        release_result = f"""- All six frozen ScienceDB V4 RAR identities passed.
- The RAR inventories contain {summary['mat_file_count']} MAT files and
  {summary['directory_count']} directories; `unrar t` passed for every archive.
- K `channelA` arrays are complex and L `channelA` arrays are real in this release.
- The official headline/table and actual MAT count are 90, while page 2 prose sums to
  86. The naming text also omits the documented 4.096 ms setting. Both conflicts remain
  explicit rather than being silently reconciled.
- Collection angles describe acquisition/radar included angle, not verified target
  aspect."""
        boundary = f"""- {summary['raw_duplicate_group_count']} exact raw and
  {summary['numeric_duplicate_group_count']} decoded duplicate groups are present.
- Removing ordinals gives {summary['recording_stem_count']} candidate stems; joining
  stem and duplicate edges gives {summary['candidate_group_count']} conservative groups.
  Neither is an authoritative recording/session identity.
- The bird archive contains simulated bird data only, not natural-bird evidence.
- No H/V polarimetry, global sampling rate, carrier/velocity axis, complete `md_stft`,
  or independent source/session key is available."""
    else:
        release_result = """- Schema-fixture RAR inventory, integrity-command behavior,
  MAT structure, finite values, duplicate accounting, and grouping arithmetic passed.
- V4 identity, official targets, band physics, collection-angle meaning, and published
  metadata were not evaluated."""
        boundary = """- Fixture target, recording, and candidate-group labels are not
  source/session identities.
- Raw-IQ, H/V, physical-axis, simulated/natural-bird, and deployment claims were not
  evaluated and cannot be inferred from this fixture."""
    return f"""# LSS-FMCWR-2.0 Read-Only RAR/MAT Audit

Status: `{summary['status']}`
Audit mode: `{summary['audit_mode']}`
RAR files: `{summary['archive_count']}`
MAT files: `{summary['mat_file_count']}`

## Passed

{release_result}

## Blockers and claim boundary

{boundary}
- Random MAT/frame/window splitting and model training remain prohibited.

The source RAR files were tested and streamed in memory. They were not extracted to
disk or modified. Detailed member paths and hashes are local-only audit evidence.
"""


def _audit_to_directory(
    *,
    dataset_root: Path,
    archives: list[Path],
    snapshots: dict[str, tuple[int, int, str]],
    output_dir: Path,
    executable: str,
    runner: Runner,
    strict_release: bool,
) -> dict[str, Any]:
    if not output_dir.is_dir() or output_dir.is_symlink() or any(output_dir.iterdir()):
        raise ValueError("staging output directory must be an empty regular directory")
    inventories, target_roots = _read_inventories(
        archives=archives,
        executable=executable,
        runner=runner,
        strict_release=strict_release,
    )
    _test_archives(archives=archives, executable=executable, runner=runner)
    records = _extract_mat_records(
        archives=archives,
        inventories=inventories,
        target_roots=target_roots,
        executable=executable,
        runner=runner,
        strict_release=strict_release,
    )
    _verify_sources_unchanged(archives, snapshots)

    candidate_ids, components = _candidate_groups(records)
    raw_unique, raw_groups, raw_members, raw_cross_target = _hash_group_statistics(
        records, "raw_sha256"
    )
    (
        numeric_unique,
        numeric_groups,
        numeric_members,
        numeric_cross_target,
    ) = _hash_group_statistics(records, "numeric_payload_sha256")

    target_stems: dict[str, set[str]] = defaultdict(set)
    target_candidate_groups: dict[str, set[str]] = defaultdict(set)
    for index, record in enumerate(records):
        target_stems[record.target_id].add(record.recording_stem)
        target_candidate_groups[record.target_id].add(candidate_ids[index])
    component_cross_target_count = sum(
        len({records[index].target_id for index in component}) > 1
        for component in components
    )

    archive_rows: list[dict[str, Any]] = []
    for archive in archives:
        entries = inventories[archive.name]
        files = [entry for entry in entries if entry.kind == "file"]
        expected = EXPECTED_ARCHIVES.get(archive.name)
        archive_records = [
            record for record in records if record.archive_name == archive.name
        ]
        archive_rows.append(
            {
                "archive_name": archive.name,
                "target_id": (
                    expected.target_id
                    if strict_release and expected
                    else target_roots[archive.name]
                ),
                "physical_size_bytes": snapshots[archive.name][0],
                "sha256": snapshots[archive.name][2],
                "entry_count": len(entries),
                "mat_file_count": len(files),
                "directory_count": len(entries) - len(files),
                "uncompressed_mat_size_bytes": sum(
                    entry.size or 0 for entry in files
                ),
                "packed_mat_size_bytes": sum(
                    entry.packed_size or 0 for entry in files
                ),
                "k_count": sum(record.band == "K" for record in archive_records),
                "l_count": sum(record.band == "L" for record in archive_records),
                "rar5_non_solid_non_volume_unencrypted": True,
                "integrity_test_passed": True,
                "release_identity_verified": strict_release,
            }
        )

    mat_rows = []
    for index, record in enumerate(records):
        mat_rows.append(
            {
                "archive_name": record.archive_name,
                "target_id": record.target_id,
                "target_label": record.target_label,
                "member_path": record.member_path,
                "band_token": record.band,
                "ordinal": record.ordinal,
                "recording_stem": record.recording_stem,
                "candidate_group_id": candidate_ids[index],
                "path_angle": record.path_angle,
                "filename_angle": record.filename_angle,
                "collection_angle": record.collection_angle,
                "angle_consistent": record.angle_consistent,
                "duration_token": record.duration_token,
                "member_size": record.member_size,
                "packed_size": record.packed_size,
                "crc32": record.crc32,
                "raw_sha256": record.raw_sha256,
                "numeric_payload_sha256": record.numeric_payload_sha256,
                "channelA_shape": _shape_text(record.channel_a_shape),
                "channelA_dtype": record.channel_a_dtype,
                "channelA_complex": record.channel_a_complex,
                "channelA_finite": True,
                "channelB_shape": _shape_text(record.channel_b_shape),
                "channelB_dtype": record.channel_b_dtype,
                "channelB_complex": record.channel_b_complex,
                "channelB_finite": True,
            }
        )

    grouped: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[
            (
                record.archive_name,
                record.target_id,
                record.band,
                record.collection_angle,
                record.duration_token,
                record.channel_a_shape,
                record.channel_b_shape,
            )
        ].append(index)
    group_rows = []
    for key, indices in sorted(grouped.items(), key=lambda item: repr(item[0])):
        (
            archive_name,
            target_id,
            band,
            collection_angle,
            duration_token,
            channel_a_shape,
            channel_b_shape,
        ) = key
        members = [records[index] for index in indices]
        group_rows.append(
            {
                "archive_name": archive_name,
                "target_id": target_id,
                "band_token": band,
                "collection_angle": collection_angle,
                "duration_token": duration_token,
                "channelA_shape": _shape_text(channel_a_shape),
                "channelB_shape": _shape_text(channel_b_shape),
                "mat_count": len(members),
                "unique_raw_mat_count": len(
                    {record.raw_sha256 for record in members}
                ),
                "unique_numeric_payload_count": len(
                    {record.numeric_payload_sha256 for record in members}
                ),
                "candidate_group_count": len(
                    {candidate_ids[index] for index in indices}
                ),
            }
        )

    all_entries = [entry for entries in inventories.values() for entry in entries]
    band_counts = Counter(record.band for record in records)
    shape_counts = Counter(
        (
            record.band,
            _shape_text(record.channel_a_shape),
            record.channel_a_dtype,
            record.channel_a_complex,
            _shape_text(record.channel_b_shape),
            record.channel_b_dtype,
            record.channel_b_complex,
        )
        for record in records
    )
    strict_facts = strict_release
    summary: dict[str, Any] = {
        "schema_version": 1,
        "dataset_id": "lss_fmcwr_2_0",
        "target_release_version": "V4",
        "release_version": "V4" if strict_facts else None,
        "target_data_doi": "10.57760/sciencedb.radars.00054",
        "data_doi": "10.57760/sciencedb.radars.00054" if strict_facts else None,
        "status": AUDIT_STATUS,
        "audit_mode": "strict_release" if strict_facts else "schema_fixture",
        "release_identity_verified": strict_facts,
        "gates": {
            "release_identity": (
                "PASS" if strict_facts else "NOT_EVALUATED_TEST_FIXTURE"
            ),
            "rar_path_safety_and_inventory": "PASS",
            "rar_integrity": "PASS",
            "mat_schema_and_finite": "PASS",
            "exact_duplicate_isolation": (
                "BLOCKED_DUPLICATES_PRESENT" if raw_groups else "PASS"
            ),
            "recording_session_identity": "BLOCKED_NOT_AVAILABLE",
            "physical_time_doppler_velocity_axis": (
                "BLOCKED_NOT_AVAILABLE"
                if strict_facts
                else "NOT_EVALUATED_TEST_FIXTURE"
            ),
            "natural_bird_evidence": (
                "BLOCKED_SIMULATION_ONLY"
                if strict_facts
                else "NOT_EVALUATED_TEST_FIXTURE"
            ),
            "model_training": "BLOCKED",
        },
        "archive_count": len(archives),
        "archive_total_size_bytes": sum(item[0] for item in snapshots.values()),
        "archive_identities": {
            name: {"size": value[0], "sha256": value[2]}
            for name, value in sorted(snapshots.items())
        },
        "rar_entry_count": len(all_entries),
        "mat_file_count": len(records),
        "directory_count": sum(entry.kind == "directory" for entry in all_entries),
        "uncompressed_mat_size_bytes": sum(record.member_size for record in records),
        "packed_mat_size_bytes": sum(record.packed_size for record in records),
        "band_counts": {band: band_counts[band] for band in sorted(band_counts)},
        "duration_tokens": sorted({record.duration_token for record in records}),
        "collection_angles": sorted({record.collection_angle for record in records}),
        "path_filename_angle_conflict_count": sum(
            record.angle_consistent is False for record in records
        ),
        "recording_stem_count": sum(len(values) for values in target_stems.values()),
        "target_recording_stem_counts": {
            target: len(values) for target, values in sorted(target_stems.items())
        },
        "candidate_group_count": len(components),
        "target_candidate_group_counts": {
            target: len(values)
            for target, values in sorted(target_candidate_groups.items())
        },
        "candidate_group_cross_target_count": component_cross_target_count,
        "candidate_groups_authoritative": False,
        "independent_recording_or_session_key_available": False,
        "unique_raw_mat_count": raw_unique,
        "raw_duplicate_group_count": raw_groups,
        "raw_duplicate_member_count": raw_members,
        "raw_cross_target_duplicate_group_count": raw_cross_target,
        "unique_numeric_payload_count": numeric_unique,
        "numeric_duplicate_group_count": numeric_groups,
        "numeric_duplicate_member_count": numeric_members,
        "numeric_cross_target_duplicate_group_count": numeric_cross_target,
        "channel_shape_dtype_counts": [
            {
                "band_token": key[0],
                "channelA_shape": key[1],
                "channelA_dtype": key[2],
                "channelA_complex": key[3],
                "channelB_shape": key[4],
                "channelB_dtype": key[5],
                "channelB_complex": key[6],
                "mat_count": count,
            }
            for key, count in sorted(shape_counts.items(), key=lambda item: repr(item[0]))
        ],
        "echoes_public_fields": ["echoes"],
        "echoes_struct_fields": ["channelA", "channelB"],
        "k_channel_a_complex_verified": True if strict_facts else None,
        "l_channel_a_real_verified": True if strict_facts else None,
        "raw_complex_iq_available_for_all_records": False if strict_facts else None,
        "h_v_polarimetry_available": False if strict_facts else None,
        "collection_angle_semantics": (
            "acquisition/radar included angle; target aspect is not verified"
            if strict_facts
            else None
        ),
        "target_aspect_angle_verified": False if strict_facts else None,
        "simulated_bird_archive_is_simulation_only": True if strict_facts else None,
        "natural_bird_evidence_available": False if strict_facts else None,
        "global_sampling_rate_available": False if strict_facts else None,
        "global_carrier_frequency_available": False if strict_facts else None,
        "physical_doppler_hz_axis_available": False if strict_facts else None,
        "physical_velocity_axis_available": False if strict_facts else None,
        "complete_md_stft_implementation_available": False if strict_facts else None,
        "single_example_fs_hz": 500_000 if strict_facts else None,
        "single_example_parameter_scope": (
            "Br=100 MHz and 0.3 ms only; not global" if strict_facts else None
        ),
        "official_headline_mat_count": 90 if strict_facts else None,
        "official_table_mat_count_sum": 90 if strict_facts else None,
        "official_page2_prose_count_sum": 86 if strict_facts else None,
        "official_90_vs_86_metadata_conflict": True if strict_facts else None,
        "documentation_omits_4_096_ms_setting_conflict": (
            True if strict_facts else None
        ),
        "simulated_bird_must_not_be_claimed_as_natural_bird": True,
        "random_mat_frame_or_window_split_allowed": False,
        "model_training_allowed": False,
        "model_training_performed": False,
        "source_archives_extracted_to_disk": False,
        "source_archives_modified": False,
        "subprocess_shell_used": False,
        "raw_data_included_in_outputs": False,
        "member_level_outputs_local_only": True,
        "allowed_use": (
            "read-only V4 archive/schema/duplicate audit and conservative grouped "
            "method design"
            if strict_facts
            else "schema-fixture test execution only"
        ),
        "prohibited_claims": [
            "90 MAT files as 90 independent samples",
            "candidate groups as authoritative sessions",
            "collection angle as verified target aspect",
            "simulated bird as natural-bird evidence",
            "global Fs, Doppler-Hz, or velocity axes",
            "H/V polarimetry or balloon recognition",
            "model performance before a separate preregistered protocol",
        ],
    }
    if strict_release:
        _strict_summary_check(summary)

    duplicate_rows = _duplicate_rows(records)
    _write_csv(output_dir / "archive_audit.csv", archive_rows, list(archive_rows[0]))
    _write_csv(output_dir / "mat_audit.csv", mat_rows, list(mat_rows[0]))
    _write_csv(output_dir / "group_summary.csv", group_rows, list(group_rows[0]))
    duplicate_fields = [
        "hash_type",
        "sha256",
        "member_count",
        "target_count",
        "archive_count",
        "bands",
        "members_json",
    ]
    _write_csv(output_dir / "duplicate_groups.csv", duplicate_rows, duplicate_fields)
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


def _validate_output_dir(
    output_dir: Path, dataset_root: Path, *, overwrite: bool
) -> None:
    if output_dir in {PROJECT_ROOT, dataset_root}:
        raise ValueError("output directory must be separate from project and source roots")
    try:
        output_dir.relative_to(dataset_root)
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


def _audit_transaction(
    *,
    dataset_root: Path,
    archives: list[Path],
    snapshots: dict[str, tuple[int, int, str]],
    output_dir: Path,
    executable: str,
    runner: Runner,
    overwrite: bool,
    strict_release: bool,
) -> dict[str, Any]:
    _validate_output_dir(output_dir, dataset_root, overwrite=overwrite)
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
        summary = _audit_to_directory(
            dataset_root=dataset_root,
            archives=archives,
            snapshots=snapshots,
            output_dir=new_output,
            executable=executable,
            runner=runner,
            strict_release=strict_release,
        )
        _validate_complete_output(new_output)
        _validate_output_dir(output_dir, dataset_root, overwrite=overwrite)
        if _output_snapshot(output_dir) != original_snapshot:
            raise RuntimeError("output directory changed while audit was running")
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
                        "automatic rollback failed; previous output remains at "
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
                    f"transaction cleanup failed; inspect {transaction_root}: "
                    f"{cleanup_error}"
                )
        raise
    if old_output.exists():
        try:
            _remove_generated_output_dir(old_output)
        except BaseException as cleanup_error:
            raise RuntimeError(
                "new output committed but previous-output backup could not be removed; "
                f"inspect {old_output}"
            ) from cleanup_error
    transaction_root.rmdir()
    return summary


def audit_dataset(
    *,
    dataset_root: Path,
    output_dir: Path,
    unrar_path: str | Path | None = None,
    overwrite: bool = False,
    strict_release: bool = True,
    runner: Runner | None = None,
) -> dict[str, Any]:
    root_input = dataset_root.expanduser()
    if not root_input.is_absolute():
        root_input = PROJECT_ROOT / root_input
    if root_input.is_symlink():
        raise ValueError("dataset root must not be a symlink")
    dataset_root = root_input.resolve()
    archives = _discover_archives(dataset_root)
    snapshots = _archive_snapshots(archives, strict_release=strict_release)

    if runner is None:
        executable = resolve_unrar(unrar_path)
        runner = _default_runner
    else:
        executable = str(unrar_path) if unrar_path is not None else "unrar"

    output_input = output_dir.expanduser()
    if not output_input.is_absolute():
        output_input = PROJECT_ROOT / output_input
    if output_input.is_symlink():
        raise ValueError("output directory must not be a symlink")
    output_dir = output_input.resolve()
    _validate_output_dir(output_dir, dataset_root, overwrite=overwrite)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(output_dir.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        unfinished = _unfinished_transactions(output_dir)
        if unfinished:
            paths = ", ".join(str(path) for path in unfinished)
            raise RuntimeError(
                "unfinished audit transaction detected; inspect and recover or "
                f"remove it before rerunning: {paths}"
            )
        return _audit_transaction(
            dataset_root=dataset_root,
            archives=archives,
            snapshots=snapshots,
            output_dir=output_dir,
            executable=executable,
            runner=runner,
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
        unrar_path=args.unrar,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
