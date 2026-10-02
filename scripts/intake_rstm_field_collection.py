#!/usr/bin/env python3
"""Create a read-only manifest for the 2025-04-25 RSTM field collection.

The script does not decode the payload or infer H/V labels.  It reads a small
header from IQ files (and from the beginning of bz2 products), which keeps the
intake reproducible without copying or rewriting the raw collection.
"""
from __future__ import annotations

import argparse
import bz2
import csv
import hashlib
import re
import struct
from datetime import datetime, timezone
from pathlib import Path


FILENAME_TIME_RE = re.compile(r"(?:ST\d+_)?(\d{8})_(\d{6})_(\d{3})")
PRODUCT_TIME_RE = re.compile(r"ST\d+_(\d{14})")
CATEGORIES = {
    "target_six_wing": ("target", "six_wing_source_label"),
    "target_reflector": ("target", "reflector_source_label"),
    "target_small_satellite": ("target", "small_satellite_source_label"),
    "target_uav": ("target", "uav_source_label"),
    "target_balloon": ("target", "balloon_source_label"),
    "background_iq": ("background", "background_iq"),
    "background_products_bz2": ("background_product", "background_product"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hash", action="store_true", help="compute SHA-256 (slow for raw IQ)")
    return parser.parse_args()


def decode_fixed(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("utf-8", errors="replace").strip()


def read_header(path: Path) -> tuple[bytes, int | None]:
    if path.name.endswith(".bin.bz2"):
        with bz2.open(path, "rb") as handle:
            return handle.read(0x220), None
    with path.open("rb") as handle:
        return handle.read(0x220), None


def u32(data: bytes, offset: int) -> int | None:
    if len(data) < offset + 4:
        return None
    return struct.unpack_from("<I", data, offset)[0]


def f32(data: bytes, offset: int) -> float | None:
    if len(data) < offset + 4:
        return None
    return float(struct.unpack_from("<f", data, offset)[0])


def parse_filename_time(name: str) -> str:
    match = FILENAME_TIME_RE.search(name)
    if not match:
        product = PRODUCT_TIME_RE.search(name)
        if product:
            return datetime.strptime(product.group(1), "%Y%m%d%H%M%S").replace(
                tzinfo=timezone.utc
            ).isoformat()
        return ""
    date, clock, millis = match.groups()
    value = datetime.strptime(date + clock, "%Y%m%d%H%M%S").replace(
        microsecond=int(millis) * 1000, tzinfo=timezone.utc
    )
    return value.isoformat()


def header_time(data: bytes) -> str:
    stamp = u32(data, 0x14C)
    if stamp is None or stamp <= 0:
        return ""
    try:
        return datetime.fromtimestamp(stamp, tz=timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return ""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_one(path: Path, root: Path, do_hash: bool) -> dict[str, object]:
    relative = path.relative_to(root).as_posix()
    category_dir = path.relative_to(root).parts[0]
    role, source_label = CATEGORIES.get(category_dir, ("unknown", "unknown"))
    data, _ = read_header(path)
    suffix = ".bin.bz2" if path.name.endswith(".bin.bz2") else path.suffix.lower()
    filename_time = parse_filename_time(path.name)
    header_time_value = header_time(data)
    delta = ""
    if filename_time and header_time_value:
        left = datetime.fromisoformat(filename_time)
        right = datetime.fromisoformat(header_time_value)
        delta = f"{(right - left).total_seconds():.3f}"
    row: dict[str, object] = {
        "relative_path": relative,
        "category_dir": category_dir,
        "role": role,
        "source_label": source_label,
        "extension": suffix,
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path) if do_hash else "",
        "magic": data[:4].decode("ascii", errors="replace"),
        "format_version_u16": u16(data, 4),
        "format_schema_u16": u16(data, 6),
        "header_kind_u32": u32(data, 8),
        "station_id": decode_fixed(data[0x20:0x28]),
        "site_name_header": decode_fixed(data[0x28:0x48]),
        "header_coord_1": f32(data, 0x48),
        "header_coord_2": f32(data, 0x4C),
        "filename_scan_type": path.name.rsplit("_", 1)[-1].split(".", 1)[0],
        "header_scan_type": decode_fixed(data[0xA0:0xA8]),
        "header_scan_description": decode_fixed(data[0xC0:0x100]),
        "header_field_0x140": u32(data, 0x140),
        "header_field_0x144": u32(data, 0x144),
        "header_field_0x148": u32(data, 0x148),
        "header_timestamp_utc": header_time_value,
        "filename_timestamp_utc": filename_time,
        "header_minus_filename_seconds": delta,
        "header_bytes_read": len(data),
        "label_status": "source_directory_only_unverified",
        "physical_axis_status": "unverified",
        "hv_status": "unverified",
    }
    return row


def u16(data: bytes, offset: int) -> int | None:
    if len(data) < offset + 2:
        return None
    return struct.unpack_from("<H", data, offset)[0]


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"missing collection root: {root}")
    paths = sorted(
        p for p in root.rglob("*") if p.is_file() and (p.suffix.lower() == ".iq" or p.name.endswith(".bin.bz2"))
    )
    rows = [parse_one(path, root, args.hash) for path in paths]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["relative_path"]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
