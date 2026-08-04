from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import pytest

from scripts.audit_dronerfc_mm_v1 import (
    GT_FIELDS,
    LABEL_FIELDS,
    audit_ground_truth,
    audit_labels,
    audit_pcd_archive,
    parse_aware_timestamp,
    validate_output_dir,
)


def pcd_bytes(
    *, seconds: int, nanoseconds: int, points: list[list[str]], declared: int | None = None
) -> bytes:
    count = len(points) if declared is None else declared
    lines = [
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION 0.7",
        (
            "FIELDS x y z rgb range azimuth elevation doppler power snr "
            "track_id cluster_id timestamp_sec timestamp_nsec radar_id"
        ),
        "SIZE " + " ".join(["4"] * 15),
        "TYPE " + " ".join(["F"] * 10 + ["I"] * 5),
        "COUNT " + " ".join(["1"] * 15),
        f"WIDTH {count}",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        f"POINTS {count}",
        "DATA ascii",
    ]
    for point in points:
        lines.append(" ".join(point[:12] + [str(seconds), str(nanoseconds)] + point[14:]))
    return ("\n".join(lines) + "\n").encode("ascii")


def valid_point() -> list[str]:
    return [
        "1.0",
        "2.0",
        "3.0",
        "4.0",
        "5.0",
        "0.1",
        "0.2",
        "0.3",
        "10.0",
        "5.0",
        "0",
        "0",
        "0",
        "0",
        "0",
    ]


def write_pcd_zip(path: Path, members: list[tuple[int, int, int, bytes]]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for frame_id, seconds, nanoseconds, payload in members:
            archive.writestr(
                f"recording/{frame_id:06d}_{seconds}.{nanoseconds:09d}.pcd",
                payload,
            )


def test_parse_aware_timestamp_accepts_single_digit_offset() -> None:
    timestamp = parse_aware_timestamp("2026-04-01T13:58:37.431000+8:00")
    assert timestamp.utcoffset() is not None
    assert timestamp.utcoffset().total_seconds() == 8 * 3600


def test_pcd_archive_is_sorted_by_frame_id_and_validates_all_rows(tmp_path: Path) -> None:
    path = tmp_path / "radar.zip"
    first = pcd_bytes(seconds=1_700_000_000, nanoseconds=10, points=[valid_point()])
    second = pcd_bytes(
        seconds=1_700_000_001,
        nanoseconds=20,
        points=[valid_point(), valid_point()],
    )
    write_pcd_zip(
        path,
        [
            (1, 1_700_000_001, 20, second),
            (0, 1_700_000_000, 10, first),
        ],
    )

    audit = audit_pcd_archive(path)

    assert audit.frame_count == 2
    assert audit.point_count == 3
    assert audit.min_points_per_frame == 1
    assert audit.max_points_per_frame == 2
    assert list(audit.timestamps) == [1_700_000_000.00000001, 1_700_000_001.00000002]


@pytest.mark.parametrize("failure", ["points", "nonfinite", "timestamp"])
def test_pcd_archive_rejects_invalid_data_rows(tmp_path: Path, failure: str) -> None:
    point = valid_point()
    declared = None
    filename_seconds = 1_700_000_000
    payload_seconds = filename_seconds
    if failure == "points":
        declared = 2
    elif failure == "nonfinite":
        point[0] = "nan"
    elif failure == "timestamp":
        payload_seconds += 1
    payload = pcd_bytes(
        seconds=payload_seconds,
        nanoseconds=10,
        points=[point],
        declared=declared,
    )
    path = tmp_path / "radar.zip"
    write_pcd_zip(path, [(0, filename_seconds, 10, payload)])

    with pytest.raises(ValueError):
        audit_pcd_archive(path)


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def gt_row(timestamp: str) -> dict[str, object]:
    row: dict[str, object] = {field: 0.0 for field in GT_FIELDS[1:]}
    row["timestamp"] = timestamp
    return row


def test_ground_truth_allows_duplicates_but_rejects_decreasing_time(tmp_path: Path) -> None:
    path = tmp_path / "gt.csv"
    write_csv(
        path,
        GT_FIELDS,
        [
            gt_row("2026-04-01T13:58:37+8:00"),
            gt_row("2026-04-01T13:58:37+08:00"),
            gt_row("2026-04-01T13:58:38.1+08:00"),
        ],
    )
    audit = audit_ground_truth(path)
    assert audit.row_count == 3
    assert audit.duplicate_timestamp_count == 1

    write_csv(
        path,
        GT_FIELDS,
        [
            gt_row("2026-04-01T13:58:38+08:00"),
            gt_row("2026-04-01T13:58:37+08:00"),
        ],
    )
    with pytest.raises(ValueError, match="decrease"):
        audit_ground_truth(path)


def test_labels_require_contiguous_ids_positive_nonoverlapping_windows(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    write_csv(
        path,
        LABEL_FIELDS,
        [
            {
                "idx": 0,
                "start_time": "2026-04-01T13:58:37+08:00",
                "end_time": "2026-04-01T13:58:42+08:00",
                "direction": "S",
            },
            {
                "idx": 1,
                "start_time": "2026-04-01T13:58:42+08:00",
                "end_time": "2026-04-01T13:58:47+08:00",
                "direction": "U",
            },
        ],
    )
    assert len(audit_labels(path)) == 2

    write_csv(
        path,
        LABEL_FIELDS,
        [
            {
                "idx": 0,
                "start_time": "2026-04-01T13:58:37+08:00",
                "end_time": "2026-04-01T13:58:42+08:00",
                "direction": "S",
            },
            {
                "idx": 1,
                "start_time": "2026-04-01T13:58:41+08:00",
                "end_time": "2026-04-01T13:58:47+08:00",
                "direction": "U",
            },
        ],
    )
    with pytest.raises(ValueError, match="overlap"):
        audit_labels(path)


def test_output_guard_refuses_source_paths_and_unknown_existing_entries(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    with pytest.raises(ValueError, match="protected|separate"):
        validate_output_dir(dataset_root, dataset_root, overwrite=True)

    output = tmp_path / "output"
    output.mkdir()
    (output / "keep.txt").write_text("keep\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        validate_output_dir(output, dataset_root, overwrite=True)
    assert (output / "keep.txt").read_text(encoding="utf-8") == "keep\n"
