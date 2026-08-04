from __future__ import annotations

import csv
import json
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scripts.audit_lss_hsr_l_v2 as hsr_audit
from scipy.io import savemat

from scripts.audit_lss_hsr_l_v2 import audit_dataset


def _mat_payload(
    frames: int,
    *,
    dpl_width: int = 512,
    track_frames: int | None = None,
    nonfinite: str | None = None,
    offset: float = 0.0,
) -> bytes:
    track_frames = frames if track_frames is None else track_frames
    dpl = np.arange(frames * dpl_width, dtype=np.float64).reshape(
        frames, dpl_width
    )
    track = np.arange(track_frames * 5, dtype=np.float64).reshape(track_frames, 5)
    dpl += offset
    track += offset
    if nonfinite == "nan":
        dpl[0, 0] = np.nan
    elif nonfinite == "inf":
        track[0, 0] = np.inf

    outer = np.empty((1, 2), dtype=object)
    dpl_cell = np.empty((1, 1), dtype=object)
    track_cell = np.empty((1, 1), dtype=object)
    dpl_cell[0, 0] = dpl
    track_cell[0, 0] = track
    outer[0, 0] = dpl_cell
    outer[0, 1] = track_cell
    buffer = BytesIO()
    savemat(buffer, {"Trace_DPL_Data": outer})
    return buffer.getvalue()


def _base_mats() -> dict[str, bytes]:
    return {
        "train/Bird/0001_6.mat": _mat_payload(6, offset=1_000.0),
        "train/Bird/0002_5.mat": _mat_payload(5, offset=2_000.0),
        "validation/UAV1/0001_8.mat": _mat_payload(8, offset=3_000.0),
        "overflow/Car/0001_10.mat": _mat_payload(10, offset=4_000.0),
    }


def _base_routes() -> dict[str, dict[str, Any]]:
    return {
        "air_route_1": {
            "tracks": [
                "train/Bird/0001_6.mat",
                "train/Bird/0002_5.mat",
            ],
            "step_length": 1,
        },
        "air_route_2": {
            "tracks": ["validation/UAV1/0001_8.mat"],
            "step_length": 1,
        },
        "air_route_3": {
            "tracks": ["overflow/Car/0001_10.mat"],
            "step_length": 5,
        },
    }


def _write_archive(
    path: Path,
    *,
    mats: dict[str, bytes] | None = None,
    routes: dict[str, dict[str, Any]] | None = None,
    route_payload: str | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    mats = _base_mats() if mats is None else mats
    routes = _base_routes() if routes is None else routes
    prefix = "release/LSS-HSR-L/"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("release/Dataset.py", "raise RuntimeError('must not run')\n")
        archive.writestr("release/user_guide.docx", b"synthetic docx")
        archive.writestr("release/user_guide.pdf", b"synthetic pdf")
        archive.writestr(
            prefix + "air_routes.json",
            route_payload
            if route_payload is not None
            else json.dumps(routes, separators=(",", ":"), sort_keys=True),
        )
        for relative_path, payload in sorted(mats.items()):
            archive.writestr(prefix + relative_path, payload)
    return path


def _snapshot(output_dir: Path) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in sorted(output_dir.iterdir(), key=lambda item: item.name)
    }


def test_synthetic_archive_schema_routes_windows_and_gates(tmp_path: Path) -> None:
    archive_path = _write_archive(tmp_path / "source" / "v2.zip")
    source_before = archive_path.read_bytes()
    source_hash_before = hsr_audit.sha256_file(archive_path)
    output_dir = tmp_path / "audit"

    summary = audit_dataset(
        archive_path=archive_path,
        output_dir=output_dir,
        strict_release=False,
    )

    assert archive_path.read_bytes() == source_before
    assert hsr_audit.sha256_file(archive_path) == source_hash_before
    assert {path.name for path in output_dir.iterdir()} == hsr_audit.OUTPUT_FILES
    assert summary["status"] == (
        "PASS_SCHEMA_BLOCKED_SOURCE_PROVENANCE_AND_PHYSICAL_AXIS"
    )
    assert summary["audit_mode"] == "schema_fixture"
    assert summary["release_identity_verified"] is False
    assert summary["gates"]["release_identity"] == "NOT_EVALUATED_TEST_FIXTURE"
    assert summary["release_version"] is None
    assert summary["data_doi"] is None
    assert summary["target_release_version"] == "V2"
    assert summary["target_data_doi"] == "10.57760/sciencedb.radars.00063"
    assert summary["mat_file_count"] == 4
    assert summary["route_count"] == 3
    assert summary["route_track_reference_count"] == 4
    assert summary["total_frame_count"] == 29
    assert summary["split_mat_counts"] == {
        "train": 2,
        "validation": 1,
        "overflow": 1,
    }
    assert summary["split_route_counts"] == {
        "train": 1,
        "validation": 1,
        "overflow": 1,
    }
    assert summary["split_published_window_counts"] == {
        "train": 6,
        "validation": 3,
        "overflow": 1,
    }
    assert summary["published_window_count"] == 10
    assert summary["step_length_counts"] == {"1": 2, "5": 1}
    assert summary["gates"]["route_mapping"] == "PASS"
    assert summary["gates"]["overflow_role"] == "NOT_EVALUATED_TEST_FIXTURE"
    assert summary["gates"]["acquisition_session_identity"] == (
        "NOT_EVALUATED_TEST_FIXTURE"
    )
    assert summary["gates"]["dpl_physical_time_axis"] == (
        "NOT_EVALUATED_TEST_FIXTURE"
    )
    assert summary["gates"]["model_training"] == "BLOCKED"
    assert summary["model_training_allowed"] is False
    assert summary["overflow_merge_allowed"] is False
    assert summary["overflow_isolated"] is True
    assert summary["train_validation_route_disjoint"] is True
    assert summary["published_train_validation_route_disjoint"] is None
    assert summary["published_train_validation_route_overlap_count"] is None
    assert summary["published_train_validation_session_disjoint_verified"] is False
    assert summary["session_identity_field_present"] is False
    assert summary["acquisition_session_key_available"] is None
    assert summary["lowest_available_grouping_key"] == "route_id"
    assert summary["minimum_split_unit"] == (
        "UNRESOLVED_SOURCE_SESSION_IDENTITY_UNAVAILABLE"
    )
    assert summary["route_id_sufficient_for_independent_evaluation"] is False
    assert summary["track_physical_units_available"] is False
    assert summary["track_physical_units_verified"] is False
    assert summary["track_features"] == [
        {"index": index, "name": None, "unit": None} for index in range(1, 6)
    ]
    assert summary["dpl_representation"] is None
    assert summary["dpl_amplitude_unit_verified"] is None
    assert summary["dpl_physical_time_axis_available"] is None
    assert summary["dpl_physical_doppler_hz_axis_available"] is None
    assert summary["dpl_physical_velocity_axis_available"] is None
    assert summary["raw_adc_or_iq_available"] is None
    assert summary["h_v_polarimetry_available"] is None
    assert summary["official_dataset_py_executed"] is False
    assert summary["source_archive_extracted"] is False
    assert summary["source_archive_modified"] is False

    persisted = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert persisted == summary
    report = (output_dir / "REPORT.md").read_text(encoding="utf-8")
    assert "Release identity was not evaluated in schema-fixture mode" in report
    assert "canonical ScienceDB V2 ZIP passed identity" not in report
    assert "official V2 user guide" not in report
    assert "processed real waterfall" not in report
    with (output_dir / "feature_summary.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        feature_rows = list(csv.DictReader(handle))
    assert len(feature_rows) == 6
    assert {row["unit"] for row in feature_rows} == {"not_evaluated"}
    assert {row["unit_source"] for row in feature_rows} == {
        "not evaluated in schema-fixture mode"
    }
    assert {row["physical_axis_available"] for row in feature_rows} == {
        "NOT_EVALUATED"
    }
    with (output_dir / "split_summary.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        split_rows = list(csv.DictReader(handle))
    assert {row["role"] for row in split_rows} == {"NOT_EVALUATED_TEST_FIXTURE"}
    with (output_dir / "route_audit.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        route_rows = list(csv.DictReader(handle))
    assert [row["route_id"] for row in route_rows] == [
        "air_route_1",
        "air_route_2",
        "air_route_3",
    ]
    assert [int(row["published_window_count"]) for row in route_rows] == [6, 3, 1]


def test_duplicate_statistics_count_unique_hashes_not_duplicate_groups(
    tmp_path: Path,
) -> None:
    payload = _mat_payload(6)
    mats = {
        "train/Bird/0001_6.mat": payload,
        "validation/UAV1/0001_6.mat": payload,
        "overflow/Car/0001_6.mat": payload,
    }
    routes = {
        f"air_route_{index}": {"tracks": [track], "step_length": 1}
        for index, track in enumerate(mats, start=1)
    }
    archive_path = _write_archive(
        tmp_path / "source" / "duplicates.zip", mats=mats, routes=routes
    )

    summary = audit_dataset(
        archive_path=archive_path,
        output_dir=tmp_path / "audit",
        strict_release=False,
    )

    assert summary["mat_file_count"] == 3
    assert summary["unique_raw_mat_count"] == 1
    assert summary["unique_numeric_payload_count"] == 1
    assert summary["exact_raw_mat_duplicate_group_count"] == 1
    assert summary["exact_numeric_payload_duplicate_group_count"] == 1


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("dpl_width", "invalid DPL shape"),
        ("length_mismatch", "DPL/track length mismatch"),
        ("nan", "contains NaN/Inf"),
        ("inf", "contains NaN/Inf"),
        ("filename_frames", "filename declares 6 frames"),
    ),
)
def test_mat_schema_failures_are_rejected_before_outputs(
    tmp_path: Path, case: str, message: str
) -> None:
    mats = _base_mats()
    if case == "dpl_width":
        mats["train/Bird/0001_6.mat"] = _mat_payload(6, dpl_width=511)
    elif case == "length_mismatch":
        mats["train/Bird/0001_6.mat"] = _mat_payload(6, track_frames=5)
    elif case in {"nan", "inf"}:
        mats["train/Bird/0001_6.mat"] = _mat_payload(6, nonfinite=case)
    else:
        mats["train/Bird/0001_6.mat"] = _mat_payload(5)
    archive_path = _write_archive(tmp_path / "source" / "invalid.zip", mats=mats)
    output_dir = tmp_path / "audit"

    with pytest.raises(ValueError, match=message):
        audit_dataset(
            archive_path=archive_path,
            output_dir=output_dir,
            strict_release=False,
        )

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".audit.transaction-*"))


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("duplicate", "assigned to both"),
        ("unlisted", "missing from air_routes.json"),
        ("mixed", "route mixes split or class"),
        ("unsafe", "unsafe track path"),
        ("missing", "listed track does not exist"),
        ("step", "step_length must be a positive integer"),
    ),
)
def test_route_mapping_failures_are_rejected(
    tmp_path: Path, case: str, message: str
) -> None:
    routes = _base_routes()
    if case == "duplicate":
        routes["air_route_2"]["tracks"].insert(0, "train/Bird/0001_6.mat")
    elif case == "unlisted":
        routes["air_route_1"]["tracks"].pop()
    elif case == "mixed":
        routes["air_route_1"]["tracks"].append(
            "validation/UAV1/0001_8.mat"
        )
    elif case == "unsafe":
        routes["air_route_1"]["tracks"][0] = "../train/Bird/0001_6.mat"
    elif case == "missing":
        routes["air_route_1"]["tracks"][0] = "train/Bird/9999_6.mat"
    else:
        routes["air_route_1"]["step_length"] = 0
    archive_path = _write_archive(
        tmp_path / "source" / "invalid-routes.zip", routes=routes
    )

    with pytest.raises(ValueError, match=message):
        audit_dataset(
            archive_path=archive_path,
            output_dir=tmp_path / "audit",
            strict_release=False,
        )

    assert not list(tmp_path.glob(".audit.transaction-*"))


def test_duplicate_json_key_is_rejected(tmp_path: Path) -> None:
    route = json.dumps(_base_routes()["air_route_1"], separators=(",", ":"))
    route_payload = f'{{"air_route_1":{route},"air_route_1":{route}}}'
    archive_path = _write_archive(
        tmp_path / "source" / "duplicate-key.zip",
        route_payload=route_payload,
    )

    with pytest.raises(ValueError, match="duplicate JSON key 'air_route_1'"):
        audit_dataset(
            archive_path=archive_path,
            output_dir=tmp_path / "audit",
            strict_release=False,
        )

    assert not list(tmp_path.glob(".audit.transaction-*"))


def test_failed_overwrite_preserves_previous_outputs(tmp_path: Path) -> None:
    archive_path = _write_archive(tmp_path / "source" / "v2.zip")
    output_dir = tmp_path / "audit"
    audit_dataset(
        archive_path=archive_path,
        output_dir=output_dir,
        strict_release=False,
    )
    previous_outputs = _snapshot(output_dir)
    mats = _base_mats()
    mats["train/Bird/0001_6.mat"] = _mat_payload(6, dpl_width=511)
    _write_archive(archive_path, mats=mats)

    with pytest.raises(ValueError, match="invalid DPL shape"):
        audit_dataset(
            archive_path=archive_path,
            output_dir=output_dir,
            overwrite=True,
            strict_release=False,
        )

    assert _snapshot(output_dir) == previous_outputs
    assert not list(tmp_path.glob(".audit.transaction-*"))


def test_staging_write_failure_preserves_previous_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_path = _write_archive(tmp_path / "source" / "v2.zip")
    output_dir = tmp_path / "audit"
    audit_dataset(
        archive_path=archive_path,
        output_dir=output_dir,
        strict_release=False,
    )
    previous_outputs = _snapshot(output_dir)
    real_write_csv = hsr_audit._write_csv
    write_count = 0

    def fail_second_csv(*args: object, **kwargs: object) -> None:
        nonlocal write_count
        write_count += 1
        if write_count == 2:
            raise OSError("injected staging write failure")
        real_write_csv(*args, **kwargs)

    monkeypatch.setattr(hsr_audit, "_write_csv", fail_second_csv)
    with pytest.raises(OSError, match="injected staging write failure"):
        audit_dataset(
            archive_path=archive_path,
            output_dir=output_dir,
            overwrite=True,
            strict_release=False,
        )

    assert _snapshot(output_dir) == previous_outputs
    assert not list(tmp_path.glob(".audit.transaction-*"))


def test_promotion_failure_rolls_back_previous_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_path = _write_archive(tmp_path / "source" / "v2.zip")
    output_dir = tmp_path / "audit"
    audit_dataset(
        archive_path=archive_path,
        output_dir=output_dir,
        strict_release=False,
    )
    previous_outputs = _snapshot(output_dir)
    real_rename = Path.rename

    def fail_new_output_promotion(self: Path, target: Path) -> Path:
        if self.name == "new" and Path(target) == output_dir:
            raise OSError("injected promotion failure")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", fail_new_output_promotion)
    with pytest.raises(OSError, match="injected promotion failure"):
        audit_dataset(
            archive_path=archive_path,
            output_dir=output_dir,
            overwrite=True,
            strict_release=False,
        )

    assert _snapshot(output_dir) == previous_outputs
    assert not list(tmp_path.glob(".audit.transaction-*"))


def test_orphan_transaction_with_literal_glob_characters_fails_closed(
    tmp_path: Path,
) -> None:
    archive_path = _write_archive(tmp_path / "source" / "v2.zip")
    output_dir = tmp_path / "audit[1]"
    audit_dataset(
        archive_path=archive_path,
        output_dir=output_dir,
        strict_release=False,
    )
    previous_outputs = _snapshot(output_dir)
    orphan = tmp_path / ".audit[1].transaction-interrupted"
    (orphan / "old").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="unfinished audit transaction"):
        audit_dataset(
            archive_path=archive_path,
            output_dir=output_dir,
            overwrite=True,
            strict_release=False,
        )

    assert _snapshot(output_dir) == previous_outputs
    assert (orphan / "old").is_dir()


def test_known_journal_bundle_is_rejected_before_zip_parsing_or_output_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal_path = hsr_audit.DEFAULT_ARCHIVE.with_name(
        "LSS-HSR-L_dataset_and_instructions.zip"
    )
    if not journal_path.is_file():
        pytest.skip("local historical journal bundle is unavailable")
    assert journal_path.stat().st_size == hsr_audit.JOURNAL_ARCHIVE_SIZE_BYTES

    def forbidden_zip_parse(*args: object, **kwargs: object) -> None:
        raise AssertionError("journal bundle reached ZIP parsing")

    monkeypatch.setattr(hsr_audit.zipfile, "ZipFile", forbidden_zip_parse)
    output_dir = tmp_path / "not-created" / "audit"
    with pytest.raises(ValueError, match="historical journal bundle"):
        audit_dataset(
            archive_path=journal_path,
            output_dir=output_dir,
            strict_release=True,
        )

    assert not output_dir.parent.exists()


def test_real_sciencedb_v2_strict_identity_inventory_and_frozen_counts(
    tmp_path: Path,
) -> None:
    archive_path = hsr_audit.DEFAULT_ARCHIVE
    if not archive_path.is_file():
        pytest.skip("local ScienceDB V2 archive is unavailable")
    source_stat_before = archive_path.stat()
    source_hash_before = hsr_audit.sha256_file(archive_path)
    output_dir = tmp_path / "audit"

    summary = audit_dataset(
        archive_path=archive_path,
        output_dir=output_dir,
        strict_release=True,
    )

    source_stat_after = archive_path.stat()
    assert source_stat_after.st_size == source_stat_before.st_size
    assert source_stat_after.st_mtime_ns == source_stat_before.st_mtime_ns
    assert hsr_audit.sha256_file(archive_path) == source_hash_before
    assert summary["archive_size_bytes"] == 237_020_946
    assert summary["archive_sha256"] == (
        "fea8a21354110a96fb9644dc1c69649b6dc6d1a1b6da512498d9c2d74d839540"
    )
    assert summary["zip_entry_count"] == 1_561
    assert summary["zip_file_count"] == 1_534
    assert summary["zip_directory_count"] == 27
    assert summary["mat_file_count"] == 1_530
    assert summary["route_count"] == 865
    assert summary["route_track_reference_count"] == 1_530
    assert summary["unique_raw_mat_count"] == 1_530
    assert summary["unique_numeric_payload_count"] == 1_530
    assert summary["total_frame_count"] == 63_148
    assert summary["split_mat_counts"] == {
        "train": 1_269,
        "validation": 250,
        "overflow": 11,
    }
    assert summary["split_route_counts"] == {
        "train": 723,
        "validation": 131,
        "overflow": 11,
    }
    assert summary["split_frame_counts"] == {
        "train": 51_789,
        "validation": 10_655,
        "overflow": 704,
    }
    assert summary["split_published_window_counts"] == {
        "train": 45_366,
        "validation": 9_336,
        "overflow": 529,
    }
    assert summary["published_window_count"] == 55_231
    assert summary["step_length_counts"] == {"1": 802, "5": 63}
    assert summary["status"] == (
        "PASS_SCHEMA_BLOCKED_SOURCE_PROVENANCE_AND_PHYSICAL_AXIS"
    )
    assert summary["audit_mode"] == "strict_release"
    assert summary["release_identity_verified"] is True
    assert summary["gates"]["release_identity"] == "PASS"
    assert summary["release_version"] == "V2"
    assert summary["data_doi"] == "10.57760/sciencedb.radars.00063"
    assert summary["model_training_allowed"] is False
    assert summary["overflow_isolated"] is True
    assert summary["published_train_validation_route_disjoint"] is True
    assert summary["published_train_validation_session_disjoint_verified"] is False
    assert summary["lowest_available_grouping_key"] == "route_id"
    assert summary["minimum_split_unit"] == (
        "UNRESOLVED_SOURCE_SESSION_IDENTITY_UNAVAILABLE"
    )
    assert summary["route_id_sufficient_for_independent_evaluation"] is False
    assert summary["track_physical_units_available"] is True
    assert summary["track_physical_units_verified"] is True
    assert summary["dpl_physical_time_axis_available"] is False
    assert summary["dpl_physical_doppler_hz_axis_available"] is False
    assert summary["dpl_physical_velocity_axis_available"] is False
    assert summary["raw_adc_or_iq_available"] is False
    assert summary["h_v_polarimetry_available"] is False
    with (output_dir / "feature_summary.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        feature_rows = list(csv.DictReader(handle))
    assert feature_rows[1]["feature_name"] == "radial_velocity"
    assert feature_rows[1]["unit"] == "m/s"
    assert feature_rows[1]["physical_axis_available"] == "True"
    assert feature_rows[1]["unit_source"].startswith("official V2 user guide")
    report = (output_dir / "REPORT.md").read_text(encoding="utf-8")
    assert "canonical ScienceDB V2 ZIP passed identity" in report
    assert "processed real waterfall" in report


def test_strict_wrong_size_fails_before_hashing_or_zip_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_path = _write_archive(tmp_path / "source" / "wrong-size.zip")

    def forbidden_hash(*args: object, **kwargs: object) -> str:
        raise AssertionError("wrong-size archive reached SHA256")

    monkeypatch.setattr(hsr_audit, "sha256_file", forbidden_hash)
    with pytest.raises(ValueError, match="unexpected V2 archive size"):
        audit_dataset(
            archive_path=archive_path,
            output_dir=tmp_path / "audit",
            strict_release=True,
        )

    assert not (tmp_path / "audit").exists()
