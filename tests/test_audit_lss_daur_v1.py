from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import scripts.audit_lss_daur_v1 as daur_audit
from scipy.io import savemat

from scripts.audit_lss_daur_v1 import (
    _hdf5_numeric,
    _validate_output_dir,
    audit_dataset,
    parse_name,
)


TR_FIELDS = ("A", "A_m", "E", "E_m", "R", "R_m", "SNR", "V", "V_m")


def _write_backup(
    path: Path,
    payload: dict[str, np.ndarray],
    *,
    width: int,
    date: tuple[int, int, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        for key, value in payload.items():
            array = np.asarray(value)
            if array.ndim >= 2:
                array = array.T
            elif array.ndim == 1:
                array = array[:, None]
            handle.create_dataset(key, data=array)
        file_head = handle.create_group("File_head")
        head = {
            "year": date[0],
            "month": date[1],
            "day": date[2],
            "nDaCf": 1360,
            "nDPLLen": width,
            "nSaveDplLen": width,
        }
        for key, value in head.items():
            file_head.create_dataset(key, data=np.asarray([[float(value)]]))


def _write_recording(
    root: Path,
    *,
    timestamp: str,
    category: str,
    serial: int,
    batch: int,
    frame_keys: list[tuple[int, int]],
    corrupt_td_backup: bool = False,
) -> None:
    stem = f"{timestamp}_DAUR_{{token}}_{category}_{serial:02d}_{batch}.mat"
    frame_count = len(frame_keys)
    data_time = np.asarray([key[0] / 1000 for key in frame_keys], dtype=np.float64)
    gps = np.asarray([key[0] for key in frame_keys], dtype=np.int32)
    iframe = np.asarray([key[1] for key in frame_keys], dtype=np.uint16)
    dpl = np.arange(frame_count * 512, dtype=np.float64).reshape(frame_count, 512)
    dpl = dpl + 1j * (dpl / 10)
    shared = {
        "DATA_time": data_time,
        "GPS_time_in_data": gps,
        "Iframecnt": iframe,
    }
    td = {**shared, "DPL": dpl, "nDaCf": 1360}
    tr = {
        **shared,
        **{
            key: np.arange(frame_count, dtype=np.float64) + index
            for index, key in enumerate(TR_FIELDS)
        },
        "nDaCf": 1360,
    }

    td_path = root / "TD Data" / category / stem.format(token="RD")
    tr_path = root / "TR Data" / category / stem.format(token="TR")
    td_path.parent.mkdir(parents=True, exist_ok=True)
    tr_path.parent.mkdir(parents=True, exist_ok=True)
    savemat(td_path, td)
    savemat(tr_path, tr)

    td_backup = {key: np.asarray(value) for key, value in td.items() if key != "nDaCf"}
    if corrupt_td_backup:
        td_backup["DPL"] = td_backup["DPL"].copy()
        td_backup["DPL"][0, 0] += 1
    tr_backup = {key: np.asarray(value) for key, value in tr.items() if key != "nDaCf"}
    date = (int(timestamp[:4]), int(timestamp[4:6]), int(timestamp[6:8]))
    _write_backup(
        root / "TD Data" / "backup_original" / category / td_path.name,
        td_backup,
        width=512,
        date=date,
    )
    _write_backup(
        root / "TR Data" / "backup_original" / category / tr_path.name,
        tr_backup,
        width=512,
        date=date,
    )


def test_parse_name_preserves_official_td_rd_token() -> None:
    path = Path("Bird/20230102123456_DAUR_RD_Bird_01_42.mat")
    parsed = parse_name(path, "TD")
    assert parsed.timestamp == "20230102123456"
    assert parsed.category == "Bird"
    assert parsed.recording_id == "20230102123456|Bird|01|42"


@pytest.mark.parametrize(
    ("path", "modality"),
    (
        (Path("Bird/20230102123456_DAUR_TR_Bird_01_42.mat"), "TD"),
        (Path("Bird/20230102_DAUR_RD_Bird_01_42.mat"), "TD"),
        (Path("Bird/20230102123456_DAUR_RD_Unknown_01_42.mat"), "TD"),
    ),
)
def test_parse_name_rejects_wrong_token_shape_or_parent(
    path: Path, modality: str
) -> None:
    with pytest.raises(ValueError):
        parse_name(path, modality)


def test_hdf5_numeric_decodes_matlab_complex_and_transposes(tmp_path: Path) -> None:
    path = tmp_path / "complex.h5"
    compound = np.zeros((2, 1), dtype=[("real", "<f8"), ("imag", "<f8")])
    compound["real"][:, 0] = [1.0, 3.0]
    compound["imag"][:, 0] = [2.0, 4.0]
    with h5py.File(path, "w") as handle:
        handle.create_dataset("DPL", data=compound)
    with h5py.File(path, "r") as handle:
        decoded = _hdf5_numeric(handle["DPL"])
    assert np.array_equal(decoded, np.array([1.0 + 2.0j, 3.0 + 4.0j]))


def test_output_guard_refuses_dataset_paths_and_unknown_entries(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    with pytest.raises(ValueError, match="protected dataset"):
        _validate_output_dir(dataset_root / "audit", dataset_root, overwrite=True)

    output_dir = tmp_path / "audit"
    output_dir.mkdir()
    marker = output_dir / "keep.txt"
    marker.write_text("keep\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        _validate_output_dir(output_dir, dataset_root, overwrite=True)
    assert marker.read_text(encoding="utf-8") == "keep\n"


def test_output_guard_does_not_remove_known_audit_files(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    output_dir = tmp_path / "audit"
    output_dir.mkdir()
    report = output_dir / "REPORT.md"
    report.write_text("old\n", encoding="utf-8")
    _validate_output_dir(output_dir, dataset_root, overwrite=True)
    assert output_dir.is_dir()
    assert report.read_text(encoding="utf-8") == "old\n"


def test_audit_pairs_four_representations_and_builds_session_groups(tmp_path: Path) -> None:
    dataset_root = tmp_path / "LSS-DAUR-1.0"
    _write_recording(
        dataset_root,
        timestamp="20230102123456",
        category="Bird",
        serial=1,
        batch=42,
        frame_keys=[(1000, 1), (1000, 2), (2000, 3)],
    )
    _write_recording(
        dataset_root,
        timestamp="20230102123456",
        category="Speedboat",
        serial=1,
        batch=43,
        frame_keys=[(1000, 1), (3000, 4)],
    )

    output_dir = tmp_path / "audit"
    summary = audit_dataset(
        dataset_root=dataset_root,
        output_dir=output_dir,
        strict_release=False,
    )

    assert summary["logical_recording_count"] == 2
    assert summary["mat_file_count"] == 8
    assert summary["td_tr_pair_count"] == 2
    assert summary["canonical_backup_equivalent_pair_count"] == 4
    assert summary["shared_frame_record_pair_count"] == 1
    assert summary["candidate_source_session_group_count"] == 1
    assert summary["status"] == (
        "PASS_SCHEMA_PAIRING_BLOCKED_GROUPING_AND_PHYSICAL_AXIS"
    )
    assert summary["gates"]["model_training"] == "BLOCKED"
    assert summary["paired_track_count"] == 2
    assert summary["frame_count"] == 5
    assert summary["duplicate_time_step_count"] == 1
    assert summary["unique_time_position_count"] == 4
    assert summary["random_frame_or_window_split_allowed"] is False
    assert summary["canonical_backup_as_extra_samples_allowed"] is False
    assert summary["model_training_allowed"] is False
    assert summary["source_files_modified"] is False
    assert (output_dir / "source_session_membership.csv").is_file()
    assert (output_dir / "source_session_group_audit.csv").is_file()
    assert (output_dir / "class_summary.csv").is_file()
    assert (output_dir / "doppler_config_audit.csv").is_file()
    assert (output_dir / "REPORT.md").is_file()


def test_audit_rejects_numerically_different_backup(tmp_path: Path) -> None:
    dataset_root = tmp_path / "LSS-DAUR-1.0"
    _write_recording(
        dataset_root,
        timestamp="20230102123456",
        category="Bird",
        serial=1,
        batch=42,
        frame_keys=[(1000, 1), (2000, 2)],
        corrupt_td_backup=True,
    )
    with pytest.raises(ValueError, match="backup field DPL differs numerically"):
        audit_dataset(
            dataset_root=dataset_root,
            output_dir=tmp_path / "audit",
            strict_release=False,
        )


def test_failed_overwrite_preserves_previous_audit_outputs(tmp_path: Path) -> None:
    dataset_root = tmp_path / "LSS-DAUR-1.0"
    recording_args = {
        "timestamp": "20230102123456",
        "category": "Bird",
        "serial": 1,
        "batch": 42,
        "frame_keys": [(1000, 1), (2000, 2)],
    }
    _write_recording(dataset_root, **recording_args)
    output_dir = tmp_path / "audit"
    audit_dataset(
        dataset_root=dataset_root,
        output_dir=output_dir,
        strict_release=False,
    )
    previous_outputs = {
        path.name: path.read_bytes() for path in sorted(output_dir.iterdir())
    }

    _write_recording(dataset_root, **recording_args, corrupt_td_backup=True)
    with pytest.raises(ValueError, match="backup field DPL differs numerically"):
        audit_dataset(
            dataset_root=dataset_root,
            output_dir=output_dir,
            overwrite=True,
            strict_release=False,
        )

    assert {
        path.name: path.read_bytes() for path in sorted(output_dir.iterdir())
    } == previous_outputs
    assert not list(tmp_path.glob(".audit.transaction-*"))


def test_partial_staging_write_failure_preserves_previous_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root = tmp_path / "LSS-DAUR-1.0"
    _write_recording(
        dataset_root,
        timestamp="20230102123456",
        category="Bird",
        serial=1,
        batch=42,
        frame_keys=[(1000, 1), (2000, 2)],
    )
    output_dir = tmp_path / "audit"
    audit_dataset(
        dataset_root=dataset_root,
        output_dir=output_dir,
        strict_release=False,
    )
    previous_outputs = {
        path.name: path.read_bytes() for path in sorted(output_dir.iterdir())
    }
    real_write_csv = daur_audit._write_csv
    write_count = 0

    def fail_second_csv(*args: object, **kwargs: object) -> None:
        nonlocal write_count
        write_count += 1
        if write_count == 2:
            raise OSError("injected staging write failure")
        real_write_csv(*args, **kwargs)

    monkeypatch.setattr(daur_audit, "_write_csv", fail_second_csv)
    with pytest.raises(OSError, match="injected staging write failure"):
        audit_dataset(
            dataset_root=dataset_root,
            output_dir=output_dir,
            overwrite=True,
            strict_release=False,
        )

    assert {
        path.name: path.read_bytes() for path in sorted(output_dir.iterdir())
    } == previous_outputs
    assert not list(tmp_path.glob(".audit.transaction-*"))


def test_promotion_failure_rolls_back_previous_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root = tmp_path / "LSS-DAUR-1.0"
    _write_recording(
        dataset_root,
        timestamp="20230102123456",
        category="Bird",
        serial=1,
        batch=42,
        frame_keys=[(1000, 1), (2000, 2)],
    )
    output_dir = tmp_path / "audit"
    audit_dataset(
        dataset_root=dataset_root,
        output_dir=output_dir,
        strict_release=False,
    )
    previous_outputs = {
        path.name: path.read_bytes() for path in sorted(output_dir.iterdir())
    }
    real_rename = Path.rename

    def fail_new_output_promotion(self: Path, target: Path) -> Path:
        if self.name == "new" and Path(target) == output_dir:
            raise OSError("injected promotion failure")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", fail_new_output_promotion)
    with pytest.raises(OSError, match="injected promotion failure"):
        audit_dataset(
            dataset_root=dataset_root,
            output_dir=output_dir,
            overwrite=True,
            strict_release=False,
        )

    assert {
        path.name: path.read_bytes() for path in sorted(output_dir.iterdir())
    } == previous_outputs
    assert not list(tmp_path.glob(".audit.transaction-*"))


def test_unfinished_transaction_fails_closed_without_touching_output(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "LSS-DAUR-1.0"
    _write_recording(
        dataset_root,
        timestamp="20230102123456",
        category="Bird",
        serial=1,
        batch=42,
        frame_keys=[(1000, 1), (2000, 2)],
    )
    output_dir = tmp_path / "audit"
    output_dir.mkdir()
    report = output_dir / "REPORT.md"
    report.write_text("previous evidence\n", encoding="utf-8")
    orphan = tmp_path / ".audit.transaction-interrupted"
    (orphan / "old").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="unfinished audit transaction"):
        audit_dataset(
            dataset_root=dataset_root,
            output_dir=output_dir,
            overwrite=True,
            strict_release=False,
        )

    assert report.read_text(encoding="utf-8") == "previous evidence\n"
    assert (orphan / "old").is_dir()
