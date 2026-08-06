from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from scripts.audit_field_iq_integrity_v1 import (
    DEFAULT_CONTRACT,
    audit_field_iq,
    load_contract,
)


def complex_pair(shape: tuple[int, int] = (4, 3)) -> tuple[np.ndarray, np.ndarray]:
    base = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    h_data = base + 1j * (base + 0.5)
    v_data = (base * 2.0 + 1.0) + 1j * (base * 0.5 + 2.0)
    return h_data, v_data


def matlab_hdf5_complex(array: np.ndarray) -> np.ndarray:
    stored = array.T
    output = np.empty(stored.shape, dtype=[("real", "<f8"), ("imag", "<f8")])
    output["real"] = np.real(stored)
    output["imag"] = np.imag(stored)
    return output


def write_contract(path: Path, expected_shape: list[int] | None) -> Path:
    contract = load_contract(DEFAULT_CONTRACT)
    contract["contract_id"] = "test_field_iq_contract"
    contract["requirements"]["expected_shape"] = expected_shape
    path.write_text(json.dumps(contract), encoding="utf-8")
    return path


def write_manifest(path: Path, iq_paths: list[str]) -> Path:
    pd.DataFrame({"iq_path": iq_paths}).to_csv(path, index=False)
    return path


def test_v5_complex_pair_passes_file_content_gate(tmp_path: Path) -> None:
    data_root = tmp_path / "collection"
    data_root.mkdir()
    h_data, v_data = complex_pair()
    savemat(data_root / "sample.mat", {"local_data_H": h_data, "local_data_V": v_data})
    manifest = write_manifest(tmp_path / "manifest.csv", ["sample.mat"])
    contract = write_contract(tmp_path / "contract.json", [4, 3])

    summary = audit_field_iq(
        manifest_path=manifest,
        data_root=data_root,
        contract_path=contract,
        output_dir=tmp_path / "output",
        overwrite=False,
    )

    assert summary["status"] == "PASS_FILE_CONTENT_ONLY"
    assert summary["cap_raw_complex_hv_candidate"] is True
    assert summary["cal_complex_iq_candidate"] is True
    assert summary["gate_status"]["hv_coherence"].startswith("BLOCKED")


def test_template_without_device_shape_remains_blocked(tmp_path: Path) -> None:
    data_root = tmp_path / "collection"
    data_root.mkdir()
    h_data, v_data = complex_pair()
    savemat(data_root / "sample.mat", {"local_data_H": h_data, "local_data_V": v_data})
    manifest = write_manifest(tmp_path / "manifest.csv", ["sample.mat"])

    summary = audit_field_iq(
        manifest_path=manifest,
        data_root=data_root,
        contract_path=DEFAULT_CONTRACT,
        output_dir=tmp_path / "output",
        overwrite=False,
    )

    assert summary["status"] == "BLOCKED_EXPECTED_SHAPE"
    assert summary["cap_raw_complex_hv_candidate"] is True
    assert summary["cal_complex_iq_candidate"] is False


def test_noncomplex_channel_fails_probe(tmp_path: Path) -> None:
    data_root = tmp_path / "collection"
    data_root.mkdir()
    h_data, v_data = complex_pair()
    savemat(
        data_root / "sample.mat",
        {"local_data_H": h_data, "local_data_V": np.real(v_data)},
    )
    manifest = write_manifest(tmp_path / "manifest.csv", ["sample.mat"])
    contract = write_contract(tmp_path / "contract.json", [4, 3])

    summary = audit_field_iq(
        manifest_path=manifest,
        data_root=data_root,
        contract_path=contract,
        output_dir=tmp_path / "output",
        overwrite=False,
    )
    detail = pd.read_csv(tmp_path / "output/file_audit_local.csv", encoding="utf-8-sig")

    assert summary["status"] == "FAIL"
    assert "NONCOMPLEX_CHANNEL" in detail.iloc[0]["error_codes"]


def test_hdf5_complex_pair_is_supported(tmp_path: Path) -> None:
    data_root = tmp_path / "collection"
    data_root.mkdir()
    h_data, v_data = complex_pair()
    with h5py.File(data_root / "sample.mat", "w") as handle:
        # MATLAB v7.3 stores logical array dimensions in reverse HDF5 order.
        handle.create_dataset("local_data_H", data=matlab_hdf5_complex(h_data))
        handle.create_dataset("local_data_V", data=matlab_hdf5_complex(v_data))
    manifest = write_manifest(tmp_path / "manifest.csv", ["sample.mat"])
    contract = write_contract(tmp_path / "contract.json", [4, 3])

    summary = audit_field_iq(
        manifest_path=manifest,
        data_root=data_root,
        contract_path=contract,
        output_dir=tmp_path / "output",
        overwrite=False,
    )
    shapes = pd.read_csv(tmp_path / "output/shape_dtype_summary.csv", encoding="utf-8-sig")

    assert summary["status"] == "PASS_FILE_CONTENT_ONLY"
    assert shapes.iloc[0]["reader"] == "h5py_mat_v7_3"


def test_nonnumeric_variable_becomes_file_failure(tmp_path: Path) -> None:
    data_root = tmp_path / "collection"
    data_root.mkdir()
    _, v_data = complex_pair()
    savemat(
        data_root / "sample.mat",
        {"local_data_H": np.array([["bad", "iq"]]), "local_data_V": v_data},
    )
    manifest = write_manifest(tmp_path / "manifest.csv", ["sample.mat"])
    contract = write_contract(tmp_path / "contract.json", [4, 3])

    summary = audit_field_iq(
        manifest_path=manifest,
        data_root=data_root,
        contract_path=contract,
        output_dir=tmp_path / "output",
        overwrite=False,
    )
    detail = pd.read_csv(tmp_path / "output/file_audit_local.csv", encoding="utf-8-sig")

    assert summary["status"] == "FAIL"
    assert "NONNUMERIC_CHANNEL" in detail.iloc[0]["error_codes"]


def test_unsafe_manifest_path_is_rejected(tmp_path: Path) -> None:
    data_root = tmp_path / "collection"
    data_root.mkdir()
    manifest = write_manifest(tmp_path / "manifest.csv", ["../outside.mat"])

    with pytest.raises(ValueError, match="unsafe iq_path"):
        audit_field_iq(
            manifest_path=manifest,
            data_root=data_root,
            contract_path=DEFAULT_CONTRACT,
            output_dir=tmp_path / "output",
            overwrite=False,
        )


def test_summary_omits_private_root_and_file_identity(tmp_path: Path) -> None:
    data_root = tmp_path / "private_collection"
    data_root.mkdir()
    h_data, v_data = complex_pair()
    savemat(data_root / "secret_name.mat", {"local_data_H": h_data, "local_data_V": v_data})
    manifest = write_manifest(tmp_path / "private_manifest.csv", ["secret_name.mat"])
    contract = write_contract(tmp_path / "contract.json", [4, 3])
    output = tmp_path / "output"

    audit_field_iq(
        manifest_path=manifest,
        data_root=data_root,
        contract_path=contract,
        output_dir=output,
        overwrite=False,
    )
    summary_text = (output / "summary.json").read_text(encoding="utf-8")

    assert str(data_root) not in summary_text
    assert "secret_name.mat" not in summary_text
    assert "file_audit_local.csv" in summary_text
