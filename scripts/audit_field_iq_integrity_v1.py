#!/usr/bin/env python3
"""Read-only H/V complex-IQ integrity probe for a field collection manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = (
    PROJECT_ROOT / "configs/field_iq_probe_contract_template_v1.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/data_audit/field_iq_integrity_v1"
LOCAL_COLUMNS = (
    "file_index",
    "iq_path",
    "sha256",
    "size_bytes",
    "reader",
    "read_success",
    "h_present",
    "v_present",
    "h_shape",
    "v_shape",
    "h_dtype",
    "v_dtype",
    "h_numeric",
    "v_numeric",
    "h_complex",
    "v_complex",
    "h_finite",
    "v_finite",
    "h_real_std",
    "h_imag_std",
    "v_real_std",
    "v_imag_std",
    "same_hv_shape",
    "expected_shape_match",
    "file_pass",
    "error_codes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit referenced MAT files for H/V complex-IQ integrity."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    path = resolve_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"missing IQ probe contract: {path}")
    contract = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "contract_id",
        "file_format",
        "h_variable",
        "v_variable",
        "requirements",
        "claim_boundaries",
    }
    missing = required - set(contract)
    if missing:
        raise ValueError(f"IQ probe contract missing keys: {sorted(missing)}")
    if contract["schema_version"] != 1:
        raise ValueError("only IQ probe schema_version=1 is supported")
    if contract["file_format"] != "mat_auto_v5_or_v7_3":
        raise ValueError("V1 only supports MAT v5 or v7.3/HDF5 files")
    if not str(contract["h_variable"]).strip() or not str(
        contract["v_variable"]
    ).strip():
        raise ValueError("H/V variable names must be nonempty")
    requirements = contract["requirements"]
    required_requirements = {
        "expected_ndim",
        "expected_shape",
        "minimum_elements_per_channel",
        "require_same_hv_shape",
        "require_complex",
        "require_finite",
        "require_real_component_variation",
        "require_imag_component_variation",
        "minimum_component_std",
    }
    if set(requirements) != required_requirements:
        raise ValueError("IQ probe requirements do not match the V1 schema")
    if int(requirements["expected_ndim"]) <= 0:
        raise ValueError("expected_ndim must be positive")
    expected_shape = requirements["expected_shape"]
    if expected_shape is not None:
        if (
            not isinstance(expected_shape, list)
            or len(expected_shape) != int(requirements["expected_ndim"])
            or any(int(value) <= 0 for value in expected_shape)
        ):
            raise ValueError("expected_shape must contain one positive size per axis")
    if int(requirements["minimum_elements_per_channel"]) < 2:
        raise ValueError("minimum_elements_per_channel must be at least two")
    if float(requirements["minimum_component_std"]) < 0:
        raise ValueError("minimum_component_std must be nonnegative")
    expected_claims = {
        "hv_coherence_established_by_probe",
        "polarimetric_calibration_established_by_probe",
        "prf_or_physical_axis_established_by_probe",
        "channel_mapping_established_by_probe",
        "model_training_allowed_by_probe",
    }
    if set(contract["claim_boundaries"]) != expected_claims:
        raise ValueError("IQ probe claim boundaries do not match the V1 schema")
    if any(contract["claim_boundaries"].values()):
        raise ValueError("all V1 IQ-probe claim boundaries must remain false")
    return contract


def decode_hdf5_array(value: np.ndarray) -> np.ndarray:
    fields = value.dtype.fields
    if fields:
        names = {name.lower(): name for name in fields}
        real_name = names.get("real") or names.get("r")
        imag_name = names.get("imag") or names.get("imaginary") or names.get("i")
        if real_name and imag_name:
            value = np.asarray(value[real_name]) + 1j * np.asarray(value[imag_name])
    # MATLAB stores dimensions in reverse order inside v7.3 HDF5 files.
    return np.ascontiguousarray(np.asarray(value).T)


def read_mat_pair(path: Path, h_variable: str, v_variable: str) -> tuple[
    np.ndarray | None, np.ndarray | None, str
]:
    try:
        payload = loadmat(path, variable_names=[h_variable, v_variable])
        return payload.get(h_variable), payload.get(v_variable), "scipy_mat_v5"
    except (NotImplementedError, ValueError, OSError):
        try:
            with h5py.File(path, "r") as handle:
                if h_variable in handle and not isinstance(
                    handle[h_variable], h5py.Dataset
                ):
                    raise ValueError("HDF5_H_VARIABLE_NOT_DATASET")
                if v_variable in handle and not isinstance(
                    handle[v_variable], h5py.Dataset
                ):
                    raise ValueError("HDF5_V_VARIABLE_NOT_DATASET")
                h_data = (
                    decode_hdf5_array(np.asarray(handle[h_variable]))
                    if h_variable in handle
                    else None
                )
                v_data = (
                    decode_hdf5_array(np.asarray(handle[v_variable]))
                    if v_variable in handle
                    else None
                )
            return h_data, v_data, "h5py_mat_v7_3"
        except OSError as exc:
            raise ValueError(f"MAT_READER_ERROR:{type(exc).__name__}") from exc


def component_std(array: np.ndarray, component: str) -> float:
    values = np.real(array) if component == "real" else np.imag(array)
    return float(np.std(values.astype(np.float64, copy=False)))


def inspect_iq_file(
    path: Path,
    relative_path: str,
    file_index: int,
    contract: dict[str, Any],
) -> dict[str, Any]:
    record: dict[str, Any] = {column: "" for column in LOCAL_COLUMNS}
    record.update(
        {
            "file_index": file_index,
            "iq_path": relative_path,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "read_success": False,
            "h_present": False,
            "v_present": False,
            "file_pass": False,
        }
    )
    errors: list[str] = []
    try:
        h_data, v_data, reader = read_mat_pair(
            path, str(contract["h_variable"]), str(contract["v_variable"])
        )
        record["reader"] = reader
        record["read_success"] = True
    except (ValueError, KeyError) as exc:
        record["error_codes"] = str(exc).split(":", 1)[0]
        return record
    record["h_present"] = h_data is not None
    record["v_present"] = v_data is not None
    if h_data is None:
        errors.append("H_VARIABLE_MISSING")
    if v_data is None:
        errors.append("V_VARIABLE_MISSING")
    if h_data is None or v_data is None:
        record["error_codes"] = ";".join(errors)
        return record

    h_data = np.asarray(h_data)
    v_data = np.asarray(v_data)
    h_numeric = bool(np.issubdtype(h_data.dtype, np.number))
    v_numeric = bool(np.issubdtype(v_data.dtype, np.number))
    requirements = contract["requirements"]
    expected_ndim = int(requirements["expected_ndim"])
    expected_shape_raw = requirements["expected_shape"]
    expected_shape = (
        tuple(int(value) for value in expected_shape_raw)
        if expected_shape_raw is not None
        else None
    )
    record.update(
        {
            "h_shape": "x".join(str(value) for value in h_data.shape),
            "v_shape": "x".join(str(value) for value in v_data.shape),
            "h_dtype": str(h_data.dtype),
            "v_dtype": str(v_data.dtype),
            "h_numeric": h_numeric,
            "v_numeric": v_numeric,
            "h_complex": bool(np.iscomplexobj(h_data)),
            "v_complex": bool(np.iscomplexobj(v_data)),
            "h_finite": bool(np.isfinite(h_data).all()) if h_numeric else False,
            "v_finite": bool(np.isfinite(v_data).all()) if v_numeric else False,
            "h_real_std": component_std(h_data, "real") if h_numeric else np.nan,
            "h_imag_std": component_std(h_data, "imag") if h_numeric else np.nan,
            "v_real_std": component_std(v_data, "real") if v_numeric else np.nan,
            "v_imag_std": component_std(v_data, "imag") if v_numeric else np.nan,
            "same_hv_shape": h_data.shape == v_data.shape,
            "expected_shape_match": (
                h_data.shape == expected_shape and v_data.shape == expected_shape
                if expected_shape is not None
                else "not_configured"
            ),
        }
    )
    minimum_elements = int(requirements["minimum_elements_per_channel"])
    if h_data.ndim != expected_ndim or v_data.ndim != expected_ndim:
        errors.append("UNEXPECTED_NDIM")
    if h_data.size < minimum_elements or v_data.size < minimum_elements:
        errors.append("TOO_FEW_ELEMENTS")
    if not h_numeric or not v_numeric:
        errors.append("NONNUMERIC_CHANNEL")
    if requirements["require_same_hv_shape"] and h_data.shape != v_data.shape:
        errors.append("H_V_SHAPE_MISMATCH")
    if expected_shape is not None and (
        h_data.shape != expected_shape or v_data.shape != expected_shape
    ):
        errors.append("EXPECTED_SHAPE_MISMATCH")
    if requirements["require_complex"] and (
        not np.iscomplexobj(h_data) or not np.iscomplexobj(v_data)
    ):
        errors.append("NONCOMPLEX_CHANNEL")
    if requirements["require_finite"] and (
        not record["h_finite"] or not record["v_finite"]
    ):
        errors.append("NONFINITE_VALUES")
    minimum_std = float(requirements["minimum_component_std"])
    if h_numeric and v_numeric and requirements["require_real_component_variation"] and (
        record["h_real_std"] <= minimum_std or record["v_real_std"] <= minimum_std
    ):
        errors.append("REAL_COMPONENT_NO_VARIATION")
    if h_numeric and v_numeric and requirements["require_imag_component_variation"] and (
        record["h_imag_std"] <= minimum_std or record["v_imag_std"] <= minimum_std
    ):
        errors.append("IMAG_COMPONENT_NO_VARIATION")
    record["error_codes"] = ";".join(errors)
    record["file_pass"] = not errors
    return record


def manifest_iq_paths(manifest: pd.DataFrame) -> list[str]:
    if "iq_path" not in manifest.columns:
        raise ValueError("manifest missing iq_path column")
    values = manifest["iq_path"].fillna("").astype(str).str.strip()
    if values.eq("").any():
        raise ValueError("manifest contains empty iq_path values")
    paths = sorted(set(values))
    for value in paths:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"manifest contains unsafe iq_path: {value}")
        if path.suffix.lower() != ".mat":
            raise ValueError(f"V1 IQ probe requires .mat files: {value}")
    return paths


def resolve_iq_file(data_root: Path, relative_path: str) -> Path:
    candidate = (data_root / relative_path).resolve()
    try:
        candidate.relative_to(data_root)
    except ValueError as exc:
        raise ValueError(f"iq_path escapes data root: {relative_path}") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"missing referenced IQ file: {relative_path}")
    return candidate


def gate_summary(
    audit: pd.DataFrame, contract: dict[str, Any]
) -> tuple[str, dict[str, str]]:
    expected_shape_configured = contract["requirements"]["expected_shape"] is not None
    gates = {
        "file_inventory": "PASS" if len(audit) > 0 else "FAIL",
        "mat_read_and_variable_pair": (
            "PASS"
            if audit["read_success"].astype(bool).all()
            and audit["h_present"].astype(bool).all()
            and audit["v_present"].astype(bool).all()
            else "FAIL"
        ),
        "complex_finite_and_component_variation": (
            "PASS" if audit["file_pass"].astype(bool).all() else "FAIL"
        ),
        "expected_device_shape": (
            "PASS"
            if expected_shape_configured
            and audit["expected_shape_match"].astype(bool).all()
            else (
                "BLOCKED_NOT_CONFIGURED"
                if not expected_shape_configured
                else "FAIL"
            )
        ),
        "hv_coherence": "BLOCKED_REQUIRES_DEVICE_TIMING_EVIDENCE",
        "polarimetric_calibration": "BLOCKED_REQUIRES_REFERENCE_MEASUREMENTS",
        "physical_axis": "BLOCKED_REQUIRES_PRF_WAVEFORM_AND_CARRIER",
    }
    if "FAIL" in gates.values():
        status = "FAIL"
    elif gates["expected_device_shape"].startswith("BLOCKED"):
        status = "BLOCKED_EXPECTED_SHAPE"
    else:
        status = "PASS_FILE_CONTENT_ONLY"
    return status, gates


def aggregate_shapes(audit: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "reader",
        "h_shape",
        "v_shape",
        "h_dtype",
        "v_dtype",
        "h_complex",
        "v_complex",
        "file_pass",
    ]
    return (
        audit.groupby(columns, dropna=False)
        .size()
        .rename("file_count")
        .reset_index()
        .sort_values(columns)
        .reset_index(drop=True)
    )


def make_readme(summary: dict[str, Any]) -> str:
    return f"""# Field IQ Integrity Probe V1

Status: `{summary['status']}`. The probe inspected
{summary['file_count']} unique manifest-referenced MAT files without modifying
them.

This output can support a raw-complex-H/V file-content check. It cannot prove
H/V channel mapping, coherent timing, polarimetric calibration, PRF, physical
axes, independent sessions, or model readiness.

`file_audit_local.csv` contains relative data paths and per-file hashes and must
remain local. `shape_dtype_summary.csv` and `summary.json` contain aggregate
facts only.
"""


def audit_field_iq(
    *,
    manifest_path: Path,
    data_root: Path,
    contract_path: Path,
    output_dir: Path,
    overwrite: bool,
) -> dict[str, Any]:
    manifest_path = resolve_path(manifest_path)
    data_root = resolve_path(data_root)
    contract_path = resolve_path(contract_path)
    output_dir = resolve_path(output_dir)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"data root is not a directory: {data_root}")
    if output_dir == PROJECT_ROOT:
        raise ValueError("output directory must not be the project root")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is nonempty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    contract = load_contract(contract_path)
    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig")
    relative_paths = manifest_iq_paths(manifest)
    records = [
        inspect_iq_file(
            resolve_iq_file(data_root, relative_path),
            relative_path,
            file_index,
            contract,
        )
        for file_index, relative_path in enumerate(relative_paths, start=1)
    ]
    audit = pd.DataFrame(records, columns=LOCAL_COLUMNS)
    status, gates = gate_summary(audit, contract)
    shapes = aggregate_shapes(audit)
    summary = {
        "schema_version": 1,
        "probe_id": "field_iq_integrity_probe_v1",
        "status": status,
        "file_count": int(len(audit)),
        "passed_file_count": int(audit["file_pass"].astype(bool).sum()),
        "failed_file_count": int((~audit["file_pass"].astype(bool)).sum()),
        "total_size_bytes": int(audit["size_bytes"].sum()),
        "expected_shape_configured": (
            contract["requirements"]["expected_shape"] is not None
        ),
        "gate_status": gates,
        "cap_raw_complex_hv_candidate": bool(
            gates["mat_read_and_variable_pair"] == "PASS"
            and gates["complex_finite_and_component_variation"] == "PASS"
        ),
        "cal_complex_iq_candidate": bool(status == "PASS_FILE_CONTENT_ONLY"),
        "claim_boundaries": contract["claim_boundaries"],
        "input": {
            "manifest_name": manifest_path.name,
            "manifest_sha256": sha256_file(manifest_path),
            "contract_name": contract_path.name,
            "contract_sha256": sha256_file(contract_path),
            "probe_script_sha256": sha256_file(Path(__file__)),
            "absolute_paths_published": False,
            "file_names_or_hashes_published_in_summary": False,
        },
        "sharing_boundary": {
            "local_only_file": "file_audit_local.csv",
            "shareable_aggregate_files": [
                "shape_dtype_summary.csv",
                "summary.json",
                "README.md",
            ],
            "raw_iq_included": False,
        },
    }
    audit.to_csv(
        output_dir / "file_audit_local.csv", index=False, encoding="utf-8-sig"
    )
    shapes.to_csv(
        output_dir / "shape_dtype_summary.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "README.md").write_text(make_readme(summary), encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    summary = audit_field_iq(
        manifest_path=args.manifest,
        data_root=args.data_root,
        contract_path=args.contract,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print("Field IQ integrity probe: COMPLETE")
    print(f"status={summary['status']}")
    print(f"files={summary['file_count']}")
    print(f"cap_raw_complex_hv_candidate={summary['cap_raw_complex_hv_candidate']}")
    print(f"cal_complex_iq_candidate={summary['cal_complex_iq_candidate']}")
    return 0 if summary["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
