#!/usr/bin/env python3
"""Synthetic/single-record processing contract for LSS-FMCWR-2.0.

This module deliberately does not know how to open RAR/MAT archives.  It accepts
one already-decoded ``echoes.channelA`` matrix in the caller-supplied processing
orientation ``[slow_time_index, fast_time_index]`` and returns normalized
bin-domain spectra.  The row/column names are processing indices only; no
physical timing or velocity is inferred.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/lss_fmcwr_normalized_processing_contract_v1.json"
PACKAGED_CONFIG = (
    PROJECT_ROOT / "assets/contracts/lss_fmcwr_normalized_processing_contract_v1.json"
)
if not DEFAULT_CONFIG.is_file():
    # The sanitized share package stores contracts under assets/contracts.
    # Keep the same script runnable both from the repository and after unzip.
    DEFAULT_CONFIG = (
        PROJECT_ROOT / "assets/contracts/lss_fmcwr_normalized_processing_contract_v1.json"
    )
CONTRACT_VERSION = "v1"
EPS = 1e-12


class ProcessingContractError(ValueError):
    """Raised when a record violates the normalized processing contract."""


def _json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_contract(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load and validate the frozen JSON contract."""

    contract_path = Path(path).expanduser()
    if not contract_path.is_absolute():
        contract_path = PROJECT_ROOT / contract_path
    if (
        contract_path == DEFAULT_CONFIG
        and not contract_path.is_file()
        and PACKAGED_CONFIG.is_file()
    ):
        # The team share package places frozen contracts under assets/contracts.
        contract_path = PACKAGED_CONFIG
    try:
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProcessingContractError(f"unable to load contract: {contract_path}") from error
    _validate_contract(payload)
    return payload


def _validate_contract(contract: Mapping[str, Any]) -> None:
    required = {
        "schema_version", "contract_id", "input", "cli_npy", "guards", "normalization",
        "fast_time_fft", "slow_time_stft", "output", "claim_gates",
    }
    missing = required - set(contract)
    if missing:
        raise ProcessingContractError(f"contract missing fields: {sorted(missing)}")
    if int(contract["schema_version"]) != 1:
        raise ProcessingContractError("only schema_version=1 is supported")
    if contract["input"].get("axis_order") != ["slow_time_index", "fast_time_index"]:
        raise ProcessingContractError("input axis_order must be slow_time_index/fast_time_index")
    cli_npy = contract["cli_npy"]
    if cli_npy.get("axis_order_argument_required") is not True:
        raise ProcessingContractError("CLI NPY axis-order declaration must remain required")
    if cli_npy.get("axis_order_choices") != ["slow-fast", "fast-slow"]:
        raise ProcessingContractError("CLI NPY axis-order choices must be slow-fast/fast-slow")
    if cli_npy.get("fast_slow_transform") != "transpose_before_library_call":
        raise ProcessingContractError("CLI fast-slow input must be transposed before processing")
    if cli_npy.get("library_axis_order") != "slow-fast":
        raise ProcessingContractError("library axis order must remain slow-fast")
    guards = contract["guards"]
    for name in ("max_elements", "max_dimension", "max_abs_value"):
        if float(guards.get(name, 0)) <= 0:
            raise ProcessingContractError(f"guard {name} must be positive")
    normalization = contract["normalization"]
    percentile = float(normalization.get("amplitude_percentile", -1))
    if not 0.0 < percentile <= 100.0:
        raise ProcessingContractError("amplitude_percentile must be in (0, 100]")
    if float(normalization.get("rms_epsilon", 0)) <= 0:
        raise ProcessingContractError("rms_epsilon must be positive")
    stft = contract["slow_time_stft"]
    if int(stft.get("window_length", 0)) < 2:
        raise ProcessingContractError("slow STFT window_length must be at least two")
    if int(stft.get("hop_length", 0)) < 1:
        raise ProcessingContractError("slow STFT hop_length must be positive")
    if int(stft.get("nfft", 0)) < int(stft["window_length"]):
        raise ProcessingContractError("slow STFT nfft must be >= window_length")
    gates = contract["claim_gates"]
    for name in ("model_training_allowed", "physical_axis", "performance_reporting_allowed"):
        if gates.get(name) is not False:
            raise ProcessingContractError(f"claim gate {name} must remain false")
    if int(guards.get("max_output_elements_per_branch", 0)) <= 0:
        raise ProcessingContractError("max_output_elements_per_branch must be positive")


def _contract_from_arg(config: Mapping[str, Any] | str | Path | None) -> dict[str, Any]:
    if config is None:
        return load_contract()
    if isinstance(config, (str, Path)):
        return load_contract(config)
    payload = copy.deepcopy(dict(config))
    _validate_contract(payload)
    return payload


def _window(name: str, length: int) -> np.ndarray:
    if name == "hann":
        # A symmetric Hann window has no nonzero samples at length two.  Use the
        # rectangular limiting case so a valid one/two-sample input is not
        # silently erased; lengths above two retain the established behavior.
        if length <= 2:
            return np.ones(length, dtype=np.float64)
        # np.hanning is available without scipy and has the expected symmetric
        # endpoint convention for this smoke contract.
        return np.hanning(length).astype(np.float64)
    if name == "rectangular":
        return np.ones(length, dtype=np.float64)
    raise ProcessingContractError(f"unsupported window: {name}")


def _validate_record(
    channel_a: Any,
    contract: Mapping[str, Any],
    *,
    band: str | None,
) -> np.ndarray:
    try:
        array = np.asarray(channel_a)
    except Exception as error:  # pragma: no cover - numpy-specific conversion errors
        raise ProcessingContractError("channelA must be a numeric 2-D array") from error
    if array.ndim != 2:
        raise ProcessingContractError("channelA must be a two-dimensional array")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ProcessingContractError("channelA must be non-empty")
    if np.issubdtype(array.dtype, np.bool_) or not np.issubdtype(array.dtype, np.number):
        raise ProcessingContractError("channelA must have a numeric dtype")
    guards = contract["guards"]
    elements = int(array.size)
    if elements > int(guards["max_elements"]):
        raise ProcessingContractError(
            f"channelA has {elements} elements; max_elements is {guards['max_elements']}"
        )
    if max(array.shape) > int(guards["max_dimension"]):
        raise ProcessingContractError(
            f"channelA dimension exceeds max_dimension={guards['max_dimension']}"
        )
    # Convert before finite/absolute checks so integer and float inputs follow
    # one deterministic path, while preserving the real-versus-complex fact.
    is_complex = np.iscomplexobj(array)
    converted = np.asarray(array, dtype=np.complex128 if is_complex else np.float64)
    if not np.isfinite(converted).all():
        raise ProcessingContractError("channelA contains NaN/Inf")
    max_abs = float(np.max(np.abs(converted)))
    if max_abs > float(guards["max_abs_value"]):
        raise ProcessingContractError(
            f"channelA absolute value {max_abs:g} exceeds max_abs_value={guards['max_abs_value']}"
        )
    if band is not None:
        normalized_band = str(band).upper()
        if normalized_band not in {"K", "L"}:
            raise ProcessingContractError("band must be K, L, or omitted")
        if normalized_band == "K" and not is_complex:
            raise ProcessingContractError("K contract input must be complex single-channel data")
        if normalized_band == "L" and is_complex:
            raise ProcessingContractError("L contract input must remain real single-channel data")
    return converted


def _channel_b_status(channel_b: Any, contract: Mapping[str, Any]) -> dict[str, Any]:
    if channel_b is None:
        return {
            "supplied": False,
            "empty": False,
            "interpreted_as_h_v": False,
            "h_v_polarimetry_available": False,
            "status": "not_supplied",
        }
    try:
        array = np.asarray(channel_b)
    except Exception as error:  # pragma: no cover - numpy-specific conversion errors
        raise ProcessingContractError("channelB must be a numeric array") from error
    if array.size == 0:
        return {
            "supplied": True,
            "empty": True,
            "shape": list(array.shape),
            "interpreted_as_h_v": False,
            "h_v_polarimetry_available": False,
            "status": "empty_not_hv",
        }
    if array.ndim != 2:
        raise ProcessingContractError("non-empty channelB must be two-dimensional")
    if not np.issubdtype(array.dtype, np.number):
        raise ProcessingContractError("non-empty channelB must be numeric")
    try:
        finite = np.isfinite(array).all()
    except TypeError as error:
        raise ProcessingContractError("channelB must be numeric") from error
    if not finite:
        raise ProcessingContractError("channelB contains NaN/Inf")
    # A second numeric array is deliberately not fused or assigned H/V meaning;
    # channel identity and calibration are unavailable in the release audit.
    return {
        "supplied": True,
        "empty": False,
        "shape": list(array.shape),
        "interpreted_as_h_v": False,
        "h_v_polarimetry_available": False,
        "status": "present_uninterpreted",
    }


def _scales(record: np.ndarray, contract: Mapping[str, Any]) -> tuple[float, float, bool, bool]:
    magnitude = np.abs(record)
    normalization = contract["normalization"]
    epsilon = float(normalization["rms_epsilon"])
    rms_raw = float(np.sqrt(np.mean(magnitude**2)))
    percentile_raw = float(np.percentile(magnitude, float(normalization["amplitude_percentile"])))
    return (
        max(rms_raw, epsilon),
        max(percentile_raw, epsilon),
        rms_raw <= epsilon,
        percentile_raw <= epsilon,
    )


def normalize_record(
    channel_a: Any,
    contract: Mapping[str, Any] | None = None,
    *,
    band: str | None = None,
) -> dict[str, Any]:
    """Return both per-record RMS and percentile normalized matrices."""

    frozen = _contract_from_arg(contract)
    record = _validate_record(channel_a, frozen, band=band)
    rms, percentile, rms_degenerate, percentile_degenerate = _scales(record, frozen)
    return {
        "record": record,
        "rms": record / rms,
        "percentile": record / percentile,
        "rms_scale": rms,
        "percentile_scale": percentile,
        "rms_degenerate": rms_degenerate,
        "percentile_degenerate": percentile_degenerate,
    }


def _fast_fft(
    record: np.ndarray,
    contract: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    settings = contract["fast_time_fft"]
    nfft_value = settings.get("nfft")
    nfft = record.shape[1] if nfft_value in (None, 0) else int(nfft_value)
    if nfft < record.shape[1]:
        raise ProcessingContractError("fast FFT nfft must be >= fast_time length")
    taper = _window(str(settings.get("window", "hann")), record.shape[1])
    spectrum = np.fft.fft(record * taper[None, :], n=nfft, axis=1)
    if bool(settings.get("fftshift", True)):
        spectrum = np.fft.fftshift(spectrum, axes=1)
    power = np.abs(spectrum) ** 2
    return spectrum, power


def compute_fast_time_fft(
    normalized_record: np.ndarray,
    contract: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute full fast-time FFT and its normalized bin axis."""

    frozen = _contract_from_arg(contract)
    record = np.asarray(normalized_record)
    if record.ndim != 2:
        raise ProcessingContractError("normalized_record must be two-dimensional")
    if record.size == 0:
        raise ProcessingContractError("normalized_record must be non-empty")
    if record.size > int(frozen["guards"]["max_elements"]):
        raise ProcessingContractError("normalized_record exceeds max_elements")
    if not np.issubdtype(record.dtype, np.number):
        raise ProcessingContractError("normalized_record must be numeric")
    if not np.isfinite(record).all():
        raise ProcessingContractError("normalized_record contains NaN/Inf")
    spectrum, power = _fast_fft(record, frozen)
    nfft = spectrum.shape[1]
    axis = np.fft.fftfreq(nfft, d=1.0)
    if bool(frozen["fast_time_fft"].get("fftshift", True)):
        axis = np.fft.fftshift(axis)
    return spectrum, power, axis


def _slow_stft(
    fast_spectrum: np.ndarray,
    contract: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    settings = contract["slow_time_stft"]
    window_length = int(settings["window_length"])
    hop_length = int(settings["hop_length"])
    nfft = int(settings["nfft"])
    if nfft < window_length:
        raise ProcessingContractError("slow STFT nfft must be >= window_length")
    slow_count = fast_spectrum.shape[0]
    if slow_count < window_length:
        starts = [0]
    else:
        starts = list(range(0, slow_count - window_length + 1, hop_length))
    output_elements = len(starts) * nfft * fast_spectrum.shape[1]
    max_output = int(contract["guards"]["max_output_elements_per_branch"])
    if output_elements > max_output:
        raise ProcessingContractError(
            "slow STFT output would have "
            f"{output_elements} elements per branch; max_output_elements_per_branch is {max_output}"
        )
    frames = np.zeros((len(starts), window_length, fast_spectrum.shape[1]), dtype=fast_spectrum.dtype)
    for frame_index, start in enumerate(starts):
        stop = min(start + window_length, slow_count)
        frames[frame_index, : stop - start, :] = fast_spectrum[start:stop, :]
    window_name = str(settings.get("window", "hann"))
    taper = _window(window_name, window_length)
    if window_name == "hann" and slow_count <= 2:
        # The sole short frame is left-aligned by contract.  Its occupied bins
        # otherwise coincide with Hann's zero/near-zero leading coefficients.
        taper = taper.copy()
        taper[:slow_count] = 1.0
    transformed = np.fft.fft(frames * taper[None, :, None], n=nfft, axis=1)
    if bool(settings.get("fftshift", True)):
        transformed = np.fft.fftshift(transformed, axes=1)
    power = np.abs(transformed) ** 2
    slow_axis = np.fft.fftfreq(nfft, d=1.0)
    if bool(settings.get("fftshift", True)):
        slow_axis = np.fft.fftshift(slow_axis)
    frame_axis = np.arange(len(starts), dtype=np.int64)
    frame_start_axis = np.asarray(starts, dtype=np.int64)
    return power, slow_axis, frame_axis, frame_start_axis


def compute_slow_time_stft(
    fast_spectrum: np.ndarray,
    contract: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute slow-time STFT for every fast-time bin."""

    frozen = _contract_from_arg(contract)
    array = np.asarray(fast_spectrum)
    if array.ndim != 2:
        raise ProcessingContractError("fast_spectrum must be two-dimensional")
    if not np.isfinite(array).all():
        raise ProcessingContractError("fast_spectrum contains NaN/Inf")
    return _slow_stft(array, frozen)


def _branch(
    normalized_record: np.ndarray,
    contract: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    fast_complex, fast_power, fast_axis = compute_fast_time_fft(normalized_record, contract)
    slow_power, slow_axis, frame_axis, frame_start_axis = compute_slow_time_stft(
        fast_complex, contract
    )
    # Summing across fast bins is an intentionally descriptive reduction, not a
    # detector score or a performance estimate.
    slow_summary = np.sum(slow_power, axis=2)
    return {
        "fast_complex": fast_complex,
        "fast_power": fast_power,
        "fast_axis": fast_axis,
        "slow_power": slow_power,
        "slow_axis": slow_axis,
        "slow_summary": slow_summary,
        "frame_axis": frame_axis,
        "frame_start_axis": frame_start_axis,
    }


def process_record(
    channel_a: Any,
    contract: Mapping[str, Any] | str | Path | None = None,
    *,
    band: str | None = None,
    channel_b: Any | None = None,
    record_id: str | None = None,
) -> dict[str, Any]:
    """Process one in-memory channelA record under the frozen contract.

    The returned mapping contains NumPy arrays and a JSON-compatible
    ``metadata`` mapping.  No labels, sessions, archives, or model calls are
    involved.
    """

    frozen = _contract_from_arg(contract)
    normalized = normalize_record(channel_a, frozen, band=band)
    record = normalized["record"]
    b_status = _channel_b_status(channel_b, frozen)
    rms_branch = _branch(normalized["rms"], frozen)
    percentile_branch = _branch(normalized["percentile"], frozen)

    slow_count, fast_count = record.shape
    metadata: dict[str, Any] = {
        "contract_id": frozen["contract_id"],
        "contract_version": CONTRACT_VERSION,
        "schema_version": int(frozen["schema_version"]),
        "contract_sha256": _json_hash(frozen),
        "record_id": record_id,
        "band": None if band is None else str(band).upper(),
        "input_shape": [int(slow_count), int(fast_count)],
        "input_dtype": str(np.asarray(channel_a).dtype),
        "input_representation": "complex_single_channel" if np.iscomplexobj(record) else "real_single_channel",
        "axis_order": ["slow_time_index", "fast_time_index"],
        "axis_semantics": {
            "slow_time_index": "row index used for processing only; physical timing is unknown",
            "fast_time_index": "column index used for processing only; physical sampling is unknown",
        },
        "channelB": b_status,
        "normalization": {
            "scope": "one_record",
            "amplitude_percentile": float(frozen["normalization"]["amplitude_percentile"]),
            "rms_scale": float(normalized["rms_scale"]),
            "percentile_scale": float(normalized["percentile_scale"]),
            "rms_degenerate": bool(normalized["rms_degenerate"]),
            "percentile_degenerate": bool(normalized["percentile_degenerate"]),
        },
        "fast_time_fft": {
            "axis": "fast_time_index",
            "nfft": int(rms_branch["fast_power"].shape[1]),
            "window": frozen["fast_time_fft"]["window"],
            "fftshift": bool(frozen["fast_time_fft"].get("fftshift", True)),
            "axis_unit": "normalized_cycles_per_fast_sample",
        },
        "slow_time_stft": {
            "axis": "slow_time_index",
            "window_length": int(frozen["slow_time_stft"]["window_length"]),
            "hop_length": int(frozen["slow_time_stft"]["hop_length"]),
            "nfft": int(frozen["slow_time_stft"]["nfft"]),
            "frame_count": int(rms_branch["slow_power"].shape[0]),
            "axis_unit": "normalized_cycles_per_slow_sample",
        },
        "physical_axis": False,
        "physical_doppler_hz_axis_available": False,
        "physical_frequency_hz_allowed": False,
        "velocity_m_per_s_allowed": False,
        "h_v_polarimetry_available": False,
        "model_training_allowed": False,
        "performance_reporting_allowed": False,
        "random_window_split_allowed": False,
    }
    return {
        "normalized_channel_rms": normalized["rms"],
        "normalized_channel_percentile": normalized["percentile"],
        "normalized_channel": normalized["rms"],
        "fast_time_spectrum_rms": rms_branch["fast_power"],
        "fast_time_spectrum_percentile": percentile_branch["fast_power"],
        "slow_time_spectrogram_rms": rms_branch["slow_power"],
        "slow_time_spectrogram_percentile": percentile_branch["slow_power"],
        "slow_time_spectrum_rms": rms_branch["slow_summary"],
        "slow_time_spectrum_percentile": percentile_branch["slow_summary"],
        "slow_index_axis": np.arange(slow_count, dtype=np.int64),
        "fast_index_axis": np.arange(fast_count, dtype=np.int64),
        "fast_bin_axis": rms_branch["fast_axis"],
        "slow_frame_index_axis": rms_branch["frame_axis"],
        "slow_frame_start_index_axis": rms_branch["frame_start_axis"],
        "slow_bin_axis": rms_branch["slow_axis"],
        # Concise aliases make the contract convenient for plotting while the
        # explicit suffixes above preserve which normalization branch was used.
        "fast_time_spectrum": rms_branch["fast_power"],
        "slow_time_spectrogram": rms_branch["slow_power"],
        "slow_time_spectrum": rms_branch["slow_summary"],
        "metadata": metadata,
    }


def _demo_record(band: str, slow_count: int, fast_count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    slow = np.arange(slow_count, dtype=np.float64)[:, None]
    fast = np.arange(fast_count, dtype=np.float64)[None, :]
    phase = 2.0 * np.pi * (0.13 * fast + 0.07 * slow + 0.0002 * slow * fast)
    if band == "K":
        signal = np.exp(1j * phase)
        noise = 0.05 * (rng.normal(size=signal.shape) + 1j * rng.normal(size=signal.shape))
        return signal + noise
    signal = np.cos(phase)
    return signal + 0.05 * rng.normal(size=signal.shape)


def _json_safe_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(metadata, ensure_ascii=False, default=str))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--smoke", action="store_true", help="process an in-memory synthetic record")
    parser.add_argument("--input-npy", type=Path, help="optional .npy channelA matrix; archives are not accepted")
    parser.add_argument(
        "--input-axis-order",
        choices=("slow-fast", "fast-slow"),
        help="required with --input-npy; fast-slow is transposed before processing",
    )
    parser.add_argument("--band", choices=("K", "L"), default="K")
    parser.add_argument("--slow-count", type=int, default=96)
    parser.add_argument("--fast-count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--record-id", default="synthetic_smoke_0001")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/lss_fmcwr_normalized_smoke"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.smoke and args.input_npy is not None:
        raise SystemExit("choose --smoke or --input-npy, not both")
    if not args.smoke and args.input_npy is None:
        raise SystemExit("--smoke or --input-npy is required")
    if args.input_npy is not None and args.input_axis_order is None:
        raise SystemExit("--input-axis-order is required with --input-npy")
    if args.smoke and args.input_axis_order is not None:
        raise SystemExit("--input-axis-order is only valid with --input-npy")
    if args.input_npy is not None and args.input_npy.suffix.lower() != ".npy":
        raise SystemExit("--input-npy accepts only a .npy matrix; RAR/MAT readers are intentionally absent")
    if args.slow_count < 1 or args.fast_count < 1:
        raise SystemExit("slow/fast counts must be positive")
    if args.smoke:
        record = _demo_record(args.band, args.slow_count, args.fast_count, args.seed)
        cli_source_shape = list(record.shape)
        cli_input_axis_order = "slow-fast"
        cli_transposed = False
    else:
        loaded_record = np.load(args.input_npy, allow_pickle=False)
        if loaded_record.ndim != 2:
            raise ProcessingContractError("--input-npy must contain a two-dimensional array")
        cli_source_shape = [int(value) for value in loaded_record.shape]
        cli_input_axis_order = str(args.input_axis_order)
        cli_transposed = cli_input_axis_order == "fast-slow"
        record = loaded_record.T if cli_transposed else loaded_record
    result = process_record(record, args.config, band=args.band, record_id=args.record_id)
    result["metadata"]["cli_input"] = {
        "source": "synthetic_smoke" if args.smoke else "npy",
        "declared_axis_order": cli_input_axis_order,
        "source_shape": cli_source_shape,
        "transposed_to_contract_axis_order": cli_transposed,
        "contract_axis_order": "slow-fast",
    }
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {name: value for name, value in result.items() if isinstance(value, np.ndarray)}
    np.savez_compressed(output_dir / "normalized_processing.npz", **arrays)
    metadata = _json_safe_metadata(result["metadata"])
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "COMPLETE_SYNTHETIC_SMOKE" if args.smoke else "COMPLETE_SINGLE_RECORD", **metadata}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
