from __future__ import annotations

import copy
import json
import sys

import numpy as np
import pytest

import scripts.process_lss_fmcwr_normalized_v1 as processing
from scripts.process_lss_fmcwr_normalized_v1 import (
    ProcessingContractError,
    load_contract,
    process_record,
)


@pytest.fixture()
def contract() -> dict:
    return load_contract()


def test_complex_k_record_has_explicit_index_axes_and_claim_gates(contract: dict) -> None:
    rng = np.random.default_rng(7)
    record = rng.normal(size=(96, 128)) + 1j * rng.normal(size=(96, 128))
    output = process_record(
        record,
        contract,
        band="K",
        channel_b=np.empty((0, 0), dtype=np.float64),
        record_id="synthetic-k",
    )

    assert output["normalized_channel_rms"].shape == record.shape
    assert output["normalized_channel_percentile"].shape == record.shape
    assert output["fast_time_spectrum"].shape == (96, 128)
    assert output["slow_time_spectrogram"].shape == (2, 64, 128)
    assert output["slow_time_spectrum"].shape == (2, 64)
    np.testing.assert_array_equal(output["slow_index_axis"], np.arange(96))
    np.testing.assert_array_equal(output["fast_index_axis"], np.arange(128))
    np.testing.assert_allclose(
        output["fast_bin_axis"], np.fft.fftshift(np.fft.fftfreq(128)), atol=1e-15
    )
    assert output["metadata"]["axis_order"] == ["slow_time_index", "fast_time_index"]
    assert output["metadata"]["channelB"]["status"] == "empty_not_hv"
    assert output["metadata"]["channelB"]["interpreted_as_h_v"] is False
    assert output["metadata"]["h_v_polarimetry_available"] is False
    assert output["metadata"]["physical_axis"] is False
    assert output["metadata"]["physical_doppler_hz_axis_available"] is False
    assert output["metadata"]["model_training_allowed"] is False
    assert len(output["metadata"]["contract_sha256"]) == 64


def test_real_l_record_remains_single_channel_and_uses_both_normalizations(contract: dict) -> None:
    record = np.ones((8, 16), dtype=np.float32) * 3.0
    output = process_record(record, contract, band="L")

    assert output["metadata"]["input_representation"] == "real_single_channel"
    assert output["metadata"]["normalization"]["rms_scale"] == pytest.approx(3.0)
    assert output["metadata"]["normalization"]["percentile_scale"] == pytest.approx(3.0)
    np.testing.assert_allclose(output["normalized_channel_rms"], 1.0)
    np.testing.assert_allclose(output["normalized_channel_percentile"], 1.0)
    assert output["fast_time_spectrum_rms"].shape == (8, 16)
    assert output["slow_time_spectrogram_percentile"].shape[0] == 1


def test_short_slow_record_is_zero_padded_to_one_stft_frame(contract: dict) -> None:
    record = np.ones((5, 16), dtype=np.float64)
    output = process_record(record, contract, band="L")

    assert output["slow_time_spectrogram"].shape == (1, 64, 16)
    np.testing.assert_array_equal(output["slow_frame_index_axis"], [0])
    np.testing.assert_array_equal(output["slow_frame_start_index_axis"], [0])


def test_one_two_sample_hann_edges_preserve_nonzero_input_energy(contract: dict) -> None:
    one_slow_sample = process_record(np.ones((1, 16)), contract, band="L")
    two_slow_samples = process_record(np.ones((2, 16)), contract, band="L")
    two_fast_samples = process_record(np.ones((4, 2)), contract, band="L")

    assert np.max(one_slow_sample["slow_time_spectrogram"]) > 0.0
    assert np.max(two_slow_samples["slow_time_spectrogram"]) > 0.0
    assert np.max(two_fast_samples["fast_time_spectrum"]) > 0.0
    assert np.max(two_fast_samples["slow_time_spectrogram"]) > 0.0
    np.testing.assert_array_equal(processing._window("hann", 2), np.ones(2))
    np.testing.assert_array_equal(processing._window("hann", 3), np.hanning(3))


@pytest.mark.parametrize(
    "bad_record, message",
    [
        (np.ones((4, 4), dtype=np.float64) * np.nan, "NaN/Inf"),
        (np.ones((4, 4), dtype=np.float64) * np.inf, "NaN/Inf"),
        (np.ones((4, 4), dtype=np.float64) * 1e13, "max_abs_value"),
        (np.ones((4, 4, 1), dtype=np.float64), "two-dimensional"),
        (np.ones((0, 4), dtype=np.float64), "non-empty"),
    ],
)
def test_invalid_inputs_are_rejected(contract: dict, bad_record: np.ndarray, message: str) -> None:
    with pytest.raises(ProcessingContractError, match=message):
        process_record(bad_record, contract, band="L")


def test_element_and_output_guards_prevent_memory_expansion(contract: dict) -> None:
    guarded = copy.deepcopy(contract)
    guarded["guards"]["max_elements"] = 32
    with pytest.raises(ProcessingContractError, match="max_elements"):
        process_record(np.ones((8, 8)), guarded, band="L")

    guarded = copy.deepcopy(contract)
    guarded["guards"]["max_output_elements_per_branch"] = 10
    with pytest.raises(ProcessingContractError, match="max_output_elements_per_branch"):
        process_record(np.ones((96, 16)), guarded, band="L")


def test_band_and_channel_b_semantics_are_explicit(contract: dict) -> None:
    with pytest.raises(ProcessingContractError, match="K contract input must be complex"):
        process_record(np.ones((8, 16)), contract, band="K")
    with pytest.raises(ProcessingContractError, match="L contract input must remain real"):
        process_record(np.ones((8, 16), dtype=np.complex128), contract, band="L")

    output = process_record(
        np.ones((8, 16), dtype=np.float64),
        contract,
        band="L",
        channel_b=np.ones((8, 16), dtype=np.float64),
    )
    assert output["metadata"]["channelB"]["status"] == "present_uninterpreted"
    assert output["metadata"]["channelB"]["interpreted_as_h_v"] is False
    assert output["metadata"]["h_v_polarimetry_available"] is False


def test_unshifted_contract_keeps_frequency_axes_in_fft_order(contract: dict) -> None:
    unshifted = copy.deepcopy(contract)
    unshifted["fast_time_fft"]["fftshift"] = False
    unshifted["slow_time_stft"]["fftshift"] = False

    output = process_record(np.ones((8, 16)), unshifted, band="L")

    np.testing.assert_allclose(output["fast_bin_axis"], np.fft.fftfreq(16))
    np.testing.assert_allclose(output["slow_bin_axis"], np.fft.fftfreq(64))


def test_share_package_contract_location_is_supported(
    tmp_path, monkeypatch: pytest.MonkeyPatch, contract: dict
) -> None:
    repository_location = tmp_path / "configs/missing.json"
    packaged_location = (
        tmp_path / "assets/contracts/lss_fmcwr_normalized_processing_contract_v1.json"
    )
    packaged_location.parent.mkdir(parents=True)
    packaged_location.write_text(
        json.dumps(contract, ensure_ascii=False), encoding="utf-8"
    )
    monkeypatch.setattr(processing, "DEFAULT_CONFIG", repository_location)
    monkeypatch.setattr(processing, "PACKAGED_CONFIG", packaged_location)

    assert processing.load_contract(repository_location)["contract_id"] == contract["contract_id"]


def test_cli_npy_requires_explicit_axis_order(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "record.npy"
    np.save(input_path, np.ones((8, 16), dtype=np.float64))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "process_lss_fmcwr_normalized_v1.py",
            "--input-npy",
            str(input_path),
            "--band",
            "L",
        ],
    )

    with pytest.raises(SystemExit, match="--input-axis-order is required"):
        processing.main()


@pytest.mark.parametrize(
    "axis_order, source_shape, expected_shape, expected_transposed",
    [
        ("slow-fast", (8, 16), (8, 16), False),
        ("fast-slow", (16, 8), (8, 16), True),
    ],
)
def test_cli_npy_axis_order_is_applied_and_recorded(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    axis_order: str,
    source_shape: tuple[int, int],
    expected_shape: tuple[int, int],
    expected_transposed: bool,
) -> None:
    input_path = tmp_path / f"record-{axis_order}.npy"
    output_dir = tmp_path / f"output-{axis_order}"
    np.save(input_path, np.ones(source_shape, dtype=np.float64))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "process_lss_fmcwr_normalized_v1.py",
            "--input-npy",
            str(input_path),
            "--input-axis-order",
            axis_order,
            "--band",
            "L",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert processing.main() == 0
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    with np.load(output_dir / "normalized_processing.npz", allow_pickle=False) as arrays:
        assert arrays["normalized_channel"].shape == expected_shape
    assert metadata["input_shape"] == list(expected_shape)
    assert metadata["cli_input"] == {
        "source": "npy",
        "declared_axis_order": axis_order,
        "source_shape": list(source_shape),
        "transposed_to_contract_axis_order": expected_transposed,
        "contract_axis_order": "slow-fast",
    }
