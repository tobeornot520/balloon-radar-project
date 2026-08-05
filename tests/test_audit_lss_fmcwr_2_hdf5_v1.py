from __future__ import annotations

from io import BytesIO

import h5py
import numpy as np
import pytest
from scipy.io import savemat

from scripts.audit_lss_fmcwr_2_hdf5_v1 import _harden_unrar_args, load_echoes


def _v73_payload(
    *,
    logical: np.ndarray,
    empty_channel_b: bool = True,
    nonfinite: bool = False,
) -> bytes:
    """Build the small subset of MATLAB 7.3 HDF5 used by this release."""

    buffer = BytesIO()
    with h5py.File(buffer, "w", userblock_size=512) as handle:
        echoes = handle.create_group("echoes")
        echoes.attrs["MATLAB_class"] = np.bytes_(b"struct")
        raw = np.asarray(logical.T)
        if np.iscomplexobj(raw):
            compound = np.empty(raw.shape, dtype=[("real", "<f8"), ("imag", "<f8")])
            compound["real"] = raw.real
            compound["imag"] = raw.imag
            channel_a = compound
        else:
            channel_a = raw
        if nonfinite:
            channel_a = channel_a.copy()
            if channel_a.dtype.names:
                channel_a["real"][0, 0] = np.nan
            else:
                channel_a[0, 0] = np.nan
        echoes.create_dataset("channelA", data=channel_a)
        if empty_channel_b:
            channel_b = echoes.create_dataset(
                "channelB", data=np.array([0, 0], dtype=np.uint64)
            )
            channel_b.attrs["MATLAB_empty"] = np.uint8(1)
        else:
            echoes.create_dataset("channelB", data=np.asarray(raw))
    payload = bytearray(buffer.getvalue())
    payload[:128] = b"MATLAB 7.3 MAT-file, synthetic HDF5".ljust(128, b" ")
    return bytes(payload)


def _v5_payload(channel_a: np.ndarray) -> bytes:
    buffer = BytesIO()
    savemat(
        buffer,
        {
            "echoes": {
                "channelA": channel_a,
                "channelB": np.empty((0, 0), dtype=np.uint8),
            }
        },
    )
    return buffer.getvalue()


def test_v73_compound_complex_is_transposed_and_empty_b_is_canonical() -> None:
    logical = np.arange(12, dtype=np.float64).reshape(3, 4) + 1j
    channel_a, channel_b = load_echoes(
        _v73_payload(logical=logical), member_path="synthetic/complex.mat"
    )

    np.testing.assert_array_equal(channel_a, logical)
    assert channel_a.shape == (3, 4)
    assert np.iscomplexobj(channel_a)
    assert channel_b.shape == (0, 0)
    assert channel_b.dtype == np.dtype("uint8")


def test_v5_real_falls_back_to_existing_loader() -> None:
    logical = np.arange(20, dtype=np.float64).reshape(4, 5)
    channel_a, channel_b = load_echoes(
        _v5_payload(logical), member_path="synthetic/real.mat"
    )

    np.testing.assert_array_equal(channel_a, logical)
    assert channel_a.shape == (4, 5)
    assert not np.iscomplexobj(channel_a)
    assert channel_b.shape == (0, 0)


def test_v73_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="contains NaN/Inf"):
        load_echoes(
            _v73_payload(
                logical=np.ones((2, 3), dtype=np.complex128), nonfinite=True
            ),
            member_path="synthetic/nonfinite.mat",
        )


def test_unrar_commands_disable_config_and_list_file_expansion() -> None:
    assert _harden_unrar_args(
        ["/tool/unrar", "p", "-inul", "-p-", "/source/data.rar", "x.mat"]
    ) == [
        "/tool/unrar",
        "p",
        "-cfg-",
        "-@-",
        "-inul",
        "-p-",
        "--",
        "/source/data.rar",
        "x.mat",
    ]
