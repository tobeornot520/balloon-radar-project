#!/usr/bin/env python3
"""HDF5-compatible read-only entry point for the LSS-FMCWR-2.0 audit.

The preserved ``audit_lss_fmcwr_2_v1.py`` development draft is intentionally
left unchanged. This adapter supplies the missing MATLAB 7.3 reader and
delegates all archive inventory, transaction, grouping, and report logic to
that draft.  It is the reproducible entry point for the mixed v5/v7.3 release.
"""
from __future__ import annotations

from io import BytesIO
import os
import subprocess

import h5py
import numpy as np

try:  # Support both ``python -m scripts...`` and direct script execution.
    from . import audit_lss_fmcwr_2_v1 as _base
except ImportError:  # pragma: no cover - exercised by the CLI, not imports.
    import audit_lss_fmcwr_2_v1 as _base


HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
_V5_LOAD_ECHOES = _base.load_echoes


def _is_matlab_hdf5(payload: bytes) -> bool:
    """Return whether a MATLAB 7.3 payload has the HDF5 signature."""

    return len(payload) >= 520 and payload[512:520] == HDF5_SIGNATURE


def _matlab_empty(dataset: h5py.Dataset) -> bool:
    value = dataset.attrs.get("MATLAB_empty", 0)
    array = np.asarray(value)
    return bool(array.size and np.all(array != 0))


def _hdf5_channel(
    dataset: h5py.Dataset, *, member_path: str, field: str
) -> np.ndarray:
    """Decode one MATLAB 7.3 numeric dataset into MATLAB logical orientation."""

    if _matlab_empty(dataset):
        # MATLAB's canonical empty reference is stored as a uint64 vector in
        # HDF5, whereas scipy.io.loadmat exposes the same value as 0-by-0.
        return np.empty((0, 0), dtype=np.uint8)

    raw = np.asarray(dataset)
    if raw.ndim != 2:
        raise ValueError(
            f"{member_path}: HDF5 {field} must be a non-empty 2-D dataset"
        )

    if raw.dtype.names is not None:
        if set(raw.dtype.names) != {"real", "imag"}:
            raise ValueError(
                f"{member_path}: HDF5 {field} has an unsupported compound dtype"
            )
        real = np.asarray(raw["real"])
        imag = np.asarray(raw["imag"])
        if not np.issubdtype(real.dtype, np.number) or not np.issubdtype(
            imag.dtype, np.number
        ):
            raise ValueError(
                f"{member_path}: HDF5 {field} real/imag fields must be numeric"
            )
        logical = real + 1j * imag
    else:
        if not np.issubdtype(raw.dtype, np.number):
            raise ValueError(f"{member_path}: HDF5 {field} must be numeric")
        logical = raw

    # MATLAB stores array dimensions reversed in the HDF5 representation.
    logical = np.ascontiguousarray(logical.T)
    if not np.isfinite(logical).all():
        raise ValueError(f"{member_path}: HDF5 {field} contains NaN/Inf")
    return logical


def _load_hdf5_echoes(payload: bytes, *, member_path: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        with h5py.File(BytesIO(payload), "r") as handle:
            public = {name for name in handle.keys() if name != "#refs#"}
            if public != {"echoes"}:
                raise ValueError(
                    f"{member_path}: expected only public HDF5 group echoes"
                )
            echoes = handle["echoes"]
            if not isinstance(echoes, h5py.Group):
                raise ValueError(f"{member_path}: echoes must be an HDF5 group")
            if set(echoes.keys()) != {"channelA", "channelB"}:
                raise ValueError(
                    f"{member_path}: echoes must contain channelA and channelB"
                )
            channel_a = _hdf5_channel(
                echoes["channelA"], member_path=member_path, field="channelA"
            )
            channel_b = _hdf5_channel(
                echoes["channelB"], member_path=member_path, field="channelB"
            )
    except (OSError, KeyError, TypeError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith(member_path):
            raise
        raise ValueError(f"{member_path}: unreadable MATLAB 7.3 file") from error

    if channel_a.ndim != 2 or channel_a.size == 0:
        raise ValueError(f"{member_path}: channelA must be a non-empty 2-D array")
    return channel_a, channel_b


def load_echoes(payload: bytes, *, member_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load either MATLAB v5 or v7.3 ``echoes.channelA/B`` data."""

    if _is_matlab_hdf5(payload):
        return _load_hdf5_echoes(payload, member_path=member_path)
    return _V5_LOAD_ECHOES(payload, member_path=member_path)


def _harden_unrar_args(args: list[str]) -> list[str]:
    """Disable config/list-file expansion and terminate option parsing."""

    if len(args) < 3 or args[1] not in {"lt", "t", "p"}:
        raise ValueError("unexpected unrar command")
    try:
        first_positional = next(
            index for index in range(2, len(args)) if not args[index].startswith("-")
        )
    except StopIteration as error:
        raise ValueError("unrar command has no archive argument") from error
    return (
        args[:2]
        + ["-cfg-", "-@-"]
        + args[2:first_positional]
        + ["--"]
        + args[first_positional:]
    )


def _default_runner(args: list[str]) -> subprocess.CompletedProcess[bytes]:
    environment = os.environ.copy()
    environment["LC_ALL"] = "C.UTF-8"
    environment["LANG"] = "C.UTF-8"
    return subprocess.run(
        _harden_unrar_args(args),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        shell=False,
        env=environment,
    )


# The delegated audit functions resolve ``load_echoes`` in their own module's
# globals, so replace only that function after importing the unchanged draft.
_base.load_echoes = load_echoes
_base._default_runner = _default_runner

audit_dataset = _base.audit_dataset
main = _base.main


if __name__ == "__main__":  # pragma: no cover - covered by CLI smoke tests.
    raise SystemExit(main())
