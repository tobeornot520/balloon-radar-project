from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat

from scripts.audit_detection_acquisition_order import (
    inspect_manifest,
    inspect_mat_file,
    parse_mat_header_created_at,
    parse_sample_timestamp,
)


def write_iq_mat(path: Path, *, include_timestamp: bool = False) -> None:
    payload: dict[str, object] = {
        "local_data_H": np.zeros((2, 2), dtype=np.float32),
        "local_data_V": np.zeros((2, 2), dtype=np.float32),
    }
    if include_timestamp:
        payload["acquisition_time"] = "2026-02-04T10:07:39.123"
    savemat(path, payload)


def test_timestamp_parsers_distinguish_sample_and_mat_header() -> None:
    assert parse_sample_timestamp("20260204_100739_beam1_az010") == datetime(
        2026, 2, 4, 10, 7, 39
    )
    header = (
        b"MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: "
        b"Thu Mar 26 16:26:39 2026"
    ).ljust(128, b" ")
    assert parse_mat_header_created_at(header) == datetime(2026, 3, 26, 16, 26, 39)


def test_mat_inspection_detects_timestamp_like_variable(tmp_path: Path) -> None:
    path = tmp_path / "sample.mat"
    write_iq_mat(path, include_timestamp=True)
    inspection = inspect_mat_file(path)
    assert set(inspection.variable_names) == {
        "local_data_H",
        "local_data_V",
        "acquisition_time",
    }
    assert inspection.timestamp_variable_names == ("acquisition_time",)


def test_manifest_audit_keeps_group_timestamp_out_of_within_group_order(
    tmp_path: Path,
) -> None:
    paths = [tmp_path / "a.mat", tmp_path / "b.mat"]
    for path in paths:
        write_iq_mat(path)
    frame = pd.DataFrame(
        {
            "source_file": ["20260204_100739", "20260204_100739"],
            "sample_id": [
                "20260204_100739_beam1_az010",
                "20260204_100739_beam1_az019",
            ],
            "target_present": [0, 0],
            "beam_layer": [1, 1],
            "azimuth_deg": [10.0, 19.0],
            "mat_path": [str(path) for path in paths],
        }
    )
    groups, sources = inspect_manifest(frame)
    assert groups.loc[0, "sample_timestamp_unique_count"] == 1
    assert bool(groups.loc[0, "inferred_order_key_unique"])
    assert not bool(groups.loc[0, "verified_within_group_order_available"])
    sample_timestamp = sources.loc[
        sources["candidate_source"].eq("sample_id_timestamp")
    ].iloc[0]
    assert not bool(sample_timestamp["timestamp_verified"])
    assert not bool(sample_timestamp["within_group_order_verified"])
