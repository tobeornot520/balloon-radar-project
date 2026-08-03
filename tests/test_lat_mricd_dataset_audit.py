from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from scripts.audit_lat_mricd_dataset_v1 import audit_dataset, reconstruct_narrow_iq


AGGREGATES = (
    ("HRRP/X波段/data_hrrp_X.mat", 2, 504),
    ("HRRP/Ku波段/data_hrrp_Ku.mat", 3, 504),
    ("Narrow/S波段/data_narrow_S.mat", 1, 1028),
    ("Narrow/X波段/data_narrow_X.mat", 2, 1028),
    ("Narrow/Ku波段/data_narrow_Ku.mat", 3, 1028),
)


def write_release(root: Path, *, invalid_width: bool = False) -> None:
    for index, (relative, band_code, width) in enumerate(AGGREGATES):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        actual_width = width - 1 if invalid_width and index == 0 else width
        matrix = np.ones((6, actual_width), dtype=np.float64)
        matrix[:, :4] = np.asarray(
            [
                [band_code, 1, 1, 1],
                [band_code, 1, 1, 2],
                [band_code, 1, 1, 3],
                [band_code, 3, 9, 1],
                [band_code, 3, 9, 4],
                [band_code, 3, 9, 5],
            ]
        )
        savemat(path, {"Data": matrix})


def test_reconstruct_narrow_iq_uses_alternating_columns() -> None:
    matrix = np.zeros((1, 1028), dtype=np.float64)
    matrix[0, 4:10] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    iq = reconstruct_narrow_iq(matrix)
    assert iq.shape == (1, 512)
    np.testing.assert_array_equal(iq[0, :3], [1 + 2j, 3 + 4j, 5 + 6j])


def test_audit_validates_release_and_flags_batch_collisions(tmp_path: Path) -> None:
    dataset_root = tmp_path / "LAT-MRICD-1.0"
    output_dir = tmp_path / "audit"
    write_release(dataset_root)

    summary = audit_dataset(dataset_root=dataset_root, output_dir=output_dir)

    assert summary["status"] == "READY_FOR_PREREGISTERED_GROUPED_BASELINE"
    assert summary["record_count"] == 30
    assert summary["aggregate_file_count"] == 5
    assert summary["batch_code_collision_count"] == 5
    assert summary["random_row_split_allowed"] is False
    assert summary["h_v_polarimetric_channels_available"] is False
    assert (output_dir / "model_split_readiness.csv").is_file()
    assert (output_dir / "REPORT.md").is_file()


def test_audit_rejects_wrong_schema(tmp_path: Path) -> None:
    dataset_root = tmp_path / "LAT-MRICD-1.0"
    write_release(dataset_root, invalid_width=True)
    with pytest.raises(ValueError, match=r"expected \(\*, 504\)"):
        audit_dataset(dataset_root=dataset_root, output_dir=tmp_path / "audit")
