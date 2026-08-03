from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from bs4 import BeautifulSoup

from scripts.build_zero_doppler_review_workbench_v1 import (
    build_workbench,
    prepare_cases,
)


def case_frame(image_file: str = "images/case.png") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "atlas_rank": [1],
            "fold": [4],
            "sample_id": ["sample-1"],
            "review_priority": ["P0_removed_by_residual"],
            "review_status": ["pending"],
            "visible_pattern": ["unreviewed"],
            "physical_class": ["unknown"],
            "evidence_source": ["prediction_and_relative_features_only"],
            "review_note": [""],
            "score_fixed": [0.3],
            "score_residual": [0.1],
            "score_delta_residual_minus_fixed": [-0.2],
            "zero_velocity_distance_bins": [2],
            "feature_rd_anchor_zero_doppler_fraction": [0.98],
            "feature_polar_roi_zdr_iqr_db": [3.5],
            "image_file": [image_file],
        }
    )


def test_build_workbench_embeds_cases_controls_and_validation(tmp_path: Path) -> None:
    atlas = tmp_path / "atlas"
    (atlas / "images").mkdir(parents=True)
    (atlas / "images/case.png").write_bytes(b"not-needed-for-html-build")
    cases_path = atlas / "cases.csv"
    case_frame().to_csv(cases_path, index=False, encoding="utf-8-sig")
    output = atlas / "review_workbench.html"

    result = build_workbench(cases_path=cases_path, output_path=output, overwrite=False)
    soup = BeautifulSoup(output.read_text(encoding="utf-8"), "html.parser")
    script_text = output.read_text(encoding="utf-8")

    assert result["case_count"] == 1
    assert soup.select_one("#case-image") is not None
    assert soup.select_one("#review-status") is not None
    assert soup.select_one("#visible-pattern") is not None
    assert soup.select_one("#physical-class")["value"] == "unknown"
    assert soup.select_one("#export") is not None
    assert '"sample_id":"sample-1"' in soup.select_one("#workbench-data").text
    assert "localStorage.setItem" in script_text
    assert "Named physical classes require an independent scene record." in script_text
    assert 'lines.join("\\r\\n")' in script_text


def test_prepare_cases_rejects_missing_image(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="atlas image not found"):
        prepare_cases(case_frame(), tmp_path, tmp_path)


def test_prepare_cases_rejects_image_outside_atlas(tmp_path: Path) -> None:
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"image")
    atlas = tmp_path / "atlas"
    atlas.mkdir()

    with pytest.raises(ValueError, match="inside atlas directory"):
        prepare_cases(case_frame("../outside.png"), atlas, atlas)


def test_build_workbench_does_not_overwrite_without_flag(tmp_path: Path) -> None:
    atlas = tmp_path / "atlas"
    (atlas / "images").mkdir(parents=True)
    (atlas / "images/case.png").write_bytes(b"image")
    cases_path = atlas / "cases.csv"
    case_frame().to_csv(cases_path, index=False, encoding="utf-8-sig")
    output = atlas / "review_workbench.html"
    output.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="output already exists"):
        build_workbench(cases_path=cases_path, output_path=output, overwrite=False)
