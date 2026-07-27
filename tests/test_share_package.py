from __future__ import annotations

import json
from pathlib import Path

import pytest
import scripts.build_project_share_package as share_package

from scripts.build_project_share_package import (
    PACKAGE_FILES,
    audit_markdown_links,
    audit_package_directory,
    sha256_file,
    validate_source_map,
    write_deterministic_zip,
)


def test_share_source_map_is_complete_and_unique() -> None:
    validate_source_map()
    destinations = [item.destination for item in PACKAGE_FILES]
    assert len(destinations) == len(set(destinations))
    assert not any("development_history" in item.source for item in PACKAGE_FILES)
    assert "docs/06_DATA_CARD_ZH.md" in destinations
    assert "docs/07_METRIC_DEFINITIONS_ZH.md" in destinations
    assert "docs/08_MODEL_SELECTION_LEDGER_ZH.md" in destinations
    assert "assets/figures/joint_fold_heterogeneity.png" in destinations
    assert "assets/tables/joint_scan_group_bootstrap.csv" in destinations
    assert "evidence/05_BC_DPG_V3_CAUSAL_CONTEXT_AUDIT.md" in destinations
    assert "assets/tables/bc_dpg_causal_context_aggregate.csv" in destinations
    assert "assets/tables/bc_dpg_causal_context_paired_deltas.csv" in destinations
    assert "assets/tables/bc_dpg_causal_context_replay_validation.csv" in destinations
    assert "assets/tables/bc_dpg_causal_context_history_coverage.csv" in destinations
    assert not any("joint_fold_false_alarms" in path for path in destinations)


def test_share_manifest_marks_causal_context_as_post_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(share_package, "current_commit", lambda: "test-commit")
    share_package.write_manifest(tmp_path, [])
    manifest = json.loads((tmp_path / "MANIFEST.json").read_text(encoding="utf-8"))
    rules = manifest["evidence_rules"]
    assert rules["causal_context_audit_role"] == (
        "post-hoc frozen-checkpoint sensitivity"
    )
    assert rules["causal_context_retraining_performed"] is False
    assert rules["causal_history_window_selected"] is False
    assert rules["past_only_order_verified_by_timestamp"] is False
    assert rules["past_only_order_columns"] == [
        "beam_layer",
        "azimuth_deg",
        "sample_id",
    ]


def test_share_audit_rejects_local_paths(tmp_path: Path) -> None:
    package = tmp_path / "share"
    package.mkdir()
    (package / "README.md").write_text(
        "private source: /home/example/project\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="sensitive marker"):
        audit_package_directory(package)


def test_share_audit_rejects_model_weights(tmp_path: Path) -> None:
    package = tmp_path / "share"
    package.mkdir()
    (package / "model.pt").write_bytes(b"not-a-real-checkpoint")
    with pytest.raises(ValueError, match="file type"):
        audit_package_directory(package)


def test_share_link_audit_rejects_missing_local_target(tmp_path: Path) -> None:
    package = tmp_path / "share"
    package.mkdir()
    (package / "README.md").write_text(
        "[missing](docs/missing.md)\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="missing link target"):
        audit_markdown_links(package)


def test_share_zip_is_deterministic(tmp_path: Path) -> None:
    package = tmp_path / "share"
    package.mkdir()
    (package / "README.md").write_text("share\n", encoding="utf-8")
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    write_deterministic_zip(package, first)
    write_deterministic_zip(package, second)
    assert sha256_file(first) == sha256_file(second)
