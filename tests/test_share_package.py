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
    assert "docs/00_ONE_PAGE_SUMMARY_ZH.md" in destinations
    assert "docs/02A_HISTORICAL_PROJECT_RECONSTRUCTION_ZH.md" in destinations
    assert "docs/12_PROJECT_TASK_LEDGER_ZH.md" in destinations
    assert "docs/13_TEAM_REPRODUCTION_GUIDE_ZH.md" in destinations
    assert "docs/09_RECENT_PROGRESS_AND_FAILURE_ANALYSIS_ZH.md" in destinations
    assert "docs/10_QUESTIONS_FOR_SENIOR_ZH.md" in destinations
    assert "docs/11_DATA_REQUEST_CHECKLIST_ZH.md" in destinations
    assert "assets/figures/joint_fold_heterogeneity.png" in destinations
    assert "assets/tables/joint_scan_group_bootstrap.csv" in destinations
    assert "evidence/05_BC_DPG_V3_CAUSAL_CONTEXT_AUDIT.md" in destinations
    assert "evidence/06_DETECTION_ACQUISITION_ORDER_AUDIT.md" in destinations
    assert "evidence/07_BC_DPG_LOCALIZATION_EVIDENCE.md" in destinations
    assert "evidence/08_CURRENT_DATA_COLLECTION_READINESS.md" in destinations
    assert "evidence/08_CURRENT_DATA_COLLECTION_READINESS.json" in destinations
    assert "evidence/09_MULTIDOMAIN_FEATURE_CATALOG.md" in destinations
    assert "evidence/10_TIAN_FCN_FOLD1_DIAGNOSTIC_CONCLUSION.md" in destinations
    assert "evidence/14_ZERO_DOPPLER_CANDIDATE_VETO.md" in destinations
    assert "evidence/15_ZERO_DOPPLER_FROZEN_SIXFOLD.md" in destinations
    assert "evidence/17_ZERO_DOPPLER_MECHANISM_CONCLUSION.md" in destinations
    assert "evidence/18_POLARIMETRIC_TRANSFER_ENCODER.md" in destinations
    assert "evidence/19_FIELD_COLLECTION_SOP.md" in destinations
    assert "assets/tables/bc_dpg_causal_context_aggregate.csv" in destinations
    assert "assets/tables/bc_dpg_causal_context_paired_deltas.csv" in destinations
    assert "assets/tables/bc_dpg_causal_context_replay_validation.csv" in destinations
    assert "assets/tables/bc_dpg_causal_context_history_coverage.csv" in destinations
    assert "assets/tables/detection_acquisition_order_sources.csv" in destinations
    assert "assets/figures/bc_dpg_localization_error_cdf.png" in destinations
    assert "evidence/figures/fig1_localization_error_cdf.png" in destinations
    assert "assets/tables/bc_dpg_localization_pooled.csv" in destinations
    assert "assets/tables/bc_dpg_localization_error_distribution.csv" in destinations
    assert "assets/tables/current_data_collection_contract_coverage.csv" in destinations
    assert "assets/contracts/data_collection_contract_v1.json" in destinations
    assert "assets/templates/data_collection_manifest_template_v1.csv" in destinations
    assert "assets/templates/field_capability_response_template_v1.csv" in destinations
    assert "assets/tables/zero_doppler_candidate_veto_tradeoff.csv" in destinations
    assert "assets/tables/zero_doppler_frozen_sixfold.csv" in destinations
    assert "docs/NEW_DATA_COLLECTION_PROTOCOL.md" in destinations
    assert not any("joint_fold_false_alarms" in path for path in destinations)


def test_current_data_readiness_sources_are_fresh() -> None:
    readiness = share_package.load_current_data_readiness()
    assert readiness["status"] == "FAIL"
    assert len(readiness["missing_columns"]) == 33
    assert readiness["gates"]["schema"] == "FAIL"
    assert set(readiness["gates"].values()) == {"FAIL", "BLOCKED"}


def test_current_data_readiness_rejects_stale_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed_contract = tmp_path / "changed_contract.json"
    changed_contract.write_text('{"schema_version": 999}\n', encoding="utf-8")
    monkeypatch.setattr(share_package, "DATA_CONTRACT", changed_contract)
    with pytest.raises(ValueError, match="contract hash is stale"):
        share_package.load_current_data_readiness()


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
    assert rules["formal_causal_training_gate_open"] is False
    assert rules["verified_within_scan_order_available"] is False
    assert rules["causal_development_smoke_test_split_loaded"] is False
    assert rules["causal_development_smoke_test_split_evaluated"] is False
    assert rules["localization_training_performed"] is False
    assert rules["localization_inference_performed"] is False
    assert rules["localization_test_threshold_retuning"] is False
    assert rules["localization_coordinates_match_raw_dpg"] is True
    assert rules["localization_sample_predictions_included"] is False
    assert rules["new_data_collection_contract_version"] == 1
    assert rules["current_manifest_contract_profile"] == "locked_evaluation"
    assert rules["current_manifest_contract_status"] == "FAIL"
    assert rules["current_manifest_missing_contract_columns"] == 33
    assert rules["current_formal_causal_training_gate_open"] is False
    assert rules["current_locked_evaluation_gate_open"] is False
    assert rules["past_only_order_columns"] == [
        "beam_layer",
        "azimuth_deg",
        "sample_id",
    ]
    assert rules["tian_reproduction_successful"] is False
    assert rules["tian_point_gt_role"] == "validation-only local-transfer ablation"
    assert rules["zero_doppler_candidate_veto_role"] == (
        "post-test mechanism diagnostic"
    )
    assert rules["zero_doppler_fixed_notch_role"] == (
        "development safety reference"
    )
    assert rules["zero_doppler_learned_sixfold_authorized"] is False
    assert rules["polarimetric_transfer_checkpoint_available"] is False
    assert rules["absolute_polarimetric_calibration_verified"] is False
    assert rules["physical_micro_doppler_timing_verified"] is False
    assert rules["field_readiness_gate_open"] is False


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
