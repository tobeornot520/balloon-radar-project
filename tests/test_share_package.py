from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import scripts.build_project_share_package as share_package

from scripts.build_project_share_package import (
    PACKAGE_FILES,
    audit_markdown_links,
    audit_package_directory,
    cleanup_staging_directory,
    ensure_output_available,
    parse_args,
    sha256_file,
    validate_output_paths,
    validate_source_map,
    write_deterministic_zip,
)


def test_share_source_map_is_complete_and_unique() -> None:
    validate_source_map()
    destinations = [item.destination for item in PACKAGE_FILES]
    sources = [item.source for item in PACKAGE_FILES]
    assert (
        share_package.PACKAGE_NAME
        == "balloon_radar_results_and_team_onboarding_20260806_v13"
    )
    assert len(destinations) == len(set(destinations))
    assert not any("development_history" in item.source for item in PACKAGE_FILES)
    assert "docs/06_DATA_CARD_ZH.md" in destinations
    assert "docs/07_METRIC_DEFINITIONS_ZH.md" in destinations
    assert "docs/08_MODEL_SELECTION_LEDGER_ZH.md" in destinations
    assert "docs/00_ONE_PAGE_SUMMARY_ZH.md" in destinations
    assert "docs/02A_HISTORICAL_PROJECT_RECONSTRUCTION_ZH.md" in destinations
    assert "docs/12_PROJECT_TASK_LEDGER_ZH.md" in destinations
    assert "docs/13_TEAM_REPRODUCTION_GUIDE_ZH.md" in destinations
    assert "docs/14_MULTIDOMAIN_FEATURE_GATE_V1_ZH.md" in destinations
    assert "docs/15_PAPER_MAINLINE_V1_ZH.md" in destinations
    assert "docs/16_EXTERNAL_FACT_REQUEST_MESSAGE_V1_ZH.md" in destinations
    assert "docs/17_NEXT_STAGE_PLAN_20260803_ZH.md" in destinations
    assert "docs/18_TIAN_REPRODUCTION_FAILURE_AND_ALTERNATIVES_20260803_ZH.md" in destinations
    assert "evidence/21_ZERO_DOPPLER_P0_REVIEW_PRESCREEN.md" in destinations
    assert "docs/EXTERNAL_PUBLIC_DATA_AUDIT_20260803.md" in destinations
    assert "TEAM_START_HERE.md" in destinations
    assert "docs/19_TEAM_QUALIFICATION_AND_ROLE_SCREENING_ZH.md" in destinations
    assert "docs/20_RECOMMENDED_PAPERS_20260805.md" in destinations
    assert "assets/contracts/lss_fmcwr_normalized_processing_contract_v1.json" in destinations
    assert "docs/LSS_FMCWR_2_NORMALIZED_PROCESSING_CONTRACT_20260805.md" in destinations
    assert "scripts/process_lss_fmcwr_normalized_v1.py" in destinations
    assert "docs/FIELD_IQ_INTEGRITY_PROBE_V1.md" in destinations
    assert "assets/contracts/field_iq_probe_contract_template_v1.json" in destinations
    assert "scripts/audit_field_iq_integrity_v1.py" in destinations
    assert "docs/ZERO_DOPPLER_FALSE_ALARM_LIBRARY_V1.md" in destinations
    assert "assets/contracts/zero_doppler_false_alarm_library_v1.json" in destinations
    assert "scripts/build_zero_doppler_false_alarm_library_v1.py" in destinations
    assert "assets/tables/zero_doppler_false_alarm_fold_summary.csv" in destinations
    assert "assets/tables/zero_doppler_false_alarm_scan_summary.csv" in destinations
    assert "assets/tables/zero_doppler_false_alarm_review_pattern_summary.csv" in destinations
    assert "evidence/29_ZERO_DOPPLER_FALSE_ALARM_LIBRARY_V1.json" in destinations
    assert "docs/ZERO_DOPPLER_TARGET_SAFETY_AUDIT_V1.md" in destinations
    assert "assets/contracts/zero_doppler_target_safety_audit_v1.json" in destinations
    assert "scripts/audit_zero_doppler_target_safety_v1.py" in destinations
    assert "assets/tables/zero_doppler_target_safety_fold_summary.csv" in destinations
    assert "assets/tables/zero_doppler_target_peak_shift_histogram.csv" in destinations
    assert "assets/tables/zero_doppler_target_score_delta_quantiles.csv" in destinations
    assert "evidence/30_ZERO_DOPPLER_TARGET_SAFETY_AUDIT_V1.json" in destinations
    assert "assets/templates/team_onboarding_checklist_template_v1.csv" in destinations
    assert "assets/templates/team_qualification_scorecard_template_v1.csv" in destinations
    assert "assets/templates/team_task_claim_template_v1.csv" in destinations
    assert "assets/templates/TEAM_WEEKLY_REPORT_TEMPLATE_ZH.md" in destinations
    assert "assets/contracts/current_direction_completion_v1.json" in destinations
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
    assert "docs/LAT_MRICD_GROUPED_BASELINE_PROTOCOL_V1.md" in destinations
    assert "evidence/22_LAT_MRICD_DATA_AUDIT.md" in destinations
    assert "evidence/22_LAT_MRICD_DATA_AUDIT.json" in destinations
    assert "assets/tables/lat_mricd_category_split_readiness.csv" in destinations
    assert "assets/tables/lat_mricd_batch_code_collisions.csv" in destinations
    assert "evidence/23_LAT_MRICD_GROUPED_BASELINES.md" in destinations
    assert "evidence/23_LAT_MRICD_GROUPED_BASELINES.json" in destinations
    grouped_table_names = {
        "aggregate_metrics",
        "batch_class_distribution",
        "batch_class_metrics",
        "claim_boundaries",
        "cluster_bootstrap_intervals",
        "confusion_matrices",
        "feature_definitions",
        "feature_importance",
        "feature_summary_by_category",
        "fold_coverage",
        "fold_metrics",
        "split_manifest",
        "subtype_pressure",
    }
    assert {
        f"assets/tables/lat_mricd_grouped_{name}.csv"
        for name in grouped_table_names
    }.issubset(destinations)
    assert "docs/LAT_MRICD_CROSS_BAND_TRANSFER_PROTOCOL_V1.md" in destinations
    assert "assets/contracts/lat_mricd_cross_band_transfer_v1.json" in destinations
    assert (
        "evidence/24_LAT_MRICD_CROSS_BAND_TRANSFER_RUN_CONSUMED.json"
        in destinations
    )
    assert "evidence/24_LAT_MRICD_CROSS_BAND_TRANSFER.md" in destinations
    assert "evidence/24_LAT_MRICD_CROSS_BAND_TRANSFER_MANIFEST.json" in destinations
    assert "evidence/24_LAT_MRICD_CROSS_BAND_TRANSFER_GATE.json" in destinations
    assert "evidence/24_LAT_MRICD_CROSS_BAND_TRANSFER_MODEL_FIT.json" in destinations
    assert "evidence/24_LAT_MRICD_CROSS_BAND_TRANSFER_SUMMARY.json" in destinations
    cross_band_table_names = {
        "aggregate_metrics",
        "bootstrap_intervals",
        "claim_boundaries",
        "confusion_matrices",
        "disjoint_sensitivity",
        "feature_definitions",
        "feature_importance",
        "raw_batch_overlap_audit",
        "target_batch_class_metrics",
        "training_weight_audit",
        "transfer_coverage",
    }
    assert {
        f"assets/tables/lat_mricd_cross_band_{name}.csv"
        for name in cross_band_table_names
    }.issubset(destinations)
    assert "evidence/25_DRONERFC_MM_READ_ONLY_AUDIT.md" in destinations
    assert "evidence/25_DRONERFC_MM_READ_ONLY_AUDIT.json" in destinations
    assert "assets/tables/dronerfc_mm_recording_audit.csv" in destinations
    assert "evidence/26_LSS_DAUR_READ_ONLY_AUDIT.md" in destinations
    assert "evidence/26_LSS_DAUR_READ_ONLY_AUDIT.json" in destinations
    daur_table_names = {
        "source_session_group_audit",
        "class_summary",
        "doppler_config_audit",
    }
    assert {
        f"assets/tables/lss_daur_{name}.csv" for name in daur_table_names
    } == {
        destination
        for destination in destinations
        if destination.startswith("assets/tables/lss_daur_")
    }
    assert {
        item.source
        for item in PACKAGE_FILES
        if item.source.startswith("results/data_audit/lss_daur_v1/")
    } == {
        "results/data_audit/lss_daur_v1/REPORT.md",
        "results/data_audit/lss_daur_v1/summary.json",
        "results/data_audit/lss_daur_v1/source_session_group_audit.csv",
        "results/data_audit/lss_daur_v1/class_summary.csv",
        "results/data_audit/lss_daur_v1/doppler_config_audit.csv",
    }
    forbidden_daur_columns = {
        "recording_id",
        "timestamp",
        "relative_path",
        "td_payload_sha256",
        "tr_payload_sha256",
    }
    for item in PACKAGE_FILES:
        if not item.destination.startswith("assets/tables/lss_daur_"):
            continue
        with (share_package.PROJECT_ROOT / item.source).open(
            encoding="utf-8", newline=""
        ) as handle:
            fieldnames = set(csv.DictReader(handle).fieldnames or ())
        assert fieldnames.isdisjoint(forbidden_daur_columns)
    hsr_sources = {
        "results/data_audit/lss_hsr_l_v2/REPORT.md",
        "results/data_audit/lss_hsr_l_v2/summary.json",
        "results/data_audit/lss_hsr_l_v2/split_summary.csv",
        "results/data_audit/lss_hsr_l_v2/split_class_summary.csv",
        "results/data_audit/lss_hsr_l_v2/feature_summary.csv",
    }
    hsr_destinations = {
        "evidence/27_LSS_HSR_L_V2_READ_ONLY_AUDIT.md",
        "evidence/27_LSS_HSR_L_V2_READ_ONLY_AUDIT.json",
        "assets/tables/lss_hsr_l_v2_split_summary.csv",
        "assets/tables/lss_hsr_l_v2_split_class_summary.csv",
        "assets/tables/lss_hsr_l_v2_feature_summary.csv",
    }
    assert {
        item.source
        for item in PACKAGE_FILES
        if item.source.startswith("results/data_audit/lss_hsr_l_v2/")
    } == hsr_sources
    assert {
        item.destination
        for item in PACKAGE_FILES
        if item.source.startswith("results/data_audit/lss_hsr_l_v2/")
    } == hsr_destinations
    assert not any(
        Path(path).name in {"route_audit.csv", "mat_audit.csv"}
        for path in sources + destinations
    )
    fmcwr_sources = {
        "results/data_audit/lss_fmcwr_2_v1/REPORT.md",
        "results/data_audit/lss_fmcwr_2_v1/summary.json",
        "results/data_audit/lss_fmcwr_2_v1/archive_audit.csv",
        "results/data_audit/lss_fmcwr_2_v1/group_summary.csv",
    }
    fmcwr_destinations = {
        "evidence/28_LSS_FMCWR_2_V1_READ_ONLY_AUDIT.md",
        "evidence/28_LSS_FMCWR_2_V1_READ_ONLY_AUDIT.json",
        "assets/tables/lss_fmcwr_2_v1_archive_audit.csv",
        "assets/tables/lss_fmcwr_2_v1_group_summary.csv",
    }
    assert {
        item.source
        for item in PACKAGE_FILES
        if item.source.startswith("results/data_audit/lss_fmcwr_2_v1/")
    } == fmcwr_sources
    assert {
        item.destination
        for item in PACKAGE_FILES
        if item.source.startswith("results/data_audit/lss_fmcwr_2_v1/")
    } == fmcwr_destinations
    assert {
        item.source: item.destination
        for item in PACKAGE_FILES
        if item.source.startswith("results/data_audit/lss_fmcwr_2_v1/")
    } == {
        "results/data_audit/lss_fmcwr_2_v1/REPORT.md": (
            "evidence/28_LSS_FMCWR_2_V1_READ_ONLY_AUDIT.md"
        ),
        "results/data_audit/lss_fmcwr_2_v1/summary.json": (
            "evidence/28_LSS_FMCWR_2_V1_READ_ONLY_AUDIT.json"
        ),
        "results/data_audit/lss_fmcwr_2_v1/archive_audit.csv": (
            "assets/tables/lss_fmcwr_2_v1_archive_audit.csv"
        ),
        "results/data_audit/lss_fmcwr_2_v1/group_summary.csv": (
            "assets/tables/lss_fmcwr_2_v1_group_summary.csv"
        ),
    }
    assert {
        item.source: item.destination
        for item in PACKAGE_FILES
        if item.source == "docs/LSS_FMCWR_2_READ_ONLY_AUDIT_20260805.md"
    } == {
        "docs/LSS_FMCWR_2_READ_ONLY_AUDIT_20260805.md": (
            "docs/LSS_FMCWR_2_READ_ONLY_AUDIT_20260805.md"
        )
    }
    assert not any(
        Path(path).name in {"mat_audit.csv", "duplicate_groups.csv"}
        for path in sources + destinations
    )
    assert "assets/registries/external_public_datasets_v1.csv" in destinations
    assert "assets/registries/external_public_artifacts_v1.csv" in destinations
    assert not any("oof_predictions.csv" in path for path in sources + destinations)
    assert not any(
        forbidden in Path(path).name.lower()
        for path in sources + destinations
        for forbidden in ("predictions", "review_queue", "sample_features")
    )
    assert not any(path.startswith("data/raw/") for path in sources + destinations)
    assert all(not Path(path).is_absolute() for path in sources + destinations)
    assert not any("joint_fold_false_alarms" in path for path in destinations)
    false_alarm_sources = {
        item.source
        for item in PACKAGE_FILES
        if item.source.startswith(
            "results/data_audit/zero_doppler_false_alarm_library_v1/"
        )
    }
    assert false_alarm_sources == {
        "results/data_audit/zero_doppler_false_alarm_library_v1/fold_transition_summary.csv",
        "results/data_audit/zero_doppler_false_alarm_library_v1/scan_transition_summary.csv",
        "results/data_audit/zero_doppler_false_alarm_library_v1/review_pattern_summary.csv",
        "results/data_audit/zero_doppler_false_alarm_library_v1/summary.json",
    }
    assert not any("case_library_local" in path for path in sources + destinations)
    target_safety_sources = {
        item.source
        for item in PACKAGE_FILES
        if item.source.startswith(
            "results/data_audit/zero_doppler_target_safety_audit_v1/"
        )
    }
    assert target_safety_sources == {
        "results/data_audit/zero_doppler_target_safety_audit_v1/fold_target_safety_summary.csv",
        "results/data_audit/zero_doppler_target_safety_audit_v1/peak_shift_histogram.csv",
        "results/data_audit/zero_doppler_target_safety_audit_v1/score_delta_quantiles.csv",
        "results/data_audit/zero_doppler_target_safety_audit_v1/summary.json",
    }
    assert not any("target_case_library_local" in path for path in sources + destinations)


def test_completion_delivery_evidence_tracks_frozen_package_name() -> None:
    project_root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (project_root / "configs/current_direction_completion_v1.json").read_text(
            encoding="utf-8"
        )
    )
    delivery = next(item for item in payload["items"] if item["id"] == "CD06")
    evidence_paths = {
        value.strip() for value in delivery["evidence"].split(";")
    }

    assert f"dist/{share_package.PACKAGE_NAME}/MANIFEST.json" in evidence_paths
    assert f"dist/{share_package.PACKAGE_NAME}.zip" in evidence_paths


@pytest.mark.parametrize(
    "arguments",
    (
        ["--output-dir", "ignored"],
        ["--zip-path", "ignored"],
        ["--overwrite"],
    ),
)
def test_share_builder_cli_rejects_path_and_overwrite_controls(
    arguments: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["share-builder", *arguments])
    with pytest.raises(SystemExit):
        parse_args()


def test_share_builder_rejects_existing_empty_destination_without_deleting_it(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "existing-package"
    output_dir.mkdir()
    zip_path = tmp_path / "package.zip"

    with pytest.raises(FileExistsError, match="directory already exists"):
        ensure_output_available(output_dir, zip_path)

    assert output_dir.is_dir()
    assert list(output_dir.iterdir()) == []


def test_share_builder_rejects_nonfrozen_output_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="frozen"):
        validate_output_paths(tmp_path / "package", tmp_path / "package.zip")


def test_share_builder_cleanup_is_limited_to_fixed_staging_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging_root = tmp_path / ".share-package-staging"
    staging_root.mkdir()
    monkeypatch.setattr(share_package, "STAGING_ROOT", staging_root)
    valid_staging = staging_root / f".{share_package.PACKAGE_NAME}.build-current"
    unrelated_staging = staging_root / "unrelated"
    outside = tmp_path / f".{share_package.PACKAGE_NAME}.build-outside"
    for directory in (valid_staging, unrelated_staging, outside):
        directory.mkdir()
        (directory / "marker.txt").write_text("preserve\n", encoding="utf-8")

    cleanup_staging_directory(valid_staging)

    assert not valid_staging.exists()
    assert (unrelated_staging / "marker.txt").read_text(encoding="utf-8") == "preserve\n"
    with pytest.raises(ValueError, match="outside the fixed staging root"):
        cleanup_staging_directory(outside)
    assert (outside / "marker.txt").read_text(encoding="utf-8") == "preserve\n"


def test_share_lat_mricd_split_manifest_is_group_level() -> None:
    source = next(
        item.source
        for item in PACKAGE_FILES
        if item.destination == "assets/tables/lat_mricd_grouped_split_manifest.csv"
    )
    with (share_package.PROJECT_ROOT / source).open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = set(reader.fieldnames or [])

    expected_columns = {
        "task_id",
        "representation",
        "band_code",
        "band",
        "batch_code",
        "heldout_fold",
        "record_count",
        "category_count",
        "model_count",
        "uav_record_count",
        "uav_present",
        "bird_record_count",
        "bird_present",
        "weather_record_count",
        "weather_present",
    }
    forbidden_columns = {
        "sample_id",
        "mat_path",
        "label_path",
        "source_path",
        "prediction",
        "predicted_category",
        "true_category",
    }
    assert columns == expected_columns
    assert columns.isdisjoint(forbidden_columns)
    assert rows
    group_keys = [
        (
            row["task_id"],
            row["representation"],
            row["band_code"],
            row["batch_code"],
        )
        for row in rows
    ]
    assert len(group_keys) == len(set(group_keys))


def test_share_lat_mricd_evidence_manifest_is_sanitized() -> None:
    evidence_root = Path(
        "results/final_evidence/lat_mricd_grouped_baselines_v1"
    )
    source = next(
        item.source
        for item in PACKAGE_FILES
        if item.destination == "evidence/23_LAT_MRICD_GROUPED_BASELINES.json"
    )
    manifest = json.loads(
        (share_package.PROJECT_ROOT / source).read_text(encoding="utf-8")
    )
    assert manifest["sample_predictions_included"] is False
    assert manifest["raw_data_included"] is False
    assert "oof_predictions.csv" in manifest["excluded_source_files"]
    published_paths = [item["relative_path"] for item in manifest["files"]]
    assert published_paths
    assert all(not Path(path).is_absolute() for path in published_paths)
    assert not any("oof_predictions.csv" in path for path in published_paths)
    assert not any(path.startswith("data/raw/") for path in published_paths)
    mapped_paths = {
        Path(item.source).relative_to(evidence_root).as_posix()
        for item in PACKAGE_FILES
        if item.source.startswith(f"{evidence_root.as_posix()}/")
    }
    assert mapped_paths == set(published_paths) | {"evidence_manifest.json"}


def test_share_lat_mricd_cross_band_evidence_is_complete_and_sanitized() -> None:
    evidence_root = Path(
        "results/final_evidence/lat_mricd_cross_band_transfer_v1"
    )
    source = next(
        item.source
        for item in PACKAGE_FILES
        if item.destination
        == "evidence/24_LAT_MRICD_CROSS_BAND_TRANSFER_MANIFEST.json"
    )
    manifest = json.loads(
        (share_package.PROJECT_ROOT / source).read_text(encoding="utf-8")
    )
    assert manifest["sample_level_predictions_included"] is False
    assert manifest["raw_data_included"] is False
    assert manifest["raw_or_absolute_paths_included"] is False
    assert manifest["model_checkpoints_included"] is False
    published_paths = [item["file"] for item in manifest["files"]]
    assert published_paths
    assert all(not Path(path).is_absolute() for path in published_paths)
    assert not any("oof_predictions.csv" in path for path in published_paths)
    assert not any(path.startswith("data/raw/") for path in published_paths)
    mapped_paths = {
        Path(item.source).relative_to(evidence_root).as_posix()
        for item in PACKAGE_FILES
        if item.source.startswith(f"{evidence_root.as_posix()}/")
    }
    assert mapped_paths == set(published_paths) | {"evidence_manifest.json"}


def test_share_external_public_registries_are_metadata_only() -> None:
    registry_destinations = {
        "assets/registries/external_public_datasets_v1.csv",
        "assets/registries/external_public_artifacts_v1.csv",
    }
    registry_items = [
        item for item in PACKAGE_FILES if item.destination in registry_destinations
    ]
    assert {item.destination for item in registry_items} == registry_destinations
    assert all(item.source.startswith("data/metadata/") for item in registry_items)
    assert all(not item.source.startswith("data/raw/") for item in registry_items)


def test_share_dronerfc_audit_is_sanitized_blocked_and_grouped() -> None:
    summary_source = next(
        item.source
        for item in PACKAGE_FILES
        if item.destination == "evidence/25_DRONERFC_MM_READ_ONLY_AUDIT.json"
    )
    table_source = next(
        item.source
        for item in PACKAGE_FILES
        if item.destination == "assets/tables/dronerfc_mm_recording_audit.csv"
    )
    summary = json.loads(
        (share_package.PROJECT_ROOT / summary_source).read_text(encoding="utf-8")
    )
    assert summary["status"] == "PASS_SCHEMA_BLOCKED_TIMESTAMP_ALIGNMENT"
    assert summary["schema_version"] == 1
    assert summary["dataset_id"] == "dronerfc_mm"
    assert summary["release_version"] == "V1"
    assert summary["data_doi"] == "10.57760/sciencedb.j00173.00094"
    assert summary["blocked_recordings"] == ["B1"]
    assert summary["model_training_allowed"] is False
    assert summary["source_files_modified"] is False
    assert summary["source_archives_extracted"] is False
    assert summary["sample_level_outputs_included"] is False
    assert summary["recording_count"] == 9
    assert summary["split_family_group_count"] == 6
    assert summary["pcd_frame_count"] == 30717
    assert summary["pcd_point_count"] == 639527
    assert summary["ground_truth_row_count"] == 35959
    assert summary["label_window_count"] == 717
    assert summary["selected_subset_file_count"] == 28
    assert summary["selected_subset_size_bytes"] == 47366902
    assert summary["selected_subset_manifest_sha256"] == (
        "6b0c2ed1a075aa9164a516af001b630a9f775fddc9f399223c1aeeb6e7047b2b"
    )
    assert summary["minimum_split_unit"] == (
        "split_family_group; keep -2 recordings with their base group"
    )
    assert "Random frame/window split" in summary["prohibited_scope"]
    assert "B1 supervised alignment" in summary["prohibited_scope"]

    with (share_package.PROJECT_ROOT / table_source).open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = set(reader.fieldnames or [])
    assert columns == {
        "recording_id",
        "split_family_group",
        "radar_zip",
        "pcd_frame_count",
        "pcd_point_count",
        "min_points_per_frame",
        "max_points_per_frame",
        "radar_start_utc",
        "radar_end_utc",
        "ground_truth_csv",
        "ground_truth_row_count",
        "ground_truth_duplicate_timestamp_count",
        "ground_truth_start_utc",
        "ground_truth_end_utc",
        "radar_gt_overlap_seconds",
        "nearest_gt_error_median_seconds",
        "nearest_gt_error_p95_seconds",
        "nearest_gt_error_max_seconds",
        "label_group",
        "label_window_overlap_count",
        "alignment_status",
    }
    assert len(rows) == 9
    assert len({row["split_family_group"] for row in rows}) == 6
    assert {row["recording_id"] for row in rows} == {
        "A1",
        "A1-2",
        "B1",
        "C1",
        "E1",
        "E1-2",
        "F1",
        "G1",
        "G1-2",
    }
    grouped = {row["recording_id"]: row["split_family_group"] for row in rows}
    assert grouped["A1-2"] == grouped["A1"] == "A1"
    assert grouped["E1-2"] == grouped["E1"] == "E1"
    assert grouped["G1-2"] == grouped["G1"] == "G1"
    assert all(row["label_group"] == row["split_family_group"] for row in rows)
    assert sum(int(row["pcd_frame_count"]) for row in rows) == 30717
    assert sum(int(row["pcd_point_count"]) for row in rows) == 639527
    assert sum(int(row["ground_truth_row_count"]) for row in rows) == 35959
    b1 = next(row for row in rows if row["recording_id"] == "B1")
    assert float(b1["radar_gt_overlap_seconds"]) == 0.0
    assert b1["alignment_status"] == "BLOCKED_NO_RADAR_GT_TIME_OVERLAP"
    assert b1["radar_end_utc"] == "2026-04-01T07:19:22.629000Z"
    assert b1["ground_truth_start_utc"] == "2026-04-01T07:27:15.240000Z"
    assert {
        row["recording_id"]
        for row in rows
        if row["alignment_status"] != "PASS_TIME_RANGE_OVERLAP"
    } == {"B1"}
    assert all(
        float(row["radar_gt_overlap_seconds"]) > 0.0
        for row in rows
        if row["recording_id"] != "B1"
    )


def test_share_lss_daur_audit_is_sanitized_blocked_and_not_doubled() -> None:
    summary_source = next(
        item.source
        for item in PACKAGE_FILES
        if item.destination == "evidence/26_LSS_DAUR_READ_ONLY_AUDIT.json"
    )
    summary = json.loads(
        (share_package.PROJECT_ROOT / summary_source).read_text(encoding="utf-8")
    )
    assert summary["status"] == (
        "PASS_SCHEMA_PAIRING_BLOCKED_GROUPING_AND_PHYSICAL_AXIS"
    )
    assert summary["gates"] == {
        "release_identity": "PASS",
        "schema_and_finite": "PASS",
        "td_tr_pairing": "PASS",
        "canonical_backup_equivalence": "PASS",
        "strict_time": "FAIL_DUPLICATE_TIMESTAMPS",
        "absolute_date_weather_join": "BLOCKED_DATE_CONFLICT",
        "session_identity": "BLOCKED_NO_AUTHORITATIVE_SESSION_KEY",
        "physical_axis_512": "PARTIAL_SCRIPT_DEFINED",
        "physical_axis_1024": "BLOCKED_UNDOCUMENTED_WIDTH",
        "model_training": "BLOCKED",
    }
    assert summary["paired_track_count"] == 77
    assert summary["unique_signal_trajectory_content_count"] == 76
    assert summary["canonical_mat_file_count"] == 154
    assert summary["backup_mat_file_count"] == 154
    assert summary["canonical_backup_observation_count_multiplier_allowed"] is False
    assert summary["frame_count"] == 11366
    assert summary["doppler_value_count"] == 7728640
    assert summary["doppler_512_track_count"] == 58
    assert summary["doppler_1024_track_count"] == 19
    assert summary["duplicate_time_step_count"] == 894
    assert summary["unique_time_position_count"] == 10472
    assert summary["tracks_with_noncontiguous_frame_counter"] == 13
    assert summary["frame_counter_gap_event_count"] == 85
    assert summary["frame_counter_missing_value_count"] == 94
    assert summary["frame_counter_repeat_event_count"] == 2
    assert summary["filename_header_date_mismatch_count"] == 6
    assert summary["filename_session_candidate_count"] == 45
    assert summary["header_date_session_candidate_count"] == 40
    assert summary["candidate_source_session_group_count"] == 39
    assert summary["header_date_scene_group_count"] == 24
    assert summary["header_date_scene_class_pure_group_count"] == 20
    assert summary["bird_uav_filename_session_overlap_count"] == 0
    assert summary["bird_uav_header_date_session_overlap_count"] == 0
    assert summary["bird_uav_connected_session_overlap_count"] == 0
    assert summary["shared_frame_record_pair_count"] == 11
    assert summary["exact_duplicate_recording_group_count"] == 1
    assert summary["exact_duplicate_recording_count"] == 2
    assert summary["v_field_constant_zero"] is True
    assert summary["authoritative_session_key_available"] is False
    assert summary["absolute_weather_join_allowed"] is False
    assert summary["raw_adc_or_iq_available"] is False
    assert summary["h_v_polarimetry_available"] is False
    assert summary["physical_micro_doppler_hz_allowed"] is False
    assert summary["random_mat_split_allowed"] is False
    assert summary["random_frame_or_window_split_allowed"] is False
    assert summary["td_tr_split_allowed"] is False
    assert summary["canonical_backup_as_extra_samples_allowed"] is False
    assert summary["model_training_allowed"] is False
    assert summary["source_files_modified"] is False
    assert summary["sample_level_outputs_included"] is False

    expected_rows = {
        "assets/tables/lss_daur_source_session_group_audit.csv": 39,
        "assets/tables/lss_daur_class_summary.csv": 6,
        "assets/tables/lss_daur_doppler_config_audit.csv": 3,
    }
    for destination, expected_count in expected_rows.items():
        source = next(
            item.source for item in PACKAGE_FILES if item.destination == destination
        )
        with (share_package.PROJECT_ROOT / source).open(
            encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == expected_count


def test_share_lss_hsr_l_v2_audit_is_aggregate_blocked_and_sanitized() -> None:
    summary = share_package.load_lss_hsr_audit()
    assert summary["status"] == (
        "PASS_SCHEMA_BLOCKED_SOURCE_PROVENANCE_AND_PHYSICAL_AXIS"
    )
    assert summary["gates"] == {
        "release_identity": "PASS",
        "zip_safety_and_integrity": "PASS",
        "mat_schema_and_finite": "PASS",
        "route_mapping": "PASS",
        "published_train_validation_route_isolation": "PASS",
        "overflow_role": "BLOCKED_UNDOCUMENTED",
        "acquisition_session_identity": "BLOCKED_NOT_AVAILABLE",
        "dpl_physical_time_axis": "BLOCKED_NOT_AVAILABLE",
        "dpl_physical_doppler_axis": "BLOCKED_NOT_AVAILABLE",
        "model_training": "BLOCKED",
    }
    assert summary["mat_file_count"] == 1530
    assert summary["total_frame_count"] == 63148
    assert summary["route_count"] == 865
    assert summary["published_window_count"] == 55231
    assert summary["split_mat_counts"] == {
        "overflow": 11,
        "train": 1269,
        "validation": 250,
    }
    assert summary["split_route_counts"] == {
        "overflow": 11,
        "train": 723,
        "validation": 131,
    }
    assert summary["split_frame_counts"] == {
        "overflow": 704,
        "train": 51789,
        "validation": 10655,
    }
    assert summary["split_published_window_counts"] == {
        "overflow": 529,
        "train": 45366,
        "validation": 9336,
    }
    assert summary["published_train_validation_route_disjoint"] is True
    assert summary["published_train_validation_route_overlap_count"] == 0
    assert summary["acquisition_session_key_available"] is False
    assert summary["published_train_validation_session_disjoint_verified"] is False
    assert summary["route_id_sufficient_for_independent_evaluation"] is False
    assert summary["overflow_isolated"] is True
    assert summary["overflow_merge_allowed"] is False
    assert summary["random_mat_split_allowed"] is False
    assert summary["random_frame_or_window_split_allowed"] is False
    assert summary["dpl_amplitude_unit_verified"] is False
    assert summary["dpl_physical_time_axis_available"] is False
    assert summary["dpl_physical_doppler_hz_axis_available"] is False
    assert summary["dpl_physical_velocity_axis_available"] is False
    assert summary["track_physical_units_verified"] is True
    assert summary["physical_micro_doppler_hz_allowed"] is False
    assert summary["model_training_allowed"] is False
    assert summary["model_training_performed"] is False
    assert summary["journal_bundle_mixing_allowed"] is False

    expected_tables = {
        "assets/tables/lss_hsr_l_v2_split_summary.csv": (
            {
                "split",
                "mat_file_count",
                "route_count",
                "frame_count",
                "published_window_count",
                "class_count",
                "role",
            },
            3,
        ),
        "assets/tables/lss_hsr_l_v2_split_class_summary.csv": (
            {
                "split",
                "category",
                "mat_file_count",
                "route_count",
                "frame_count",
                "published_window_count",
            },
            22,
        ),
        "assets/tables/lss_hsr_l_v2_feature_summary.csv": (
            {
                "representation",
                "feature_index",
                "feature_name",
                "width",
                "unit",
                "unit_source",
                "physical_axis_available",
            },
            6,
        ),
    }
    forbidden_detail_columns = {
        "route_id",
        "route_number",
        "track_path",
        "mat_path",
        "relative_path",
        "zip_member",
        "raw_mat_sha256",
        "numeric_payload_sha256",
        "raw_sha256",
        "payload_sha256",
        "dpl_sha256",
        "track_sha256",
    }
    for destination, (expected_columns, expected_row_count) in expected_tables.items():
        source = next(
            item.source for item in PACKAGE_FILES if item.destination == destination
        )
        with (share_package.PROJECT_ROOT / source).open(
            encoding="utf-8", newline=""
        ) as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            columns = set(reader.fieldnames or ())
        assert columns == expected_columns
        assert columns.isdisjoint(forbidden_detail_columns)
        assert len(rows) == expected_row_count


def test_share_lss_fmcwr_audit_is_aggregate_blocked_and_sanitized() -> None:
    summary = share_package.load_lss_fmcwr_audit()
    assert summary["dataset_id"] == "lss_fmcwr_2_0"
    assert summary["target_release_version"] == "V4"
    assert summary["data_doi"] == "10.57760/sciencedb.radars.00054"
    assert summary["status"] == (
        "PASS_ARCHIVE_SCHEMA_BLOCKED_GROUPING_PROVENANCE_AND_PHYSICAL_AXIS"
    )
    assert summary["archive_count"] == 6
    assert summary["rar_entry_count"] == 116
    assert summary["mat_file_count"] == 90
    assert summary["directory_count"] == 26
    assert summary["band_counts"] == {"K": 64, "L": 26}
    assert summary["recording_stem_count"] == 66
    assert summary["candidate_group_count"] == 48
    assert summary["unique_raw_mat_count"] == 71
    assert summary["raw_duplicate_group_count"] == 11
    assert summary["raw_duplicate_member_count"] == 30
    assert summary["unique_numeric_payload_count"] == 71
    assert summary["numeric_duplicate_group_count"] == 11
    assert summary["numeric_duplicate_member_count"] == 30
    assert summary["candidate_groups_authoritative"] is False
    assert summary["independent_recording_or_session_key_available"] is False
    assert summary["h_v_polarimetry_available"] is False
    assert summary["natural_bird_evidence_available"] is False
    assert summary["physical_doppler_hz_axis_available"] is False
    assert summary["physical_velocity_axis_available"] is False
    assert summary["model_training_allowed"] is False
    assert summary["model_training_performed"] is False
    assert summary["source_archives_extracted_to_disk"] is False
    assert summary["source_archives_modified"] is False
    assert summary["raw_data_included_in_outputs"] is False
    assert summary["member_level_outputs_local_only"] is True
    assert summary["gates"] == {
        "release_identity": "PASS",
        "rar_path_safety_and_inventory": "PASS",
        "rar_integrity": "PASS",
        "mat_schema_and_finite": "PASS",
        "exact_duplicate_isolation": "BLOCKED_DUPLICATES_PRESENT",
        "recording_session_identity": "BLOCKED_NOT_AVAILABLE",
        "physical_time_doppler_velocity_axis": "BLOCKED_NOT_AVAILABLE",
        "natural_bird_evidence": "BLOCKED_SIMULATION_ONLY",
        "model_training": "BLOCKED",
    }
    assert summary["target_recording_stem_counts"] == {
        "ac311": 4,
        "hexacopter": 15,
        "inspire2": 15,
        "m350": 15,
        "mavic2": 15,
        "simulated_bird": 2,
    }
    assert summary["target_candidate_group_counts"] == {
        "ac311": 4,
        "hexacopter": 10,
        "inspire2": 12,
        "m350": 10,
        "mavic2": 10,
        "simulated_bird": 2,
    }

    expected_tables = {
        "assets/tables/lss_fmcwr_2_v1_archive_audit.csv": (
            {
                "archive_name",
                "target_id",
                "physical_size_bytes",
                "sha256",
                "entry_count",
                "mat_file_count",
                "directory_count",
                "uncompressed_mat_size_bytes",
                "packed_mat_size_bytes",
                "k_count",
                "l_count",
                "rar5_non_solid_non_volume_unencrypted",
                "integrity_test_passed",
                "release_identity_verified",
            },
            6,
        ),
        "assets/tables/lss_fmcwr_2_v1_group_summary.csv": (
            {
                "archive_name",
                "target_id",
                "band_token",
                "collection_angle",
                "duration_token",
                "channelA_shape",
                "channelB_shape",
                "mat_count",
                "unique_raw_mat_count",
                "unique_numeric_payload_count",
                "candidate_group_count",
            },
            68,
        ),
    }
    forbidden_detail_columns = {
        "member_path",
        "members_json",
        "member_sha256",
        "raw_sha256",
        "raw_mat_sha256",
        "numeric_payload_sha256",
        "mat_path",
        "relative_path",
        "recording_stem",
        "ordinal",
        "crc32",
    }
    for destination, (expected_columns, expected_row_count) in expected_tables.items():
        source = next(
            item.source for item in PACKAGE_FILES if item.destination == destination
        )
        with (share_package.PROJECT_ROOT / source).open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            columns = set(reader.fieldnames or ())
        assert columns == expected_columns
        assert columns.isdisjoint(forbidden_detail_columns)
        assert len(rows) == expected_row_count
        assert all(not any(key in row for key in forbidden_detail_columns) for row in rows)


def test_lss_fmcwr_audit_loader_rejects_training_gate_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = json.loads(
        share_package.LSS_FMCWR_AUDIT_SUMMARY.read_text(encoding="utf-8")
    )
    payload["model_training_allowed"] = True
    changed_summary = tmp_path / "changed_fmcwr_summary.json"
    changed_summary.write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    monkeypatch.setattr(share_package, "LSS_FMCWR_AUDIT_SUMMARY", changed_summary)
    with pytest.raises(ValueError, match="model_training_allowed"):
        share_package.load_lss_fmcwr_audit()


def test_lss_fmcwr_audit_loader_rejects_gate_state_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = json.loads(
        share_package.LSS_FMCWR_AUDIT_SUMMARY.read_text(encoding="utf-8")
    )
    payload["gates"]["model_training"] = "PASS"
    changed_summary = tmp_path / "changed_fmcwr_gate_summary.json"
    changed_summary.write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    monkeypatch.setattr(share_package, "LSS_FMCWR_AUDIT_SUMMARY", changed_summary)
    with pytest.raises(ValueError, match="gate states"):
        share_package.load_lss_fmcwr_audit()


def test_lss_fmcwr_audit_loader_rejects_member_level_detail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = json.loads(
        share_package.LSS_FMCWR_AUDIT_SUMMARY.read_text(encoding="utf-8")
    )
    payload["member_path"] = "archive/member.mat"
    changed_summary = tmp_path / "changed_fmcwr_member_summary.json"
    changed_summary.write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    monkeypatch.setattr(share_package, "LSS_FMCWR_AUDIT_SUMMARY", changed_summary)
    with pytest.raises(ValueError, match="member-level detail"):
        share_package.load_lss_fmcwr_audit()


def test_lss_hsr_audit_loader_rejects_training_gate_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = json.loads(
        share_package.LSS_HSR_AUDIT_SUMMARY.read_text(encoding="utf-8")
    )
    payload["model_training_allowed"] = True
    changed_summary = tmp_path / "changed_hsr_summary.json"
    changed_summary.write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    monkeypatch.setattr(share_package, "LSS_HSR_AUDIT_SUMMARY", changed_summary)
    with pytest.raises(ValueError, match="model_training_allowed"):
        share_package.load_lss_hsr_audit()


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
    assert rules["tian_exact_reproduction_status"] == "blocked_unverifiable"
    assert rules["tian_reproduction_conditions_available"] is False
    assert rules["tian_point_gt_role"] == "validation-only local-transfer ablation"
    assert rules["tian_fallback_mainline"] == (
        "DPG-FCN zero-Doppler development plus grouped LAT-MRICD baselines"
    )
    assert rules["zero_doppler_candidate_veto_role"] == (
        "post-test mechanism diagnostic"
    )
    assert rules["zero_doppler_fixed_notch_role"] == (
        "development safety reference"
    )
    assert rules["zero_doppler_learned_sixfold_authorized"] is False
    assert rules["zero_doppler_target_safety_audit_included"] is True
    assert rules["zero_doppler_target_sample_count"] == 318
    assert rules["zero_doppler_target_fixed_detected"] == 302
    assert rules["zero_doppler_target_residual_detected"] == 301
    assert rules["zero_doppler_target_detection_lost"] == 1
    assert rules["zero_doppler_target_fixed_joint_success"] == 290
    assert rules["zero_doppler_target_residual_joint_success"] == 290
    assert rules["zero_doppler_target_peak_changed"] == 6
    assert rules["zero_doppler_target_large_peak_shift_count"] == 2
    assert rules["zero_doppler_target_row_level_library_included"] is False
    assert rules["zero_doppler_target_source_identifiers_included"] is False
    assert rules["zero_doppler_target_threshold_retuning_performed"] is False
    assert rules["zero_doppler_target_deployment_safety_established"] is False
    assert rules["polarimetric_transfer_checkpoint_available"] is False
    assert rules["absolute_polarimetric_calibration_verified"] is False
    assert rules["physical_micro_doppler_timing_verified"] is False
    assert rules["field_readiness_gate_open"] is False
    assert rules["field_iq_integrity_probe_included"] is True
    assert rules["field_iq_integrity_probe_schema_version"] == 1
    assert rules["field_iq_integrity_probe_supports_mat_v5_and_v7_3"] is True
    assert (
        rules["field_iq_integrity_probe_default_expected_shape_configured"]
        is False
    )
    assert rules["field_iq_integrity_probe_real_data_audit_completed"] is False
    assert rules["field_iq_integrity_probe_raw_data_included"] is False
    assert rules["field_iq_integrity_probe_establishes_hv_coherence"] is False
    assert (
        rules["field_iq_integrity_probe_establishes_polarimetric_calibration"]
        is False
    )
    assert rules["field_iq_integrity_probe_establishes_physical_axes"] is False
    assert rules["field_iq_integrity_probe_opens_model_training"] is False
    assert rules["lat_mricd_raw_data_included"] is False
    assert rules["lat_mricd_random_row_split_allowed"] is False
    assert rules["lat_mricd_physical_micro_doppler_hz_allowed"] is False
    assert rules["lat_mricd_grouped_baseline_included"] is True
    assert rules["lat_mricd_group_key"] == [
        "representation",
        "band_code",
        "batch_code",
    ]
    assert rules["lat_mricd_sample_predictions_included"] is False
    assert rules["lat_mricd_cross_band_transfer_included"] is True
    assert rules["lat_mricd_cross_band_sealed_run_consumed"] is True
    assert rules["lat_mricd_cross_band_primary_gate_passed"] is False
    assert rules["lat_mricd_cross_band_target_bands_consumed"] == ["S", "Ku"]
    assert (
        rules["lat_mricd_cross_band_same_target_confirmatory_reuse_allowed"]
        is False
    )
    assert rules["lat_mricd_cross_band_raw_data_included"] is False
    assert rules["lat_mricd_cross_band_sample_predictions_included"] is False
    assert rules["dronerfc_mm_read_only_audit_included"] is True
    assert rules["dronerfc_mm_audit_status"] == (
        "PASS_SCHEMA_BLOCKED_TIMESTAMP_ALIGNMENT"
    )
    assert rules["dronerfc_mm_raw_data_included"] is False
    assert rules["dronerfc_mm_sample_level_outputs_included"] is False
    assert rules["dronerfc_mm_training_performed"] is False
    assert rules["dronerfc_mm_model_training_allowed"] is False
    assert rules["dronerfc_mm_blocked_recordings"] == ["B1"]
    assert rules["dronerfc_mm_b1_supervised_alignment_allowed"] is False
    assert rules["dronerfc_mm_random_frame_window_split_allowed"] is False
    assert rules["dronerfc_mm_group_key"] == "split_family_group"
    assert rules["dronerfc_mm_minimum_split_unit"] == "split_family_group"
    assert rules["dronerfc_mm_split_family_group_count"] == 6
    assert rules["lss_daur_read_only_audit_included"] is True
    assert rules["lss_daur_audit_status"] == (
        "PASS_SCHEMA_PAIRING_BLOCKED_GROUPING_AND_PHYSICAL_AXIS"
    )
    assert rules["lss_daur_paired_observation_count"] == 77
    assert rules["lss_daur_unique_signal_trajectory_content_count"] == 76
    assert rules["lss_daur_candidate_source_session_group_count"] == 39
    assert rules["lss_daur_canonical_mat_file_count"] == 154
    assert rules["lss_daur_backup_mat_file_count"] == 154
    assert rules["lss_daur_canonical_backup_observation_multiplier_allowed"] is False
    assert rules["lss_daur_canonical_backup_as_extra_samples_allowed"] is False
    assert rules["lss_daur_authoritative_session_key_available"] is False
    assert rules["lss_daur_random_mat_split_allowed"] is False
    assert rules["lss_daur_random_frame_window_split_allowed"] is False
    assert rules["lss_daur_td_tr_split_allowed"] is False
    assert rules["lss_daur_physical_micro_doppler_hz_allowed"] is False
    assert rules["lss_daur_model_training_allowed"] is False
    assert rules["lss_daur_raw_data_included"] is False
    assert rules["lss_daur_sample_level_outputs_included"] is False
    assert rules["lss_hsr_l_v2_read_only_audit_included"] is True
    assert rules["lss_hsr_l_v2_audit_status"] == (
        "PASS_SCHEMA_BLOCKED_SOURCE_PROVENANCE_AND_PHYSICAL_AXIS"
    )
    assert rules["lss_hsr_l_v2_release_version"] == "V2"
    assert rules["lss_hsr_l_v2_release_identity_verified"] is True
    assert rules["lss_hsr_l_v2_data_doi"] == "10.57760/sciencedb.radars.00063"
    assert rules["lss_hsr_l_v2_archive_size_bytes"] == 237020946
    assert rules["lss_hsr_l_v2_archive_sha256"] == (
        "fea8a21354110a96fb9644dc1c69649b6dc6d1a1b6da512498d9c2d74d839540"
    )
    assert rules["lss_hsr_l_v2_mat_file_count"] == 1530
    assert rules["lss_hsr_l_v2_total_frame_count"] == 63148
    assert rules["lss_hsr_l_v2_route_count"] == 865
    assert rules["lss_hsr_l_v2_published_window_count"] == 55231
    assert rules["lss_hsr_l_v2_split_mat_counts"] == {
        "overflow": 11,
        "train": 1269,
        "validation": 250,
    }
    assert rules["lss_hsr_l_v2_split_route_counts"] == {
        "overflow": 11,
        "train": 723,
        "validation": 131,
    }
    assert rules["lss_hsr_l_v2_split_frame_counts"] == {
        "overflow": 704,
        "train": 51789,
        "validation": 10655,
    }
    assert rules["lss_hsr_l_v2_split_published_window_counts"] == {
        "overflow": 529,
        "train": 45366,
        "validation": 9336,
    }
    assert rules["lss_hsr_l_v2_gates"] == {
        "release_identity": "PASS",
        "zip_safety_and_integrity": "PASS",
        "mat_schema_and_finite": "PASS",
        "route_mapping": "PASS",
        "published_train_validation_route_isolation": "PASS",
        "overflow_role": "BLOCKED_UNDOCUMENTED",
        "acquisition_session_identity": "BLOCKED_NOT_AVAILABLE",
        "dpl_physical_time_axis": "BLOCKED_NOT_AVAILABLE",
        "dpl_physical_doppler_axis": "BLOCKED_NOT_AVAILABLE",
        "model_training": "BLOCKED",
    }
    assert rules["lss_hsr_l_v2_authoritative_route_id_available"] is True
    assert rules["lss_hsr_l_v2_every_mat_assigned_to_exactly_one_route"] is True
    assert rules["lss_hsr_l_v2_published_train_validation_route_disjoint"] is True
    assert (
        rules["lss_hsr_l_v2_published_train_validation_route_overlap_count"] == 0
    )
    assert rules["lss_hsr_l_v2_lowest_available_grouping_key"] == "route_id"
    assert rules["lss_hsr_l_v2_minimum_split_unit"] == (
        "UNRESOLVED_SOURCE_SESSION_IDENTITY_UNAVAILABLE"
    )
    assert rules["lss_hsr_l_v2_acquisition_session_key_available"] is False
    assert rules["lss_hsr_l_v2_published_session_disjoint_verified"] is False
    assert (
        rules["lss_hsr_l_v2_route_id_sufficient_for_independent_evaluation"]
        is False
    )
    assert rules["lss_hsr_l_v2_published_split_preservation_required"] is True
    assert rules["lss_hsr_l_v2_overflow_isolated"] is True
    assert rules["lss_hsr_l_v2_overflow_merge_allowed"] is False
    assert rules["lss_hsr_l_v2_overflow_role_documented"] is False
    assert rules["lss_hsr_l_v2_random_mat_split_allowed"] is False
    assert rules["lss_hsr_l_v2_random_frame_window_split_allowed"] is False
    assert rules["lss_hsr_l_v2_dpl_amplitude_unit_verified"] is False
    assert rules["lss_hsr_l_v2_dpl_physical_time_axis_available"] is False
    assert (
        rules["lss_hsr_l_v2_dpl_physical_doppler_hz_axis_available"] is False
    )
    assert rules["lss_hsr_l_v2_dpl_physical_velocity_axis_available"] is False
    assert rules["lss_hsr_l_v2_track_physical_units_verified"] is True
    assert rules["lss_hsr_l_v2_physical_micro_doppler_hz_allowed"] is False
    assert rules["lss_hsr_l_v2_raw_adc_or_iq_available"] is False
    assert rules["lss_hsr_l_v2_h_v_polarimetry_available"] is False
    assert rules["lss_hsr_l_v2_model_training_allowed"] is False
    assert rules["lss_hsr_l_v2_model_training_performed"] is False
    assert rules["lss_hsr_l_v2_source_archive_extracted"] is False
    assert rules["lss_hsr_l_v2_source_archive_modified"] is False
    assert rules["lss_hsr_l_v2_journal_bundle_mixing_allowed"] is False
    assert rules["lss_hsr_l_v2_raw_data_included"] is False
    assert rules["lss_hsr_l_v2_route_mat_detail_tables_included"] is False
    assert rules["lss_hsr_l_v2_sample_level_outputs_included"] is False
    fmcwr = share_package.load_lss_fmcwr_audit()
    fmcwr_manifest_fields = {
        "lss_fmcwr_2_audit_status": "status",
        "lss_fmcwr_2_target_release_version": "target_release_version",
        "lss_fmcwr_2_release_identity_verified": "release_identity_verified",
        "lss_fmcwr_2_data_doi": "data_doi",
        "lss_fmcwr_2_archive_count": "archive_count",
        "lss_fmcwr_2_archive_total_size_bytes": "archive_total_size_bytes",
        "lss_fmcwr_2_rar_entry_count": "rar_entry_count",
        "lss_fmcwr_2_mat_file_count": "mat_file_count",
        "lss_fmcwr_2_directory_count": "directory_count",
        "lss_fmcwr_2_band_counts": "band_counts",
        "lss_fmcwr_2_recording_stem_count": "recording_stem_count",
        "lss_fmcwr_2_candidate_group_count": "candidate_group_count",
        "lss_fmcwr_2_candidate_groups_authoritative": (
            "candidate_groups_authoritative"
        ),
        "lss_fmcwr_2_independent_recording_or_session_key_available": (
            "independent_recording_or_session_key_available"
        ),
        "lss_fmcwr_2_unique_raw_mat_count": "unique_raw_mat_count",
        "lss_fmcwr_2_raw_duplicate_group_count": "raw_duplicate_group_count",
        "lss_fmcwr_2_raw_duplicate_member_count": "raw_duplicate_member_count",
        "lss_fmcwr_2_unique_numeric_payload_count": "unique_numeric_payload_count",
        "lss_fmcwr_2_numeric_duplicate_group_count": (
            "numeric_duplicate_group_count"
        ),
        "lss_fmcwr_2_numeric_duplicate_member_count": (
            "numeric_duplicate_member_count"
        ),
        "lss_fmcwr_2_channel_shape_dtype_counts": "channel_shape_dtype_counts",
        "lss_fmcwr_2_target_recording_stem_counts": "target_recording_stem_counts",
        "lss_fmcwr_2_target_candidate_group_counts": "target_candidate_group_counts",
        "lss_fmcwr_2_gates": "gates",
        "lss_fmcwr_2_raw_complex_iq_available_for_all_records": (
            "raw_complex_iq_available_for_all_records"
        ),
        "lss_fmcwr_2_h_v_polarimetry_available": "h_v_polarimetry_available",
        "lss_fmcwr_2_target_aspect_angle_verified": "target_aspect_angle_verified",
        "lss_fmcwr_2_natural_bird_evidence_available": (
            "natural_bird_evidence_available"
        ),
        "lss_fmcwr_2_global_sampling_rate_available": "global_sampling_rate_available",
        "lss_fmcwr_2_global_carrier_frequency_available": "global_carrier_frequency_available",
        "lss_fmcwr_2_physical_doppler_hz_axis_available": (
            "physical_doppler_hz_axis_available"
        ),
        "lss_fmcwr_2_physical_velocity_axis_available": (
            "physical_velocity_axis_available"
        ),
        "lss_fmcwr_2_complete_md_stft_implementation_available": (
            "complete_md_stft_implementation_available"
        ),
        "lss_fmcwr_2_simulated_bird_archive_is_simulation_only": (
            "simulated_bird_archive_is_simulation_only"
        ),
        "lss_fmcwr_2_random_mat_frame_or_window_split_allowed": (
            "random_mat_frame_or_window_split_allowed"
        ),
        "lss_fmcwr_2_model_training_allowed": "model_training_allowed",
        "lss_fmcwr_2_model_training_performed": "model_training_performed",
        "lss_fmcwr_2_source_archives_extracted_to_disk": (
            "source_archives_extracted_to_disk"
        ),
        "lss_fmcwr_2_source_archives_modified": "source_archives_modified",
    }
    for manifest_key, summary_key in fmcwr_manifest_fields.items():
        assert rules[manifest_key] == fmcwr[summary_key]
    assert rules["lss_fmcwr_2_read_only_audit_included"] is True
    assert rules["lss_fmcwr_2_raw_data_included"] is False
    assert rules["lss_fmcwr_2_member_level_outputs_included"] is False
    assert rules["lss_fmcwr_2_normalized_axis_contract_only"] is True
    assert rules["lss_fmcwr_2_normalized_processing_contract_version"] == 1
    assert rules["lss_fmcwr_2_normalized_processing_performance_reported"] is False
    assert rules["lss_fmcwr_2_normalized_processing_code_included"] is True
    assert rules["external_public_data_registries_included"] is True
    assert rules["team_onboarding_manual_included"] is True
    assert rules["team_qualification_policy_included"] is True
    assert rules["team_qualification_scorecard_included"] is True
    assert rules["team_task_claim_template_included"] is True
    assert rules["team_weekly_report_template_included"] is True


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


def test_share_audit_rejects_sensitive_markers_in_binary_metadata(
    tmp_path: Path,
) -> None:
    package = tmp_path / "share"
    package.mkdir()
    (package / "figure.png").write_bytes(
        b"not-a-real-png\x00creator=/home/example/project"
    )
    with pytest.raises(ValueError, match="sensitive marker"):
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
