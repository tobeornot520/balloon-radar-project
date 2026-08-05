#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "balloon_radar_results_and_team_onboarding_20260805_v9"
DIST_ROOT = PROJECT_ROOT / "dist"
DEFAULT_OUTPUT_DIR = DIST_ROOT / PACKAGE_NAME
DEFAULT_ZIP_PATH = DIST_ROOT / f"{PACKAGE_NAME}.zip"
STAGING_ROOT = DIST_ROOT / ".share-package-staging"
PACKAGE_DATE = "2026-08-05"
ZIP_TIMESTAMP = (2026, 8, 5, 0, 0, 0)
CURRENT_DATA_READINESS = (
    PROJECT_ROOT
    / "results"
    / "data_audit"
    / "data_collection_readiness_v1"
    / "preflight.json"
)
CURRENT_DATA_MANIFEST = (
    PROJECT_ROOT
    / "results"
    / "data_audit"
    / "dataset_v4_multifold"
    / "fold_01_manifest.csv"
)
DATA_CONTRACT = PROJECT_ROOT / "configs" / "data_collection_contract_v1.json"
DATA_CONTRACT_VALIDATOR = PROJECT_ROOT / "scripts" / "validate_data_collection_manifest.py"
LSS_DAUR_AUDIT_SUMMARY = (
    PROJECT_ROOT / "results" / "data_audit" / "lss_daur_v1" / "summary.json"
)
LSS_HSR_AUDIT_SUMMARY = (
    PROJECT_ROOT / "results" / "data_audit" / "lss_hsr_l_v2" / "summary.json"
)
LSS_FMCWR_AUDIT_SUMMARY = (
    PROJECT_ROOT / "results" / "data_audit" / "lss_fmcwr_2_v1" / "summary.json"
)

ALLOWED_SUFFIXES = {".md", ".csv", ".png", ".pdf", ".json", ".txt", ".py"}
FORBIDDEN_SUFFIXES = {
    ".mat",
    ".h5",
    ".hdf5",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".log",
    ".doc",
    ".docx",
    ".zip",
    ".rar",
    ".7z",
    ".tar",
    ".gz",
}
FORBIDDEN_NAME_PARTS = {
    ".git",
    "raw_transcript",
    "chat_transcript",
    "聊天记录",
}
SENSITIVE_TEXT_MARKERS = (
    "/home/",
    "tobeornot8259748",
    "C:\\Users\\",
    "BEGIN PRIVATE KEY",
    "api_key=",
    "password=",
    "secret=",
)
MARKDOWN_LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


@dataclass(frozen=True)
class PackageFile:
    source: str
    destination: str
    category: str


PACKAGE_FILES = (
    PackageFile("docs/share/README_SHARE_ZH.md", "README.md", "share_document"),
    PackageFile(
        "docs/share/TEAM_START_HERE.md",
        "TEAM_START_HERE.md",
        "team_onboarding_manual",
    ),
    PackageFile(
        "docs/share/TEAM_QUALIFICATION_AND_ROLE_SCREENING_ZH.md",
        "docs/19_TEAM_QUALIFICATION_AND_ROLE_SCREENING_ZH.md",
        "team_qualification_policy",
    ),
    PackageFile(
        "configs/team_onboarding_checklist_template_v1.csv",
        "assets/templates/team_onboarding_checklist_template_v1.csv",
        "team_onboarding_template",
    ),
    PackageFile(
        "configs/team_qualification_scorecard_template_v1.csv",
        "assets/templates/team_qualification_scorecard_template_v1.csv",
        "team_qualification_template",
    ),
    PackageFile(
        "configs/team_task_claim_template_v1.csv",
        "assets/templates/team_task_claim_template_v1.csv",
        "team_task_template",
    ),
    PackageFile(
        "docs/share/TEAM_WEEKLY_REPORT_TEMPLATE_ZH.md",
        "assets/templates/TEAM_WEEKLY_REPORT_TEMPLATE_ZH.md",
        "team_reporting_template",
    ),
    PackageFile(
        "docs/share/01_PROJECT_OVERVIEW_ZH.md",
        "docs/01_PROJECT_OVERVIEW_ZH.md",
        "share_document",
    ),
    PackageFile(
        "docs/share/02_DEVELOPMENT_HISTORY_ZH.md",
        "docs/02_DEVELOPMENT_HISTORY_ZH.md",
        "share_document",
    ),
    PackageFile(
        "docs/HISTORICAL_PROJECT_RECONSTRUCTION_20260801.md",
        "docs/02A_HISTORICAL_PROJECT_RECONSTRUCTION_ZH.md",
        "historical_reconstruction",
    ),
    PackageFile(
        "docs/PROJECT_TASK_LEDGER.md",
        "docs/12_PROJECT_TASK_LEDGER_ZH.md",
        "project_governance_document",
    ),
    PackageFile(
        "docs/RECOMMENDED_PAPERS_20260805.md",
        "docs/20_RECOMMENDED_PAPERS_20260805.md",
        "reading_registry",
    ),
    PackageFile(
        "docs/TEAM_REPRODUCTION_GUIDE_ZH.md",
        "docs/13_TEAM_REPRODUCTION_GUIDE_ZH.md",
        "reproduction_guide",
    ),
    PackageFile(
        "docs/MULTIDOMAIN_FEATURE_GATE_V1.md",
        "docs/14_MULTIDOMAIN_FEATURE_GATE_V1_ZH.md",
        "feature_governance_document",
    ),
    PackageFile(
        "docs/PAPER_MAINLINE_V1.md",
        "docs/15_PAPER_MAINLINE_V1_ZH.md",
        "paper_scope_document",
    ),
    PackageFile(
        "docs/EXTERNAL_FACT_REQUEST_MESSAGE_V1.md",
        "docs/16_EXTERNAL_FACT_REQUEST_MESSAGE_V1_ZH.md",
        "external_fact_request",
    ),
    PackageFile(
        "docs/NEXT_STAGE_PLAN_20260803.md",
        "docs/17_NEXT_STAGE_PLAN_20260803_ZH.md",
        "next_stage_plan",
    ),
    PackageFile(
        "docs/TIAN_REPRODUCTION_FAILURE_AND_ALTERNATIVES_20260803.md",
        "docs/18_TIAN_REPRODUCTION_FAILURE_AND_ALTERNATIVES_20260803_ZH.md",
        "reproduction_failure_and_fallback_decision",
    ),
    PackageFile(
        "docs/ZERO_DOPPLER_P0_REVIEW_PRESCREEN_V1.md",
        "evidence/21_ZERO_DOPPLER_P0_REVIEW_PRESCREEN.md",
        "human_review_prescreen",
    ),
    PackageFile(
        "docs/EXTERNAL_PUBLIC_DATA_AUDIT_20260803.md",
        "docs/EXTERNAL_PUBLIC_DATA_AUDIT_20260803.md",
        "external_public_data_audit",
    ),
    PackageFile(
        "docs/LSS_FMCWR_2_READ_ONLY_AUDIT_20260805.md",
        "docs/LSS_FMCWR_2_READ_ONLY_AUDIT_20260805.md",
        "external_public_data_audit",
    ),
    PackageFile(
        "docs/LSS_FMCWR_2_NORMALIZED_PROCESSING_CONTRACT_20260805.md",
        "docs/LSS_FMCWR_2_NORMALIZED_PROCESSING_CONTRACT_20260805.md",
        "fmcwr_normalized_processing_contract",
    ),
    PackageFile(
        "configs/lss_fmcwr_normalized_processing_contract_v1.json",
        "assets/contracts/lss_fmcwr_normalized_processing_contract_v1.json",
        "fmcwr_normalized_processing_contract",
    ),
    PackageFile(
        "scripts/process_lss_fmcwr_normalized_v1.py",
        "scripts/process_lss_fmcwr_normalized_v1.py",
        "fmcwr_normalized_processing_code",
    ),
    PackageFile(
        "docs/LAT_MRICD_GROUPED_BASELINE_PROTOCOL_V1.md",
        "docs/LAT_MRICD_GROUPED_BASELINE_PROTOCOL_V1.md",
        "grouped_public_data_baseline_protocol",
    ),
    PackageFile(
        "results/data_audit/lat_mricd_v1/REPORT.md",
        "evidence/22_LAT_MRICD_DATA_AUDIT.md",
        "external_data_audit_report",
    ),
    PackageFile(
        "results/data_audit/lat_mricd_v1/summary.json",
        "evidence/22_LAT_MRICD_DATA_AUDIT.json",
        "external_data_audit_manifest",
    ),
    PackageFile(
        "results/data_audit/lat_mricd_v1/category_split_readiness.csv",
        "assets/tables/lat_mricd_category_split_readiness.csv",
        "external_data_audit_table",
    ),
    PackageFile(
        "results/data_audit/lat_mricd_v1/batch_code_collisions.csv",
        "assets/tables/lat_mricd_batch_code_collisions.csv",
        "external_data_audit_table",
    ),
    PackageFile(
        "results/final_evidence/lat_mricd_grouped_baselines_v1/REPORT.md",
        "evidence/23_LAT_MRICD_GROUPED_BASELINES.md",
        "grouped_public_data_baseline_report",
    ),
    PackageFile(
        "results/final_evidence/lat_mricd_grouped_baselines_v1/evidence_manifest.json",
        "evidence/23_LAT_MRICD_GROUPED_BASELINES.json",
        "grouped_public_data_baseline_manifest",
    ),
    PackageFile(
        "results/final_evidence/lat_mricd_grouped_baselines_v1/tables/aggregate_metrics.csv",
        "assets/tables/lat_mricd_grouped_aggregate_metrics.csv",
        "grouped_public_data_baseline_table",
    ),
    PackageFile(
        "results/final_evidence/lat_mricd_grouped_baselines_v1/tables/batch_class_distribution.csv",
        "assets/tables/lat_mricd_grouped_batch_class_distribution.csv",
        "grouped_public_data_baseline_table",
    ),
    PackageFile(
        "results/final_evidence/lat_mricd_grouped_baselines_v1/tables/batch_class_metrics.csv",
        "assets/tables/lat_mricd_grouped_batch_class_metrics.csv",
        "grouped_public_data_baseline_table",
    ),
    PackageFile(
        "results/final_evidence/lat_mricd_grouped_baselines_v1/tables/claim_boundaries.csv",
        "assets/tables/lat_mricd_grouped_claim_boundaries.csv",
        "grouped_public_data_baseline_table",
    ),
    PackageFile(
        "results/final_evidence/lat_mricd_grouped_baselines_v1/tables/cluster_bootstrap_intervals.csv",
        "assets/tables/lat_mricd_grouped_cluster_bootstrap_intervals.csv",
        "grouped_public_data_baseline_table",
    ),
    PackageFile(
        "results/final_evidence/lat_mricd_grouped_baselines_v1/tables/confusion_matrices.csv",
        "assets/tables/lat_mricd_grouped_confusion_matrices.csv",
        "grouped_public_data_baseline_table",
    ),
    PackageFile(
        "results/final_evidence/lat_mricd_grouped_baselines_v1/tables/feature_definitions.csv",
        "assets/tables/lat_mricd_grouped_feature_definitions.csv",
        "grouped_public_data_baseline_table",
    ),
    PackageFile(
        "results/final_evidence/lat_mricd_grouped_baselines_v1/tables/feature_importance.csv",
        "assets/tables/lat_mricd_grouped_feature_importance.csv",
        "grouped_public_data_baseline_table",
    ),
    PackageFile(
        "results/final_evidence/lat_mricd_grouped_baselines_v1/tables/feature_summary_by_category.csv",
        "assets/tables/lat_mricd_grouped_feature_summary_by_category.csv",
        "grouped_public_data_baseline_table",
    ),
    PackageFile(
        "results/final_evidence/lat_mricd_grouped_baselines_v1/tables/fold_coverage.csv",
        "assets/tables/lat_mricd_grouped_fold_coverage.csv",
        "grouped_public_data_baseline_table",
    ),
    PackageFile(
        "results/final_evidence/lat_mricd_grouped_baselines_v1/tables/fold_metrics.csv",
        "assets/tables/lat_mricd_grouped_fold_metrics.csv",
        "grouped_public_data_baseline_table",
    ),
    PackageFile(
        "results/final_evidence/lat_mricd_grouped_baselines_v1/tables/split_manifest.csv",
        "assets/tables/lat_mricd_grouped_split_manifest.csv",
        "grouped_public_data_baseline_table",
    ),
    PackageFile(
        "results/final_evidence/lat_mricd_grouped_baselines_v1/tables/subtype_pressure.csv",
        "assets/tables/lat_mricd_grouped_subtype_pressure.csv",
        "grouped_public_data_baseline_table",
    ),
    PackageFile(
        "docs/LAT_MRICD_CROSS_BAND_TRANSFER_PROTOCOL_V1.md",
        "docs/LAT_MRICD_CROSS_BAND_TRANSFER_PROTOCOL_V1.md",
        "cross_band_transfer_protocol",
    ),
    PackageFile(
        "configs/lat_mricd_cross_band_transfer_v1.json",
        "assets/contracts/lat_mricd_cross_band_transfer_v1.json",
        "cross_band_transfer_contract",
    ),
    PackageFile(
        "results/final_evidence/lat_mricd_cross_band_transfer_v1.run_consumed.json",
        "evidence/24_LAT_MRICD_CROSS_BAND_TRANSFER_RUN_CONSUMED.json",
        "sealed_run_consumption_record",
    ),
    PackageFile(
        "results/final_evidence/lat_mricd_cross_band_transfer_v1/REPORT.md",
        "evidence/24_LAT_MRICD_CROSS_BAND_TRANSFER.md",
        "cross_band_transfer_report",
    ),
    PackageFile(
        "results/final_evidence/lat_mricd_cross_band_transfer_v1/evidence_manifest.json",
        "evidence/24_LAT_MRICD_CROSS_BAND_TRANSFER_MANIFEST.json",
        "cross_band_transfer_evidence_manifest",
    ),
    PackageFile(
        "results/final_evidence/lat_mricd_cross_band_transfer_v1/gate_decision.json",
        "evidence/24_LAT_MRICD_CROSS_BAND_TRANSFER_GATE.json",
        "cross_band_transfer_gate_decision",
    ),
    PackageFile(
        "results/final_evidence/lat_mricd_cross_band_transfer_v1/model_fit_manifest.json",
        "evidence/24_LAT_MRICD_CROSS_BAND_TRANSFER_MODEL_FIT.json",
        "cross_band_transfer_model_fit_manifest",
    ),
    PackageFile(
        "results/final_evidence/lat_mricd_cross_band_transfer_v1/summary.json",
        "evidence/24_LAT_MRICD_CROSS_BAND_TRANSFER_SUMMARY.json",
        "cross_band_transfer_summary",
    ),
    PackageFile(
        "results/final_evidence/lat_mricd_cross_band_transfer_v1/tables/aggregate_metrics.csv",
        "assets/tables/lat_mricd_cross_band_aggregate_metrics.csv",
        "cross_band_transfer_table",
    ),
    PackageFile(
        "results/final_evidence/lat_mricd_cross_band_transfer_v1/tables/bootstrap_intervals.csv",
        "assets/tables/lat_mricd_cross_band_bootstrap_intervals.csv",
        "cross_band_transfer_table",
    ),
    PackageFile(
        "results/final_evidence/lat_mricd_cross_band_transfer_v1/tables/claim_boundaries.csv",
        "assets/tables/lat_mricd_cross_band_claim_boundaries.csv",
        "cross_band_transfer_table",
    ),
    PackageFile(
        "results/final_evidence/lat_mricd_cross_band_transfer_v1/tables/confusion_matrices.csv",
        "assets/tables/lat_mricd_cross_band_confusion_matrices.csv",
        "cross_band_transfer_table",
    ),
    PackageFile(
        "results/final_evidence/lat_mricd_cross_band_transfer_v1/tables/disjoint_sensitivity.csv",
        "assets/tables/lat_mricd_cross_band_disjoint_sensitivity.csv",
        "cross_band_transfer_table",
    ),
    PackageFile(
        "results/final_evidence/lat_mricd_cross_band_transfer_v1/tables/feature_definitions.csv",
        "assets/tables/lat_mricd_cross_band_feature_definitions.csv",
        "cross_band_transfer_table",
    ),
    PackageFile(
        "results/final_evidence/lat_mricd_cross_band_transfer_v1/tables/feature_importance.csv",
        "assets/tables/lat_mricd_cross_band_feature_importance.csv",
        "cross_band_transfer_table",
    ),
    PackageFile(
        "results/final_evidence/lat_mricd_cross_band_transfer_v1/tables/raw_batch_overlap_audit.csv",
        "assets/tables/lat_mricd_cross_band_raw_batch_overlap_audit.csv",
        "cross_band_transfer_table",
    ),
    PackageFile(
        "results/final_evidence/lat_mricd_cross_band_transfer_v1/tables/target_batch_class_metrics.csv",
        "assets/tables/lat_mricd_cross_band_target_batch_class_metrics.csv",
        "cross_band_transfer_table",
    ),
    PackageFile(
        "results/final_evidence/lat_mricd_cross_band_transfer_v1/tables/training_weight_audit.csv",
        "assets/tables/lat_mricd_cross_band_training_weight_audit.csv",
        "cross_band_transfer_table",
    ),
    PackageFile(
        "results/final_evidence/lat_mricd_cross_band_transfer_v1/tables/transfer_coverage.csv",
        "assets/tables/lat_mricd_cross_band_transfer_coverage.csv",
        "cross_band_transfer_table",
    ),
    PackageFile(
        "results/data_audit/dronerfc_mm_v1/README.md",
        "evidence/25_DRONERFC_MM_READ_ONLY_AUDIT.md",
        "dronerfc_mm_read_only_audit_report",
    ),
    PackageFile(
        "results/data_audit/dronerfc_mm_v1/summary.json",
        "evidence/25_DRONERFC_MM_READ_ONLY_AUDIT.json",
        "dronerfc_mm_read_only_audit_summary",
    ),
    PackageFile(
        "results/data_audit/dronerfc_mm_v1/recording_audit.csv",
        "assets/tables/dronerfc_mm_recording_audit.csv",
        "dronerfc_mm_recording_audit_table",
    ),
    PackageFile(
        "results/data_audit/lss_daur_v1/REPORT.md",
        "evidence/26_LSS_DAUR_READ_ONLY_AUDIT.md",
        "lss_daur_read_only_audit_report",
    ),
    PackageFile(
        "results/data_audit/lss_daur_v1/summary.json",
        "evidence/26_LSS_DAUR_READ_ONLY_AUDIT.json",
        "lss_daur_read_only_audit_summary",
    ),
    PackageFile(
        "results/data_audit/lss_daur_v1/source_session_group_audit.csv",
        "assets/tables/lss_daur_source_session_group_audit.csv",
        "lss_daur_aggregate_audit_table",
    ),
    PackageFile(
        "results/data_audit/lss_daur_v1/class_summary.csv",
        "assets/tables/lss_daur_class_summary.csv",
        "lss_daur_aggregate_audit_table",
    ),
    PackageFile(
        "results/data_audit/lss_daur_v1/doppler_config_audit.csv",
        "assets/tables/lss_daur_doppler_config_audit.csv",
        "lss_daur_aggregate_audit_table",
    ),
    PackageFile(
        "results/data_audit/lss_hsr_l_v2/REPORT.md",
        "evidence/27_LSS_HSR_L_V2_READ_ONLY_AUDIT.md",
        "lss_hsr_l_v2_read_only_audit_report",
    ),
    PackageFile(
        "results/data_audit/lss_hsr_l_v2/summary.json",
        "evidence/27_LSS_HSR_L_V2_READ_ONLY_AUDIT.json",
        "lss_hsr_l_v2_read_only_audit_summary",
    ),
    PackageFile(
        "results/data_audit/lss_hsr_l_v2/split_summary.csv",
        "assets/tables/lss_hsr_l_v2_split_summary.csv",
        "lss_hsr_l_v2_aggregate_audit_table",
    ),
    PackageFile(
        "results/data_audit/lss_hsr_l_v2/split_class_summary.csv",
        "assets/tables/lss_hsr_l_v2_split_class_summary.csv",
        "lss_hsr_l_v2_aggregate_audit_table",
    ),
    PackageFile(
        "results/data_audit/lss_hsr_l_v2/feature_summary.csv",
        "assets/tables/lss_hsr_l_v2_feature_summary.csv",
        "lss_hsr_l_v2_aggregate_audit_table",
    ),
    PackageFile(
        "results/data_audit/lss_fmcwr_2_v1/REPORT.md",
        "evidence/28_LSS_FMCWR_2_V1_READ_ONLY_AUDIT.md",
        "lss_fmcwr_2_v1_read_only_audit_report",
    ),
    PackageFile(
        "results/data_audit/lss_fmcwr_2_v1/summary.json",
        "evidence/28_LSS_FMCWR_2_V1_READ_ONLY_AUDIT.json",
        "lss_fmcwr_2_v1_read_only_audit_summary",
    ),
    PackageFile(
        "results/data_audit/lss_fmcwr_2_v1/archive_audit.csv",
        "assets/tables/lss_fmcwr_2_v1_archive_audit.csv",
        "lss_fmcwr_2_v1_aggregate_audit_table",
    ),
    PackageFile(
        "results/data_audit/lss_fmcwr_2_v1/group_summary.csv",
        "assets/tables/lss_fmcwr_2_v1_group_summary.csv",
        "lss_fmcwr_2_v1_aggregate_audit_table",
    ),
    PackageFile(
        "data/metadata/external_public_datasets_v1.csv",
        "assets/registries/external_public_datasets_v1.csv",
        "external_public_data_registry",
    ),
    PackageFile(
        "data/metadata/external_public_artifacts_v1.csv",
        "assets/registries/external_public_artifacts_v1.csv",
        "external_public_artifact_registry",
    ),
    PackageFile(
        "configs/current_direction_completion_v1.json",
        "assets/contracts/current_direction_completion_v1.json",
        "direction_completion_contract",
    ),
    PackageFile(
        "docs/share/03_RESULTS_AND_EVIDENCE_ZH.md",
        "docs/03_RESULTS_AND_EVIDENCE_ZH.md",
        "share_document",
    ),
    PackageFile(
        "docs/share/04_SHARING_TALK_TRACK_ZH.md",
        "docs/04_SHARING_TALK_TRACK_ZH.md",
        "share_document",
    ),
    PackageFile(
        "docs/share/05_REPRODUCTION_AND_NEXT_STEPS_ZH.md",
        "docs/05_REPRODUCTION_AND_NEXT_STEPS_ZH.md",
        "share_document",
    ),
    PackageFile(
        "docs/share/00_ONE_PAGE_SUMMARY_ZH.md",
        "docs/00_ONE_PAGE_SUMMARY_ZH.md",
        "share_document",
    ),
    PackageFile(
        "docs/share/06_RECENT_PROGRESS_AND_FAILURE_ANALYSIS_ZH.md",
        "docs/09_RECENT_PROGRESS_AND_FAILURE_ANALYSIS_ZH.md",
        "share_document",
    ),
    PackageFile(
        "docs/share/07_QUESTIONS_FOR_SENIOR_ZH.md",
        "docs/10_QUESTIONS_FOR_SENIOR_ZH.md",
        "share_document",
    ),
    PackageFile(
        "docs/share/08_DATA_REQUEST_CHECKLIST_ZH.md",
        "docs/11_DATA_REQUEST_CHECKLIST_ZH.md",
        "share_document",
    ),
    PackageFile(
        "docs/DATA_CARD.md",
        "docs/06_DATA_CARD_ZH.md",
        "governance_document",
    ),
    PackageFile(
        "docs/METRIC_DEFINITIONS.md",
        "docs/07_METRIC_DEFINITIONS_ZH.md",
        "governance_document",
    ),
    PackageFile(
        "docs/MODEL_SELECTION_LEDGER.md",
        "docs/08_MODEL_SELECTION_LEDGER_ZH.md",
        "governance_document",
    ),
    PackageFile(
        "docs/NEW_DATA_COLLECTION_PROTOCOL.md",
        "docs/NEW_DATA_COLLECTION_PROTOCOL.md",
        "governance_document",
    ),
    PackageFile(
        "docs/EXPERIMENT_RECORDING_PROTOCOL.md",
        "docs/EXPERIMENT_RECORDING_PROTOCOL.md",
        "governance_document",
    ),
    PackageFile(
        "docs/FIELD_COLLECTION_SOP_V1.md",
        "docs/FIELD_COLLECTION_SOP_V1.md",
        "field_protocol",
    ),
    PackageFile(
        "results/final_evidence/bc_dpg_v3_final/FINAL_EVIDENCE_REPORT.md",
        "evidence/01_BC_DPG_V3_FINAL_REPORT.md",
        "frozen_report",
    ),
    PackageFile(
        "docs/polarimetric_stage3/STAGE3_FROZEN_CONCLUSION.md",
        "evidence/02_POLARIMETRIC_STAGE3_FROZEN_CONCLUSION.md",
        "frozen_report",
    ),
    PackageFile(
        "results/final_evidence/roi_stage4_twofold/STAGE4_TWOFOLD_FROZEN_ANALYSIS.md",
        "evidence/03_ROI_STAGE4_TWOFOLD_FROZEN_ANALYSIS.md",
        "frozen_report",
    ),
    PackageFile(
        "results/final_evidence/roi_bc_dpg_joint_fixed_threshold/JOINT_AUDIT_REPORT.md",
        "evidence/04_ROI_BC_DPG_JOINT_AUDIT_REPORT.md",
        "frozen_report",
    ),
    PackageFile(
        "results/data_audit/bc_dpg_v3_causal_context_audit/CAUSAL_CONTEXT_AUDIT.md",
        "evidence/05_BC_DPG_V3_CAUSAL_CONTEXT_AUDIT.md",
        "post_test_sensitivity_report",
    ),
    PackageFile(
        "results/data_audit/detection_acquisition_order/ACQUISITION_ORDER_AUDIT.md",
        "evidence/06_DETECTION_ACQUISITION_ORDER_AUDIT.md",
        "causal_training_readiness_report",
    ),
    PackageFile(
        "results/final_evidence/bc_dpg_localization/LOCALIZATION_EVIDENCE_REPORT.md",
        "evidence/07_BC_DPG_LOCALIZATION_EVIDENCE.md",
        "frozen_localization_report",
    ),
    PackageFile(
        "results/data_audit/data_collection_readiness_v1/PRECHECK_REPORT.md",
        "evidence/08_CURRENT_DATA_COLLECTION_READINESS.md",
        "data_contract_readiness_report",
    ),
    PackageFile(
        "results/data_audit/data_collection_readiness_v1/preflight.json",
        "evidence/08_CURRENT_DATA_COLLECTION_READINESS.json",
        "data_contract_readiness_manifest",
    ),
    PackageFile(
        "results/data_audit/multidomain_feature_catalog_v1/REPORT.md",
        "evidence/09_MULTIDOMAIN_FEATURE_CATALOG.md",
        "development_feature_audit",
    ),
    PackageFile(
        "docs/TIAN_FCN_FOLD1_DIAGNOSTIC_CONCLUSION.md",
        "evidence/10_TIAN_FCN_FOLD1_DIAGNOSTIC_CONCLUSION.md",
        "failed_reproduction_diagnostic",
    ),
    PackageFile(
        "docs/TIAN_FCN_FOLD1_COMPONENT_MECHANISM.md",
        "evidence/11_TIAN_FCN_FOLD1_COMPONENT_MECHANISM.md",
        "failed_reproduction_mechanism",
    ),
    PackageFile(
        "docs/TIAN_FCN_REPRODUCTION_PROTOCOL.md",
        "evidence/12_TIAN_FCN_REPRODUCTION_PROTOCOL.md",
        "reproduction_protocol",
    ),
    PackageFile(
        "docs/TIAN_FCN_REPRODUCTION_CONDITIONS_REQUEST.md",
        "evidence/13_TIAN_FCN_REPRODUCTION_CONDITIONS_REQUEST.md",
        "consultation_request",
    ),
    PackageFile(
        "results/data_audit/zero_doppler_candidate_veto_v1/REPORT.md",
        "evidence/14_ZERO_DOPPLER_CANDIDATE_VETO.md",
        "post_test_mechanism_diagnostic",
    ),
    PackageFile(
        "results/data_audit/zero_doppler_mechanism_v1/REPORT_frozen_sixfold_baseline_fixed.md",
        "evidence/15_ZERO_DOPPLER_FROZEN_SIXFOLD.md",
        "development_mechanism_comparison",
    ),
    PackageFile(
        "results/data_audit/zero_doppler_mechanism_v1/REPORT_comparison_fold01_04_all.md",
        "evidence/16_ZERO_DOPPLER_FOLD01_04_COMPARISON.md",
        "development_mechanism_gate",
    ),
    PackageFile(
        "docs/ZERO_DOPPLER_MECHANISM_V1_CONCLUSION.md",
        "evidence/17_ZERO_DOPPLER_MECHANISM_CONCLUSION.md",
        "development_decision",
    ),
    PackageFile(
        "docs/POLARIMETRIC_TRANSFER_ENCODER_V1.md",
        "evidence/18_POLARIMETRIC_TRANSFER_ENCODER.md",
        "architecture_preparation",
    ),
    PackageFile(
        "docs/FIELD_COLLECTION_SOP_V1.md",
        "evidence/19_FIELD_COLLECTION_SOP.md",
        "field_protocol",
    ),
    PackageFile(
        "docs/FIELD_CAPABILITY_REQUEST_V1.md",
        "evidence/20_FIELD_CAPABILITY_REQUEST.md",
        "field_consultation_request",
    ),
    PackageFile(
        "results/final_evidence/bc_dpg_v3_final/figures/fig1_deployment_false_alarms.png",
        "assets/figures/bc_dpg_deployment_false_alarms.png",
        "figure",
    ),
    PackageFile(
        "results/final_evidence/bc_dpg_v3_final/figures/fig2_ablation_false_alarms.png",
        "assets/figures/bc_dpg_ablation_false_alarms.png",
        "figure",
    ),
    PackageFile(
        "results/final_evidence/bc_dpg_v3_final/figures/fig3_false_alarm_target_shift_tradeoff.png",
        "assets/figures/bc_dpg_false_alarm_target_shift_tradeoff.png",
        "figure",
    ),
    PackageFile(
        "results/final_evidence/roi_bc_dpg_joint_fixed_threshold/figures/fig1_pooled_detection_tradeoff.png",
        "assets/figures/joint_pooled_detection_tradeoff.png",
        "figure",
    ),
    PackageFile(
        "results/final_evidence/roi_bc_dpg_joint_fixed_threshold/figures/fig1_pooled_detection_tradeoff.pdf",
        "assets/figures/joint_pooled_detection_tradeoff.pdf",
        "figure",
    ),
    PackageFile(
        "results/final_evidence/roi_bc_dpg_joint_fixed_threshold/figures/fig2_fold_heterogeneity.png",
        "assets/figures/joint_fold_heterogeneity.png",
        "figure",
    ),
    PackageFile(
        "results/final_evidence/roi_bc_dpg_joint_fixed_threshold/figures/fig2_fold_heterogeneity.pdf",
        "assets/figures/joint_fold_heterogeneity.pdf",
        "figure",
    ),
    PackageFile(
        "results/final_evidence/roi_bc_dpg_joint_fixed_threshold/figures/fig3_complementarity.png",
        "assets/figures/joint_complementarity.png",
        "figure",
    ),
    PackageFile(
        "results/final_evidence/roi_bc_dpg_joint_fixed_threshold/figures/fig3_complementarity.pdf",
        "assets/figures/joint_complementarity.pdf",
        "figure",
    ),
    PackageFile(
        "results/final_evidence/bc_dpg_localization/figures/fig1_localization_error_cdf.png",
        "assets/figures/bc_dpg_localization_error_cdf.png",
        "figure",
    ),
    PackageFile(
        "results/final_evidence/bc_dpg_localization/figures/fig1_localization_error_cdf.png",
        "evidence/figures/fig1_localization_error_cdf.png",
        "report_figure",
    ),
    PackageFile(
        "results/final_evidence/bc_dpg_localization/figures/fig1_localization_error_cdf.pdf",
        "assets/figures/bc_dpg_localization_error_cdf.pdf",
        "figure",
    ),
    PackageFile(
        "results/final_evidence/bc_dpg_v3_final/tables/table_01_main_model_comparison.csv",
        "assets/tables/bc_dpg_main_model_comparison.csv",
        "table",
    ),
    PackageFile(
        "results/final_evidence/bc_dpg_v3_final/tables/table_03_ablation_summary.csv",
        "assets/tables/bc_dpg_ablation_summary.csv",
        "table",
    ),
    PackageFile(
        "results/final_evidence/roi_stage4_twofold/paper_table_stage4_twofold_main.csv",
        "assets/tables/roi_stage4_twofold_main.csv",
        "table",
    ),
    PackageFile(
        "results/final_evidence/roi_bc_dpg_joint_fixed_threshold/tables/table_01_pooled_detection.csv",
        "assets/tables/joint_pooled_detection.csv",
        "table",
    ),
    PackageFile(
        "results/final_evidence/roi_bc_dpg_joint_fixed_threshold/tables/table_02_fold_detection.csv",
        "assets/tables/joint_fold_detection.csv",
        "table",
    ),
    PackageFile(
        "results/final_evidence/roi_bc_dpg_joint_fixed_threshold/tables/table_03_complementarity.csv",
        "assets/tables/joint_complementarity.csv",
        "table",
    ),
    PackageFile(
        "results/final_evidence/roi_bc_dpg_joint_fixed_threshold/tables/table_04_simple_combination_diagnostics.csv",
        "assets/tables/joint_combination_diagnostics.csv",
        "table",
    ),
    PackageFile(
        "results/final_evidence/roi_bc_dpg_joint_fixed_threshold/tables/table_05_claim_boundaries.csv",
        "assets/tables/joint_claim_boundaries.csv",
        "table",
    ),
    PackageFile(
        "results/final_evidence/roi_bc_dpg_joint_fixed_threshold/tables/table_06_fold_distribution_summary.csv",
        "assets/tables/joint_fold_distribution_summary.csv",
        "table",
    ),
    PackageFile(
        "results/final_evidence/roi_bc_dpg_joint_fixed_threshold/tables/table_07_derived_metrics_and_wilson_ci.csv",
        "assets/tables/joint_derived_metrics_and_wilson_ci.csv",
        "table",
    ),
    PackageFile(
        "results/final_evidence/roi_bc_dpg_joint_fixed_threshold/tables/table_08_scan_group_bootstrap.csv",
        "assets/tables/joint_scan_group_bootstrap.csv",
        "table",
    ),
    PackageFile(
        "results/final_evidence/roi_bc_dpg_joint_fixed_threshold/tables/table_09_paired_mcnemar_diagnostics.csv",
        "assets/tables/joint_paired_mcnemar_diagnostics.csv",
        "table",
    ),
    PackageFile(
        "results/data_audit/bc_dpg_v3_causal_context_audit/context_metrics_aggregate.csv",
        "assets/tables/bc_dpg_causal_context_aggregate.csv",
        "post_test_sensitivity_table",
    ),
    PackageFile(
        "results/data_audit/bc_dpg_v3_causal_context_audit/paired_deltas_vs_complete_scan.csv",
        "assets/tables/bc_dpg_causal_context_paired_deltas.csv",
        "post_test_sensitivity_table",
    ),
    PackageFile(
        "results/data_audit/bc_dpg_v3_causal_context_audit/complete_replay_validation.csv",
        "assets/tables/bc_dpg_causal_context_replay_validation.csv",
        "replay_validation_table",
    ),
    PackageFile(
        "results/data_audit/bc_dpg_v3_causal_context_audit/history_coverage_by_fold.csv",
        "assets/tables/bc_dpg_causal_context_history_coverage.csv",
        "post_test_sensitivity_table",
    ),
    PackageFile(
        "results/data_audit/detection_acquisition_order/order_source_summary.csv",
        "assets/tables/detection_acquisition_order_sources.csv",
        "causal_training_readiness_table",
    ),
    PackageFile(
        "results/final_evidence/bc_dpg_localization/tables/table_01_pooled_localization.csv",
        "assets/tables/bc_dpg_localization_pooled.csv",
        "frozen_localization_table",
    ),
    PackageFile(
        "results/final_evidence/bc_dpg_localization/tables/table_02_fold_localization.csv",
        "assets/tables/bc_dpg_localization_by_fold.csv",
        "frozen_localization_table",
    ),
    PackageFile(
        "results/final_evidence/bc_dpg_localization/tables/table_03_error_distribution.csv",
        "assets/tables/bc_dpg_localization_error_distribution.csv",
        "frozen_localization_table",
    ),
    PackageFile(
        "results/final_evidence/bc_dpg_localization/tables/table_04_distance_strata.csv",
        "assets/tables/bc_dpg_localization_distance_strata.csv",
        "frozen_localization_table",
    ),
    PackageFile(
        "results/final_evidence/bc_dpg_localization/tables/table_05_velocity_strata.csv",
        "assets/tables/bc_dpg_localization_velocity_strata.csv",
        "frozen_localization_table",
    ),
    PackageFile(
        "results/data_audit/data_collection_readiness_v1/column_coverage.csv",
        "assets/tables/current_data_collection_contract_coverage.csv",
        "data_contract_readiness_table",
    ),
    PackageFile(
        "configs/data_collection_contract_v1.json",
        "assets/contracts/data_collection_contract_v1.json",
        "data_collection_contract",
    ),
    PackageFile(
        "configs/data_collection_manifest_template_v1.csv",
        "assets/templates/data_collection_manifest_template_v1.csv",
        "data_collection_template",
    ),
    PackageFile(
        "results/data_audit/multidomain_feature_catalog_v1/feature_schema.csv",
        "assets/tables/multidomain_feature_schema.csv",
        "development_feature_table",
    ),
    PackageFile(
        "results/data_audit/multidomain_feature_catalog_v1/detection_univariate_separability.csv",
        "assets/tables/multidomain_detection_separability.csv",
        "development_feature_table",
    ),
    PackageFile(
        "results/data_audit/multidomain_feature_catalog_v1/detection_background_group_stress.csv",
        "assets/tables/multidomain_background_group_stress.csv",
        "development_feature_table",
    ),
    PackageFile(
        "results/data_audit/zero_doppler_candidate_veto_v1/radius_tradeoff.csv",
        "assets/tables/zero_doppler_candidate_veto_tradeoff.csv",
        "post_test_mechanism_table",
    ),
    PackageFile(
        "results/data_audit/zero_doppler_mechanism_v1/aggregate_frozen_sixfold_baseline_fixed.csv",
        "assets/tables/zero_doppler_frozen_sixfold.csv",
        "development_mechanism_table",
    ),
    PackageFile(
        "results/data_audit/zero_doppler_mechanism_v1/aggregate_comparison_fold01_04_all.csv",
        "assets/tables/zero_doppler_fold01_04_comparison.csv",
        "development_mechanism_table",
    ),
    PackageFile(
        "configs/field_capability_response_template_v1.csv",
        "assets/templates/field_capability_response_template_v1.csv",
        "field_template",
    ),
    PackageFile(
        "configs/field_readiness_evidence_template_v1.csv",
        "assets/templates/field_readiness_evidence_template_v1.csv",
        "field_template",
    ),
    PackageFile(
        "configs/pilot_scenario_matrix_v1.csv",
        "assets/templates/pilot_scenario_matrix_v1.csv",
        "field_template",
    ),
    PackageFile(
        "configs/pilot_session_log_template_v1.csv",
        "assets/templates/pilot_session_log_template_v1.csv",
        "field_template",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a sanitized project-overview share package without raw data, "
            "checkpoints, predictions, logs, or development transcripts."
        )
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def load_current_data_readiness() -> dict[str, object]:
    payload = json.loads(CURRENT_DATA_READINESS.read_text(encoding="utf-8"))
    expected = {
        "profile": "locked_evaluation",
        "status": "FAIL",
        "formal_causal_training_gate_open": False,
        "locked_evaluation_gate_open": False,
        "row_count": 1148,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(
                f"Unexpected current data readiness {field}: {payload.get(field)!r}"
            )
    hashes = (
        (payload.get("input", {}).get("sha256"), CURRENT_DATA_MANIFEST, "input"),
        (payload.get("contract", {}).get("sha256"), DATA_CONTRACT, "contract"),
        (
            payload.get("implementation", {}).get("sha256"),
            DATA_CONTRACT_VALIDATOR,
            "implementation",
        ),
    )
    for recorded, path, label in hashes:
        actual = sha256_file(path)
        if recorded != actual:
            raise ValueError(
                f"Current data readiness {label} hash is stale; rebuild precheck"
            )
    gates = payload.get("gates", {})
    if gates.get("schema") != "FAIL" or any(
        status != "BLOCKED" for gate, status in gates.items() if gate != "schema"
    ):
        raise ValueError("Unexpected current data readiness gate states")
    return payload


def load_lss_daur_audit() -> dict[str, object]:
    payload = json.loads(LSS_DAUR_AUDIT_SUMMARY.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "dataset_id": "lss_daur_1_0",
        "release_version": "V3",
        "status": "PASS_SCHEMA_PAIRING_BLOCKED_GROUPING_AND_PHYSICAL_AXIS",
        "official_file_count": 314,
        "official_size_bytes": 148763512,
        "canonical_mat_file_count": 154,
        "backup_mat_file_count": 154,
        "paired_track_count": 77,
        "unique_signal_trajectory_content_count": 76,
        "frame_count": 11366,
        "doppler_value_count": 7728640,
        "doppler_512_track_count": 58,
        "doppler_1024_track_count": 19,
        "duplicate_time_step_count": 894,
        "unique_time_position_count": 10472,
        "tracks_with_noncontiguous_frame_counter": 13,
        "frame_counter_gap_event_count": 85,
        "frame_counter_missing_value_count": 94,
        "frame_counter_repeat_event_count": 2,
        "filename_header_date_mismatch_count": 6,
        "filename_session_candidate_count": 45,
        "header_date_session_candidate_count": 40,
        "header_date_scene_group_count": 24,
        "header_date_scene_class_pure_group_count": 20,
        "candidate_source_session_group_count": 39,
        "bird_uav_filename_session_overlap_count": 0,
        "bird_uav_header_date_session_overlap_count": 0,
        "bird_uav_connected_session_overlap_count": 0,
        "shared_frame_record_pair_count": 11,
        "exact_duplicate_recording_group_count": 1,
        "exact_duplicate_recording_count": 2,
        "v_field_constant_zero": True,
        "canonical_backup_observation_count_multiplier_allowed": False,
        "canonical_backup_as_extra_samples_allowed": False,
        "random_mat_split_allowed": False,
        "random_frame_or_window_split_allowed": False,
        "td_tr_split_allowed": False,
        "authoritative_session_key_available": False,
        "absolute_weather_join_allowed": False,
        "raw_adc_or_iq_available": False,
        "h_v_polarimetry_available": False,
        "physical_micro_doppler_hz_allowed": False,
        "model_training_allowed": False,
        "source_files_modified": False,
        "sample_level_outputs_included": False,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(
                f"Unexpected LSS-DAUR audit {field}: {payload.get(field)!r}"
            )
    expected_gates = {
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
    if payload.get("gates") != expected_gates:
        raise ValueError("Unexpected LSS-DAUR audit gate states")
    return payload


def load_lss_hsr_audit() -> dict[str, object]:
    payload = json.loads(LSS_HSR_AUDIT_SUMMARY.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "dataset_id": "lss_hsr_l",
        "release_version": "V2",
        "audit_mode": "strict_release",
        "release_identity_verified": True,
        "data_doi": "10.57760/sciencedb.radars.00063",
        "status": "PASS_SCHEMA_BLOCKED_SOURCE_PROVENANCE_AND_PHYSICAL_AXIS",
        "archive_size_bytes": 237020946,
        "archive_sha256": (
            "fea8a21354110a96fb9644dc1c69649b6dc6d1a1b6da512498d9c2d74d839540"
        ),
        "zip_entry_count": 1561,
        "zip_file_count": 1534,
        "zip_directory_count": 27,
        "zip_integrity_passed": True,
        "zip_paths_safe": True,
        "mat_file_count": 1530,
        "route_count": 865,
        "route_track_reference_count": 1530,
        "unique_raw_mat_count": 1530,
        "unique_numeric_payload_count": 1530,
        "exact_raw_mat_duplicate_group_count": 0,
        "exact_numeric_payload_duplicate_group_count": 0,
        "total_frame_count": 63148,
        "doppler_value_count": 32331776,
        "track_value_count": 315740,
        "published_window_count": 55231,
        "window_size": 10,
        "first_frame_repeat": 4,
        "authoritative_route_id_available": True,
        "every_mat_assigned_to_exactly_one_route": True,
        "published_train_validation_route_disjoint": True,
        "published_train_validation_route_overlap_count": 0,
        "published_train_validation_session_disjoint_verified": False,
        "published_split_preservation_required": True,
        "overflow_merge_allowed": False,
        "overflow_isolated": True,
        "overflow_role_documented_in_published_statistics": False,
        "acquisition_session_key_available": False,
        "random_mat_split_allowed": False,
        "random_frame_or_window_split_allowed": False,
        "lowest_available_grouping_key": "route_id",
        "minimum_split_unit": "UNRESOLVED_SOURCE_SESSION_IDENTITY_UNAVAILABLE",
        "route_id_sufficient_for_independent_evaluation": False,
        "dpl_representation": "processed real 512-bin Doppler waterfall",
        "dpl_amplitude_unit_verified": False,
        "dpl_physical_time_axis_available": False,
        "dpl_physical_doppler_hz_axis_available": False,
        "dpl_physical_velocity_axis_available": False,
        "track_physical_units_available": True,
        "track_physical_units_verified": True,
        "raw_adc_or_iq_available": False,
        "h_v_polarimetry_available": False,
        "physical_micro_doppler_hz_allowed": False,
        "model_training_allowed": False,
        "model_training_performed": False,
        "official_dataset_py_executed": False,
        "source_archive_extracted": False,
        "source_archive_modified": False,
        "raw_data_included_in_outputs": False,
        "local_route_and_mat_audit_tables_included": True,
        "journal_bundle_mixing_allowed": False,
        "allowed_use": (
            "read-only loader/schema verification, route-grouped method design, and "
            "exact reproduction of published window counts"
        ),
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(
                f"Unexpected LSS-HSR-L V2 audit {field}: {payload.get(field)!r}"
            )

    expected_gates = {
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
    expected_mappings = {
        "gates": expected_gates,
        "step_length_counts": {"1": 802, "5": 63},
        "split_mat_counts": {"overflow": 11, "train": 1269, "validation": 250},
        "split_route_counts": {"overflow": 11, "train": 723, "validation": 131},
        "split_frame_counts": {
            "overflow": 704,
            "train": 51789,
            "validation": 10655,
        },
        "split_published_window_counts": {
            "overflow": 529,
            "train": 45366,
            "validation": 9336,
        },
        "mat_schema": {
            "public_fields": ["Trace_DPL_Data"],
            "dpl_shape": "[T, 512]",
            "track_shape": "[T, 5]",
            "storage_dtype": "float64",
            "real_values": True,
            "all_values_finite": True,
        },
        "track_features": [
            {"index": 1, "name": "radial_velocity", "unit": "m/s"},
            {"index": 2, "name": "range", "unit": "km"},
            {"index": 3, "name": "azimuth", "unit": "degree"},
            {"index": 4, "name": "height", "unit": "m"},
            {"index": 5, "name": "normalized_snr", "unit": "dB"},
        ],
        "blockers": [
            "overflow is present in V2 but omitted from the published train/validation "
            "statistics and Dataset.py main loop",
            "route_id exists but acquisition day, flight, scene, and source-session "
            "identities are unavailable",
            "the processed DPL has no machine-readable CPI duration, PRF, or "
            "bin-to-Hz/velocity mapping",
            "no preregistered modeling protocol has been approved for this release",
        ],
        "prohibited_claims": [
            "random MAT/frame/window split performance",
            "overflow as train, validation, test, or independent evidence",
            "independent-session or deployment generalization",
            "physical-Hz or physical-time micro-Doppler features",
            "raw IQ, H/V polarimetry, or balloon recognition",
            "model performance before a separate preregistered protocol",
        ],
    }
    for field, value in expected_mappings.items():
        if payload.get(field) != value:
            raise ValueError(f"Unexpected LSS-HSR-L V2 audit {field}")
    return payload


def _summary_contains_forbidden_member_keys(value: object) -> bool:
    forbidden = {
        "member_path",
        "members_json",
        "raw_sha256",
        "numeric_payload_sha256",
        "candidate_group_id",
        "recording_stem",
        "ordinal",
        "crc32",
    }
    if isinstance(value, dict):
        return any(
            key in forbidden or _summary_contains_forbidden_member_keys(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_summary_contains_forbidden_member_keys(item) for item in value)
    return False


def load_lss_fmcwr_audit() -> dict[str, object]:
    """Load and freeze only aggregate, share-safe FMCWR audit facts."""

    payload = json.loads(LSS_FMCWR_AUDIT_SUMMARY.read_text(encoding="utf-8"))
    if _summary_contains_forbidden_member_keys(payload):
        raise ValueError("LSS-FMCWR audit summary contains member-level detail")
    expected = {
        "schema_version": 1,
        "dataset_id": "lss_fmcwr_2_0",
        "target_release_version": "V4",
        "release_version": "V4",
        "target_data_doi": "10.57760/sciencedb.radars.00054",
        "data_doi": "10.57760/sciencedb.radars.00054",
        "status": "PASS_ARCHIVE_SCHEMA_BLOCKED_GROUPING_PROVENANCE_AND_PHYSICAL_AXIS",
        "audit_mode": "strict_release",
        "release_identity_verified": True,
        "archive_count": 6,
        "archive_total_size_bytes": 1013456629,
        "rar_entry_count": 116,
        "mat_file_count": 90,
        "directory_count": 26,
        "uncompressed_mat_size_bytes": 1041307141,
        "packed_mat_size_bytes": 1013436117,
        "band_counts": {"K": 64, "L": 26},
        "recording_stem_count": 66,
        "candidate_group_count": 48,
        "candidate_group_cross_target_count": 0,
        "candidate_groups_authoritative": False,
        "independent_recording_or_session_key_available": False,
        "path_filename_angle_conflict_count": 1,
        "unique_raw_mat_count": 71,
        "raw_duplicate_group_count": 11,
        "raw_duplicate_member_count": 30,
        "raw_cross_target_duplicate_group_count": 0,
        "unique_numeric_payload_count": 71,
        "numeric_duplicate_group_count": 11,
        "numeric_duplicate_member_count": 30,
        "numeric_cross_target_duplicate_group_count": 0,
        "k_channel_a_complex_verified": True,
        "l_channel_a_real_verified": True,
        "raw_complex_iq_available_for_all_records": False,
        "h_v_polarimetry_available": False,
        "target_aspect_angle_verified": False,
        "natural_bird_evidence_available": False,
        "global_sampling_rate_available": False,
        "global_carrier_frequency_available": False,
        "physical_doppler_hz_axis_available": False,
        "physical_velocity_axis_available": False,
        "complete_md_stft_implementation_available": False,
        "simulated_bird_archive_is_simulation_only": True,
        "random_mat_frame_or_window_split_allowed": False,
        "model_training_allowed": False,
        "model_training_performed": False,
        "source_archives_extracted_to_disk": False,
        "source_archives_modified": False,
        "raw_data_included_in_outputs": False,
        "member_level_outputs_local_only": True,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(
                f"Unexpected LSS-FMCWR audit {field}: {payload.get(field)!r}"
            )
    expected_gates = {
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
    if payload.get("gates") != expected_gates:
        raise ValueError("Unexpected LSS-FMCWR audit gate states")
    expected_targets = {
        "ac311": 4,
        "hexacopter": 15,
        "inspire2": 15,
        "m350": 15,
        "mavic2": 15,
        "simulated_bird": 2,
    }
    if payload.get("target_recording_stem_counts") != expected_targets:
        raise ValueError("Unexpected LSS-FMCWR target stem counts")
    expected_groups = {
        "ac311": 4,
        "hexacopter": 10,
        "inspire2": 12,
        "m350": 10,
        "mavic2": 10,
        "simulated_bird": 2,
    }
    if payload.get("target_candidate_group_counts") != expected_groups:
        raise ValueError("Unexpected LSS-FMCWR candidate group counts")
    expected_shapes = [
        {"band_token": "K", "channelA_shape": "150x6000", "channelA_dtype": "<c16", "channelA_complex": True, "channelB_shape": "0x0", "channelB_dtype": "|u1", "channelB_complex": False, "mat_count": 29},
        {"band_token": "K", "channelA_shape": "2000x5704", "channelA_dtype": "<c16", "channelA_complex": True, "channelB_shape": "0x0", "channelB_dtype": "|u1", "channelB_complex": False, "mat_count": 1},
        {"band_token": "K", "channelA_shape": "2000x6000", "channelA_dtype": "<c16", "channelA_complex": True, "channelB_shape": "0x0", "channelB_dtype": "|u1", "channelB_complex": False, "mat_count": 3},
        {"band_token": "K", "channelA_shape": "500x102400", "channelA_dtype": "<c16", "channelA_complex": True, "channelB_shape": "0x0", "channelB_dtype": "|u1", "channelB_complex": False, "mat_count": 1},
        {"band_token": "K", "channelA_shape": "500x43439", "channelA_dtype": "<c16", "channelA_complex": True, "channelB_shape": "0x0", "channelB_dtype": "|u1", "channelB_complex": False, "mat_count": 1},
        {"band_token": "K", "channelA_shape": "500x6000", "channelA_dtype": "<c16", "channelA_complex": True, "channelB_shape": "0x0", "channelB_dtype": "|u1", "channelB_complex": False, "mat_count": 16},
        {"band_token": "K", "channelA_shape": "512x6000", "channelA_dtype": "<c16", "channelA_complex": True, "channelB_shape": "0x0", "channelB_dtype": "|u1", "channelB_complex": False, "mat_count": 13},
        {"band_token": "L", "channelA_shape": "150x4000", "channelA_dtype": "<f8", "channelA_complex": False, "channelB_shape": "0x0", "channelB_dtype": "|u1", "channelB_complex": False, "mat_count": 5},
        {"band_token": "L", "channelA_shape": "150x6000", "channelA_dtype": "<f8", "channelA_complex": False, "channelB_shape": "0x0", "channelB_dtype": "|u1", "channelB_complex": False, "mat_count": 21},
    ]
    if payload.get("channel_shape_dtype_counts") != expected_shapes:
        raise ValueError("Unexpected LSS-FMCWR channel shape/type counts")
    return payload


def validate_source_map(files: Iterable[PackageFile] = PACKAGE_FILES) -> None:
    files = tuple(files)
    destinations = [item.destination for item in files]
    if len(destinations) != len(set(destinations)):
        raise ValueError("Package file map contains duplicate destinations")

    hsr_source_prefix = "results/data_audit/lss_hsr_l_v2/"
    expected_hsr_map = {
        f"{hsr_source_prefix}REPORT.md": (
            "evidence/27_LSS_HSR_L_V2_READ_ONLY_AUDIT.md"
        ),
        f"{hsr_source_prefix}summary.json": (
            "evidence/27_LSS_HSR_L_V2_READ_ONLY_AUDIT.json"
        ),
        f"{hsr_source_prefix}split_summary.csv": (
            "assets/tables/lss_hsr_l_v2_split_summary.csv"
        ),
        f"{hsr_source_prefix}split_class_summary.csv": (
            "assets/tables/lss_hsr_l_v2_split_class_summary.csv"
        ),
        f"{hsr_source_prefix}feature_summary.csv": (
            "assets/tables/lss_hsr_l_v2_feature_summary.csv"
        ),
    }
    actual_hsr_map = {
        item.source: item.destination
        for item in files
        if item.source.startswith(hsr_source_prefix)
    }
    if actual_hsr_map != expected_hsr_map:
        raise ValueError(
            "LSS-HSR-L V2 share evidence must contain exactly the five approved "
            "aggregate source/destination mappings"
        )
    if any(
        Path(path).name in {"route_audit.csv", "mat_audit.csv"}
        for item in files
        if item.source.startswith(hsr_source_prefix)
        for path in (item.source, item.destination)
    ):
        raise ValueError("LSS-HSR-L route/MAT detail tables are forbidden")

    fmcwr_source_prefix = "results/data_audit/lss_fmcwr_2_v1/"
    expected_fmcwr_map = {
        f"{fmcwr_source_prefix}REPORT.md": (
            "evidence/28_LSS_FMCWR_2_V1_READ_ONLY_AUDIT.md"
        ),
        f"{fmcwr_source_prefix}summary.json": (
            "evidence/28_LSS_FMCWR_2_V1_READ_ONLY_AUDIT.json"
        ),
        f"{fmcwr_source_prefix}archive_audit.csv": (
            "assets/tables/lss_fmcwr_2_v1_archive_audit.csv"
        ),
        f"{fmcwr_source_prefix}group_summary.csv": (
            "assets/tables/lss_fmcwr_2_v1_group_summary.csv"
        ),
    }
    actual_fmcwr_map = {
        item.source: item.destination
        for item in files
        if item.source.startswith(fmcwr_source_prefix)
    }
    if actual_fmcwr_map != expected_fmcwr_map:
        raise ValueError(
            "LSS-FMCWR-2.0 share evidence must contain exactly the four approved "
            "aggregate source/destination mappings"
        )

    fmcwr_report_map = {
        item.source: item.destination
        for item in files
        if item.source == "docs/LSS_FMCWR_2_READ_ONLY_AUDIT_20260805.md"
    }
    if fmcwr_report_map != {
        "docs/LSS_FMCWR_2_READ_ONLY_AUDIT_20260805.md": (
            "docs/LSS_FMCWR_2_READ_ONLY_AUDIT_20260805.md"
        )
    }:
        raise ValueError("The standalone FMCWR read-only audit document is required")

    forbidden_fmcwr_names = {
        "mat_audit.csv",
        "duplicate_groups.csv",
    }
    if any(
        Path(path).name in forbidden_fmcwr_names
        for item in files
        if item.source.startswith(fmcwr_source_prefix)
        for path in (item.source, item.destination)
    ):
        raise ValueError("LSS-FMCWR member-level audit tables are forbidden")
    if any(
        item.source.startswith("data/raw/") or item.destination.startswith("data/raw/")
        for item in files
    ):
        raise ValueError("Raw data paths are forbidden in the share package")

    missing: list[str] = []
    errors: list[str] = []
    for item in files:
        source = PROJECT_ROOT / item.source
        destination = Path(item.destination)
        if not source.is_file():
            missing.append(item.source)
        if destination.is_absolute() or ".." in destination.parts:
            errors.append(f"unsafe destination: {item.destination}")
        if destination.suffix.lower() not in ALLOWED_SUFFIXES:
            errors.append(f"unsupported destination type: {item.destination}")
        if destination.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden destination type: {item.destination}")
    if missing:
        raise FileNotFoundError(f"Missing share-package sources: {missing}")
    if errors:
        raise ValueError("Invalid share-package mapping: " + "; ".join(errors))


def ensure_output_available(output_dir: Path, zip_path: Path) -> None:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Package directory already exists: {output_dir}")
    if zip_path.exists() or zip_path.is_symlink():
        raise FileExistsError(f"Package ZIP already exists: {zip_path}")


def validate_output_paths(output_dir: Path, zip_path: Path) -> None:
    frozen_output_dir = PROJECT_ROOT / "dist" / PACKAGE_NAME
    frozen_zip_path = PROJECT_ROOT / "dist" / f"{PACKAGE_NAME}.zip"
    if output_dir != frozen_output_dir or zip_path != frozen_zip_path:
        raise ValueError(
            "Share-package outputs are frozen to the project dist directory"
        )


def prepare_staging_root() -> Path:
    expected_dist_root = PROJECT_ROOT / "dist"
    expected_staging_root = expected_dist_root / ".share-package-staging"
    if DIST_ROOT != expected_dist_root or STAGING_ROOT != expected_staging_root:
        raise ValueError("Share-package staging root is not the frozen project path")
    if DIST_ROOT.is_symlink():
        raise ValueError("Share-package dist root must not be a symlink")
    DIST_ROOT.mkdir(parents=True, exist_ok=True)
    if not DIST_ROOT.is_dir():
        raise ValueError("Share-package dist root is not a directory")
    if STAGING_ROOT.is_symlink():
        raise ValueError("Share-package staging root must not be a symlink")
    STAGING_ROOT.mkdir(exist_ok=True)
    if not STAGING_ROOT.is_dir():
        raise ValueError("Share-package staging root is not a directory")
    if STAGING_ROOT.resolve().parent != DIST_ROOT.resolve():
        raise ValueError("Share-package staging root escapes the project dist directory")
    return STAGING_ROOT.resolve()


def cleanup_staging_directory(staging_dir: Path) -> None:
    staging_root = STAGING_ROOT.resolve()
    if staging_dir.is_symlink():
        raise ValueError("Refusing to clean a symlinked staging directory")
    resolved = staging_dir.resolve()
    expected_prefix = f".{PACKAGE_NAME}.build-"
    if resolved.parent != staging_root or not resolved.name.startswith(expected_prefix):
        raise ValueError("Refusing to clean outside the fixed staging root")
    if resolved.exists():
        shutil.rmtree(resolved)


def copy_package_files(staging_dir: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for item in PACKAGE_FILES:
        source = PROJECT_ROOT / item.source
        destination = staging_dir / item.destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        records.append(
            {
                "category": item.category,
                "packaged_path": item.destination,
                "repository_source": item.source,
                "size_bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
        )
    return records


def write_manifest(staging_dir: Path, records: list[dict[str, object]]) -> None:
    readiness = load_current_data_readiness()
    daur = load_lss_daur_audit()
    hsr = load_lss_hsr_audit()
    fmcwr = load_lss_fmcwr_audit()
    manifest = {
        "schema_version": 1,
        "package_name": PACKAGE_NAME,
        "package_date": PACKAGE_DATE,
        "language": "zh-CN",
        "purpose": (
            "sanitized results analysis, technical consultation, and zero-context "
            "team-onboarding package with traceable frozen evidence, task requirements, "
            "and clearly labeled development diagnostics; not a self-contained "
            "reproduction package"
        ),
        "source_commit": current_commit(),
        "current_scope": "H/V UAV detection, localization, and false-alarm suppression front end",
        "long_term_scope": "balloon payload and motion-state recognition after new data collection",
        "evidence_rules": {
            "test_threshold_retuning": False,
            "joint_model_trained": False,
            "and_or_rules_selected": False,
            "complete_scan_bc_is_causal": False,
            "causal_context_audit_role": "post-hoc frozen-checkpoint sensitivity",
            "causal_context_retraining_performed": False,
            "causal_history_window_selected": False,
            "leave_one_out_is_causal": False,
            "past_only_order_verified_by_timestamp": False,
            "past_only_order_columns": ["beam_layer", "azimuth_deg", "sample_id"],
            "formal_causal_training_gate_open": False,
            "verified_within_scan_order_available": False,
            "causal_development_smoke_test_split_loaded": False,
            "causal_development_smoke_test_split_evaluated": False,
            "localization_evidence_role": "frozen aggregate analysis",
            "localization_training_performed": False,
            "localization_inference_performed": False,
            "localization_test_threshold_retuning": False,
            "localization_coordinates_match_raw_dpg": True,
            "localization_sample_predictions_included": False,
            "new_data_collection_contract_version": 1,
            "current_manifest_contract_profile": readiness["profile"],
            "current_manifest_contract_status": readiness["status"],
            "current_manifest_missing_contract_columns": len(
                readiness["missing_columns"]
            ),
            "current_formal_causal_training_gate_open": readiness[
                "formal_causal_training_gate_open"
            ],
            "current_locked_evaluation_gate_open": readiness[
                "locked_evaluation_gate_open"
            ],
            "evaluation_role": "internal development estimate",
            "stage4_development_folds_reused_in_sixfold": [1, 4],
            "class_and_acquisition_date_confounded": True,
            "tian_reproduction_successful": False,
            "tian_exact_reproduction_status": "blocked_unverifiable",
            "tian_reproduction_conditions_available": False,
            "tian_point_gt_role": "validation-only local-transfer ablation",
            "tian_fallback_mainline": "DPG-FCN zero-Doppler development plus grouped LAT-MRICD baselines",
            "zero_doppler_candidate_veto_role": "post-test mechanism diagnostic",
            "zero_doppler_fixed_notch_role": "development safety reference",
            "zero_doppler_learned_sixfold_authorized": False,
            "polarimetric_transfer_checkpoint_available": False,
            "absolute_polarimetric_calibration_verified": False,
            "physical_micro_doppler_timing_verified": False,
            "field_readiness_gate_open": False,
            "lat_mricd_raw_data_included": False,
            "lat_mricd_random_row_split_allowed": False,
            "lat_mricd_physical_micro_doppler_hz_allowed": False,
            "lat_mricd_grouped_baseline_included": True,
            "lat_mricd_group_key": [
                "representation",
                "band_code",
                "batch_code",
            ],
            "lat_mricd_sample_predictions_included": False,
            "lat_mricd_cross_band_transfer_included": True,
            "lat_mricd_cross_band_sealed_run_consumed": True,
            "lat_mricd_cross_band_primary_gate_passed": False,
            "lat_mricd_cross_band_target_bands_consumed": ["S", "Ku"],
            "lat_mricd_cross_band_same_target_confirmatory_reuse_allowed": False,
            "lat_mricd_cross_band_raw_data_included": False,
            "lat_mricd_cross_band_sample_predictions_included": False,
            "dronerfc_mm_read_only_audit_included": True,
            "dronerfc_mm_audit_status": "PASS_SCHEMA_BLOCKED_TIMESTAMP_ALIGNMENT",
            "dronerfc_mm_raw_data_included": False,
            "dronerfc_mm_sample_level_outputs_included": False,
            "dronerfc_mm_training_performed": False,
            "dronerfc_mm_model_training_allowed": False,
            "dronerfc_mm_blocked_recordings": ["B1"],
            "dronerfc_mm_b1_supervised_alignment_allowed": False,
            "dronerfc_mm_random_frame_window_split_allowed": False,
            "dronerfc_mm_group_key": "split_family_group",
            "dronerfc_mm_minimum_split_unit": "split_family_group",
            "dronerfc_mm_split_family_group_count": 6,
            "lss_daur_read_only_audit_included": True,
            "lss_daur_audit_status": daur["status"],
            "lss_daur_paired_observation_count": daur["paired_track_count"],
            "lss_daur_unique_signal_trajectory_content_count": daur[
                "unique_signal_trajectory_content_count"
            ],
            "lss_daur_candidate_source_session_group_count": daur[
                "candidate_source_session_group_count"
            ],
            "lss_daur_canonical_mat_file_count": daur["canonical_mat_file_count"],
            "lss_daur_backup_mat_file_count": daur["backup_mat_file_count"],
            "lss_daur_canonical_backup_observation_multiplier_allowed": daur[
                "canonical_backup_observation_count_multiplier_allowed"
            ],
            "lss_daur_canonical_backup_as_extra_samples_allowed": daur[
                "canonical_backup_as_extra_samples_allowed"
            ],
            "lss_daur_authoritative_session_key_available": daur[
                "authoritative_session_key_available"
            ],
            "lss_daur_random_mat_split_allowed": daur["random_mat_split_allowed"],
            "lss_daur_random_frame_window_split_allowed": daur[
                "random_frame_or_window_split_allowed"
            ],
            "lss_daur_td_tr_split_allowed": daur["td_tr_split_allowed"],
            "lss_daur_physical_micro_doppler_hz_allowed": daur[
                "physical_micro_doppler_hz_allowed"
            ],
            "lss_daur_model_training_allowed": daur["model_training_allowed"],
            "lss_daur_raw_data_included": False,
            "lss_daur_sample_level_outputs_included": daur[
                "sample_level_outputs_included"
            ],
            "lss_hsr_l_v2_read_only_audit_included": True,
            "lss_hsr_l_v2_audit_status": hsr["status"],
            "lss_hsr_l_v2_release_version": hsr["release_version"],
            "lss_hsr_l_v2_release_identity_verified": hsr[
                "release_identity_verified"
            ],
            "lss_hsr_l_v2_data_doi": hsr["data_doi"],
            "lss_hsr_l_v2_archive_size_bytes": hsr["archive_size_bytes"],
            "lss_hsr_l_v2_archive_sha256": hsr["archive_sha256"],
            "lss_hsr_l_v2_mat_file_count": hsr["mat_file_count"],
            "lss_hsr_l_v2_total_frame_count": hsr["total_frame_count"],
            "lss_hsr_l_v2_route_count": hsr["route_count"],
            "lss_hsr_l_v2_published_window_count": hsr[
                "published_window_count"
            ],
            "lss_hsr_l_v2_split_mat_counts": hsr["split_mat_counts"],
            "lss_hsr_l_v2_split_route_counts": hsr["split_route_counts"],
            "lss_hsr_l_v2_split_frame_counts": hsr["split_frame_counts"],
            "lss_hsr_l_v2_split_published_window_counts": hsr[
                "split_published_window_counts"
            ],
            "lss_hsr_l_v2_gates": hsr["gates"],
            "lss_hsr_l_v2_authoritative_route_id_available": hsr[
                "authoritative_route_id_available"
            ],
            "lss_hsr_l_v2_every_mat_assigned_to_exactly_one_route": hsr[
                "every_mat_assigned_to_exactly_one_route"
            ],
            "lss_hsr_l_v2_published_train_validation_route_disjoint": hsr[
                "published_train_validation_route_disjoint"
            ],
            "lss_hsr_l_v2_published_train_validation_route_overlap_count": hsr[
                "published_train_validation_route_overlap_count"
            ],
            "lss_hsr_l_v2_lowest_available_grouping_key": hsr[
                "lowest_available_grouping_key"
            ],
            "lss_hsr_l_v2_minimum_split_unit": hsr["minimum_split_unit"],
            "lss_hsr_l_v2_acquisition_session_key_available": hsr[
                "acquisition_session_key_available"
            ],
            "lss_hsr_l_v2_published_session_disjoint_verified": hsr[
                "published_train_validation_session_disjoint_verified"
            ],
            "lss_hsr_l_v2_route_id_sufficient_for_independent_evaluation": hsr[
                "route_id_sufficient_for_independent_evaluation"
            ],
            "lss_hsr_l_v2_published_split_preservation_required": hsr[
                "published_split_preservation_required"
            ],
            "lss_hsr_l_v2_overflow_isolated": hsr["overflow_isolated"],
            "lss_hsr_l_v2_overflow_merge_allowed": hsr[
                "overflow_merge_allowed"
            ],
            "lss_hsr_l_v2_overflow_role_documented": hsr[
                "overflow_role_documented_in_published_statistics"
            ],
            "lss_hsr_l_v2_random_mat_split_allowed": hsr[
                "random_mat_split_allowed"
            ],
            "lss_hsr_l_v2_random_frame_window_split_allowed": hsr[
                "random_frame_or_window_split_allowed"
            ],
            "lss_hsr_l_v2_dpl_amplitude_unit_verified": hsr[
                "dpl_amplitude_unit_verified"
            ],
            "lss_hsr_l_v2_dpl_physical_time_axis_available": hsr[
                "dpl_physical_time_axis_available"
            ],
            "lss_hsr_l_v2_dpl_physical_doppler_hz_axis_available": hsr[
                "dpl_physical_doppler_hz_axis_available"
            ],
            "lss_hsr_l_v2_dpl_physical_velocity_axis_available": hsr[
                "dpl_physical_velocity_axis_available"
            ],
            "lss_hsr_l_v2_track_physical_units_verified": hsr[
                "track_physical_units_verified"
            ],
            "lss_hsr_l_v2_physical_micro_doppler_hz_allowed": hsr[
                "physical_micro_doppler_hz_allowed"
            ],
            "lss_hsr_l_v2_raw_adc_or_iq_available": hsr[
                "raw_adc_or_iq_available"
            ],
            "lss_hsr_l_v2_h_v_polarimetry_available": hsr[
                "h_v_polarimetry_available"
            ],
            "lss_hsr_l_v2_model_training_allowed": hsr[
                "model_training_allowed"
            ],
            "lss_hsr_l_v2_model_training_performed": hsr[
                "model_training_performed"
            ],
            "lss_hsr_l_v2_source_archive_extracted": hsr[
                "source_archive_extracted"
            ],
            "lss_hsr_l_v2_source_archive_modified": hsr[
                "source_archive_modified"
            ],
            "lss_hsr_l_v2_journal_bundle_mixing_allowed": hsr[
                "journal_bundle_mixing_allowed"
            ],
            "lss_hsr_l_v2_raw_data_included": False,
            "lss_hsr_l_v2_route_mat_detail_tables_included": False,
            "lss_hsr_l_v2_sample_level_outputs_included": False,
            "lss_fmcwr_2_read_only_audit_included": True,
            "lss_fmcwr_2_audit_status": fmcwr["status"],
            "lss_fmcwr_2_target_release_version": fmcwr[
                "target_release_version"
            ],
            "lss_fmcwr_2_release_identity_verified": fmcwr[
                "release_identity_verified"
            ],
            "lss_fmcwr_2_data_doi": fmcwr["data_doi"],
            "lss_fmcwr_2_archive_count": fmcwr["archive_count"],
            "lss_fmcwr_2_archive_total_size_bytes": fmcwr[
                "archive_total_size_bytes"
            ],
            "lss_fmcwr_2_rar_entry_count": fmcwr["rar_entry_count"],
            "lss_fmcwr_2_mat_file_count": fmcwr["mat_file_count"],
            "lss_fmcwr_2_directory_count": fmcwr["directory_count"],
            "lss_fmcwr_2_band_counts": fmcwr["band_counts"],
            "lss_fmcwr_2_recording_stem_count": fmcwr[
                "recording_stem_count"
            ],
            "lss_fmcwr_2_candidate_group_count": fmcwr[
                "candidate_group_count"
            ],
            "lss_fmcwr_2_candidate_groups_authoritative": fmcwr[
                "candidate_groups_authoritative"
            ],
            "lss_fmcwr_2_independent_recording_or_session_key_available": fmcwr[
                "independent_recording_or_session_key_available"
            ],
            "lss_fmcwr_2_unique_raw_mat_count": fmcwr[
                "unique_raw_mat_count"
            ],
            "lss_fmcwr_2_raw_duplicate_group_count": fmcwr[
                "raw_duplicate_group_count"
            ],
            "lss_fmcwr_2_raw_duplicate_member_count": fmcwr[
                "raw_duplicate_member_count"
            ],
            "lss_fmcwr_2_unique_numeric_payload_count": fmcwr[
                "unique_numeric_payload_count"
            ],
            "lss_fmcwr_2_numeric_duplicate_group_count": fmcwr[
                "numeric_duplicate_group_count"
            ],
            "lss_fmcwr_2_numeric_duplicate_member_count": fmcwr[
                "numeric_duplicate_member_count"
            ],
            "lss_fmcwr_2_channel_shape_dtype_counts": fmcwr[
                "channel_shape_dtype_counts"
            ],
            "lss_fmcwr_2_target_recording_stem_counts": fmcwr[
                "target_recording_stem_counts"
            ],
            "lss_fmcwr_2_target_candidate_group_counts": fmcwr[
                "target_candidate_group_counts"
            ],
            "lss_fmcwr_2_gates": fmcwr["gates"],
            "lss_fmcwr_2_raw_complex_iq_available_for_all_records": fmcwr[
                "raw_complex_iq_available_for_all_records"
            ],
            "lss_fmcwr_2_h_v_polarimetry_available": fmcwr[
                "h_v_polarimetry_available"
            ],
            "lss_fmcwr_2_target_aspect_angle_verified": fmcwr[
                "target_aspect_angle_verified"
            ],
            "lss_fmcwr_2_natural_bird_evidence_available": fmcwr[
                "natural_bird_evidence_available"
            ],
            "lss_fmcwr_2_global_sampling_rate_available": fmcwr[
                "global_sampling_rate_available"
            ],
            "lss_fmcwr_2_global_carrier_frequency_available": fmcwr[
                "global_carrier_frequency_available"
            ],
            "lss_fmcwr_2_physical_doppler_hz_axis_available": fmcwr[
                "physical_doppler_hz_axis_available"
            ],
            "lss_fmcwr_2_physical_velocity_axis_available": fmcwr[
                "physical_velocity_axis_available"
            ],
            "lss_fmcwr_2_complete_md_stft_implementation_available": fmcwr[
                "complete_md_stft_implementation_available"
            ],
            "lss_fmcwr_2_simulated_bird_archive_is_simulation_only": fmcwr[
                "simulated_bird_archive_is_simulation_only"
            ],
            "lss_fmcwr_2_random_mat_frame_or_window_split_allowed": fmcwr[
                "random_mat_frame_or_window_split_allowed"
            ],
            "lss_fmcwr_2_model_training_allowed": fmcwr[
                "model_training_allowed"
            ],
            "lss_fmcwr_2_model_training_performed": fmcwr[
                "model_training_performed"
            ],
            "lss_fmcwr_2_source_archives_extracted_to_disk": fmcwr[
                "source_archives_extracted_to_disk"
            ],
            "lss_fmcwr_2_source_archives_modified": fmcwr[
                "source_archives_modified"
            ],
            "lss_fmcwr_2_raw_data_included": False,
            "lss_fmcwr_2_member_level_outputs_included": False,
            "lss_fmcwr_2_normalized_axis_contract_only": True,
            "lss_fmcwr_2_normalized_processing_contract_version": 1,
            "lss_fmcwr_2_normalized_processing_performance_reported": False,
            "lss_fmcwr_2_normalized_processing_code_included": True,
            "external_public_data_registries_included": True,
            "team_onboarding_manual_included": True,
            "team_qualification_policy_included": True,
            "team_qualification_scorecard_included": True,
            "team_task_claim_template_included": True,
            "team_weekly_report_template_included": True,
        },
        "full_reproduction_requires": [
            "internal source code",
            "raw data and manifests",
            "sample-level frozen predictions",
            "model checkpoints",
        ],
        "excluded_content": [
            "raw MAT, IQ, or PCD data and external source archives",
            "sample labels and sample-level predictions",
            "model checkpoints and training logs",
            "development transcripts and local recovery archives",
            "personal paths and credentials",
        ],
        "files": sorted(records, key=lambda record: str(record["packaged_path"])),
    }
    (staging_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_checksums(staging_dir: Path) -> None:
    files = sorted(
        path
        for path in staging_dir.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS.txt"
    )
    lines = [
        f"{sha256_file(path)}  {path.relative_to(staging_dir).as_posix()}"
        for path in files
    ]
    (staging_dir / "SHA256SUMS.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def audit_package_directory(package_dir: Path) -> dict[str, object]:
    errors: list[str] = []
    files = sorted(path for path in package_dir.rglob("*") if path.is_file())
    for path in package_dir.rglob("*"):
        relative = path.relative_to(package_dir).as_posix()
        lower = relative.lower()
        if path.is_symlink():
            errors.append(f"symlink is not allowed: {relative}")
        if any(part.lower() in lower for part in FORBIDDEN_NAME_PARTS):
            errors.append(f"forbidden name in package: {relative}")
        if path.is_file():
            suffix = path.suffix.lower()
            if suffix in FORBIDDEN_SUFFIXES or suffix not in ALLOWED_SUFFIXES:
                errors.append(f"forbidden or unsupported file type: {relative}")
            if path.stat().st_size > 10 * 1024 * 1024:
                errors.append(f"unexpected large file: {relative}")
            payload = path.read_bytes().lower()
            for marker in SENSITIVE_TEXT_MARKERS:
                if marker.lower().encode("utf-8") in payload:
                    errors.append(f"sensitive marker {marker!r} in {relative}")
    if errors:
        raise ValueError("Share-package audit failed: " + "; ".join(errors))
    link_count = audit_markdown_links(package_dir)
    return {
        "status": "PASS",
        "file_count": len(files),
        "size_bytes": sum(path.stat().st_size for path in files),
        "raw_data_included": False,
        "checkpoints_included": False,
        "sample_predictions_included": False,
        "development_transcripts_included": False,
        "sensitive_markers_found": 0,
        "markdown_links_checked": link_count,
    }


def audit_markdown_links(package_dir: Path) -> int:
    errors: list[str] = []
    checked = 0
    package_dir = package_dir.resolve()
    for markdown in sorted(package_dir.rglob("*.md")):
        text = markdown.read_text(encoding="utf-8-sig")
        for match in MARKDOWN_LINK_PATTERN.finditer(text):
            target = match.group(1).strip().strip("<>")
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            target = target.split("#", 1)[0]
            resolved = (markdown.parent / target).resolve()
            checked += 1
            try:
                resolved.relative_to(package_dir)
            except ValueError:
                errors.append(
                    f"link escapes package: {markdown.relative_to(package_dir)} -> {target}"
                )
                continue
            if not resolved.is_file():
                errors.append(
                    f"missing link target: {markdown.relative_to(package_dir)} -> {target}"
                )
    if errors:
        raise ValueError("Share-package Markdown link audit failed: " + "; ".join(errors))
    return checked


def write_deterministic_zip(package_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in sorted(item for item in package_dir.rglob("*") if item.is_file()):
            relative = path.relative_to(package_dir).as_posix()
            info = zipfile.ZipInfo(f"{package_dir.name}/{relative}", ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100644 & 0xFFFF) << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED)


def audit_zip(zip_path: Path, package_name: str) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        bad_file = archive.testzip()
        if bad_file is not None:
            raise ValueError(f"ZIP integrity check failed at {bad_file}")
        names = archive.namelist()
        expected_prefix = f"{package_name}/"
        if not names or any(
            not name.startswith(expected_prefix)
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            for name in names
        ):
            raise ValueError("ZIP contains an unsafe or unexpected archive path")


def build_share_package() -> tuple[Path, Path, dict[str, object]]:
    output_dir = DEFAULT_OUTPUT_DIR
    zip_path = DEFAULT_ZIP_PATH
    validate_source_map()
    validate_output_paths(output_dir, zip_path)
    ensure_output_available(output_dir, zip_path)

    staging_root = prepare_staging_root()
    staging_parent = Path(
        tempfile.mkdtemp(prefix=f".{PACKAGE_NAME}.build-", dir=staging_root)
    )
    if staging_parent.resolve().parent != staging_root:
        raise ValueError("Created staging directory is outside the fixed staging root")
    staging_dir = staging_parent / output_dir.name
    staging_zip = staging_parent / zip_path.name
    try:
        staging_dir.mkdir()
        records = copy_package_files(staging_dir)
        write_manifest(staging_dir, records)
        write_checksums(staging_dir)
        audit = audit_package_directory(staging_dir)
        write_deterministic_zip(staging_dir, staging_zip)
        audit_zip(staging_zip, staging_dir.name)

        ensure_output_available(output_dir, zip_path)
        staging_dir.rename(output_dir)
        staging_zip.rename(zip_path)
    finally:
        cleanup_staging_directory(staging_parent)
    return output_dir, zip_path, audit


def main() -> int:
    parse_args()
    try:
        output_dir, zip_path, audit = build_share_package()
    except (FileNotFoundError, FileExistsError, ValueError, OSError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    print("Project share package: PASS")
    print(f"directory={output_dir}")
    print(f"zip={zip_path}")
    print(f"files={audit['file_count']}")
    print(f"size_bytes={audit['size_bytes']}")
    print("raw_data_included=False")
    print("checkpoints_included=False")
    print("development_transcripts_included=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
