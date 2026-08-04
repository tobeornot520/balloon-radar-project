# Script entry points

`scripts/` contains command-line entry points, orchestration, audits, evidence builders and plots.
Reusable model, feature, loss and metric logic belongs in the corresponding source package rather
than being copied into a script.

## Active workflows

| Workflow | Primary entry points |
|---|---|
| Project checks | `check_project_health.py`, `check_current_direction_completion_v1.py` |
| Experiment provenance | `run_recorded_experiment.py`, `manage_experiment_ledger.py` |
| Dataset contracts | `validate_data_collection_manifest.py`, `audit_detection_acquisition_order.py` |
| Public external data | `audit_lat_mricd_dataset_v1.py`, `run_lat_mricd_grouped_baseline_v1.py`, `run_lat_mricd_cross_band_transfer_v1.py` |
| Field readiness | `initialize_field_readiness_evidence.py`, `audit_field_readiness_v1.py` |
| BC-DPG | `run_bc_dpg_v3.py`, `audit_bc_dpg_v3_causal_context.py` |
| Polarimetric Stage 3 | `run_polarimetric_representation_benchmark_v2.py` |
| ROI Stage 4 | `run_roi_stage4_selected_sixfold_v1.py` |
| Zero-Doppler review | `build_zero_doppler_human_review_queue_v1.py`, `build_zero_doppler_review_atlas_v1.py`, `build_zero_doppler_review_workbench_v1.py`, `audit_zero_doppler_human_review_v1.py` |
| Tian reproduction | `run_tian_fcn_reproduction_smoke.py`, `run_tian_fcn_sixfold.py` |
| Frozen evidence | `build_bc_dpg_localization_evidence.py`, `build_roi_bc_dpg_joint_paper_assets.py`, `build_lat_mricd_cross_band_evidence_v1.py` |
| Sharing | `build_project_share_package.py` |

Files named `test_*.py` in this directory are explicit historical smoke tools. Automated tests live
only in `tests/` and are collected by pytest.

## Placement rule

- Put a reusable implementation in `datasets/`, `features/`, `models/`, `training/`, `evaluation/`
  or `utils/`.
- Put a user-invoked command in `scripts/`.
- Do not keep identical copies in two active directories.
- New formal experiment commands must use the experiment ledger wrapper.
