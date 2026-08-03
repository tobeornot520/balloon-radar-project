# Configuration index

Configuration is grouped by stable filename prefixes because experiment ledgers and frozen reports
refer to these paths directly. Existing files must not be moved after an experiment is recorded.

| Prefix | Scope |
|---|---|
| `bc_dpg_*` | Background-calibrated DPG experiments |
| `polarimetric_*` | Dense polarimetric representation experiments |
| `polarimetric_transfer_*` | Reusable ROI encoders and future task-head transfer |
| `roi_*` | Candidate-ROI polarimetric refinement |
| `tian_fcn_*` | Tian reproduction and local-transfer diagnostics |
| `field_*`, `pilot_*` | Field capability, readiness and Pilot templates |
| `data_collection_*` | Versioned capture and evaluation contracts |
| `team_*` | Team onboarding, task-claim and collaboration templates |
| `radar_config.yaml` | Legacy/local radar geometry reference |

Create a new versioned file when semantics change. Do not overwrite a frozen configuration or reuse
an experiment name for a different setting.
