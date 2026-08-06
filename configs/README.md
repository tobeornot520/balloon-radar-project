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
| `field_iq_probe_*` | Device-variable and expected-shape template for read-only H/V MAT content checks |
| `field_sync_event_*` | Controlled radar/video/truth synchronization events and frozen numeric limits |
| `data_collection_*` | Versioned capture and evaluation contracts |
| `lat_mricd_grouped_*` | Frozen LAT-MRICD X-band grouped baselines |
| `lat_mricd_cross_band_*` | Preregistered one-shot LAT-MRICD band-held-out transfer |
| `zero_doppler_false_alarm_*` | Frozen prediction hashes, audit counts and sharing boundaries for the local false-alarm library |
| `zero_doppler_target_safety_*` | Frozen paired target behavior, score/peak-shift counts and no-retuning claim boundaries |
| `team_*` | Team onboarding, task-claim and collaboration templates |
| `radar_config.yaml` | Legacy/local radar geometry reference |

Create a new versioned file when semantics change. Do not overwrite a frozen configuration or reuse
an experiment name for a different setting.
