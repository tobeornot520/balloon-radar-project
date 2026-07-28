# Data collection manifest precheck

## Decision

**Profile `locked_evaluation`: FAIL.**

Rows: 1148; targets: 318; backgrounds: 830.

Formal causal-training gate open: `false`.
Locked-evaluation gate open: `false`.

## Gates

| Gate | Status |
|---|---|
| schema | FAIL |
| row_integrity | BLOCKED |
| event_timing | BLOCKED |
| causal_order | BLOCKED |
| channel_integrity | BLOCKED |
| partition_isolation | BLOCKED |
| same_condition_class_control | BLOCKED |

## Issues

- `MISSING_COLUMNS` [schema]: Manifest is missing columns: ['iq_path', 'target_class', 'background_type', 'payload_class', 'motion_state', 'acquisition_timestamp_utc', 'hardware_sequence', 'scan_id', 'scan_sequence', 'order_source', 'order_verified', 'clock_id', 'clock_reset_counter', 'timestamp_resolution_ns', 'dropped_frames_before', 'session_id', 'observation_id', 'event_id', 'event_start_utc', 'event_end_utc', 'sample_duration_s', 'collection_date', 'site_id', 'flight_id', 'platform_id', 'snr_db', 'radar_config_id', 'calibration_id', 'h_channel_valid', 'v_channel_valid', 'weather_id', 'evaluation_partition', 'outer_group_id'].
- `UNEXPECTED_COLUMNS` [schema]: Manifest contains unexpected columns: ['class_name', 'mat_path', 'new_split', 'original_split', 'source_file'].

## Interpretation

A passing capture profile establishes record completeness only. A passing causal profile establishes that the manifest can support verified ordered context construction. Only a passing locked_evaluation profile supports opening the external locked-evaluation gate, after model and metric choices are frozen.
