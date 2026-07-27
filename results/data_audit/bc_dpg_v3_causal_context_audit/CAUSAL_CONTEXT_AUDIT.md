# BC-DPG v3 causal-context sensitivity audit

## Audit status

This is a deterministic replay of frozen checkpoints and frozen test features. It performs no training, checkpoint selection, threshold selection, or test-threshold retuning. The complete-scan replay must reproduce the authoritative frozen decisions before any sensitivity result is written.

The full model was trained with complete-scan context. Leave-one-out and past-only rows substitute a different context at inference time and are therefore post-hoc out-of-distribution sensitivity diagnostics, not newly trained causal models.

## Six-fold aggregate

| mode | false_alarms | pooled_pfa | correct_detections | pooled_joint_pd | worst_fold_pfa | worst_fold_joint_pd | training_context_match |
|---|---|---|---|---|---|---|---|
| raw_dpg | 186 | 0.2241 | 289 | 0.9088 | 0.6667 | 0.7917 | True |
| sample_independent_bc | 122 | 0.1470 | 289 | 0.9088 | 0.4533 | 0.7917 | True |
| full_complete_scan | 56 | 0.0675 | 289 | 0.9088 | 0.2800 | 0.7917 | True |
| full_leave_one_out | 54 | 0.0651 | 289 | 0.9088 | 0.2600 | 0.7917 | False |
| full_past_only_w04 | 148 | 0.1783 | 288 | 0.9057 | 0.5933 | 0.7917 | False |
| full_past_only_w16 | 138 | 0.1663 | 288 | 0.9057 | 0.5467 | 0.7917 | False |
| full_past_only_w64 | 105 | 0.1265 | 288 | 0.9057 | 0.3800 | 0.7917 | False |
| full_past_only_all | 93 | 0.1120 | 288 | 0.9057 | 0.3467 | 0.7917 | False |

The matched complete-scan replay has 56 false alarms and 289/318 joint successes. The separately trained sample-independent BC reference has 122 false alarms and 289/318 joint successes.

Removing only the current sample from each complete scan gives 54 false alarms and 289/318 joint successes. This isolates self-inclusion sensitivity but remains non-causal because later scan samples are still available.

Using all assumed-order prior samples with the frozen complete-scan model gives 93 false alarms and 288/318 joint successes. Relative to the complete-scan replay, the paired changes are: complete-only alarms removed=5, candidate-only alarms added=42, complete-only joint successes lost=1, and candidate-only joint successes gained=0.

## Context definitions

- `full_complete_scan`: all samples in the scan, including the current sample and possible future samples; matched to training but non-causal.
- `full_leave_one_out`: all other samples in the scan; excludes self but still uses possible future samples.
- `full_past_only_w*`: the latest 4, 16, 64 prior samples under the assumed order.
- `full_past_only_all`: all prior samples under the assumed order.
- `sample_independent_bc`: separately trained with all 12 group features fixed to zero; it is not the full checkpoint with context removed.

Past-only order is inferred from `(beam_layer, azimuth_deg, sample_id)`. The source data do not provide a verified per-sample acquisition timestamp, so these rows are causal only under that ordering assumption.

## Replay validation

| fold | rows | stored_context_max_abs_delta | frozen_score_max_abs_delta | frozen_decision_mismatches |
|---|---|---|---|---|
| 01 | 203 | 0.0000 | 0.0000 | 0 |
| 02 | 203 | 0.0000 | 0.0000 | 0 |
| 03 | 167 | 0.0000 | 0.0000 | 0 |
| 04 | 202 | 0.0000 | 0.0000 | 0 |
| 05 | 210 | 0.0000 | 0.0000 | 0 |
| 06 | 163 | 0.0000 | 0.0000 | 0 |

## Claim boundary

This audit measures inference sensitivity of an already inspected model. It does not establish a deployable causal BC-DPG, does not remove the class/date confounding in the current data, and must not be used to select a preferred history window from test outcomes. A deployable next model must be trained and selected with causal context on training/validation data, then evaluated once on a locked external test set with recorded sample timestamps.
