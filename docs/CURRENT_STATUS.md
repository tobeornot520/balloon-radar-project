# Current research status

## Frozen stages

1. DPG-FCN provides the H/V detection and localization baseline.
2. Sample-independent BC is the current online-oriented calibration baseline.
3. Complete-scan BC-DPG-FCN v3 is an offline scan-aware upper bound; its context may include later samples from the same scan.
4. Stage 3 evaluates dense explicit polarimetric representations and retains Power2 as the most reliable detection representation.
5. Stage 4 freezes Power2 candidate locations and applies suppression-only ROI refinement.
6. The final audit aligns frozen BC-DPG and ROI predictions without retuning test thresholds or selecting a joint rule.
7. A causal-context sensitivity audit replays the frozen full checkpoint with leave-one-out and assumed-order past-only contexts. It does not retrain or select a causal model.
8. An acquisition-order readiness audit finds no verified within-scan sample order. Formal causal training remains gated; a bounded validation-only interface smoke has passed.
9. A frozen localization build aggregates range-velocity errors from the six base-threshold BC-DPG test tables without training, inference, or retuning.
10. A versioned new-data contract now enforces capture, causal-order, and locked-evaluation readiness before new data can enter formal experiments.
11. Five-fold LAT-MRICD Narrow-X and HRRP-X interpretable baselines are frozen with metadata-only batch-code assignment, fixed models, grouped metrics, and batch-code cluster intervals.

## Active method reproduction

The Tian et al. 2024 FCN method-level reproduction has an independent model,
target objective, PIR/MDP postprocessing, validation-only threshold selection,
one-fold trainer, and six-fold orchestrator. A first H-only six-fold run was
completed, but it produced zero joint detections and was subsequently found to
contain non-paper L1 distance metrics and incorrect d_min/d_5/d_avg definitions.
It is retained only as failed-transfer diagnostic evidence and must not be cited
as a successful or metric-valid reproduction.

The corrected Fold 1 diagnosis found that all 53 validation target responsible
cells passed PIR but none was selected by MDP. A preregistered train/validation-
only point-GT rescue changed one target-construction setting and reached 22/53
joint successes with 2/150 background alarms. It did not load test. Because it
changes the paper's expanded GT and still has only 15.1% responsible-cell MDP
selection with a severe Doppler-error tail, it is a local transfer ablation, not
a Tian reproduction result. Component and probability-template audits show two
nearly fixed Doppler bands: the mean target-map correlation to the shared
template is 0.99818. A preregistered 16-negative floor reduced background Pfa
but degraded joint Pd from 0.4151 to 0.2453 and did not shrink the bands, so it
is rejected. No further six-fold, V/HV, random-negative scan, or PIR-threshold
scan is authorized. A second preregistered same-range-column dense-negative
diagnostic also failed (joint Pd 0.1132; template correlation 0.99817). The
point-GT classification-supervision branch is closed pending reproduction-
condition and local-data identifiability evidence.

As of 2026-08-03, the project owner reports that the attributable H/V/IQ/PRF/
coordinate facts and the Tian preprocessing/label/configuration or aligned
example cannot currently be obtained. This records availability only; the
underlying technical values remain unknown. Exact Tian reproduction is frozen,
the existing implementation is retained as method-level failure evidence, and
work proceeds through the DPG-FCN zero-Doppler mainline, grouped LAT-MRICD
baselines, and synthetic contract tests. See
[TIAN_REPRODUCTION_FAILURE_AND_ALTERNATIVES_20260803.md](TIAN_REPRODUCTION_FAILURE_AND_ALTERNATIVES_20260803.md).

The August-September field preparation is now represented by five auditable
gates: capability, synchronization, polarimetric calibration, dry run, and
pilot. The checklist, evidence initializer, auditor, four-scenario matrix, and
field SOP are implemented. All gates remain blocked until the team supplies
real device, timing, calibration, dry-run, and pilot evidence; no readiness is
inferred from the legacy data.

## Active polarimetric transfer preparation

A reusable candidate-ROI polarimetric encoder scaffold is now implemented. It
separates H/V power, complex RD, and gated explicit polarimetric channels, then
fuses them into a task-independent embedding with a replaceable classifier
head. A per-channel validity mask allows unverified coherent phase channels to
be disabled instead of treating them as calibrated physical measurements.

This is architecture and interface preparation only. No pretrained checkpoint
exists and no formal pretraining run has been authorized. Existing UAV and
background labels may support a preregistered auxiliary representation study,
but they cannot establish balloon-payload recognition. Formal work must keep
train/validation acquisition groups isolated, avoid test-driven selection, and
audit whether the embedding mainly encodes date, source, range, or velocity.
The intended later transfer path is to replace the task head and fuse the
polarimetric embedding with time-domain and micro-Doppler representations from
new synchronized, calibrated balloon data.

## External public-data grouped baseline

LAT-MRICD-1.0 was obtained from the journal's official ScienceDB supplement and
audited locally. Its 33 MAT files contain 7,119 X/Ku HRRP records and 16,072
S/X/Ku narrowband I/Q records. All matrices pass schema, finite-value and label
code checks; alternating narrowband columns reconstruct to 512 complex samples.

The audit forbids random row splitting. Across 911 `(representation, band,
batch)` groups, 46 batch codes collide across models/categories and 94.95% of
groups are category-pure. Twelve fine-model strata have fewer than three
batches. The frozen X-band experiment assigns whole
`(representation, band_code, batch_code)` groups to five held-out folds using
metadata only, then retains a dummy prior, fixed batch-balanced logistic
regression, and a fixed batch-balanced random forest without held-out-driven
model selection.

For Narrow-X, batch-class macro accuracy is 0.7999 for logistic regression
(batch-code cluster-bootstrap 95% CI 0.7659 to 0.8313) and 0.7872 for random
forest (0.7373 to 0.8340). For HRRP-X, the corresponding values are 0.6617
(0.5826 to 0.7404) and 0.6481 (0.5764 to 0.7240). The paired random-forest-minus-
logistic intervals cross zero for both tasks, so neither model is selected as a
winner. D17-NX and D17-HX are complete; the next public-data task is a separately
preregistered cross-band transfer study.

These are within-release, batch-code-held-out category baselines for already
represented submodels. They are not unseen-model, independent-session, or
external-blind evidence, and HRRP-X is not an independent external replication
of Narrow-X. The dataset still provides no H/V polarization, balloon labels,
verified PRF/timestamps, causal session order, or Tian alignment facts. This
completion does not change the main UAV direction gate, which remains 4/6
`BLOCKED_EXTERNAL`. See
[EXTERNAL_PUBLIC_DATA_AUDIT_20260803.md](EXTERNAL_PUBLIC_DATA_AUDIT_20260803.md).

## Preregistered LAT-MRICD cross-band transfer

D17-XBAND is preregistered but has not been run. The frozen V1 protocol uses
Narrow X as the source and S/Ku as separate locked targets for the common
UAV/weather task. Target rows cannot affect scaling, fitting, model selection,
feature design or thresholds. Dummy prior, batch-balanced logistic regression
and a fixed random forest are all retained; logistic regression is the primary
gate model and random forest is a fixed sensitivity model.

At the preregistration checkpoint, no S/Ku target performance has been viewed,
and the formal result directory, external consumption record and final evidence
directory do not exist. After a clean pre-result commit, the runner may execute
exactly once. Any failed or interrupted run consumes the targets. Both locked
targets must pass the preregistered accuracy, per-class recall and paired
LR-minus-dummy interval gates; otherwise the negative result is frozen and
CNN/domain-adaptation development on these consumed targets stops. See
[LAT_MRICD_CROSS_BAND_TRANSFER_PROTOCOL_V1.md](LAT_MRICD_CROSS_BAND_TRANSFER_PROTOCOL_V1.md).

## Newly acquired public inputs

LSS-DAUR-1.0 V3 and LSS-FMCWR-2.0 V4 were downloaded from their official
ScienceDB file trees on 2026-08-04. DAUR matches the release count and size at
314 files and 148,763,512 bytes. FMCWR-2.0 matches at eight files and
1,013,535,036 bytes; all six RAR archives match the official MD5 values. These
are local research inputs only and have not entered a model result.

DAUR is pending a full MAT/pairing/group audit. FMCWR-2.0 is pending RAR
extraction and schema/group audit. The simulated-bird archive must not be cited
as natural-bird evidence. A 209,569,478-byte LSS-HSR-L journal bundle also
passed ZIP integrity checks, but it has not been proven equivalent to the
237,020,946-byte ScienceDB V2 archive and is pending a read-only loader and
group audit. Dataset decisions and download receipts are tracked in
`data/metadata/external_public_datasets_v1.csv` and
`data/metadata/external_public_artifacts_v1.csv`.

## Zero-Doppler development gate

The six-fold frozen fixed-notch diagnostic improves the current false-alarm
count while retaining joint detections, but it remains a development reference.
The first learned dense-negative and clutter-aware variants failed to beat that
reference on the Fold 1/4 gate and will not be expanded to six folds in their
current form. See
[ZERO_DOPPLER_MECHANISM_V1_CONCLUSION.md](ZERO_DOPPLER_MECHANISM_V1_CONCLUSION.md).

A preregistered V2 subsequently froze that fixed notch and learned only a
bounded, zero-localized residual with direct background peak pressure. It passed
the Fold 1/4 gate and, without parameter changes, reduced the six-fold CPU
comparison from 120 to 109 false alarms while retaining 290 joint hits. All 11
removed alarms are in Fold 4; no alarm was added and no paired joint hit changed.
This is the current learned development reference, not blind or deployment
evidence. See
[ZERO_DOPPLER_FIXED_RESIDUAL_V2_RESULT.md](ZERO_DOPPLER_FIXED_RESIDUAL_V2_RESULT.md).

The 11 removed Fold 4 alarms have now passed human visible-pattern review: 9
were recorded as `near_zero_doppler_peak`, 2 as `broad_structure`, and all 11
physical classes remain `unknown` because no independent scene record exists.
This closes the P0 review workflow but does not identify a physical clutter
mechanism. See
[ZERO_DOPPLER_P0_REVIEW_PRESCREEN_V1.md](ZERO_DOPPLER_P0_REVIEW_PRESCREEN_V1.md).

## Authoritative evidence

- BC-DPG v3: `results/final_evidence/bc_dpg_v3_final/`
- Stage 3: `docs/polarimetric_stage3/STAGE3_FROZEN_CONCLUSION.md`
- Stage 4: `results/data_audit/roi_stage4_selected_sixfold_v1/`
- Joint audit: `results/data_audit/final_roi_bc_dpg_joint_v2_base_threshold/`
- Joint paper evidence: `results/final_evidence/roi_bc_dpg_joint_fixed_threshold/`
- Causal-context sensitivity audit: `results/data_audit/bc_dpg_v3_causal_context_audit/`
- Acquisition-order readiness audit: `results/data_audit/detection_acquisition_order/`
- Causal-training protocol: [BC_DPG_CAUSAL_TRAINING_PROTOCOL.md](BC_DPG_CAUSAL_TRAINING_PROTOCOL.md)
- Frozen localization evidence: `results/final_evidence/bc_dpg_localization/`
- New-data collection protocol: [NEW_DATA_COLLECTION_PROTOCOL.md](NEW_DATA_COLLECTION_PROTOCOL.md)
- Current contract gap baseline: `results/data_audit/data_collection_readiness_v1/`
- Data card: [DATA_CARD.md](DATA_CARD.md)
- Metric definitions: [METRIC_DEFINITIONS.md](METRIC_DEFINITIONS.md)
- Model-selection ledger: [MODEL_SELECTION_LEDGER.md](MODEL_SELECTION_LEDGER.md)
- Tian FCN reproduction protocol: [TIAN_FCN_REPRODUCTION_PROTOCOL.md](TIAN_FCN_REPRODUCTION_PROTOCOL.md)
- Tian Fold 1 diagnostic conclusion: [TIAN_FCN_FOLD1_DIAGNOSTIC_CONCLUSION.md](TIAN_FCN_FOLD1_DIAGNOSTIC_CONCLUSION.md)
- Tian Fold 1 component mechanism: [TIAN_FCN_FOLD1_COMPONENT_MECHANISM.md](TIAN_FCN_FOLD1_COMPONENT_MECHANISM.md)
- Tian reproduction-condition request: [TIAN_FCN_REPRODUCTION_CONDITIONS_REQUEST.md](TIAN_FCN_REPRODUCTION_CONDITIONS_REQUEST.md)
- Tian failure analysis and fallback decision: [TIAN_REPRODUCTION_FAILURE_AND_ALTERNATIVES_20260803.md](TIAN_REPRODUCTION_FAILURE_AND_ALTERNATIVES_20260803.md)
- Field collection SOP: [FIELD_COLLECTION_SOP_V1.md](FIELD_COLLECTION_SOP_V1.md)
- Field readiness pending baseline: `results/data_audit/field_readiness_v1/pending_audit/`
- Polarimetric transfer encoder V1: [POLARIMETRIC_TRANSFER_ENCODER_V1.md](POLARIMETRIC_TRANSFER_ENCODER_V1.md)
- External public-data audit: [EXTERNAL_PUBLIC_DATA_AUDIT_20260803.md](EXTERNAL_PUBLIC_DATA_AUDIT_20260803.md)
- LAT-MRICD grouped-baseline protocol: [LAT_MRICD_GROUPED_BASELINE_PROTOCOL_V1.md](LAT_MRICD_GROUPED_BASELINE_PROTOCOL_V1.md)
- Frozen LAT-MRICD grouped-baseline evidence: `results/final_evidence/lat_mricd_grouped_baselines_v1/`
- Preregistered LAT-MRICD cross-band protocol: [LAT_MRICD_CROSS_BAND_TRANSFER_PROTOCOL_V1.md](LAT_MRICD_CROSS_BAND_TRANSFER_PROTOCOL_V1.md)
- External public dataset registry: `data/metadata/external_public_datasets_v1.csv`
- External public artifact registry: `data/metadata/external_public_artifacts_v1.csv`

The deterministic joint evidence build contains the formal report, pooled and fold-level tables, fold-distribution summaries, Wilson and scan-group bootstrap intervals, paired McNemar diagnostics, PNG/PDF figures, and a SHA256 manifest. Build it with:

```bash
python scripts/build_roi_bc_dpg_joint_paper_assets.py
```

The earlier `final_roi_bc_dpg_joint` output used the wrong BC decision source and has been moved to the local recovery archive. It must not be cited.

## Current result

The complete-scan BC-DPG-FCN v3 has the best observed result in the current six-fold internal development evaluation: 56/830 background alarms and 289/318 joint detection-localization successes. This is an offline scan-aware upper bound, not a causal deployment result. Sample-independent BC has 122 false alarms and is the closer online-oriented reference.

The 56 complete-scan BC false alarms are concentrated entirely in Folds 1 and 4. Macro Pfa is 0.0622, median Pfa is 0, worst-fold Pfa is 0.2800 in Fold 1, and worst-fold joint Pd is 0.7917 in Fold 6. The scan-group bootstrap 95% intervals are 0 to 0.1618 for Pfa and 0.8599 to 0.9521 for joint Pd; only six independent background scan groups are available.

ROI refinement remains an independent suppression study. The reported AND/intersection, OR/union, and McNemar results are post-test diagnostics only. No combination was trained or selected from these outcomes.

## Causal-context sensitivity

The frozen complete-scan checkpoint was replayed with the original fold thresholds under several context substitutions. Complete-scan replay reproduced all six frozen decision tables with zero decision mismatches. Leave-one-out produced 54/830 background alarms and 289/318 joint successes, indicating that direct self-inclusion is not the main source of the complete-scan gain; it remains non-causal because later samples are still available.

Using all prior samples under the inferred `(beam_layer, azimuth_deg, sample_id)` order produced 93/830 alarms and 288/318 joint successes. Windows of 4, 16, and 64 produced 148, 138, and 105 alarms respectively. These are post-hoc out-of-distribution sensitivity results from a model trained with complete-scan context. They must not be used to select a deployment window. The separately trained sample-independent BC remains the valid online-oriented reference at 122 alarms.

## Selection and data limitations

The absence of test-threshold retuning does not make the evaluation blind. Stage 4 modes were screened on development Folds 1 and 4, and those folds were reused in the six-fold ROI summary. BC-DPG structure and loss design were also informed by development feedback.

All target scan groups in the frozen audit are dated `20260202`; all background scan groups are dated `20260204`. Class and acquisition date are fully confounded. The current results therefore cannot establish cross-date, cross-site, or external blind generalization.

The source tables do not contain verified per-sample acquisition timestamps. Past-only order is inferred from beam, azimuth, and sample ID, so causal status holds only under that ordering assumption. Each scan also has a cold start: 71 target samples and 6 background samples have zero prior context.

The readiness audit checked all 1,148 MAT files. They contain H/V IQ arrays but no timestamp-like variable. MAT header creation times are at least 49.1 days after the filename timestamp and follow filesystem mtime within 3 seconds, so neither source is acquisition order. The formal causal-training gate is therefore closed.

A Fold 1 development smoke using inferred order, a four-sample history, two epochs, and 12 samples per class per split completed on CPU. It loaded only train and validation data; no test split or test metric was produced. This establishes interface readiness only and contributes no performance evidence or window choice.

## Frozen localization evidence

The six frozen base-threshold folds contain 318 target samples. Of these, 302 pass the score threshold, 297 meet the 2-gate/3-bin localization tolerance regardless of score, and 289 meet both conditions. This gives pooled score Pd 0.9497, localization-ok rate 0.9340, and joint Pd 0.9088; 289/302 score-detected targets meet the localization tolerance.

Across all targets, range error has MAE 1.418 gates, median 1, P90 2, and maximum 39. Velocity error has MAE 1.154 bins, median 0, P90 1, and maximum 40. The long tail means MAE, median/P90, maximum, conditional-on-detection errors, and joint success must be reported together. All six calibrated coordinate tables match their raw DPG tables exactly; BC-DPG changes scores, not candidate locations.

## New-data readiness contract

The version-1 collection contract defines 40 required manifest columns and three validation profiles: capture, causal, and locked evaluation. It requires storage-root-relative paths, UTC timestamps, hardware sequence provenance, clock-reset tracking, event and observation timing, SNR, radar/configuration identifiers, H/V validity, same-condition target/background controls, and outer-group partition isolation.

The current 1,148-row V4 manifest fails the locked-evaluation profile because 33 contract columns are absent. Schema failure blocks all downstream gates; neither formal causal training nor locked external evaluation is open. Existing filename time, beam/azimuth order, MAT header time, filesystem mtime, and historical test labels cannot be promoted into the missing verified metadata.

## Claim boundaries

The current data support an internal development-stage H/V UAV detection and distance-velocity localization front end. They do not establish balloon payload classification, event-level or hourly false-alarm performance, cross-environment blind generalization, or a trained and independently evaluated real-time causal scan adapter.
