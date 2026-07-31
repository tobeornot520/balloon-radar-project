# Multi-domain Current-data Mining V1

Date: 2026-07-31

## Decision

The reusable algorithm scaffold is ready for feature extraction and future
multi-domain fusion, but the current data do not support training or claiming a
balloon-payload classifier. The next current-data experiment should be a
development-only comparison of zero-Doppler handling mechanisms across all six
background scans. It must not be described as a new blind test.

## Evidence Inventory

| Dataset | Usable records | Independent grouping | Supported role |
|---|---:|---|---|
| Detection | 1,148: 318 UAV, 830 background | 77 source files; only 6 background scans | Detector-candidate and clutter diagnostics |
| Classification | 256 long windows from 23 files | Capture file/event | UAV-only representation and normalized-frequency diagnostics |

All 1,404 extracted records completed without error. The catalog contains 56
features in five domains: quality, time, range-Doppler, relative polarimetry,
and normalized-frequency time-frequency.

## Anchor Policy

Two feature locations have different roles and must remain separate:

- Scene features describe the strongest raw combined H/V range-Doppler cell.
- Detection-local time, polarimetric, spectral, and time-frequency features use
  the frozen out-of-fold DPG candidate.
- UAV-only long-window records have no detector candidate, so their local
  features currently use the strongest raw cell.
- Labels and truth coordinates are never accepted as feature anchors.

The frozen DPG candidates localize 297 of 318 targets within two range gates
and three Doppler bins, a rate of 93.40%. The earlier raw-global-peak anchor
localized none of the 318 targets, so it is retained only for scene context.

## Current Findings

### Range-Doppler morphology is the strongest current signal

With feature direction fixed from the pooled data and then evaluated against
each background scan separately:

| Candidate-local feature | Pooled oriented AUC | Worst of 6 background scans |
|---|---:|---:|
| Zero-Doppler energy fraction | 0.9687 | 0.9535 |
| Peak energy fraction | 0.9619 | 0.9238 |
| Main-band energy fraction | 0.9327 | 0.8587 |

These are exploratory diagnostics, not unbiased performance estimates,
because target and background acquisition dates are confounded. They do show
that candidate-local spectral morphology is the most defensible input for the
next detector mechanism experiment.

### Relative polarimetry is useful but not stable enough alone

Candidate-local relative polarimetric features show pooled separation, for
example ZDR-like IQR at 0.8567 AUC and phase resultant at 0.8378. Their worst
background-scan AUC falls to about 0.60 and 0.58 respectively. Therefore:

- keep polarimetry as an auxiliary fusion domain;
- mask it when H/V validity or calibration evidence is unavailable;
- do not claim absolute ZDR, PhiDP, or calibrated target physics;
- do not use it as a standalone detector with the current six scans.

### Time-domain and long-window features are group-sensitive

Background scan identity explains 62.6% and 56.1% of the observed H/V
magnitude-kurtosis variance. In the UAV-only long windows, capture-file
identity explains 72.9% of zero-Doppler-energy variance and more than half of
several amplitude/coherence features. Random window splitting would therefore
measure file recognition rather than target generalization.

### Current micro-Doppler evidence is limited

All 256 UAV-only records meet the long-window gate, so normalized-frequency
STFT descriptors can be computed. The 128-pulse detection records do not meet
that gate. Without PRF, continuous timing, and target-aligned events, the
current outputs cannot be converted to physical micro-Doppler or rotor-rate
claims.

## Algorithm Scaffold

The reusable path is now:

1. A detector emits a candidate range/Doppler coordinate.
2. The extractor builds scene context plus candidate-local time, RD, relative
   polarimetric, and time-frequency descriptors.
3. Scalar normalization is fitted only on training acquisition groups.
4. Domain-specific encoders create embeddings.
5. Validity-masked learned fusion gives missing or untrusted domains exactly
   zero weight.
6. Replaceable heads later predict target class, balloon loaded state, payload
   class, or motion state.

The full catalog retains coordinates and quality fields for auditing. Absolute
range/Doppler coordinates, source identity, and acquisition metadata should be
excluded from the default payload-classification input unless a grouped
ablation demonstrates transferable value. The fusion model is a tested
scaffold only; no classifier is trained in this stage.

## Next Analysis Gates

### Current-data development

Compare zero-Doppler exclusion, dense negative supervision, and a clutter-aware
auxiliary objective under identical grouped development folds. Selection must
use pooled and worst-background-group false alarms. The previously evaluated
held-out partition is consumed historical evidence and cannot become blind
again.

### New-data analysis

- Split by event/session/file before creating windows.
- Record PRF, pulse timing, hardware order, and continuous track IDs.
- Verify simultaneous H/V capture and obtain amplitude/phase calibration.
- Collect balloon unloaded/loaded states and payload classes in matched
  sessions, including background scans interleaved in the same sessions.
- Freeze preprocessing, feature schema, group split, thresholds, and metrics
  before the final untouched evaluation.

## Reproducible Outputs

The untracked generated catalog is under
`results/data_audit/multidomain_feature_catalog_v1/`. Key files are
`summary.json`, `feature_schema.csv`,
`detection_background_group_stress.csv`, and `REPORT.md`. Generated records are
kept out of Git; the extractor, catalog builder, contract, tests, and this
conclusion are tracked.
