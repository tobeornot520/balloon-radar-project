# Zero-Doppler mechanism V1 conclusion

Date: 2026-07-31

## Decision

Do not expand the current `dense_negative` or `clutter_aware` settings from
Folds 1/4 to all six development folds. The fixed soft notch remains the current
safety reference, not the final deployable solution.

All folds are consumed development evidence. None of the comparisons below is
a blind-test estimate.

## Evidence

The frozen six-fold heatmap comparison reduced test-path false alarms from 187
for the baseline to 120 for the fixed soft notch while retaining 290 joint hits
instead of 289. The one-false-alarm difference from the archived baseline count
of 186 is consistent with CPU/GPU half-precision re-inference drift.

On the deliberately difficult Fold 1/4 gate:

| Fold | Mode | Selected epoch | Validation joint Pd | Validation Pfa | Test false alarms | Test joint Pd |
|---|---|---:|---:|---:|---:|---:|
| 1 | dense negative | 1 | 0.9434 | 0.0000 | 71 | 1.0000 |
| 1 | clutter aware | 12 | 0.9434 | 0.0133 | 85 | 1.0000 |
| 4 | dense negative | 0 | 0.9333 | 0.0133 | 100 | 0.9038 |
| 4 | clutter aware | 12 | 0.9333 | 0.0133 | 100 | 0.9038 |

The corresponding fixed-notch references were 53 false alarms and 1.0000 joint
Pd on Fold 1, and 67 false alarms and 0.9231 joint Pd on Fold 4. Neither learned
mechanism beats that gate.

Epoch 0 is a valid checkpoint candidate. Dense-negative Fold 4 selecting epoch
0 therefore means training did not improve the frozen starting point under the
validation rule; it is not a training-system failure. The clutter-aware head
respected its non-increasing-logit contract, but its learned suppression was too
weak or too scan-specific to generalize to the hard test backgrounds.

## Next revision gate

The next implementation should start from the fixed notch and learn only a
residual, target-protected correction. It should use paired hard-background
scans or pooled worst-group validation and explicitly test shift selectivity.
No six-fold learned run is justified until both Fold 1 and Fold 4 beat the fixed
reference at the frozen validation threshold without reducing joint detection.

This conclusion does not establish a physical stationary-clutter mechanism.
The current target/background dates remain confounded, and micro-Doppler or
absolute polarimetric interpretation still requires timing and H/V calibration.
