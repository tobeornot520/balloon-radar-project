# Fixed-notch residual V2 result

Date: 2026-07-31

## Decision

The preregistered Fold 1/4 advancement gate passed, so the unchanged model was
expanded to all six consumed development folds. The six-fold comparison passes
the same safety gate and replaces the fixed notch as the current learned
development reference. It is not a deployment rule or blind-test result.

## Six-fold result

| Mode | False alarms | Pooled Pfa | Worst-fold Pfa | Joint hits | Pooled joint Pd |
|---|---:|---:|---:|---:|---:|
| CPU baseline replay | 187 | 0.2253 | 0.6667 | 289 | 0.9088 |
| Fixed soft notch | 120 | 0.1446 | 0.4467 | 290 | 0.9119 |
| Fixed notch + residual | 109 | 0.1313 | 0.3733 | 290 | 0.9119 |

The residual removes 11 paired background alarms, all in Fold 4. It adds no
background alarm and causes no paired joint-hit loss or gain. Fold 1 remains at
53 alarms and 53 joint hits; Fold 4 changes from 67/48 to 56/48. Folds 2, 3, 5,
and 6 retain zero alarms and their fixed-notch joint-hit counts.

Selected epochs by fold are 3, 3, 9, 4, 9, and 3. Validation decisions are
identical to the fixed-notch decisions: 3 pooled false alarms and 298 joint
hits. Selection differences are therefore driven by later validation tie
breakers such as AUC and loss, not by validation decision-count improvement.

## Paired behavior

The residual changes the predicted peak location on 48 of 1,148 test samples.
These changes do not alter any paired joint-success decision under the frozen
2-gate/3-bin tolerance. This is still behaviorally meaningful and must be
reported; the model is not merely a scalar score adjustment.

Every residual score is no greater than its paired fixed-notch score. The model
therefore satisfies both the fixed-notch safety baseline and the residual
non-increasing-logit contract.

## Interpretation

The result supports a narrow claim: direct background top-k pressure plus a
zero-localized, target-protected residual can improve the difficult Fold 4
background without degrading the current six-fold decision counts. It does not
prove a physical stationary-clutter mechanism because target and background
dates remain confounded and all outer folds are consumed development evidence.

No further tuning on these six folds is authorized. The next meaningful test
requires new same-condition target/background scans, verified acquisition
timing, and a locked outer evaluation. The fixed notch remains the deterministic
fallback; V2 is the learned development candidate.
