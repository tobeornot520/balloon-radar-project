# Zero-Doppler Mechanism V1

Date: 2026-07-31

## Decision

Proceed with a target-protected zero-Doppler mechanism comparison. Use a
seven-bin half-width as the initial hard-negative protection boundary, not as a
frozen deployment veto. Compare a fixed soft notch, dense zero-Doppler negative
weighting, and a learned clutter-aware non-increasing suppression head under
the same grouped development protocol.

No neural-network training was run in this stage. The feasibility audit reads
only the existing six-fold frozen prediction tables and thresholds.

## Candidate-veto Diagnostic

The diagnostic combines each physical sample's single out-of-fold DPG test
prediction. It reuses the original fold-specific validation threshold and then
rejects a candidate when its predicted velocity falls inside the tested
zero-Doppler band.

| Candidate-veto half-width | False alarms | Pooled Pfa | Worst scan Pfa | Joint hits | Joint Pd | Lost joint hits |
|---:|---:|---:|---:|---:|---:|---:|
| No veto | 186 | 0.2241 | 0.6667 | 289 | 0.9088 | 0 |
| 7 bins | 19 | 0.0229 | 0.0667 | 289 | 0.9088 | 0 |
| 8 bins | 14 | 0.0169 | 0.0600 | 288 | 0.9057 | 1 |
| 9 bins | 13 | 0.0157 | 0.0600 | 282 | 0.8868 | 7 |

The loss cliff after eight bins is the important result. A wider fixed notch
buys almost no additional false-alarm reduction while rapidly suppressing slow
targets. Seven bins is the largest radius with no observed loss among current
joint hits. Eight bins is the diagnostic optimum only when an absolute
one-percentage-point Joint-Pd drop is allowed.

## Mechanisms Now Available

### Fixed soft notch

`FixedZeroDopplerNotch` applies a symmetric Gaussian log-odds suppression. It
cannot increase any heatmap score. Unlike the candidate-veto audit, it operates
on the full heatmap, so another velocity/range cell can become the output peak.

### Dense zero-Doppler negatives

`DenseZeroDopplerMSE` keeps the existing sigmoid heatmap-MSE objective but gives
extra weight to negative pixels in the zero-Doppler band. Pixels inside the
target heatmap guard are exempt from the additional negative weight. This is
different from merely adding more background samples: the current background
target is already a dense all-zero heatmap.

### Clutter-aware suppression

`ClutterAwareSuppressionHead` summarizes H/V RD context over range and predicts
a non-negative suppression for each velocity row. The calibrated logit can
never exceed the frozen detector logit. Its objective combines dense detection,
target-probability preservation, and suppression regularization.

## Claim Boundaries

- A candidate veto is not equivalent to full-heatmap notching because it does
  not select the next peak.
- All six outer folds have already been consumed as development evidence.
- Target and background dates remain confounded.
- Radius seven is a development initialization, not a physical clutter width.
- A final mechanism requires new grouped data and a newly untouched evaluation
  partition.

## Next Training Gate

Run short smoke comparisons on Folds 1 and 4, which contain the two severe
zero-Doppler false-alarm scans. Use identical initialization, batches, epochs,
threshold policy, and seed for all mechanisms.

A learned mechanism advances to the six-fold development run only if:

1. validation Joint Pd is no more than 0.01 below the frozen DPG baseline;
2. worst-background-scan Pfa is no worse than the fixed soft notch;
3. calibrated scores never increase;
4. target-region probability suppression satisfies the configured protection
   limit;
5. no threshold is selected from a test scan.

The formal comparison must report pooled and per-scan results, especially the
two failure scans `20260204_100739` and `20260204_100845`.

## Reproducibility

The generated local audit is under
`results/data_audit/zero_doppler_candidate_veto_v1/`. Rebuild it with:

```bash
conda run -n radar-torch python \
  scripts/audit_zero_doppler_candidate_veto_v1.py --overwrite
```

The source module, objective, configuration, tests, and this decision document
are tracked. Generated CSV and JSON outputs remain local.
