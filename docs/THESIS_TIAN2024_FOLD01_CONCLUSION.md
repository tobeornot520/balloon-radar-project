# Thesis Tian2024 Fold 1 Conclusion

Date: 2026-07-30

## Decision

`thesis_tian2024_local_adaptation_v1` successfully fixes the fixed-template
failure of the earlier original-Tian migration on Fold 1 validation, but fails
to generalize to the historical local held-out background scan. The selected
model is rejected as a generalizing detector and must not be used as balloon
payload-recognition evidence.

## What Worked

- Six H/V channels, `2 x 2` pooling, a `32 x 25` output grid, and direct-max
  decoding produce sample-dependent responses rather than the previous fixed
  velocity-band template.
- The deployment-stable `sample_channel` variant reaches validation joint Pd
  `50/53` at Pfa `1/150` with a frozen threshold of `0.7036690711975098`.
- Probability-map template correlation is `0.3557`, far below the failed
  original migration's approximately `0.99818`.

## What Failed

The single frozen held-out evaluation obtains joint Pd `30/53` and Pfa
`134/150`. Validation and held-out backgrounds each contain only one independent
scan group. Their peak mechanisms are nearly opposite:

| Background partition | Scan group | Edge Doppler peaks | Zero-Doppler peaks | Above frozen threshold |
|---|---|---:|---:|---:|
| Validation | `20260204_100802` | 145/150 | 5/150 | 1/150 |
| Held-out | `20260204_100739` | 5/150 | 125/150 | 134/150 |

The validation threshold therefore estimates one background scan condition,
not a stable false-alarm distribution. Raising it using held-out scores would
be test-set tuning and would not solve the underlying scan-group shift.

## Next Experiment Gate

1. Treat the current held-out split as consumed historical evidence.
2. Develop only on folds with multiple background scan groups contributing to
   training and validation aggregation.
3. Audit the zero-Doppler row before selecting a mechanism: an exclusion notch,
   explicit dense negative supervision, or a clutter-aware auxiliary target
   must be compared under the same development folds.
4. Select thresholds from pooled and worst-group validation behavior, not one
   background scan.
5. Preserve a different outer group for one final evaluation after model,
   threshold, tolerances, and zero-Doppler policy are frozen.

This remains a method-level local adaptation because the thesis sample IDs and
source code are unavailable. The current data contain UAV and background only.
