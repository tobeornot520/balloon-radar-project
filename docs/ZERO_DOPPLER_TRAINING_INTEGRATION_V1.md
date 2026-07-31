# Zero-Doppler Training Integration V1

Date: 2026-07-31

## Decision

The unified real-checkpoint training and evaluation path is operational. A
fixed soft notch is safe enough to retain as the reference mechanism, but it
does not suppress the two severe zero-Doppler scans enough to become the final
solution. The next expensive run should train only the dense-negative and
clutter-aware mechanisms on full Fold 1/4 development data.

## Unified Interface

The training entry point supports four modes under one manifest, checkpoint,
seed, batch, localization tolerance, and threshold policy:

- `baseline`: frozen DPG re-inference;
- `fixed_notch`: frozen DPG plus deterministic non-increasing soft notch;
- `dense_negative`: only the DPG fusion head is trainable;
- `clutter_aware`: the DPG is frozen and a 977-parameter non-increasing
  suppression head is trainable.

Every mode reuses the base checkpoint's validation threshold. Test scans never
select a threshold. The checkpoint's recorded manifest must exactly match the
requested manifest.

For learned modes, epoch 0 is a checkpoint-selection candidate. This was added
after the first smoke exposed a Fold 4 localization regression: when both
trained epochs were worse, the corrected selector restored epoch 0 instead of
being forced to keep a degraded model.

## Fold 1/4 Mechanical Smoke

Each split used only eight background and eight UAV records. Two epochs were
run for learned modes. These numbers verify mechanics only and must not select
a mechanism.

- All eight runs completed.
- Dense-negative training updated 23,153 fusion-head parameters.
- Clutter-aware training updated 977 parameters.
- Fixed-notch and clutter-aware test scores never exceeded raw DPG scores.
- Fold 4 dense-negative correctly selected epoch 0 after the regression guard
  was added.

## Six-fold Frozen Soft-notch Audit

The baseline and fixed soft notch were then run on every full validation and
test split without training any parameters.

| Mode | Test false alarms | Pooled Pfa | Worst-fold Pfa | Joint hits | Pooled Joint Pd |
|---|---:|---:|---:|---:|---:|
| Re-inferred baseline | 187 | 0.2253 | 0.6667 | 289 | 0.9088 |
| Fixed soft notch | 120 | 0.1446 | 0.4467 | 290 | 0.9119 |

On validation folds, false alarms fell from 13 to 3 while joint hits stayed at
298. On the severe test folds:

- Fold 1 (`20260204_100739`): 87 to 53 false alarms;
- Fold 4 (`20260204_100845`): 100 to 67 false alarms, with joint hits changing
  from 47 to 48 because the full-heatmap notch moved one peak to a correct
  location.

The paired result supports the non-increasing full-heatmap implementation, but
120 false alarms remain. The fixed notch therefore becomes the safety
reference, not the selected detector.

## Precision Note

The archived candidate-table audit counted 186 raw false alarms, while current
CPU full-heatmap re-inference counts 187. Some archived DPG predictions were
generated under GPU half-precision inference. The one-sample difference is
treated as numerical re-inference drift. Mechanism effects in this stage use
paired outputs from the same CPU pass; they must not silently replace the
previously frozen evidence tables.

## Next Training Gate

Run full-data Fold 1/4 development training for `dense_negative` and
`clutter_aware` only. Do not rerun baseline training, and do not spend six-fold
training time yet.

A learned mechanism advances only if validation selection satisfies:

1. Joint Pd is no more than 0.01 below the epoch-0 baseline;
2. worst severe-scan Pfa is lower than the fixed soft-notch reference;
3. the clutter-aware non-increase contract holds for every sample;
4. epoch selection uses validation only;
5. the result is reported as consumed development evidence, not a blind test.

## Local Outputs

- Smoke summary: `results/data_audit/zero_doppler_mechanism_v1/REPORT_smoke.md`
- Six-fold frozen comparison:
  `results/data_audit/zero_doppler_mechanism_v1/REPORT_development.md`
- Per-run tables:
  `results/experiments/zero_doppler_v1_<mode>_fold<NN>_seed42*/tables/`

Generated predictions and checkpoints remain outside Git.
