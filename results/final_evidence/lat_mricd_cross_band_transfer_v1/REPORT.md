# LAT-MRICD Cross-Band Transfer V1

Status: `COMPLETE_PREREGISTERED_CROSS_BAND_TRANSFER`
Class contract: UAV versus weather
Fit scope: released source bands only
Target bands: S and Ku

## Locked primary results

| Transfer | Model | Target batch-class macro accuracy | UAV batch recall | Weather batch recall |
|---|---|---:|---:|---:|
| narrow_x_to_ku_shared_binary | dummy_prior | 0.5000 | 0.0000 | 1.0000 |
| narrow_x_to_ku_shared_binary | logistic_batch_balanced | 0.8400 | 0.8493 | 0.8307 |
| narrow_x_to_ku_shared_binary | random_forest_batch_balanced | 0.6285 | 0.3217 | 0.9354 |
| narrow_x_to_s_shared_binary | dummy_prior | 0.5000 | 0.0000 | 1.0000 |
| narrow_x_to_s_shared_binary | logistic_batch_balanced | 0.6517 | 0.4433 | 0.8600 |
| narrow_x_to_s_shared_binary | random_forest_batch_balanced | 0.5108 | 0.0851 | 0.9365 |

## Evaluation contract

- The StandardScaler, model, weighted source prior and fixed argmax decision are fit from source
  rows only. Calibration and threshold tuning are disabled.
- Target rows are used once for final aggregate evaluation. Passing the stopping gate does not
  authorize reusing either target for a new confirmatory model comparison.
- Bootstrap intervals resample complete target raw batch codes and are conditional on the fixed
  source fit. Logistic-minus-dummy intervals use identical target-batch draws.
- Raw-code-disjoint sensitivity removes overlapping batch codes from source rows only; target
  rows remain unchanged. It is not primary evidence.

## Claim boundary

The only permitted interpretation is dataset-internal band-held-out UAV/weather performance
using fixed interpretable features. The result does not establish physical-frequency
invariance, physical micro-Doppler, same-event fusion, unseen-model or independent-scene
generalization, H/V polarimetry, balloon recognition, causal deployment, or Tian reproduction.
