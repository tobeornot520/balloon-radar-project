# LAT-MRICD Grouped Interpretable Baseline V1

Status: `COMPLETE_GROUPED_PUBLIC_DATA_BASELINE`  
Implementation commit: `a102ea0c81925a3e0686bccc763a1856d6da319e`  
Grouping: `(representation, band_code, batch_code)`  
Held-out folds: `5`

## Primary grouped results

| Task | Model | Fold-macro balanced accuracy | Worst-fold balanced accuracy | Batch-macro accuracy | Batch accuracy P10 | Worst-batch accuracy | Batch-class macro accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| hrrp_x_category | dummy_prior | 0.3333 | 0.3333 | 0.2895 | 0.0000 | 0.0000 | 0.3176 |
| hrrp_x_category | logistic_batch_balanced | 0.6042 | 0.4946 | 0.6716 | 0.2295 | 0.0131 | 0.6617 |
| hrrp_x_category | random_forest_batch_balanced | 0.6313 | 0.4934 | 0.6737 | 0.2402 | 0.0952 | 0.6481 |
| narrow_x_category | dummy_prior | 0.3333 | 0.3333 | 0.3921 | 0.0000 | 0.0000 | 0.3400 |
| narrow_x_category | logistic_batch_balanced | 0.7852 | 0.7204 | 0.8217 | 0.6150 | 0.0000 | 0.7999 |
| narrow_x_category | random_forest_batch_balanced | 0.7542 | 0.6973 | 0.8635 | 0.6405 | 0.0000 | 0.7872 |

## Experimental contract

- The metadata-only batch manifest was frozen before any signal feature was extracted.
- Every physical row appears in exactly one held-out fold.
- No batch code appears in both training and held-out data within a fold.
- All held-out folds contain UAV, bird and weather records.
- Hyperparameters are fixed in `configs/lat_mricd_grouped_baseline_v1.json`; no search or
  held-out-driven model selection is performed.
- The dummy, balanced logistic and balanced random-forest results are all retained. A larger
  number does not authorize choosing a model on an external locked test.
- Training weights give each class equal total weight and each batch-class cell equal weight
  within its class. The primary batch-class metric applies the same hierarchy at evaluation.
- Metrics include sample, fold, batch and batch-class views because record counts are highly
  imbalanced across batches and classes.

## Feature scope

The HRRP branch uses per-record normalized amplitude geometry, entropy, quantile width,
roughness and autocorrelation. The Narrow branch uses scale/global-phase-invariant envelope,
phase-increment, autocorrelation and normalized Doppler-spectrum summaries. Frequencies are
reported only in cycles/sample.

## Claim boundary

This is an internal grouped public-data baseline, not an external blind test. It evaluates new
batch codes for already represented submodels; it is not unseen-model generalization. Batch
semantics are not independently verified, so batch isolation is a conservative proxy for
acquisition grouping.
The dataset contains no H/V pair, no balloon label and no verified PRF or continuous timestamp.
These results cannot establish physical micro-Doppler in Hz, polarimetric performance, causal
deployment, Tian reproduction or balloon-payload recognition.
