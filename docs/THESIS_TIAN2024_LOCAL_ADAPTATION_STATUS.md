# Thesis Tian2024 Local Adaptation Status

Date: 2026-07-30

## Full Fold 1 Result

The full validation experiment used the deterministic 80-UAV/208-background
training subset and all 53 UAV plus 150 background validation samples. The
reported batch-dependent normalization and the deployment-stable per-sample
normalization were compared before any held-out evaluation.

| Normalization | Best epoch | Validation joint Pd | Validation Pfa | Responsible grid | Range / velocity MAE | Template correlation |
|---|---:|---:|---:|---:|---:|---:|
| `batch_channel` | 29 | 49/53 = 0.9245 | 1/150 = 0.0067 | 0.7925 | 1.34 / 1.51 | 0.2497 |
| `sample_channel` | 34 | 50/53 = 0.9434 | 1/150 = 0.0067 | 0.7736 | 1.53 / 1.45 | 0.3557 |

Both outputs passed the preregistered non-degeneration gate. `sample_channel`
was frozen because it had the higher joint Pd and does not depend on inference
batch composition. Its checkpoint SHA256, manifest SHA256, threshold
`0.7036690711975098`, tolerances, and selection evidence are recorded in
`configs/thesis_tian2024_adapter_fold01_selected_v1.yaml`.

A single local held-out evaluation was then run with the frozen checkpoint and
threshold. No threshold tuning was performed on held-out data.

| Held-out metric | Result |
|---|---:|
| Joint Pd | 30/53 = 0.5660 |
| Pfa | 134/150 = 0.8933 |
| Responsible-grid selection | 0.4528 |
| Range MAE | 6.02 gates |
| Velocity MAE | 5.68 bins |

This candidate is rejected as a generalizing detector. The failure is a
background scan-group shift, not a marginal threshold miss: all validation
backgrounds come from `20260204_100802`, while all held-out backgrounds come
from `20260204_100739`. Of 150 validation backgrounds, 145 peak on the Doppler
output edges and only 1 exceeds the frozen threshold. Of 150 held-out
backgrounds, 125 peak at the zero-Doppler row (`grid_y=16`) and 134 exceed the
threshold. Retuning on held-out scores would hide this failure and is forbidden.

The next model experiment must use multiple background scan groups for model
selection, explicitly supervise or suppress the zero-Doppler clutter mechanism,
and reserve an untouched outer group. The current held-out partition has now
been consumed and must not be presented as a future blind test.

## Outcome

The existing local H/V IQ data now drive an independent implementation of the
six-channel Tian2024 adaptation described in `参考资料/thesis.pdf`. The complete
train/validation-only path has been exercised without network access. No test
dataset was constructed or loaded.

This branch is deliberately separate from the failed original Tian migration:

| Branch | Input | Pooling / output | Decoding | Current conclusion |
|---|---|---|---|---|
| Original Tian local migration | H, optionally V/HV | `2 x 4`, `32 x 7` | PIR/MDP | Closed after fixed-template degeneration |
| Thesis local adaptation | H/V six-channel | `2 x 2`, `32 x 25` | Direct maximum | Interface and sample-dependent response verified; localization not yet verified |
| Polarimetric transfer encoder | Candidate ROI features | Encoder interface | Downstream task dependent | Reusable interface only, no balloon result claim |

## Validation-Only Smoke Evidence

Two bounded CPU runs used 4 background and 4 UAV samples in each of train and
validation. These runs are mechanism checks, not performance estimates.

| Run | Epochs | Unique target peak grids | Mean target-map correlation | Responsible-grid rate | Joint Pd | Pfa |
|---|---:|---:|---:|---:|---:|---:|
| `thesis_tian2024_adapter_smoke_v1` | 3 | 1/4 | 0.8149 | 0 | 0 | 0.25 |
| `thesis_tian2024_adapter_smoke30_v1` selected epoch 23 | 30 | 4/4 | 0.9479 | 0 | 0 | 0.25 |
| `thesis_tian2024_adapter_fold01_preflight_v1` | 1 | 3/53 | 0.9579 | 0 | 0 | 0.0067 |

The 30-epoch run passes the narrow output-diversity audit: unlike the previous
original-Tian branch (`~0.99818` template correlation and near-fixed bands), its
peak varies by sample. It does **not** pass a useful localization gate because
no validation target selected its responsible grid. Test evaluation therefore
remains closed.

The selected smoke checkpoint and detailed run tables remain local under
ignored `results/experiments/` and are not part of Git history.

The one-epoch preflight exercised the deterministic 80-UAV/208-background
training contract against the complete 53-UAV/150-background Fold 1 validation
partition. It confirms the full-size interface and false-alarm threshold path,
but one epoch is not a performance experiment and did not pass the diversity or
localization gates.

## Next Decision

The thesis-scale Fold 1 experiment and normalization sensitivity check are now
complete. The next bounded experiment should address the demonstrated
zero-Doppler background-group failure using development folds only. It must not
reuse the consumed Fold 1 held-out results for threshold or architecture
selection.

Because the thesis subset identifiers and original code are unavailable, any
result remains a method-level local adaptation rather than exact numerical
reproduction. Current data support UAV/background detection research only and
cannot validate balloon payload recognition.

## Verification

- Focused adapter tests: 7 passed.
- Full test suite: 92 passed.
- Project health: 166 Python files, 0 syntax errors, 0 duplicate groups, 43/43
  required files present.
