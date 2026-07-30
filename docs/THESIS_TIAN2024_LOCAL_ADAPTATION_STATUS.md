# Thesis Tian2024 Local Adaptation Status

Date: 2026-07-30

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

The next bounded experiment should use the thesis-reported scale as closely as
the leakage-controlled manifest permits: 80 UAV and 208 background training
samples, all Fold 1 validation samples, 50 Adam epochs, and no test access. Run
the reported `batch_channel` normalization first, followed by the preregistered
`sample_channel` sensitivity check. Only a model with non-degenerate output and
non-zero responsible-grid localization should be considered for a frozen test
proposal.

Because the thesis subset identifiers and original code are unavailable, any
result remains a method-level local adaptation rather than exact numerical
reproduction. Current data support UAV/background detection research only and
cannot validate balloon payload recognition.

## Verification

- Focused adapter tests: 7 passed.
- Full test suite: 91 passed.
- Project health: 166 Python files, 0 syntax errors, 0 duplicate groups, 43/43
  required files present.
