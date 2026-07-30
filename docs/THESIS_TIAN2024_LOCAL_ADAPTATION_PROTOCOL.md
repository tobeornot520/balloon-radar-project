# Thesis Tian2024 Local Adaptation Protocol

## Purpose

This branch implements the local six-channel adaptation described in
`参考资料/thesis.pdf` using only the existing train and validation partitions.
It is separate from the original Tian et al. reproduction branch, whose local
`4 x 16` output-stride migration was closed after fixed-template degeneration.

## Frozen Method Contract

- Input: `Re(H), Im(H), Re(V), Im(V), sZdr, sRhoCo`, shape `6 x 128 x 100`.
- `sZdr`: pointwise relative H/V RD power ratio in dB. It is not absolute ZDR.
- `sRhoCo`: local H/V complex coherence magnitude from a `3 x 3` RD window.
- Shared first convolution: `6 -> 16`, kernel `3 x 5`.
- Separate classification and offset branches with two `2 x 2` max pools.
- Output grid: `32 x 25`, stride `4 x 4`.
- Classification target: `7` Doppler bins by `5` range gates, max-pooled `4 x 4`.
- Regression target: normalized offset at the responsible grid cell only.
- Loss: balanced randomly sampled BCE plus `10 * SmoothL1`.
- Inference: direct global classification maximum plus its offset; no PIR/MDP.
- Optimizer: Adam, learning rate `1e-4`, batch size `4`, seed `42`.

The thesis reports batch Z-score preprocessing. The primary reproduction keeps
that behavior as `batch_channel`, but this makes predictions depend on batch
composition. A `sample_channel` sensitivity run is required before any model is
treated as deployable.

## Validation Gate

The training entry never constructs a test dataset. A test run remains closed
unless validation has at least two distinct target peak grids, a distinct-grid
ratio of at least `0.25`, and mean pairwise target probability-map correlation
below `0.98`. The validation report also records responsible-grid selection,
distance/velocity error, joint Pd, Pfa, threshold source, and `test_split_loaded`.

Run a bounded interface and mechanism smoke:

```bash
python training/train_thesis_tian2024_adapter.py \
  --name thesis_tian2024_adapter_smoke_v1 \
  --scope smoke \
  --epochs 3 \
  --device cpu
```

Run the full train/validation experiment only after the smoke passes:

```bash
python training/train_thesis_tian2024_adapter.py \
  --name thesis_tian2024_adapter_fold01_v1 \
  --scope validation
```

The validation command deterministically samples 208 background and 80 UAV
records from the leakage-controlled training partition using seed 42, matching
the reported class counts without pretending that the unavailable thesis sample
IDs are known. It evaluates the complete Fold 1 validation partition.

This is a method-level local adaptation, not an exact numerical reproduction:
the thesis subset IDs and original source code are unavailable. It also cannot
support claims about balloon payload recognition because the current detection
data contain UAV and background only.
