# Fixed-notch residual V2 preregistration

Date: 2026-07-31

## Scope

This is a development-only revision on consumed Folds 1 and 4. It reuses each
fold's frozen DPG checkpoint and validation threshold. It does not create a new
blind test and must not be expanded to six folds unless the gate below passes.

## Frozen design

1. Apply the existing Gaussian soft notch (`sigma=4`, odds floor `0.05`).
2. Freeze the DPG detector and fixed notch.
3. Learn only a bounded, non-negative velocity-wise residual.
4. Multiply the residual by a fixed zero-Doppler proximity envelope
   (`sigma=8`) so distant rows are protected structurally.
5. Train with dense heatmap loss, target-region probability-drop protection,
   background top-16 peak loss, and residual magnitude regularization.
6. Include epoch 0 and select by validation joint Pd, worst background-group
   Pfa, pooled Pfa, AUC, then loss, in that order.

No test threshold, notch width, target tolerance, residual envelope, or loss
weight may be changed after viewing the Fold 1/4 run.

## Advancement gate

Against the fixed-notch reference on both development folds:

- joint-hit count must not decrease on either fold;
- false-alarm count must not increase on either fold;
- pooled Fold 1/4 false alarms must decrease strictly;
- the non-increasing-logit contract must hold for every test sample.

Failure closes this V2 setting. Any further revision requires a new mechanism
hypothesis and preregistration; it must not become a hyperparameter scan over
the consumed folds.
