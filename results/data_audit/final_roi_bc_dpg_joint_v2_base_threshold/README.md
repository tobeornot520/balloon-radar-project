# Fixed-threshold ROI and BC-DPG joint audit

This is the authoritative six-fold, sample-aligned joint audit.

## Decision sources

- BC-DPG: `base_threshold_test_predictions.csv`
- ROI: `refined_fixed_*` columns from `test_predictions.csv`
- Test threshold retuning: disabled

All 1,148 test rows align exactly by fold, label, sample ID, and MAT path.

## Pooled results

| Model | False alarms | Correct detections |
|---|---:|---:|
| BC-DPG-FCN v3 | 56 | 289/318 |
| Power2 baseline | 300 | 268/318 |
| ROI power control | 237 | 268/318 |
| ROI RI4 | 196 | 268/318 |

BC-DPG and ROI-RI4 share 36 false alarms and 263 correct detections. Their union
recovers five ROI-only correct targets but raises the false-alarm count to 216;
their intersection lowers false alarms to 36 but loses 26 BC-only correct
detections. The audit therefore does not justify a naive fixed combination.

Regenerate into a new directory before comparing with this frozen output:

```bash
python scripts/build_final_roi_bc_dpg_joint_audit.py \
  --output-dir results/data_audit/final_roi_bc_dpg_joint_rebuild
```
