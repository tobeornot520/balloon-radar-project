# Fixed-threshold ROI and BC-DPG joint audit: paper evidence

## Evidence status

This package is generated only from the authoritative frozen six-fold joint audit. All 1,148 test rows are aligned exactly by fold, label, sample ID, and MAT path. BC decisions come from `base_threshold_test_predictions.csv`; ROI decisions come from the frozen `refined_fixed_*` columns. Test thresholds were not retuned.

Source: `results/data_audit/final_roi_bc_dpg_joint_v2_base_threshold/`

## Pooled detector results

| display_name | false_alarms | pfa | correct_detections | joint_pd |
|---|---|---|---|---|
| BC-DPG-FCN v3 | 56 | 0.0675 | 289 | 0.9088 |
| Power2 baseline | 300 | 0.3614 | 268 | 0.8428 |
| ROI power control | 237 | 0.2855 | 268 | 0.8428 |
| ROI RI4 | 196 | 0.2361 | 268 | 0.8428 |

BC-DPG-FCN v3 remains the strongest current detector with 56 false alarms and 289/318 correct detections.

## BC-DPG and ROI RI4 complementarity

The two methods share 36 false alarms. BC-DPG contributes 20 BC-only false alarms, while ROI RI4 contributes 160 ROI-only false alarms. For targets, they share 263 correct detections; 26 are BC-only and 5 are ROI-only.

## Simple logical-combination diagnostics

| rule | false_alarms | correct_detections | missed_targets | selection_status |
|---|---|---|---|---|
| BC-DPG-FCN v3 alone | 56 | 289 | 29 | frozen current detector |
| ROI RI4 alone | 196 | 268 | 50 | independent suppression study |
| AND / intersection | 36 | 263 | 55 | diagnostic only, not selected |
| OR / union | 216 | 294 | 24 | diagnostic only, not selected |

The OR/union diagnostic recovers five targets missed by BC-DPG but raises false alarms from 56 to 216. The AND/intersection diagnostic reduces false alarms to 36 but loses 26 BC-only correct detections. These counts are descriptive test-set diagnostics, not candidate rules selected for deployment.

## Interpretation and claim boundary

ROI remains an independent suppression study rather than a trained joint model. The fixed-threshold audit does not support selecting a naive AND, OR, or serial combination. Any future learned combination must be selected using training or validation data and evaluated once with frozen rules.

The evidence supports an internal six-fold H/V UAV detection and localization front end. It does not establish balloon-payload classification, cross-site blind generalization, or strict real-time causal scan adaptation.

## Reproduction

```bash
python scripts/build_roi_bc_dpg_joint_paper_assets.py
```

The build validates source status, row alignment, and prediction-derived metrics before writing tables or figures. `evidence_manifest.json` records SHA256 hashes for all source and generated artifacts.
