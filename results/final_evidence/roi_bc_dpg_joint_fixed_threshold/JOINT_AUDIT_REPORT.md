# Fixed-threshold ROI and BC-DPG audit: transparent internal evidence

## Evidence status

This package is generated only from the authoritative frozen six-fold joint audit. All 1,148 test rows are aligned exactly by fold, label, sample ID, and MAT path. BC decisions come from `base_threshold_test_predictions.csv`; ROI decisions come from the frozen `refined_fixed_*` columns. Test thresholds were not retuned.

The absence of test-threshold retuning does not make this a blind evaluation. BC-DPG design was informed by development-fold feedback, and Stage 4 modes were screened on Folds 1 and 4 before those folds were included in the six-fold extension.

Source: `results/data_audit/final_roi_bc_dpg_joint_v2_base_threshold/`

## Pooled detector results

| display_name | false_alarms | pfa | correct_detections | joint_pd |
|---|---|---|---|---|
| BC-DPG-FCN v3 (offline) | 56 | 0.0675 | 289 | 0.9088 |
| Power2 baseline | 300 | 0.3614 | 268 | 0.8428 |
| ROI power control | 237 | 0.2855 | 268 | 0.8428 |
| ROI RI4 | 196 | 0.2361 | 268 | 0.8428 |

The complete-scan BC-DPG-FCN v3 has the best observed pooled result among the audited branches, with 56/830 background alarms and 289/318 joint successes. Because its context can include later samples from the same scan, this is an offline scan-aware upper bound, not strict causal deployment performance.

A correct detection requires a target score above the frozen threshold and localization within 2 range gates and 3 velocity bins. It is therefore a joint detection-localization success, not score detection alone.

## Fold heterogeneity

| display_name | macro_pfa | median_pfa | worst_fold_pfa | worst_pfa_fold | macro_joint_pd | worst_fold_joint_pd | worst_joint_pd_fold |
|---|---|---|---|---|---|---|---|
| BC-DPG-FCN v3 (offline) | 0.0622 | 0.0000 | 0.2800 | 01 | 0.9059 | 0.7917 | 06 |
| Power2 baseline | 0.3333 | 0.0067 | 1.0000 | 01 | 0.8389 | 0.6667 | 06 |
| ROI power control | 0.2633 | 0.0000 | 0.9267 | 04 | 0.8389 | 0.6667 | 06 |
| ROI RI4 | 0.2178 | 0.0000 | 0.6667 | 01 | 0.8389 | 0.6667 | 06 |

All 56 BC-DPG false alarms occur in Folds 1 and 4. The worst-fold Pfa is 0.2800 (Fold 01), while the worst-fold joint Pd is 0.7917 (Fold 06). Pooled Pfa alone therefore understates the concentration of errors in difficult backgrounds.

## Derived metrics and uncertainty

| display_name | joint_precision | joint_f1 | specificity | joint_pd_wilson95_low | joint_pd_wilson95_high | pfa_wilson95_low | pfa_wilson95_high |
|---|---|---|---|---|---|---|---|
| BC-DPG-FCN v3 (offline) | 0.8377 | 0.8718 | 0.9325 | 0.8721 | 0.9358 | 0.0523 | 0.0866 |
| Power2 baseline | 0.4718 | 0.6050 | 0.6386 | 0.7987 | 0.8787 | 0.3295 | 0.3947 |
| ROI power control | 0.5307 | 0.6513 | 0.7145 | 0.7987 | 0.8787 | 0.2559 | 0.3172 |
| ROI RI4 | 0.5776 | 0.6854 | 0.7639 | 0.7987 | 0.8787 | 0.2085 | 0.2662 |

A scan-group bootstrap gives BC-DPG Pfa 95% interval [0.0000, 0.1618] and joint Pd 95% interval [0.8599, 0.9521]. The resampling unit is the scan group, but only 6 independent background scan groups are available, so uncertainty remains weakly identified.

Wilson intervals are included only as sample-level references; they ignore within-scan correlation and should not be the primary uncertainty claim.

## BC-DPG and ROI RI4 complementarity

The two methods share 36 false alarms. BC-DPG contributes 20 BC-only false alarms, while ROI RI4 contributes 160 ROI-only false alarms. For targets, they share 263 correct detections; 26 are BC-only and 5 are ROI-only.

## Simple logical-combination diagnostics

| rule | false_alarms | correct_detections | missed_targets | selection_status |
|---|---|---|---|---|
| BC-DPG-FCN v3 alone | 56 | 289 | 29 | offline complete-scan upper bound |
| ROI RI4 alone | 196 | 268 | 50 | independent suppression study |
| AND / intersection | 36 | 263 | 55 | diagnostic only, not selected |
| OR / union | 216 | 294 | 24 | diagnostic only, not selected |

The OR/union diagnostic recovers five targets missed by BC-DPG but raises false alarms from 56 to 216. The AND/intersection diagnostic reduces false alarms to 36 but loses 26 BC-only correct detections. These counts are descriptive test-set diagnostics, not candidate rules selected for deployment.

| paired_outcome | bc_only | roi_ri4_only | discordant_pairs | two_sided_exact_mcnemar_p | status |
|---|---|---|---|---|---|
| background false-alarm decision | 20 | 160 | 180 | 2.607e-28 | post-test paired diagnostic only; not a selection rule |
| target joint-success decision | 26 | 5 | 31 | 1.922e-04 | post-test paired diagnostic only; not a selection rule |

The paired tests quantify differences on the already inspected test outcomes. They are post-test diagnostics and do not authorize model or rule selection.

## Data and selection limitations

Target scan dates in the frozen audit: 20260202. Background scan dates: 20260204. Class and acquisition date are fully confounded in this dataset, so the current results cannot establish cross-date generalization or exclude date-specific acquisition effects.

## Interpretation and claim boundary

ROI remains an independent suppression study rather than a trained joint model. The fixed-threshold audit does not support selecting a naive AND, OR, or serial combination. Any future learned combination must be selected using training or validation data and evaluated once with frozen rules.

The evidence supports an internal development-stage H/V UAV detection and localization front end. It does not establish balloon-payload classification, cross-date or cross-site blind generalization, or strict real-time causal scan adaptation.

## Internal regeneration

```bash
python scripts/build_roi_bc_dpg_joint_paper_assets.py
```

The build validates source status, row alignment, and prediction-derived metrics before writing tables or figures. Full regeneration requires the internal frozen prediction CSVs. A sanitized share package containing only result excerpts is traceable and hash-verifiable, but is not independently reproducible without source data, code, and checkpoints.
