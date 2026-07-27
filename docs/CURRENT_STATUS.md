# Current research status

## Frozen stages

1. DPG-FCN provides the H/V detection and localization baseline.
2. Sample-independent BC is the current online-oriented calibration baseline.
3. Complete-scan BC-DPG-FCN v3 is an offline scan-aware upper bound; its context may include later samples from the same scan.
4. Stage 3 evaluates dense explicit polarimetric representations and retains Power2 as the most reliable detection representation.
5. Stage 4 freezes Power2 candidate locations and applies suppression-only ROI refinement.
6. The final audit aligns frozen BC-DPG and ROI predictions without retuning test thresholds or selecting a joint rule.

## Authoritative evidence

- BC-DPG v3: `results/final_evidence/bc_dpg_v3_final/`
- Stage 3: `docs/polarimetric_stage3/STAGE3_FROZEN_CONCLUSION.md`
- Stage 4: `results/data_audit/roi_stage4_selected_sixfold_v1/`
- Joint audit: `results/data_audit/final_roi_bc_dpg_joint_v2_base_threshold/`
- Joint paper evidence: `results/final_evidence/roi_bc_dpg_joint_fixed_threshold/`
- Data card: [DATA_CARD.md](DATA_CARD.md)
- Metric definitions: [METRIC_DEFINITIONS.md](METRIC_DEFINITIONS.md)
- Model-selection ledger: [MODEL_SELECTION_LEDGER.md](MODEL_SELECTION_LEDGER.md)

The deterministic joint evidence build contains the formal report, pooled and fold-level tables, fold-distribution summaries, Wilson and scan-group bootstrap intervals, paired McNemar diagnostics, PNG/PDF figures, and a SHA256 manifest. Build it with:

```bash
python scripts/build_roi_bc_dpg_joint_paper_assets.py
```

The earlier `final_roi_bc_dpg_joint` output used the wrong BC decision source and has been moved to the local recovery archive. It must not be cited.

## Current result

The complete-scan BC-DPG-FCN v3 has the best observed result in the current six-fold internal development evaluation: 56/830 background alarms and 289/318 joint detection-localization successes. This is an offline scan-aware upper bound, not a causal deployment result. Sample-independent BC has 122 false alarms and is the closer online-oriented reference.

The 56 complete-scan BC false alarms are concentrated entirely in Folds 1 and 4. Macro Pfa is 0.0622, median Pfa is 0, worst-fold Pfa is 0.2800 in Fold 1, and worst-fold joint Pd is 0.7917 in Fold 6. The scan-group bootstrap 95% intervals are 0 to 0.1618 for Pfa and 0.8599 to 0.9521 for joint Pd; only six independent background scan groups are available.

ROI refinement remains an independent suppression study. The reported AND/intersection, OR/union, and McNemar results are post-test diagnostics only. No combination was trained or selected from these outcomes.

## Selection and data limitations

The absence of test-threshold retuning does not make the evaluation blind. Stage 4 modes were screened on development Folds 1 and 4, and those folds were reused in the six-fold ROI summary. BC-DPG structure and loss design were also informed by development feedback.

All target scan groups in the frozen audit are dated `20260202`; all background scan groups are dated `20260204`. Class and acquisition date are fully confounded. The current results therefore cannot establish cross-date, cross-site, or external blind generalization.

## Claim boundaries

The current data support an internal development-stage H/V UAV detection and distance-velocity localization front end. They do not establish balloon payload classification, event-level or hourly false-alarm performance, cross-environment blind generalization, or strict real-time causal scan adaptation.
