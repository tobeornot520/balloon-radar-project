# Current research status

## Frozen stages

1. DPG-FCN provides the H/V detection and localization baseline.
2. BC-DPG-FCN v3 applies offline scan-aware background calibration.
3. Stage 3 evaluates dense explicit polarimetric representations and freezes
   Power2 as the most reliable detection representation.
4. Stage 4 freezes Power2 candidate locations and applies suppression-only ROI
   refinement.
5. The final joint audit aligns BC-DPG and ROI predictions without retuning a
   threshold on test data.

## Authoritative evidence

- BC-DPG v3: `results/final_evidence/bc_dpg_v3_final/`
- Stage 3: `docs/polarimetric_stage3/STAGE3_FROZEN_CONCLUSION.md`
- Stage 4: `results/data_audit/roi_stage4_selected_sixfold_v1/`
- Joint audit: `results/data_audit/final_roi_bc_dpg_joint_v2_base_threshold/`
- Joint paper evidence: `results/final_evidence/roi_bc_dpg_joint_fixed_threshold/`

The joint paper-evidence package is generated deterministically from the frozen
audit. It contains the formal report, pooled and fold-level tables, logical
combination diagnostics, PNG/PDF figures, and a SHA256 evidence manifest. Build
it with:

```bash
python scripts/build_roi_bc_dpg_joint_paper_assets.py
```

The earlier `final_roi_bc_dpg_joint` output used the wrong BC decision source and
has been moved to the local recovery archive. It must not be cited.

## Current decision

BC-DPG-FCN v3 remains the strongest current detector. ROI refinement provides a
useful independent suppression study, but the fixed-threshold audit does not
support a naive AND, OR, or serial combination. Any learned combination must be
selected only on training or validation data and evaluated once with frozen
rules.

The reported AND/intersection and OR/union counts are diagnostic only. They are
not trained models or deployment rules, and no combination was selected from
the test outcomes.

## Claim boundaries

The current data support an H/V UAV detection and localization front end. They do
not establish balloon payload classification, cross-site blind generalization,
or real-time causal scan adaptation. The complete-scan BC-DPG model is an offline
enhancement because it can use later samples from the same scan.
