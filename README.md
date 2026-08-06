# Balloon radar research project

This repository contains the active code, protocols, and compact evidence for
H/V radar UAV detection and localization research that is intended to support
later balloon-payload recognition work.

The current data only support UAV/background development experiments. They do
not establish balloon-payload classification, cross-date generalization,
physical micro-Doppler interpretation, or calibrated polarimetric claims.

## Start here

- [Current status](docs/CURRENT_STATUS.md)
- [Documentation index](docs/INDEX.md)
- [Project structure](docs/PROJECT_STRUCTURE.md)
- [Data card](docs/DATA_CARD.md)
- [Model-selection ledger](docs/MODEL_SELECTION_LEDGER.md)

## Active workflows

| Workflow | Entry point |
|---|---|
| BC-DPG detection | `scripts/run_bc_dpg_v3.py` |
| Polarimetric representation benchmark | `scripts/run_polarimetric_representation_benchmark_v2.py` |
| ROI polarimetric refinement | `scripts/run_roi_stage4_selected_sixfold_v1.py` |
| Tian FCN method reproduction | `scripts/run_tian_fcn_sixfold.py` |
| Zero-Doppler mechanism comparison | `scripts/run_zero_doppler_mechanism_v1.py` |
| New-data contract validation | `scripts/validate_data_collection_manifest.py` |
| Field readiness audit | `scripts/audit_field_readiness_v1.py` |
| Data-free multidomain feature smoke | `scripts/run_multidomain_feature_smoke_v1.py` |
| Multidomain feature contract audit | `scripts/audit_multidomain_feature_contract_v1.py` |
| LAT-MRICD grouped baseline | `scripts/run_lat_mricd_grouped_baseline_v1.py` |
| LAT-MRICD cross-band transfer | `scripts/run_lat_mricd_cross_band_transfer_v1.py` |
| Experiment ledger | `scripts/run_recorded_experiment.py` |

Use `python <entry-point> --help` before running an experiment. Frozen claims,
selection limits, and known confounding are maintained in the status and model
selection documents rather than duplicated here.

## Environment and checks

The formal environment is defined by `environment.yml` and uses Python 3.11.

```bash
conda env create -f environment.yml
conda activate radar-torch
python scripts/check_project_health.py
python -m pytest
```

Raw radar data, checkpoints, full experiment runs, reference PDFs, development
transcripts, and generated distribution packages stay local and are excluded
from Git. Compact frozen evidence may be committed only after its provenance
and claim boundary have been reviewed.
