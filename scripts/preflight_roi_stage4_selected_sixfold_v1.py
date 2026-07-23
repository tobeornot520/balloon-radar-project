#!/usr/bin/env python3
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];folds=[1,2,3,4,5,6]
required=['scripts/run_polarimetric_representation_benchmark_v2.py','training/train_polarimetric_representation_fcn_v2.py','scripts/build_roi_polarimetric_cache_v1.py','training/train_roi_polarimetric_refiner_v1.py']
status={'required_files':{p:(ROOT/p).is_file() for p in required},'manifests':{},'power2_checkpoints':{}}
for f in folds:
 status['manifests'][str(f)]=(ROOT/f'results/data_audit/dataset_v4_multifold/fold_{f:02d}_manifest.csv').is_file()
 status['power2_checkpoints'][str(f)]=(ROOT/f'results/experiments/polar_repr_v2_power2_v4_fold{f:02d}_seed42_formal/checkpoints/best.pt').is_file()
status['missing_required']=[k for k,v in status['required_files'].items() if not v];status['missing_manifests']=[k for k,v in status['manifests'].items() if not v];status['missing_power2_folds']=[int(k) for k,v in status['power2_checkpoints'].items() if not v]
out=ROOT/'results/data_audit/roi_stage4_selected_sixfold_v1';out.mkdir(parents=True,exist_ok=True);(out/'preflight_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2));print(json.dumps(status,ensure_ascii=False,indent=2))
if status['missing_required'] or status['missing_manifests']:sys.exit(2)
print('PREFLIGHT=PASS')
