#!/usr/bin/env python3
from pathlib import Path
import json,zipfile
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'results/data_audit/roi_bc_dpg_joint_context_v1';out.mkdir(parents=True,exist_ok=True);patterns=['models/*scan*calibrator*.py','training/*scan*calibrator*.py','scripts/run_bc_dpg_v3.py','scripts/compare_bc_dpg_v3.py','configs/*bc_dpg*v3*.yaml','results/data_audit/roi_stage4_selected_sixfold_v1/*.csv','results/data_audit/roi_stage4_selected_sixfold_v1/*.json'];found=[]
for pat in patterns:
 found.extend([p for p in ROOT.glob(pat) if p.is_file()])
st={'found':[str(p.relative_to(ROOT)) for p in found],'raw_data_included':False,'checkpoint_bytes_included':False};(out/'status.json').write_text(json.dumps(st,ensure_ascii=False,indent=2));zp=ROOT/'roi_bc_dpg_joint_context_acceptance_v1.zip'
with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z:
 z.write(out/'status.json',(out/'status.json').relative_to(ROOT))
 for p in found:z.write(p,p.relative_to(ROOT))
print(zp)
