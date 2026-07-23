#!/usr/bin/env python3
from pathlib import Path
import argparse,zipfile,json,hashlib
ROOT=Path(__file__).resolve().parents[1];MODES=('power2_baseline','power2_roi_power_control','power2_roi_ri4')
def name(m,f,s):return f'roi_polar_stage4_v1_{m}_v4_fold{f:02d}_seed42_{s}'
def main():
 p=argparse.ArgumentParser();p.add_argument('--scope',default='formal');p.add_argument('--folds',nargs='+',type=int,default=[1,2,3,4,5,6]);p.add_argument('--terminal-log',default='roi_stage4_selected_sixfold_formal_terminal_v1.log');a=p.parse_args();zp=ROOT/f'roi_stage4_selected_sixfold_{a.scope}_acceptance_v1.zip';items=[]
 for rel in ['configs/roi_stage4_selected_sixfold_v1.json','docs/STAGE4_SIXFOLD_PREREGISTRATION.md','scripts/run_roi_stage4_selected_sixfold_v1.py','scripts/summarize_roi_stage4_selected_sixfold_v1.py','scripts/audit_roi_stage4_selected_sixfold_v1.py','scripts/generate_roi_stage4_paper_assets_v1.py']:
  q=ROOT/rel
  if q.is_file():items.append(q)
 audit=ROOT/'results/data_audit/roi_stage4_selected_sixfold_v1'
 if audit.is_dir():items.extend([x for x in audit.rglob('*') if x.is_file()])
 for f in a.folds:
  for m in MODES:
   t=ROOT/'results/experiments'/name(m,f,a.scope)/'tables'
   for fn in ['summary.json','val_predictions.csv','test_predictions.csv','training_history.csv']:
    q=t/fn
    if q.is_file():items.append(q)
 log=ROOT/a.terminal_log
 if log.is_file():items.append(log)
 inv=[]
 for f in a.folds:
  q=ROOT/f'results/experiments/polar_repr_v2_power2_v4_fold{f:02d}_seed42_formal/checkpoints/best.pt'
  if q.is_file():
   h=hashlib.sha256();
   with q.open('rb') as fh:
    for c in iter(lambda:fh.read(1048576),b''):h.update(c)
   inv.append({'fold':f,'path':str(q.relative_to(ROOT)),'size_bytes':q.stat().st_size,'sha256':h.hexdigest()})
 audit.mkdir(parents=True,exist_ok=True);iv=audit/f'base_power2_checkpoint_inventory_{a.scope}.json';iv.write_text(json.dumps(inv,indent=2));items.append(iv)
 with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z:
  seen=set()
  for q in items:
   arc=q.relative_to(ROOT)
   if str(arc) in seen:continue
   seen.add(str(arc));z.write(q,arc)
 print(zp);print('files=',len(seen));print('checkpoint bytes included=false');print('raw MAT included=false');print('cache tensor bytes included=false')
if __name__=='__main__':main()
