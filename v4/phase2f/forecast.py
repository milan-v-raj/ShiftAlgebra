#!/usr/bin/env python3
"""Apply frozen full-vector methods to the sealed confirmation bank."""
import csv,json
from pathlib import Path
from v4.phase2.train_bank import sha256
from v4.phase2f import METRICS,PROTOCOL_ID
from v4.phase2f.method import apply,load_behavior,rows
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'v4/phase2f';RESULTS=BASE/'results'
def run():
 frozen=json.loads((BASE/'FROZEN_FULL_VECTOR_METHODS.json').read_text())
 if frozen['status']!='FULL_VECTOR_METHODS_FROZEN_CONFIRMATION_OUTCOMES_UNOPENED':raise ValueError('method not frozen')
 if sha256(ROOT/frozen['development_behavior'])!=frozen['development_behavior_sha256']:raise ValueError('development behavior changed after method freeze')
 behavior=load_behavior(RESULTS/'confirmation-behavior.csv',{'calibration','confirmation'});lookups={}
 for name,d in frozen['designs'].items():
  rs=rows(behavior,'confirmation',d['pair_groups'],frozen['q_by_metric']);lookups[name]={(r['model_id'],r['word']):r for r in rs}
 base=lookups['full15'];out=[]
 for key in sorted(base):
  r=base[key];row={k:r[k] for k in ('model_id','lineage_id','family','recipe','seed','epoch','parameter_count','word','reverse_group','length')};row['protocol']=PROTOCOL_ID
  for metric in METRICS:
   row[f'true_{metric}']=r['target'][metric];row[f'g0_{metric}']=apply(r,frozen['g0_models'][metric])
   for name,d in frozen['designs'].items():row[f'{name}_{metric}']=apply(lookups[name][key],d['models'][metric])
  out.append(row)
 path=RESULTS/'confirmation-forecast.csv'
 with path.open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=list(out[0]),lineterminator='\n');w.writeheader();w.writerows(out)
 meta={'protocol':PROTOCOL_ID,'status':'FULL_VECTOR_CONFIRMATION_FORECAST','rows':len(out),'metrics':list(METRICS),'sha256':None};meta['sha256']=sha256(path);(RESULTS/'forecast.json').write_text(json.dumps(meta,indent=2,sort_keys=True)+'\n');print(f'wrote {len(out)} full-vector rows');return out
if __name__=='__main__':run()
