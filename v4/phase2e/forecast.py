#!/usr/bin/env python3
"""Apply byte-frozen methods to Phase-2E outcomes."""
import csv,hashlib,json
from pathlib import Path
from v4.phase2.forecast import apply_ridge
from v4.phase2b.method import EXPERTS,apply_gate,forecast_rows,load_behavior
from v4.phase2e import PROTOCOL_ID
ROOT=Path(__file__).resolve().parents[2]; BASE=ROOT/'v4/phase2e'; RESULTS=BASE/'results'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run():
 contract=json.loads((BASE/'FROZEN_BREADTH_METHODS.json').read_text()); source=ROOT/contract['method_source'];
 if contract['status']!='INHERITED_METHODS_AND_GATES_FROZEN_OUTCOMES_UNOPENED' or sha(source)!=contract['method_source_sha256']:raise ValueError('frozen method mismatch')
 frozen=json.loads(source.read_text()); priorpath=ROOT/contract['prior_regressions']; prior=json.loads(priorpath.read_text());
 if sha(priorpath)!=contract['prior_regressions_sha256']:raise ValueError('prior mismatch')
 behavior=load_behavior(RESULTS/'behavior.csv',{'calibration','architecture_breadth'}); methods={}
 for name,d in frozen['selectors'].items():methods[name]=(d,forecast_rows(behavior,d['pair_groups'],'architecture_breadth'))
 for name,d in frozen['random_designs'].items():methods[name]=(d,forecast_rows(behavior,d['pair_groups'],'architecture_breadth'))
 full=frozen['full15']; fullrows=forecast_rows(behavior,full['pair_groups'],'architecture_breadth'); fullmap={(r['model_id'],r['word']):r for r in fullrows}; maps={n:{(r['model_id'],r['word']):r for r in rs} for n,(d,rs) in methods.items()}; out=[]
 for key in sorted(fullmap):
  r=fullmap[key]; expert=dict(zip(EXPERTS,r['expert_accuracies'])); priorrow={**r,'hazard_ordered_q':expert['ordered'],'hazard_symmetric_q':expert['symmetric'],'product_retention':expert['independent'],'max_damage':expert['dominant']}; row={k:r[k] for k in ('model_id','lineage_id','family','recipe','seed','epoch','parameter_count','word','reverse_group','length','true_accuracy','clean_accuracy')}; row['protocol']=PROTOCOL_ID; row['g0_prior']=apply_ridge(priorrow,prior['ridge_g0']); row['g1_prior']=apply_ridge(priorrow,prior['ridge_g1']); row['full15']=apply_gate(r,full['gate'])['prediction']
  for name,(d,_) in methods.items():row[name]=apply_gate(maps[name][key],d['gate'])['prediction']
  out.append(row)
 path=RESULTS/'forecast.csv'
 with path.open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=list(out[0]),lineterminator='\n');w.writeheader();w.writerows(out)
 print(f'wrote {len(out)} breadth rows');return out
if __name__=='__main__':run()
