#!/usr/bin/env python3
"""Preregistered family-hierarchical Phase-2E analysis."""
import csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from v4.phase2b.analyze import method_metrics
from v4.phase2e import PROTOCOL_ID
ROOT=Path(__file__).resolve().parents[2]; RESULTS=ROOT/'v4/phase2e/results'; SELECTORS=('active','magnitude','variance','coverage','d_optimal'); RANDOM=tuple(f'random_{i:02d}' for i in range(20)); DRAWS=10000; SEED=22603
def load_rows():
 out=[]
 for raw in csv.DictReader((RESULTS/'forecast.csv').open()):
  r=dict(raw)
  for n in ('seed','epoch','parameter_count','length'):r[n]=int(r[n])
  for n in ('true_accuracy','clean_accuracy','g0_prior','g1_prior','full15',*SELECTORS,*RANDOM):r[n]=float(r[n])
  out.append(r)
 if len(out)!=1440:raise ValueError(len(out))
 return out
def bootstrap(rows):
 g=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
 for r in rows:g[r['family']][r['reverse_group']][r['lineage_id']].append(abs(r['g0_prior']-r['true_accuracy'])-abs(r['g1_prior']-r['true_accuracy']))
 fams=sorted(g); groups=sorted({r['reverse_group'] for r in rows}); lines={f:sorted({r['lineage_id'] for r in rows if r['family']==f}) for f in fams}
 def est(fs,gs,line_samples):return 100*float(np.mean([np.mean(g[f][gr][line]) for oi,f in enumerate(fs) for gr in gs for line in line_samples[oi]]))
 point=est(fams,groups,[lines[f] for f in fams]); rng=np.random.default_rng(SEED); draws=[]
 for _ in range(DRAWS):
  fs=[fams[i] for i in rng.integers(0,len(fams),len(fams))]; gs=[groups[i] for i in rng.integers(0,len(groups),len(groups))]; ls=[[lines[f][i] for i in rng.integers(0,len(lines[f]),len(lines[f]))] for f in fs]; draws.append(est(fs,gs,ls))
 return {'point_pp':point,'ci95_pp':[float(np.quantile(draws,.025)),float(np.quantile(draws,.975))],'draws':DRAWS,'seed':SEED,'hierarchy':'architecture family -> lineage; crossed with reversal group'}
def analyze():
 rows=load_rows(); names=('g0_prior','g1_prior','full15',*SELECTORS,*RANDOM); metrics={n:method_metrics(rows,n) for n in names}; inc=bootstrap(rows); byfamily={}
 for f in sorted({r['family'] for r in rows}):
  s=[r for r in rows if r['family']==f]; byfamily[f]={'g0_mae_pp':100*float(np.mean([abs(r['g0_prior']-r['true_accuracy']) for r in s])),'g1_mae_pp':100*float(np.mean([abs(r['g1_prior']-r['true_accuracy']) for r in s]))}; byfamily[f]['improvement_pp']=byfamily[f]['g0_mae_pp']-byfamily[f]['g1_mae_pp']
 g0,g1=metrics['g0_prior'],metrics['g1_prior']; gates={'g1_mae_better':bool(g1['accuracy_mae_pp']<g0['accuracy_mae_pp']),'family_lineage_reversal_ci_excludes_zero':bool(inc['ci95_pp'][0]>0),'at_least_three_of_four_family_means_positive':bool(sum(x['improvement_pp']>0 for x in byfamily.values())>=3),'g1_controlled_rank_better':bool((g1['controlled_model_rank_spearman'] or -1)>(g0['controlled_model_rank_spearman'] or -1)),'g1_regret_not_worse':bool(g1['top1_selection_regret_pp']<=g0['top1_selection_regret_pp'])}; gates['architecture_breadth_gate_passed']=all(gates.values())
 random_mae=[metrics[n]['accuracy_mae_pp'] for n in RANDOM]; result={'protocol':PROTOCOL_ID,'status':'ARCHITECTURE_BREADTH_GATE_PASSED' if gates['architecture_breadth_gate_passed'] else 'ARCHITECTURE_BREADTH_GATE_FAILED','rows':len(rows),'models':24,'lineages':12,'families':4,'words':60,'methods':{n:metrics[n] for n in ('g0_prior','g1_prior','full15',*SELECTORS)},'primary_increment':inc,'primary_gates':gates,'by_family':byfamily,'random_strategy':{'designs':20,'mae_mean':float(np.mean(random_mae)),'mae_median':float(np.median(random_mae)),'mae_range':[float(np.min(random_mae)),float(np.max(random_mae))]},'per_length_mae_pp':{str(l):{n:100*float(np.mean([abs(r[n]-r['true_accuracy']) for r in rows if r['length']==l])) for n in ('g0_prior','g1_prior','full15',*SELECTORS)} for l in (4,5,6)}}
 (RESULTS/'PHASE2E_ANALYSIS.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n');print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False));return result
if __name__=='__main__':analyze()
