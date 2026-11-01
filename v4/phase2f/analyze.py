#!/usr/bin/env python3
"""Preregistered full-vector confirmation analysis."""
import csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from v4.analysis.metrics import model_selection_regret,partial_spearman,rank_models_per_environment
from v4.phase2f import METRICS,PROTOCOL_ID
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'v4/phase2f';RESULTS=BASE/'results';RANDOM=tuple(f'random_{i:02d}' for i in range(20));DRAWS=10000;SEED=22703
def load_rows():
 out=[]
 for raw in csv.DictReader((RESULTS/'confirmation-forecast.csv').open()):
  r=dict(raw)
  for n in ('seed','epoch','parameter_count','length'):r[n]=int(r[n])
  for n in list(r):
   if n.startswith('true_') or any(n.startswith(f'{x}_') for x in ('g0','full15','magnitude5',*RANDOM)):r[n]=float(r[n])
  out.append(r)
 if len(out)!=2160:raise ValueError(len(out))
 return out
def matrices(rows,method,metric):
 words=sorted({r['word'] for r in rows});models=sorted({r['model_id'] for r in rows});d={(r['word'],r['model_id']):r for r in rows};p=np.array([[d[w,m][f'{method}_{metric}'] for m in models] for w in words]);t=np.array([[d[w,m][f'true_{metric}'] for m in models] for w in words]);return p,t,models
def controls(rows,models):
 d={r['model_id']:r for r in rows};f=sorted({r['family'] for r in rows});q=sorted({r['recipe'] for r in rows});return np.array([[np.log1p(d[m]['parameter_count']),d[m]['epoch'],*[float(d[m]['family']==x) for x in f],*[float(d[m]['recipe']==x) for x in q]] for m in models])
def metric_summary(rows,method,metric):
 p,t,models=matrices(rows,method,metric);c=controls(rows,models);partial=[partial_spearman(p[i],t[i],c) for i in range(len(p))];finite=[x for x in partial if np.isfinite(x)];out={'mae':float(np.mean(abs(p-t))),'rank_models_per_environment':float(rank_models_per_environment(p,t)['macro_mean']),'controlled_model_rank_spearman':float(np.mean(finite)) if finite else None}
 if metric=='accuracy':out['top1_selection_regret_pp']=model_selection_regret(p,t,1)
 return out
def summary(rows,method,scales):
 per={m:metric_summary(rows,method,m) for m in METRICS};return {'per_metric':per,'macro_normalized_mae':float(np.mean([per[m]['mae']/scales[m] for m in METRICS])),'macro_controlled_rank':float(np.mean([per[m]['controlled_model_rank_spearman'] for m in METRICS])),'accuracy_top1_regret_pp':per['accuracy']['top1_selection_regret_pp']}
def row_error(r,method,scales):return float(np.mean([abs(r[f'{method}_{m}']-r[f'true_{m}'])/scales[m] for m in METRICS]))
def bootstrap(rows,a,b,scales,random_expected=False,seed=SEED):
 g=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
 for r in rows:
  ea=row_error(r,a,scales);eb=np.mean([row_error(r,x,scales) for x in RANDOM]) if random_expected else row_error(r,b,scales);g[r['family']][r['reverse_group']][r['lineage_id']].append(eb-ea)
 fams=sorted(g);groups=sorted({r['reverse_group'] for r in rows});lines={f:sorted({r['lineage_id'] for r in rows if r['family']==f}) for f in fams}
 def est(fs,gs,ls):return float(np.mean([np.mean(g[f][q][line]) for oi,f in enumerate(fs) for q in gs for line in ls[oi]]))
 point=est(fams,groups,[lines[f] for f in fams]);rng=np.random.default_rng(seed);draws=[]
 for _ in range(DRAWS):
  fs=[fams[i] for i in rng.integers(0,len(fams),len(fams))];gs=[groups[i] for i in rng.integers(0,len(groups),len(groups))];ls=[[lines[f][i] for i in rng.integers(0,len(lines[f]),len(lines[f]))] for f in fs];draws.append(est(fs,gs,ls))
 return {'point_normalized_mae':point,'ci95':[float(np.quantile(draws,.025)),float(np.quantile(draws,.975))],'draws':DRAWS,'seed':seed,'hierarchy':'architecture family -> lineage; crossed with reversal group'}
def analyze():
 rows=load_rows();frozen=json.loads((BASE/'FROZEN_FULL_VECTOR_METHODS.json').read_text());scales=frozen['target_scales'];names=('g0','full15','magnitude5',*RANDOM);s={n:summary(rows,n,scales) for n in names};inc=bootstrap(rows,'full15','g0',scales);better=sum(s['full15']['per_metric'][m]['mae']<s['g0']['per_metric'][m]['mae'] for m in METRICS);gates={'g1_macro_normalized_mae_better':bool(s['full15']['macro_normalized_mae']<s['g0']['macro_normalized_mae']),'hierarchical_ci_excludes_zero':bool(inc['ci95'][0]>0),'at_least_four_metrics_better':bool(better>=4),'g1_macro_controlled_rank_better':bool(s['full15']['macro_controlled_rank']>s['g0']['macro_controlled_rank']),'accuracy_regret_not_worse':bool(s['full15']['accuracy_top1_regret_pp']<=s['g0']['accuracy_top1_regret_pp'])};gates['full_vector_gate_passed']=all(gates.values())
 random_mae=np.array([s[n]['macro_normalized_mae'] for n in RANDOM]);random_rank=np.array([s[n]['macro_controlled_rank'] for n in RANDOM]);random_reg=np.array([s[n]['accuracy_top1_regret_pp'] for n in RANDOM]);mag=s['magnitude5'];mb=bootstrap(rows,'magnitude5','',scales,True,SEED+1);mg={'mae_below_random_mean':bool(mag['macro_normalized_mae']<random_mae.mean()),'hierarchical_ci_excludes_zero':bool(mb['ci95'][0]>0),'macro_controlled_rank_at_least_random_median':bool(mag['macro_controlled_rank']>=np.median(random_rank)),'accuracy_regret_no_worse_random_median':bool(mag['accuracy_top1_regret_pp']<=np.median(random_reg))};mg['magnitude_gate_passed']=all(mg.values())
 result={'protocol':PROTOCOL_ID,'status':'FULL_VECTOR_GATE_PASSED' if gates['full_vector_gate_passed'] else 'FULL_VECTOR_GATE_FAILED','rows':len(rows),'models':36,'lineages':18,'families':6,'words':60,'metrics':list(METRICS),'balanced_accuracy_redundancy':'CIFAR-100 partition is class-balanced; balanced and overall accuracy are numerically equal. Both are retained by user requirement and historical schema.','methods':{n:s[n] for n in ('g0','full15','magnitude5')},'primary_increment':inc,'primary_gates':gates,'metrics_improved':better,'magnitude_confirmation':{'gates':mg,'bootstrap':mb,'random_macro_normalized_mae_mean':float(random_mae.mean()),'random_macro_normalized_mae_median':float(np.median(random_mae)),'random_range':[float(random_mae.min()),float(random_mae.max())]},'random_methods':{n:s[n] for n in RANDOM}}
 (RESULTS/'PHASE2F_ANALYSIS.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n');print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False));return result
if __name__=='__main__':analyze()
