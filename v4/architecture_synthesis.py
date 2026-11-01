#!/usr/bin/env python3
"""Retrospective family-level synthesis of prospective Phases 2D and 2E."""
import csv,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'v4/results'; DRAWS=10000; SEED=22604
def load():
 rows=[]
 for bank,path in [('phase2d',ROOT/'v4/phase2d/results/forecast.csv'),('phase2e',ROOT/'v4/phase2e/results/forecast.csv')]:
  for r in csv.DictReader(path.open()):
   truth=float(r['true_accuracy']);rows.append({'bank':bank,'family':f"{bank}:{r['family']}",'lineage':f"{bank}:{r['lineage_id']}",'group':r['reverse_group'],'delta':100*(abs(float(r['g0_prior'])-truth)-abs(float(r['g1_prior'])-truth))})
 return rows
def run():
 rows=load();g=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
 for r in rows:g[r['family']][r['group']][r['lineage']].append(r['delta'])
 fams=sorted(g)
 def fam_est(f,groups=None,lines=None):
  groups=sorted(g[f]) if groups is None else groups;lines=sorted({x for q in g[f].values() for x in q}) if lines is None else lines;return float(np.mean([np.mean(g[f][q][l]) for q in groups for l in lines if l in g[f][q]]))
 by={f:fam_est(f) for f in fams};point=float(np.mean(list(by.values())));rng=np.random.default_rng(SEED);draws=[]
 for _ in range(DRAWS):
  fs=[fams[i] for i in rng.integers(0,len(fams),len(fams))];values=[]
  for f in fs:
   groups=sorted(g[f]);lines=sorted({x for q in g[f].values() for x in q});sg=[groups[i] for i in rng.integers(0,len(groups),len(groups))];sl=[lines[i] for i in rng.integers(0,len(lines),len(lines))];values.append(fam_est(f,sg,sl))
  draws.append(np.mean(values))
 wins=sum(x>0 for x in by.values());signp=sum(math.comb(len(fams),k) for k in range(wins,len(fams)+1))/2**len(fams)
 p={'status':'RETROSPECTIVE_ARCHITECTURE_SYNTHESIS_COMPLETE_NO_GATE_CHANGE','families':len(fams),'lineages':len({r['lineage'] for r in rows}),'snapshots':36,'rows':len(rows),'family_improvement_pp':by,'families_favoring_g1':wins,'macro_family_improvement_pp':point,'bootstrap':{'ci95_pp':[float(np.quantile(draws,.025)),float(np.quantile(draws,.975))],'draws':DRAWS,'seed':SEED,'hierarchy':'architecture family -> reversal group and lineage'},'exact_one_sided_sign_p_assuming_independent_families':signp,'phase_gates':{'phase2d':'FAIL','phase2e':'FAIL'},'interpretation':'Retrospective sensitivity only. Families share CIFAR-10, transformations, recipes, and tiny-model design constraints.'};OUT.mkdir(exist_ok=True);(OUT/'ARCHITECTURE_SYNTHESIS.json').write_text(json.dumps(p,indent=2,sort_keys=True)+'\n')
 lines=['# Retrospective Architecture Synthesis','','Phases 2D and 2E are combined descriptively without changing either failed gate.','','| Family | g0 error minus g1 error |','|---|---:|']+[f"| {f} | {v:+.3f}pp |" for f,v in by.items()]+['',f"Macro-family improvement: **{point:.3f}pp**; hierarchical sensitivity interval **[{p['bootstrap']['ci95_pp'][0]:.3f}, {p['bootstrap']['ci95_pp'][1]:.3f}]pp**.",f"Families favoring `g1`: **{wins}/{len(fams)}**; independence-assuming one-sided sign p={signp:.4f}.",'','This is not a new confirmation: all families use CIFAR-10, the same transformations and recipes, and lightweight custom models. Phase 2D and Phase 2E both remain failed.'];(OUT/'ARCHITECTURE_SYNTHESIS.md').write_text('\n'.join(lines)+'\n');print(json.dumps(p,indent=2,sort_keys=True));return p
if __name__=='__main__':run()
