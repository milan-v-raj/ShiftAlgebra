#!/usr/bin/env python3
"""Frozen architecture-holdout and selector analysis."""
from __future__ import annotations
import csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from v4.phase2b.analyze import method_metrics
from v4.phase2d import PROTOCOL_ID
ROOT=Path(__file__).resolve().parents[2]; RESULTS=ROOT/"v4/phase2d/results"; RANDOM=tuple(f"random_{i:02d}" for i in range(20)); SELECTORS=("active","magnitude","variance","coverage","d_optimal"); DRAWS=10000

def load_rows():
 rows=[]
 for raw in csv.DictReader((RESULTS/"forecast.csv").open(newline="")):
  r=dict(raw)
  for n in ("seed","epoch","parameter_count","length"): r[n]=int(r[n])
  for n in ("true_accuracy","clean_accuracy","g0_prior","g1_prior","full15",*SELECTORS,*RANDOM): r[n]=float(r[n])
  rows.append(r)
 if len(rows)!=768: raise ValueError(len(rows))
 return rows

def bootstrap(rows,a,b,random_expected=False,seed=22503):
 grouped=defaultdict(lambda:defaultdict(list))
 for r in rows:
  ea=abs(r[a]-r["true_accuracy"])
  eb=np.mean([abs(r[x]-r["true_accuracy"]) for x in RANDOM]) if random_expected else abs(r[b]-r["true_accuracy"])
  grouped[r["reverse_group"]][r["lineage_id"]].append(eb-ea)
 groups=sorted(grouped); lines=sorted({r["lineage_id"] for r in rows})
 def est(gs,ls): return 100*float(np.mean([np.mean(grouped[g][l]) for g in gs for l in ls if l in grouped[g]]))
 point=est(groups,lines); rng=np.random.default_rng(seed); draws=[]
 for _ in range(DRAWS): draws.append(est([groups[i] for i in rng.integers(0,len(groups),len(groups))],[lines[i] for i in rng.integers(0,len(lines),len(lines))]))
 return {"point_pp":point,"ci95_pp":[float(np.quantile(draws,.025)),float(np.quantile(draws,.975))],"draws":DRAWS,"clusters":"reversal group and lineage"}

def analyze():
 rows=load_rows(); names=("g0_prior","g1_prior","full15",*SELECTORS,*RANDOM); metrics={n:method_metrics(rows,n) for n in names}; inc=bootstrap(rows,"g1_prior","g0_prior"); g0,g1=metrics["g0_prior"],metrics["g1_prior"]
 primary={"g1_mae_better":bool(g1["accuracy_mae_pp"]<g0["accuracy_mae_pp"]),"ci_excludes_zero":bool(inc["ci95_pp"][0]>0),"g1_controlled_rank_better":bool((g1["controlled_model_rank_spearman"] or -1)>(g0["controlled_model_rank_spearman"] or -1)),"g1_regret_not_worse":bool(g1["top1_selection_regret_pp"]<=g0["top1_selection_regret_pp"])}; primary["architecture_holdout_gate_passed"]=all(primary.values())
 random_mae=np.array([metrics[n]["accuracy_mae_pp"] for n in RANDOM]); random_ctrl=np.array([metrics[n]["controlled_model_rank_spearman"] for n in RANDOM]); random_reg=np.array([metrics[n]["top1_selection_regret_pp"] for n in RANDOM]); act=metrics["active"]; activeboot=bootstrap(rows,"active","",True,22504)
 activegate={"mae_below_random_mean":bool(act["accuracy_mae_pp"]<random_mae.mean()),"ci_excludes_zero":bool(activeboot["ci95_pp"][0]>0),"controlled_at_least_random_median":bool(act["controlled_model_rank_spearman"]>=np.median(random_ctrl)),"regret_no_worse_random_median":bool(act["top1_selection_regret_pp"]<=np.median(random_reg))}; activegate["active_replication_passed"]=all(activegate.values())
 byfamily={}
 for fam in sorted({r["family"] for r in rows}):
  subset=[r for r in rows if r["family"]==fam]; byfamily[fam]={n:100*float(np.mean([abs(r[n]-r["true_accuracy"]) for r in subset])) for n in ("g0_prior","g1_prior","full15",*SELECTORS)}
 result={"protocol":PROTOCOL_ID,"status":"ARCHITECTURE_HOLDOUT_GATE_PASSED" if primary["architecture_holdout_gate_passed"] else "ARCHITECTURE_HOLDOUT_GATE_FAILED","rows":len(rows),"models":12,"lineages":6,"words":64,"methods":{n:metrics[n] for n in ("g0_prior","g1_prior","full15",*SELECTORS)},"primary_increment":inc,"primary_gates":primary,"active_replication":{"bootstrap":activeboot,"gates":activegate,"random_mae_mean":float(random_mae.mean()),"random_mae_median":float(np.median(random_mae)),"random_mae_range":[float(random_mae.min()),float(random_mae.max())],"active_percentile":float(np.mean(random_mae>=act["accuracy_mae_pp"]))},"alternative_selectors":{n:metrics[n] for n in SELECTORS},"by_family_mae_pp":byfamily}
 (RESULTS/"PHASE2D_ANALYSIS.json").write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n"); print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)); return result
if __name__=="__main__": analyze()
