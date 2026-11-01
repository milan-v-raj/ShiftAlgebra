#!/usr/bin/env python3
"""Cross-phase synthesis of all independently frozen CPU confirmation banks."""
from __future__ import annotations
import csv,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
from v4.phase2.forecast import apply_ridge
from v4.phase2b.analyze import method_metrics
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"v4/results"; DRAWS=10000; SEED=22601

def typed(raw):
 r=dict(raw)
 for n in ("seed","epoch","parameter_count","length"): r[n]=int(r[n])
 for n in list(r):
  if n in {"true_accuracy","clean_accuracy","g0","g1"}: r[n]=float(r[n])
 return r

def load_banks():
 prior=json.loads((ROOT/"v4/phase2/FROZEN_METHOD.json").read_text()); banks={}
 rows=[]
 for x in csv.DictReader((ROOT/"v4/phase2/results/confirmation-forecast.csv").open()):
  if x["word_role"]!="confirmation": continue
  r=typed(x); r["g0"]=float(x["g0_source_word"]); r["g1"]=float(x["g1_plus_hazard"]); rows.append(r)
 banks["phase2"]=rows
 for bank,path,g0col,g1col in [
  ("phase2b",ROOT/"v4/phase2b/results/fresh-forecast.csv","g0_prior","g1_prior"),
  ("phase2c",ROOT/"v4/phase2c/results/fresh-forecast.csv",None,"g1_prior"),
  ("phase2d",ROOT/"v4/phase2d/results/forecast.csv","g0_prior","g1_prior")]:
  rows=[]
  for x in csv.DictReader(path.open()):
   r=typed(x); r["g1"]=float(x[g1col])
   if g0col: r["g0"]=float(x[g0col])
   else:
    dummy={"clean_accuracy":float(x["clean_accuracy"]),"parameter_count":int(x["parameter_count"]),"epoch":int(x["epoch"]),"length":int(x["length"]),"word":x["word"],"family":x["family"],"recipe":x["recipe"]}
    r["g0"]=apply_ridge(dummy,prior["ridge_g0"])
   r.setdefault("word_role","confirmation"); rows.append(r)
  banks[bank]=rows
 for bank,rows in banks.items():
  for r in rows:
   r["model_id"]=f"{bank}:{r['model_id']}"; r["lineage_id"]=f"{bank}:{r['lineage_id']}"
 return banks

def meta_bootstrap(banks):
 grouped={}
 for bank,rows in banks.items():
  g=defaultdict(lambda:defaultdict(list))
  for r in rows: g[r["reverse_group"]][r["lineage_id"]].append(abs(r["g0"]-r["true_accuracy"])-abs(r["g1"]-r["true_accuracy"]))
  grouped[bank]=g
 def bank_est(bank,gs=None,ls=None):
  g=grouped[bank]; groups=sorted(g) if gs is None else gs; lines=sorted({r["lineage_id"] for r in banks[bank]}) if ls is None else ls
  return 100*float(np.mean([np.mean(g[x][l]) for x in groups for l in lines if l in g[x]]))
 point=float(np.mean([bank_est(b) for b in banks])); rng=np.random.default_rng(SEED); names=list(banks); draws=[]
 for _ in range(DRAWS):
  values=[]
  for bank in [names[i] for i in rng.integers(0,len(names),len(names))]:
   g=grouped[bank]; gs=sorted(g); ls=sorted({r["lineage_id"] for r in banks[bank]}); sgs=[gs[i] for i in rng.integers(0,len(gs),len(gs))]; sls=[ls[i] for i in rng.integers(0,len(ls),len(ls))]; values.append(bank_est(bank,sgs,sls))
  draws.append(np.mean(values))
 return {"macro_bank_improvement_pp":point,"ci95_pp":[float(np.quantile(draws,.025)),float(np.quantile(draws,.975))],"draws":DRAWS,"seed":SEED,"hierarchy":"bank -> reversal group and lineage","banks_favoring_g1":int(wins:=sum(bank_est(b)>0 for b in banks)),"bank_count":len(banks),"exact_one_sided_sign_p_assuming_independent_banks":float(sum(math.comb(len(banks),k) for k in range(wins,len(banks)+1))/2**len(banks)),"interpretation":"Retrospective descriptive stability interval. Four banks share CIFAR-10, transformations, and some architecture classes; do not interpret as independent-dataset confirmation."}

def run():
 banks=load_banks(); per={}
 for bank,rows in banks.items(): per[bank]={"rows":len(rows),"models":len({r['model_id'] for r in rows}),"lineages":len({r['lineage_id'] for r in rows}),"words":len({r['word'] for r in rows}),"g0":method_metrics(rows,"g0"),"g1":method_metrics(rows,"g1")}
 macro={m:float(np.mean([per[b][m]["accuracy_mae_pp"] for b in per])) for m in ("g0","g1")}; meta=meta_bootstrap(banks)
 p2c=json.loads((ROOT/"v4/phase2c/results/PHASE2C_ANALYSIS.json").read_text()); p2d=json.loads((ROOT/"v4/phase2d/results/PHASE2D_ANALYSIS.json").read_text())
 result={"status":"CPU_CROSS_PHASE_SYNTHESIS_COMPLETE","banks":per,"totals":{"rows":sum(x["rows"] for x in per.values()),"models":sum(x["models"] for x in per.values()),"lineages":sum(x["lineages"] for x in per.values()),"bank_word_instances":sum(x["words"] for x in per.values())},"macro_bank_mae_pp":macro,"incremental_meta_bootstrap":meta,"gate_summary":{"phase2":"PASS","phase2b_achm":"FAIL","phase2c_active_pairs":"PASS","phase2d_architecture_holdout":"FAIL"},"active_pair_transfer":{"phase2c":p2c["gates"],"phase2d":p2d["active_replication"]["gates"]}}
 OUT.mkdir(parents=True,exist_ok=True); (OUT/"CPU_CROSS_PHASE_SYNTHESIS.json").write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
 lines=["# CPU Cross-Phase Synthesis","","Four independently frozen confirmation banks are summarized without refitting any method. This retrospective aggregation is not a new confirmation and does not revise any stage gate.","","| Bank | Lineages | Models | Words | g0 MAE | g1 MAE | Controlled rank g0 → g1 |","|---|---:|---:|---:|---:|---:|---:|"]
 for b,x in per.items(): lines.append(f"| {b} | {x['lineages']} | {x['models']} | {x['words']} | {x['g0']['accuracy_mae_pp']:.3f}pp | {x['g1']['accuracy_mae_pp']:.3f}pp | {x['g0']['controlled_model_rank_spearman']:.3f} → {x['g1']['controlled_model_rank_spearman']:.3f} |")
 lines += ["",f"Macro-bank MAE: **{macro['g0']:.3f}pp (`g0`) → {macro['g1']:.3f}pp (`g1`)**.",f"Hierarchical macro-bank improvement: **{meta['macro_bank_improvement_pp']:.3f}pp**, bootstrap sensitivity interval **[{meta['ci95_pp'][0]:.3f}, {meta['ci95_pp'][1]:.3f}]pp**.",f"All {meta['banks_favoring_g1']}/{meta['bank_count']} banks favor `g1`; an independence-assuming one-sided sign test gives p={meta['exact_one_sided_sign_p_assuming_independent_banks']:.4f}.","","**Inference warning:** the four banks share CIFAR-10, transformation semantics, and some architecture classes. The interval is descriptive retrospective stability analysis, not independent-dataset confirmation.","",f"Total: {result['totals']['lineages']} lineages, {result['totals']['models']} snapshots, {result['totals']['bank_word_instances']} bank-word environments, and {result['totals']['rows']} prediction rows.","","Gate record: Phase 2 PASS; Phase 2B FAIL; Phase 2C PASS; Phase 2D FAIL. These outcomes are not pooled into a single all-positive claim."]
 (OUT/"CPU_CROSS_PHASE_SYNTHESIS.md").write_text("\n".join(lines)+"\n"); print(json.dumps(result,indent=2,sort_keys=True)); return result
if __name__=="__main__": run()
