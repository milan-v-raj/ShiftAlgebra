#!/usr/bin/env python3
"""Apply every frozen method to architecture-holdout outcomes."""
from __future__ import annotations
import csv,json
from pathlib import Path
from v4.phase2.forecast import apply_ridge
from v4.phase2b.method import EXPERTS,apply_gate,forecast_rows,load_behavior
from v4.phase2d import PROTOCOL_ID
ROOT=Path(__file__).resolve().parents[2]; RESULTS=ROOT/"v4/phase2d/results"

def run():
 frozen=json.loads((ROOT/"v4/phase2d/FROZEN_HOLDOUT_METHODS.json").read_text()); prior=json.loads((ROOT/"v4/phase2/FROZEN_METHOD.json").read_text())
 if frozen["status"]!="ALL_METHODS_FROZEN_ARCHITECTURE_OUTCOMES_UNOPENED": raise ValueError("not frozen")
 behavior=load_behavior(RESULTS/"behavior.csv",{"calibration","architecture_holdout"}); methods={}
 for name,d in frozen["selectors"].items(): methods[name]=(d,forecast_rows(behavior,d["pair_groups"],"architecture_holdout"))
 for name,d in frozen["random_designs"].items(): methods[name]=(d,forecast_rows(behavior,d["pair_groups"],"architecture_holdout"))
 full=frozen["full15"]; fullrows=forecast_rows(behavior,full["pair_groups"],"architecture_holdout"); lookup_full={(r["model_id"],r["word"]):r for r in fullrows}; lookups={n:{(r["model_id"],r["word"]):r for r in rs} for n,(d,rs) in methods.items()}; out=[]
 for key in sorted(lookup_full):
  r=lookup_full[key]; fullres=apply_gate(r,full["gate"]); expert=dict(zip(EXPERTS,r["expert_accuracies"])); priorrow={**r,"hazard_ordered_q":expert["ordered"],"hazard_symmetric_q":expert["symmetric"],"product_retention":expert["independent"],"max_damage":expert["dominant"]}
  row={k:r[k] for k in ("model_id","lineage_id","family","recipe","seed","epoch","parameter_count","word","reverse_group","length","true_accuracy","clean_accuracy")}; row["protocol"]=PROTOCOL_ID; row["g0_prior"]=apply_ridge(priorrow,prior["ridge_g0"]); row["g1_prior"]=apply_ridge(priorrow,prior["ridge_g1"]); row["full15"]=fullres["prediction"]
  for name,(d,_) in methods.items(): row[name]=apply_gate(lookups[name][key],d["gate"])["prediction"]
  out.append(row)
 path=RESULTS/"forecast.csv"
 with path.open("w",newline="") as h:
  w=csv.DictWriter(h,fieldnames=list(out[0]),lineterminator="\n"); w.writeheader(); w.writerows(out)
 print(f"wrote {len(out)} architecture-holdout rows"); return out
if __name__=="__main__": run()
