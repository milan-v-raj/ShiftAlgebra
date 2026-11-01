#!/usr/bin/env python3
"""Independently verify the compact cross-phase synthesis artifact."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from v4.cpu_synthesis import load_banks

def verify():
 p=json.loads((ROOT/"v4/results/CPU_CROSS_PHASE_SYNTHESIS.json").read_text()); banks=load_banks(); checks=[]
 assert set(banks)==set(p["banks"])
 for bank,rows in banks.items():
  for method in ("g0","g1"):
   mae=100*float(np.mean([abs(r[method]-r["true_accuracy"]) for r in rows])); assert abs(mae-p["banks"][bank][method]["accuracy_mae_pp"])<1e-10
 checks.append("all bank/method MAEs")
 for method in ("g0","g1"):
  expected=np.mean([p["banks"][b][method]["accuracy_mae_pp"] for b in banks]); assert abs(expected-p["macro_bank_mae_pp"][method])<1e-12
 checks.append("macro-bank arithmetic")
 assert p["totals"]=={"rows":7308,"models":102,"lineages":51,"bank_word_instances":282}; assert p["incremental_meta_bootstrap"]["banks_favoring_g1"]==4 and p["incremental_meta_bootstrap"]["exact_one_sided_sign_p_assuming_independent_banks"]==0.0625; checks.append("cross-bank accounting and directional sensitivity")
 q=json.loads((ROOT/"v4/phase2d/results/PHASE2D_ANALYSIS.json").read_text()); assert q["status"]=="ARCHITECTURE_HOLDOUT_GATE_FAILED" and not q["active_replication"]["gates"]["active_replication_passed"]; checks.append("failed Phase-2D gates retained")
 return checks
if __name__=="__main__":
 c=verify(); print(f"CPU synthesis verification: {len(c)}/{len(c)} passed"); [print("PASS",x) for x in c]
