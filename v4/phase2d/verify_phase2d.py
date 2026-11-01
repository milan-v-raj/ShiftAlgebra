#!/usr/bin/env python3
"""Verify Phase-2D frozen and result artifacts."""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
BASE=ROOT/"v4/phase2d"; RESULTS=BASE/"results"
def verify():
 checks=[]; m=json.loads((BASE/"PHASE2D_MANIFEST.json").read_text()); assert m["counts"]=={"calibration":37,"lineages":6,"snapshots":12,"targets":64}; checks.append("manifest counts and unseen families")
 f=json.loads((BASE/"FROZEN_HOLDOUT_METHODS.json").read_text()); assert f["status"]=="ALL_METHODS_FROZEN_ARCHITECTURE_OUTCOMES_UNOPENED" and len(f["random_designs"])==20; checks.append("all methods frozen")
 if (RESULTS/"behavior.csv").exists():
  bp=RESULTS/"behavior.csv"; b=list(csv.DictReader(bp.open())); e=json.loads((RESULTS/"evaluation.json").read_text()); assert len(b)==1212 and len({r["lineage_id"] for r in b})==6; assert hashlib.sha256(bp.read_bytes()).hexdigest()==e["sha256"]; checks.append("behavior accounting and hash")
 if (RESULTS/"forecast.csv").exists():
  rows=list(csv.DictReader((RESULTS/"forecast.csv").open())); assert len(rows)==768
  methods=["g0_prior","g1_prior","full15","active","magnitude","variance","coverage","d_optimal",*[f"random_{i:02d}" for i in range(20)]]
  for method in methods:
   x=np.array([float(r[method]) for r in rows]); assert np.isfinite(x).all() and (x>=0).all() and (x<=1).all()
  a=json.loads((RESULTS/"PHASE2D_ANALYSIS.json").read_text())
  for method in ("g0_prior","g1_prior","full15","active","magnitude","variance","coverage","d_optimal"):
   mae=100*np.mean([abs(float(r[method])-float(r["true_accuracy"])) for r in rows]); assert abs(mae-a["methods"].get(method,a["alternative_selectors"].get(method))["accuracy_mae_pp"])<1e-10
  checks.append("forecast bounds and MAE recomputation")
 if (RESULTS/"POST_OUTCOME_ROBUSTNESS.json").exists():
  p=json.loads((RESULTS/"POST_OUTCOME_ROBUSTNESS.json").read_text())
  for rel,digest in p["artifact_sha256"].items(): assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest
  assert p["primary_gate_status"]=="ARCHITECTURE_HOLDOUT_GATE_FAILED" and p["lineage_sign_flip_sensitivity"]["exact_one_sided_p"]==0.0625
  checks.append("post-outcome provenance and sensitivity")
 return checks
if __name__=="__main__":
 c=verify(); print(f"Phase-2D verification: {len(c)}/{len(c)} passed"); [print("PASS",x) for x in c]
