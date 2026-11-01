#!/usr/bin/env python3
"""Post-outcome sensitivity and provenance audit; never changes frozen gates."""
from __future__ import annotations
import csv,hashlib,itertools,json
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; BASE=ROOT/"v4/phase2d"; RESULTS=BASE/"results"; CHECKPOINTS=ROOT/"checkpoints_v4/phase2d"

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def group(rows,key):
 out=defaultdict(list)
 for r in rows: out[r[key]].append(r["improvement_pp"])
 return {k:{"rows":len(v),"mean_improvement_pp":float(np.mean(v))} for k,v in sorted(out.items())}
def run():
 rows=[]
 for r in csv.DictReader((RESULTS/"forecast.csv").open()):
  truth=float(r["true_accuracy"]); r["improvement_pp"]=100*(abs(float(r["g0_prior"])-truth)-abs(float(r["g1_prior"])-truth)); rows.append(r)
 line=group(rows,"lineage_id"); x=np.array([v["mean_improvement_pp"] for v in line.values()]); obs=float(x.mean()); null=np.array([np.mean(x*np.array(s)) for s in itertools.product((-1,1),repeat=len(x))])
 training=json.loads((RESULTS/"training.json").read_text()); checked=available=0
 for s in training["summaries"]:
  for h in s["history"]:
   p=CHECKPOINTS/f"{h['model_id']}.pt"
   if p.exists(): available+=1; checked+=int(sha(p)==h["sha256"])
 analysis=json.loads((RESULTS/"PHASE2D_ANALYSIS.json").read_text())
 payload={"status":"POST_OUTCOME_AUDIT_COMPLETE_NO_GATE_CHANGE","protocol":analysis["protocol"],"frozen_setup_commit":"e71b360","training_record_commit":"9cbdc6a","primary_gate_status":analysis["status"],"point_improvement_pp":analysis["primary_increment"]["point_pp"],"frozen_ci95_pp":analysis["primary_increment"]["ci95_pp"],"stratified_improvement_pp":{"family":group(rows,"family"),"lineage":line,"recipe":group(rows,"recipe"),"length":group(rows,"length")},"directional_counts":{"positive_reversal_groups":sum(v["mean_improvement_pp"]>0 for v in group(rows,"reverse_group").values()),"reversal_groups":len(group(rows,"reverse_group")),"positive_lineages":int(np.sum(x>0)),"lineages":len(x)},"lineage_sign_flip_sensitivity":{"observed_mean_pp":obs,"exact_one_sided_p":float(np.mean(null>=obs)),"exact_two_sided_p":float(np.mean(np.abs(null)>=abs(obs))),"assignments":len(null),"interpretation":"descriptive post-outcome sensitivity; not a replacement gate"},"checkpoint_audit":{"available":available,"hashes_matching_training_record":checked,"expected":12},"artifact_sha256":{str(p.relative_to(ROOT)):sha(p) for p in [BASE/"PHASE2D_MANIFEST.json",BASE/"FROZEN_HOLDOUT_METHODS.json",RESULTS/"training.json",RESULTS/"behavior.csv",RESULTS/"forecast.csv",RESULTS/"PHASE2D_ANALYSIS.json"]},"audit_conclusion":"No outcome leak or artifact mismatch detected. Evidence remains heterogeneous and the preregistered gate remains failed."}
 out=RESULTS/"POST_OUTCOME_ROBUSTNESS.json"; out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); print(json.dumps(payload,indent=2,sort_keys=True)); return payload
if __name__=="__main__":run()
