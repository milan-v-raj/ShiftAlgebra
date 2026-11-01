#!/usr/bin/env python3
"""Freeze exact inherited methods and Phase-2E gates before outcomes."""
import hashlib,json
from pathlib import Path
from v4.phase2e import PROTOCOL_ID
ROOT=Path(__file__).resolve().parents[2]; BASE=ROOT/'v4/phase2e'; SOURCE=ROOT/'v4/phase2d/FROZEN_HOLDOUT_METHODS.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def freeze():
 p={'protocol':PROTOCOL_ID,'status':'INHERITED_METHODS_AND_GATES_FROZEN_OUTCOMES_UNOPENED','method_source':'v4/phase2d/FROZEN_HOLDOUT_METHODS.json','method_source_sha256':sha(SOURCE),'prior_regressions':'v4/phase2/FROZEN_METHOD.json','prior_regressions_sha256':sha(ROOT/'v4/phase2/FROZEN_METHOD.json'),'no_refitting':True,'primary_method':'g1_prior','primary_comparator':'g0_prior','primary_gate':{'all_required':True,'criteria':['g1_mae_better','family_lineage_reversal_ci_excludes_zero','at_least_three_of_four_family_means_positive','g1_controlled_rank_better','g1_regret_not_worse']},'secondary_selectors':'descriptive_only'}
 (BASE/'FROZEN_BREADTH_METHODS.json').write_text(json.dumps(p,indent=2,sort_keys=True)+'\n'); print(BASE/'FROZEN_BREADTH_METHODS.json'); return p
if __name__=='__main__':freeze()
