#!/usr/bin/env python3
"""Develop and freeze all full-vector methods before confirmation outcomes."""
import hashlib,json,random
from pathlib import Path
import numpy as np
from v4.phase2.train_bank import sha256
from v4.phase2f import METRICS,PROTOCOL_ID
from v4.phase2f.method import all_groups,apply,fit,load_behavior,magnitude_design,rows,select_q
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'v4/phase2f';RESULTS=BASE/'results';OUTPUT=BASE/'FROZEN_FULL_VECTOR_METHODS.json';RANDOM_SEED=22702
def score(rs,models,scales):
 raw={m:float(np.mean([abs(apply(r,models[m])-r['target'][m]) for r in rs])) for m in METRICS};return {'raw_mae':raw,'macro_normalized_mae':float(np.mean([raw[m]/scales[m] for m in METRICS]))}
def freeze():
 source=RESULTS/'development-behavior.csv';behavior=load_behavior(source,{'calibration','development'});targets=[r for model in behavior.values() for r in model.values() if r['word_role']=='development'];scales={m:max(float(np.std([r[m] for r in targets])),1e-6) for m in METRICS};q,search=select_q(behavior);mag=magnitude_design(behavior,scales);rng=random.Random(RANDOM_SEED);groups=list(all_groups());randoms={};seen={tuple(sorted(mag))}
 while len(randoms)<20:
  x=tuple(sorted(rng.sample(groups,5)))
  if x in seen:continue
  seen.add(x);randoms[f'random_{len(randoms):02d}']=list(x)
 designs={'full15':list(all_groups()),'magnitude5':mag,**randoms};fullrows=rows(behavior,'development',designs['full15'],q);g0={m:fit(fullrows,m,False) for m in METRICS};frozen={}
 for name,selected in designs.items():
  rs=rows(behavior,'development',selected,q);models={m:fit(rs,m,True) for m in METRICS};frozen[name]={'pair_groups':selected,'directed_measurements':2*len(selected),'models':models,'development':score(rs,models,scales)}
 payload={'protocol':PROTOCOL_ID,'status':'FULL_VECTOR_METHODS_FROZEN_CONFIRMATION_OUTCOMES_UNOPENED','development_behavior':'v4/phase2f/results/development-behavior.csv','development_behavior_sha256':sha256(source),'metrics':list(METRICS),'target_scales':scales,'q_by_metric':q,'q_search':search,'g0_models':g0,'g0_development':score(fullrows,g0,scales),'designs':frozen,'random_seed':RANDOM_SEED,'information_boundary':'development calibration and target roles only; confirmation forbidden'};OUTPUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');print(OUTPUT);print('q',q);print('magnitude',mag);return payload
if __name__=='__main__':freeze()
