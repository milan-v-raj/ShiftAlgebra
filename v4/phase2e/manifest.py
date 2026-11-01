#!/usr/bin/env python3
"""Freeze the independent Phase-2E architecture-breadth bank."""
from __future__ import annotations
import hashlib,itertools,json
from dataclasses import asdict,dataclass
from pathlib import Path
from v4.phase2e import PRIMITIVES,PROTOCOL_ID
from v4.phase2e.models import FAMILIES,make_model
ROOT=Path(__file__).resolve().parents[2]; OUTPUT=ROOT/'v4/phase2e/PHASE2E_MANIFEST.json'; RECIPES=('erm_flip','label_smooth','noise_aug'); EPOCHS=(1,2); SEED_BASE=63100; WORD_SEED=22602
@dataclass(frozen=True)
class ModelRecord: model_id:str; lineage_id:str; family:str; recipe:str; seed:int; epoch:int; parameter_count:int
def model_records():
 out=[]
 for fi,f in enumerate(FAMILIES):
  n=sum(p.numel() for p in make_model(f).parameters())
  for ri,r in enumerate(RECIPES):
   seed=SEED_BASE+100*fi+ri; line=f'p2e-{f}-{r}-s{seed}'
   for e in EPOCHS: out.append(ModelRecord(f'{line}-e{e}',line,f,r,seed,e,n))
 return out
def reverse_group(w):
 a,b='>'.join(w),'>'.join(reversed(w)); return min(a,b)
def excluded_words():
 from v4.benchmark.generate_mason_manifest import exact_words
 from v4.phase2b.manifest import fresh_words as b
 from v4.phase2c.manifest import fresh_words as c
 from v4.phase2d.manifest import fresh_words as d
 return set(exact_words())|set(b())|set(c())|set(d())
def fresh_words():
 excluded=excluded_words(); selected=[]
 for length in (4,5,6):
  candidates={}
  for word in itertools.permutations(PRIMITIVES,length):
   rev=tuple(reversed(word))
   if word not in excluded and rev not in excluded: candidates[reverse_group(word)]=(word,rev)
  score=lambda g:hashlib.sha256(f'{WORD_SEED}|{g}'.encode()).hexdigest()
  for group in sorted(candidates,key=score)[:10]: selected.extend(sorted(candidates[group]))
 if len(selected)!=60 or not set(selected).isdisjoint(excluded): raise RuntimeError('invalid Phase-2E fresh words')
 return selected
def calibration_words(): return [('Identity',)]+[(p,) for p in PRIMITIVES]+[(a,b) for a in PRIMITIVES for b in PRIMITIVES if a!=b]
def evaluation_words(): return calibration_words()+fresh_words()
def word_records():
 cal=set(calibration_words()); return [{'condition_index':i,'word':list(w),'word_string':'>'.join(w),'length':0 if w==('Identity',) else len(w),'role':'calibration' if w in cal else 'architecture_breadth','reverse_group':reverse_group(w)} for i,w in enumerate(evaluation_words())]
def payload():
 return {'protocol':PROTOCOL_ID,'status':'ARCHITECTURE_BREADTH_FROZEN_OUTCOMES_UNOPENED','dataset':'CIFAR-10','image_partition':{'per_class':100,'offset':500},'unseen_families':list(FAMILIES),'recipes':list(RECIPES),'models':[asdict(x) for x in model_records()],'words':word_records(),'counts':{'families':4,'lineages':12,'snapshots':24,'calibration':37,'targets':60},'target_lengths':[4,5,6],'word_overlap_with_prior_phases':0,'word_seed':WORD_SEED}
def main(): OUTPUT.write_text(json.dumps(payload(),indent=2,sort_keys=True)+'\n'); print(OUTPUT)
if __name__=='__main__':main()
