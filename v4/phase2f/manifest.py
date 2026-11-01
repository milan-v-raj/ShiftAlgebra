#!/usr/bin/env python3
"""Freeze CIFAR-100 full-vector development and confirmation banks."""
from __future__ import annotations
import hashlib,itertools,json
from dataclasses import asdict,dataclass
from pathlib import Path
from v4.phase2f import METRICS,PRIMITIVES,PROTOCOL_ID
from v4.phase2f.models import FAMILIES,make_model
ROOT=Path(__file__).resolve().parents[2];OUTPUT=ROOT/'v4/phase2f/PHASE2F_MANIFEST.json';RECIPES=('erm_flip','label_smooth','noise_aug');EPOCHS=(1,2);WORD_SEED=22701
@dataclass(frozen=True)
class ModelRecord: model_id:str;lineage_id:str;bank_split:str;family:str;recipe:str;seed:int;epoch:int;parameter_count:int
def model_records():
 out=[]
 for split,base in [('development',64100),('confirmation',65100)]:
  for fi,f in enumerate(FAMILIES):
   n=sum(p.numel() for p in make_model(f).parameters())
   for ri,r in enumerate(RECIPES):
    seed=base+100*fi+ri;line=f'p2f-{split}-{f}-{r}-s{seed}'
    for e in EPOCHS:out.append(ModelRecord(f'{line}-e{e}',line,split,f,r,seed,e,n))
 return out
def reverse_group(w):
 a,b='>'.join(w),'>'.join(reversed(w));return min(a,b)
def excluded_words():
 from v4.benchmark.generate_mason_manifest import exact_words
 from v4.phase2b.manifest import fresh_words as b
 from v4.phase2c.manifest import fresh_words as c
 from v4.phase2d.manifest import fresh_words as d
 from v4.phase2e.manifest import fresh_words as e
 return set(exact_words())|set(b())|set(c())|set(d())|set(e())
def target_words():
 excluded=excluded_words();out={'development':[],'confirmation':[]}
 for length in (4,5,6):
  candidates={}
  for w in itertools.permutations(PRIMITIVES,length):
   rev=tuple(reversed(w))
   if w not in excluded and rev not in excluded:candidates[reverse_group(w)]=(w,rev)
  score=lambda g:hashlib.sha256(f'{WORD_SEED}|{g}'.encode()).hexdigest();chosen=sorted(candidates,key=score)[:20]
  for split,groups in [('development',chosen[:10]),('confirmation',chosen[10:])]:
   for g in groups:out[split].extend(sorted(candidates[g]))
 if any(len(out[s])!=60 for s in out) or set(out['development'])&set(out['confirmation']):raise RuntimeError('invalid words')
 return out
def calibration_words():return [('Identity',)]+[(p,) for p in PRIMITIVES]+[(a,b) for a in PRIMITIVES for b in PRIMITIVES if a!=b]
def words_for(split):return calibration_words()+target_words()[split]
def word_records(split):
 cal=set(calibration_words());return [{'condition_index':i,'word':list(w),'word_string':'>'.join(w),'length':0 if w==('Identity',) else len(w),'role':'calibration' if w in cal else split,'reverse_group':reverse_group(w)} for i,w in enumerate(words_for(split))]
def payload():
 prov=json.loads((ROOT/'v4/phase2f/DATA_PROVENANCE.json').read_text());return {'protocol':PROTOCOL_ID,'status':'BANKS_AND_FULL_VECTOR_PROTOCOL_FROZEN_CONFIRMATION_OUTCOMES_UNOPENED','dataset':'CIFAR-100 fine labels','classes':100,'chance':.01,'metrics':list(METRICS),'models':[asdict(x) for x in model_records()],'words':{s:word_records(s) for s in ('development','confirmation')},'image_partitions':{'development':{'per_class':20,'offset':0},'confirmation':{'per_class':20,'offset':20}},'counts':{'families':6,'recipes':3,'lineages_per_split':18,'snapshots_per_split':36,'calibration_conditions':37,'target_words_per_split':60},'word_seed':WORD_SEED,'data_provenance':prov}
def main():OUTPUT.write_text(json.dumps(payload(),indent=2,sort_keys=True)+'\n');print(OUTPUT)
if __name__=='__main__':main()
