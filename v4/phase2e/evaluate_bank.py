#!/usr/bin/env python3
"""Evaluate the sealed architecture-breadth bank."""
import csv,json,time
from pathlib import Path
import torch
from v4.phase1.classifiers import normalization
from v4.phase1.data import balanced_subset,load_cifar10
from v4.phase1.pipeline import DEFAULT_CIFAR
from v4.phase2.evaluate_bank import clean_statistics,evaluate_condition
from v4.phase2.train_bank import sha256
from v4.phase2e import PROTOCOL_ID
from v4.phase2e.manifest import evaluation_words,model_records,word_records
from v4.phase2e.models import make_model
from v4.phase2e.train_bank import CHECKPOINTS,RESULTS
ROOT=Path(__file__).resolve().parents[2]; STRENGTHS=ROOT/'v4/phase1b/FROZEN_STRENGTHS.json'
def run():
 torch.set_num_threads(min(2,torch.get_num_threads())); start=time.time(); bundle=load_cifar10(DEFAULT_CIFAR); mean,std=normalization(bundle); subset=balanced_subset(bundle.test_labels,100,offset=500); images,labels=bundle.test_images[subset],bundle.test_labels[subset]; records=model_records(); training=json.loads((RESULTS/'training.json').read_text()); expected={h['model_id']:h['sha256'] for s in training['summaries'] for h in s['history']}; models=[]
 if training['status']!='ARCHITECTURE_BREADTH_TRAINED' or set(expected)!={r.model_id for r in records}:raise ValueError('training record mismatch')
 for r in records:
  p=CHECKPOINTS/f'{r.model_id}.pt'
  if sha256(p)!=expected[r.model_id]:raise ValueError(f'checkpoint hash mismatch: {r.model_id}')
  m=make_model(r.family); m.load_state_dict(torch.load(p,map_location='cpu',weights_only=True)); m.eval(); models.append(m)
 clean=clean_statistics(models,images,labels,mean,std,256); strengths=json.loads(STRENGTHS.read_text())['datasets']['cifar10']['strengths']; roles={r['word_string']:r for r in word_records()}; rows=[]; words=evaluation_words()
 for ci,word in enumerate(words):
  metrics=evaluate_condition(models,clean,images,labels,subset,word,strengths,mean,std,256); ws='>'.join(word); role=roles[ws]
  for r,v,c in zip(records,metrics,clean):rows.append({'protocol':PROTOCOL_ID,'bank_split':'architecture_breadth','model_id':r.model_id,'lineage_id':r.lineage_id,'family':r.family,'recipe':r.recipe,'seed':r.seed,'epoch':r.epoch,'parameter_count':r.parameter_count,'condition_index':ci,'word':ws,'word_role':role['role'],'reverse_group':role['reverse_group'],'length':role['length'],'sample_count':labels.numel(),'clean_accuracy':c['accuracy'],**v})
  if ci%10==0 or ci+1==len(words):print(f'[p2e] {ci+1}/{len(words)} {ws}',flush=True)
 RESULTS.mkdir(parents=True,exist_ok=True); path=RESULTS/'behavior.csv'
 with path.open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]),lineterminator='\n'); w.writeheader(); w.writerows(rows)
 out={'protocol':PROTOCOL_ID,'status':'ARCHITECTURE_BREADTH_EVALUATED','models':24,'lineages':12,'conditions':97,'samples_per_condition':1000,'partition':{'per_class':100,'offset':500},'behavior_csv':'v4/phase2e/results/behavior.csv','sha256':sha256(path),'elapsed_seconds':time.time()-start}; (RESULTS/'evaluation.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); return out
if __name__=='__main__':run()
