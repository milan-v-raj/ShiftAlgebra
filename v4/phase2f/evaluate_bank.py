#!/usr/bin/env python3
"""Evaluate all six behavior coordinates on a sealed CIFAR-100 bank split."""
import argparse,csv,json,time
from pathlib import Path
import torch
from v4.phase1b.corruptions import apply_calibrated_word
from v4.phase2.train_bank import sha256
from v4.phase2f import PROTOCOL_ID
from v4.phase2f.data import balanced_indices,load
from v4.phase2f.manifest import model_records,word_records,words_for
from v4.phase2f.models import logits_and_features,make_model
ROOT=Path(__file__).resolve().parents[2];RESULTS=ROOT/'v4/phase2f/results';CHECKPOINTS=ROOT/'checkpoints_v4/phase2f';STRENGTHS=ROOT/'v4/phase1b/FROZEN_STRENGTHS.json'
def partition(split):return (20,0) if split=='development' else (20,20)
def load_models(split):
 records=[r for r in model_records() if r.bank_split==split];training=json.loads((RESULTS/f'{split}-training.json').read_text());expected={h['model_id']:h['sha256'] for s in training['summaries'] for h in s['history']};models=[]
 if set(expected)!={r.model_id for r in records}:raise ValueError('training record mismatch')
 for r in records:
  p=CHECKPOINTS/f'{r.model_id}.pt'
  if sha256(p)!=expected[r.model_id]:raise ValueError(f'hash mismatch {r.model_id}')
  m=make_model(r.family);m.load_state_dict(torch.load(p,map_location='cpu',weights_only=True));m.eval();models.append(m)
 mean=torch.tensor(training['normalization_mean']).view(1,3,1,1);std=torch.tensor(training['normalization_std']).view(1,3,1,1);return records,models,mean,std
@torch.inference_mode()
def evaluate(models,records,images,labels,sample_ids,word,strengths,mean,std,clean_centroids=None,batch=256):
 n=len(labels);correct=[torch.zeros(100,dtype=torch.long) for _ in models];counts=torch.bincount(labels,minlength=100);conf=[0.]*len(models);margin=[0.]*len(models);bin_n=[torch.zeros(10) for _ in models];bin_conf=[torch.zeros(10) for _ in models];bin_correct=[torch.zeros(10) for _ in models];feature_sum=[None]*len(models)
 for start in range(0,n,batch):
  stop=min(start+batch,n);pixels=images[start:stop].float().div(255);shifted=apply_calibrated_word(pixels,word,sample_ids[start:stop],strengths);x=(shifted-mean)/std;y=labels[start:stop]
  for i,(m,r) in enumerate(zip(models,records)):
   logits,z=logits_and_features(m,r.family,x);p=logits.softmax(1);top=p.topk(2,dim=1).values;c,pred=p.max(1);good=pred==y;correct[i]+=torch.bincount(y[good],minlength=100);conf[i]+=float(c.sum());margin[i]+=float((top[:,0]-top[:,1]).sum());feature_sum[i]=z.double().sum(0) if feature_sum[i] is None else feature_sum[i]+z.double().sum(0);bins=torch.clamp((c*10).long(),max=9)
   for b in range(10):
    mask=bins==b;bin_n[i][b]+=int(mask.sum());bin_conf[i][b]+=float(c[mask].sum());bin_correct[i][b]+=int(good[mask].sum())
 out=[];centroids=[]
 for i in range(len(models)):
  centroid=feature_sum[i]/n;centroids.append(centroid);ece=0.
  for b in range(10):
   if bin_n[i][b]>0:ece+=float(bin_n[i][b]/n*abs(bin_correct[i][b]/bin_n[i][b]-bin_conf[i][b]/bin_n[i][b]))
  total=int(correct[i].sum());drift=0. if clean_centroids is None else float(torch.linalg.vector_norm(centroid-clean_centroids[i]));out.append({'margin':margin[i]/n,'feature_drift':drift,'ece':ece,'balanced_accuracy':float((correct[i]/counts).double().mean()),'accuracy':total/n,'mean_confidence':conf[i]/n})
 return out,centroids
def run(split):
 torch.set_num_threads(min(2,torch.get_num_threads()));start=time.time();data=load();per,offset=partition(split);ix=balanced_indices(data['test_labels'],per,offset);images=data['test_images'][ix];labels=data['test_labels'][ix];records,models,mean,std=load_models(split);strengths=json.loads(STRENGTHS.read_text())['datasets']['cifar10']['strengths'];roles={r['word_string']:r for r in word_records(split)};words=words_for(split);rows=[];clean_centroids=None
 for ci,word in enumerate(words):
  values,centroids=evaluate(models,records,images,labels,ix,word,strengths,mean,std,clean_centroids)
  if ci==0:clean_centroids=centroids
  ws='>'.join(word);role=roles[ws]
  for r,v in zip(records,values):rows.append({'protocol':PROTOCOL_ID,'bank_split':split,'model_id':r.model_id,'lineage_id':r.lineage_id,'family':r.family,'recipe':r.recipe,'seed':r.seed,'epoch':r.epoch,'parameter_count':r.parameter_count,'condition_index':ci,'word':ws,'word_role':role['role'],'reverse_group':role['reverse_group'],'length':role['length'],'sample_count':len(labels),**v})
  if ci%10==0 or ci+1==len(words):print(f'[p2f {split}] {ci+1}/{len(words)} {ws}',flush=True)
 path=RESULTS/f'{split}-behavior.csv'
 with path.open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 out={'protocol':PROTOCOL_ID,'status':f'{split.upper()}_FULL_VECTOR_EVALUATED','split':split,'models':36,'lineages':18,'conditions':97,'samples_per_condition':len(labels),'partition':{'per_class':per,'offset':offset},'metrics':['margin','feature_drift','ece','balanced_accuracy','accuracy','mean_confidence'],'behavior_csv':str(path.relative_to(ROOT)),'sha256':sha256(path),'elapsed_seconds':time.time()-start};(RESULTS/f'{split}-evaluation.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--split',choices=('development','confirmation'),required=True);run(p.parse_args().split)
