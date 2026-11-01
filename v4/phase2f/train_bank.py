#!/usr/bin/env python3
"""Train deterministic CIFAR-100 development/confirmation banks on CPU."""
import argparse,json,time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from v4.phase2.train_bank import augment,seed_everything,sha256
from v4.phase2f import PROTOCOL_ID
from v4.phase2f.data import load
from v4.phase2f.manifest import model_records
from v4.phase2f.models import make_model
ROOT=Path(__file__).resolve().parents[2];CHECKPOINTS=ROOT/'checkpoints_v4/phase2f';RESULTS=ROOT/'v4/phase2f/results'
def moments(x):
 s=torch.zeros(3,dtype=torch.float64);ss=torch.zeros(3,dtype=torch.float64);n=0
 for z in x.split(2000):
  z=z.double().div(255);s+=z.sum((0,2,3));ss+=(z*z).sum((0,2,3));n+=z.shape[0]*z.shape[2]*z.shape[3]
 mean=s/n;std=(ss/n-mean*mean).sqrt();return mean.float().view(1,3,1,1),std.float().view(1,3,1,1)
def train(split):
 torch.set_num_threads(min(2,torch.get_num_threads()));data=load();mean,std=moments(data['train_images']);records=[r for r in model_records() if r.bank_split==split];templates={}
 for r in records:templates.setdefault(r.lineage_id,r)
 summaries=[];allstart=time.time()
 for i,t in enumerate(templates.values(),1):
  gen=seed_everything(t.seed);model=make_model(t.family);opt=torch.optim.AdamW(model.parameters(),lr=3e-3,weight_decay=1e-4);loader=DataLoader(TensorDataset(data['train_images'],data['train_labels']),batch_size=512,shuffle=True,generator=gen,num_workers=0);hist=[];start=time.time()
  for epoch in (1,2):
   model.train();correct=count=0;loss_sum=0.
   for raw,y in loader:
    x=(augment(raw,t.recipe,gen)-mean)/std;opt.zero_grad(set_to_none=True);logits=model(x);loss=F.cross_entropy(logits,y,label_smoothing=.1 if t.recipe=='label_smooth' else 0.);loss.backward();opt.step();correct+=int((logits.argmax(1)==y).sum());count+=len(y);loss_sum+=float(loss.detach())*len(y)
   rec=next(r for r in records if r.lineage_id==t.lineage_id and r.epoch==epoch);p=CHECKPOINTS/f'{rec.model_id}.pt';p.parent.mkdir(parents=True,exist_ok=True);torch.save(model.state_dict(),p);hist.append({'epoch':epoch,'model_id':rec.model_id,'train_accuracy':correct/count,'train_loss':loss_sum/count,'checkpoint':str(p.relative_to(ROOT)),'sha256':sha256(p)});print(f'[p2f {split} {i}/18] {t.lineage_id} epoch={epoch} acc={correct/count:.4f}',flush=True)
  summaries.append({'lineage_id':t.lineage_id,'family':t.family,'recipe':t.recipe,'seed':t.seed,'parameter_count':t.parameter_count,'elapsed_seconds':time.time()-start,'history':hist})
 out={'protocol':PROTOCOL_ID,'status':f'{split.upper()}_BANK_TRAINED','split':split,'lineages':18,'models':36,'normalization_mean':mean.flatten().tolist(),'normalization_std':std.flatten().tolist(),'elapsed_seconds':time.time()-allstart,'summaries':summaries};RESULTS.mkdir(parents=True,exist_ok=True);(RESULTS/f'{split}-training.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--split',choices=('development','confirmation'),required=True);train(p.parse_args().split)
