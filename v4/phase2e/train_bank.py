#!/usr/bin/env python3
"""Train twelve fresh architecture-breadth CPU lineages."""
import json,time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from v4.phase1.classifiers import normalization
from v4.phase1.data import load_cifar10
from v4.phase1.pipeline import DEFAULT_CIFAR
from v4.phase2.train_bank import augment,seed_everything,sha256
from v4.phase2e import PROTOCOL_ID
from v4.phase2e.manifest import model_records
from v4.phase2e.models import make_model
ROOT=Path(__file__).resolve().parents[2]; CHECKPOINTS=ROOT/'checkpoints_v4/phase2e'; RESULTS=ROOT/'v4/phase2e/results'
def train():
 torch.set_num_threads(min(2,torch.get_num_threads())); bundle=load_cifar10(DEFAULT_CIFAR); mean,std=normalization(bundle); records=model_records(); templates={}
 for r in records:templates.setdefault(r.lineage_id,r)
 summaries=[]; allstart=time.time()
 for i,t in enumerate(templates.values(),1):
  gen=seed_everything(t.seed); model=make_model(t.family); opt=torch.optim.AdamW(model.parameters(),lr=3e-3,weight_decay=1e-4); loader=DataLoader(TensorDataset(bundle.train_images,bundle.train_labels),batch_size=512,shuffle=True,generator=gen,num_workers=0); hist=[]; start=time.time()
  for epoch in (1,2):
   model.train(); correct=count=0; loss_sum=0.
   for raw,labels in loader:
    x=(augment(raw,t.recipe,gen)-mean)/std; opt.zero_grad(set_to_none=True); logits=model(x); loss=F.cross_entropy(logits,labels,label_smoothing=.1 if t.recipe=='label_smooth' else 0.); loss.backward(); opt.step(); correct+=int((logits.argmax(-1)==labels).sum()); count+=labels.numel(); loss_sum+=float(loss.detach())*labels.numel()
   rec=next(r for r in records if r.lineage_id==t.lineage_id and r.epoch==epoch); path=CHECKPOINTS/f'{rec.model_id}.pt'; path.parent.mkdir(parents=True,exist_ok=True); torch.save(model.state_dict(),path); hist.append({'epoch':epoch,'model_id':rec.model_id,'train_accuracy':correct/count,'train_loss':loss_sum/count,'checkpoint':str(path.relative_to(ROOT)),'sha256':sha256(path)}); print(f'[p2e {i}/12] {t.lineage_id} epoch={epoch} train_acc={correct/count:.4f}',flush=True)
  summaries.append({'lineage_id':t.lineage_id,'family':t.family,'recipe':t.recipe,'seed':t.seed,'parameter_count':t.parameter_count,'elapsed_seconds':time.time()-start,'history':hist})
 out={'protocol':PROTOCOL_ID,'status':'ARCHITECTURE_BREADTH_TRAINED','lineages':12,'models':24,'elapsed_seconds':time.time()-allstart,'summaries':summaries}; RESULTS.mkdir(parents=True,exist_ok=True); (RESULTS/'training.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); return out
if __name__=='__main__':train()
