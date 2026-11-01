"""Four additional lightweight families absent from Phases 2 and 2D."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionBlock(nn.Module):
 def __init__(self,c,o):
  super().__init__(); q=o//4
  self.a=nn.Conv2d(c,q,1); self.b=nn.Sequential(nn.Conv2d(c,q,1),nn.ReLU(),nn.Conv2d(q,q,3,padding=1)); self.d=nn.Sequential(nn.Conv2d(c,q,1),nn.ReLU(),nn.Conv2d(q,q,5,padding=2)); self.p=nn.Conv2d(c,o-3*q,1)
 def forward(self,x): return F.relu(torch.cat((self.a(x),self.b(x),self.d(x),self.p(F.avg_pool2d(x,3,1,1))),1))
class TinyInception(nn.Module):
 def __init__(self):
  super().__init__(); self.f=nn.Sequential(nn.Conv2d(3,12,3,padding=1),nn.ReLU(),InceptionBlock(12,24),nn.MaxPool2d(2),InceptionBlock(24,32)); self.h=nn.Linear(32,10)
 def forward(self,x): return self.h(F.adaptive_avg_pool2d(self.f(x),1).flatten(1))
class Fire(nn.Module):
 def __init__(self,c,s,e):
  super().__init__(); self.s=nn.Conv2d(c,s,1); self.a=nn.Conv2d(s,e,1); self.b=nn.Conv2d(s,e,3,padding=1)
 def forward(self,x):
  x=F.relu(self.s(x)); return F.relu(torch.cat((self.a(x),self.b(x)),1))
class TinySqueeze(nn.Module):
 def __init__(self):
  super().__init__(); self.f=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),Fire(16,6,12),Fire(24,8,16),nn.MaxPool2d(2),Fire(32,10,20)); self.h=nn.Linear(40,10)
 def forward(self,x): return self.h(F.adaptive_avg_pool2d(self.f(x),1).flatten(1))
class MixerBlock(nn.Module):
 def __init__(self,n,c):
  super().__init__(); self.n1=nn.LayerNorm(c); self.tm=nn.Sequential(nn.Linear(n,n),nn.GELU(),nn.Linear(n,n)); self.n2=nn.LayerNorm(c); self.cm=nn.Sequential(nn.Linear(c,2*c),nn.GELU(),nn.Linear(2*c,c))
 def forward(self,x):
  x=x+self.tm(self.n1(x).transpose(1,2)).transpose(1,2); return x+self.cm(self.n2(x))
class TinyMixer(nn.Module):
 def __init__(self,c=24):
  super().__init__(); self.patch=nn.Conv2d(3,c,4,4); self.blocks=nn.Sequential(MixerBlock(64,c),MixerBlock(64,c)); self.n=nn.LayerNorm(c); self.h=nn.Linear(c,10)
 def forward(self,x):
  x=self.patch(x).flatten(2).transpose(1,2); return self.h(self.n(self.blocks(x)).mean(1))
class TinyAttention(nn.Module):
 def __init__(self,c=24):
  super().__init__(); self.patch=nn.Conv2d(3,c,4,4); self.pos=nn.Parameter(torch.zeros(1,64,c)); layer=nn.TransformerEncoderLayer(c,3,48,batch_first=True,norm_first=True,dropout=0.); self.blocks=nn.TransformerEncoder(layer,2); self.n=nn.LayerNorm(c); self.h=nn.Linear(c,10)
 def forward(self,x):
  x=self.patch(x).flatten(2).transpose(1,2)+self.pos; return self.h(self.n(self.blocks(x)).mean(1))
FAMILIES=("inception_holdout","squeeze_holdout","mixer_holdout","attention_holdout")
def make_model(family):
 return {"inception_holdout":TinyInception,"squeeze_holdout":TinySqueeze,"mixer_holdout":TinyMixer,"attention_holdout":TinyAttention}[family]()
