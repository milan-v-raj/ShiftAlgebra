"""Six previously defined holdout families with CIFAR-100 heads and features."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from v4.phase2d.models import TinyVGG,TinyDenseNet
from v4.phase2e.models import TinyInception,TinySqueeze,TinyMixer,TinyAttention
FAMILIES=('vgg_holdout','dense_holdout','inception_holdout','squeeze_holdout','mixer_holdout','attention_holdout')
def make_model(family,classes=100):
 m={'vgg_holdout':TinyVGG,'dense_holdout':TinyDenseNet,'inception_holdout':TinyInception,'squeeze_holdout':TinySqueeze,'mixer_holdout':TinyMixer,'attention_holdout':TinyAttention}[family]()
 if hasattr(m,'classifier'):m.classifier=nn.Linear(m.classifier.in_features,classes)
 else:m.h=nn.Linear(m.h.in_features,classes)
 return m
def features(model,family,x):
 if family=='vgg_holdout':return F.adaptive_avg_pool2d(model.features(x),1).flatten(1)
 if family=='dense_holdout':return F.adaptive_avg_pool2d(F.relu(model.norm(model.features(model.stem(x)))),1).flatten(1)
 if family=='inception_holdout':return F.adaptive_avg_pool2d(model.f(x),1).flatten(1)
 if family=='squeeze_holdout':return F.adaptive_avg_pool2d(model.f(x),1).flatten(1)
 if family=='mixer_holdout':
  z=model.patch(x).flatten(2).transpose(1,2);return model.n(model.blocks(z)).mean(1)
 if family=='attention_holdout':
  z=model.patch(x).flatten(2).transpose(1,2)+model.pos;return model.n(model.blocks(z)).mean(1)
 raise KeyError(family)
def logits_and_features(model,family,x):
 z=features(model,family,x);head=model.classifier if hasattr(model,'classifier') else model.h;return head(z),z
