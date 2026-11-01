"""Full-vector composition features and ridge models for Phase 2F."""
from __future__ import annotations
import csv,math
from collections import defaultdict
import numpy as np
from v4.phase2f import BOUNDED_METRICS,METRICS,PERFORMANCE_METRICS,PRIMITIVES
FAMILIES=('vgg_holdout','dense_holdout','inception_holdout','squeeze_holdout','mixer_holdout','attention_holdout');RECIPES=('erm_flip','label_smooth','noise_aug');ALPHA=1.;Q_GRID=tuple(round(.1*i,1) for i in range(21));CHANCE=.01
def pair_group(a,b):return '|'.join(sorted((a,b)))
def all_groups():return tuple(pair_group(PRIMITIVES[i],PRIMITIVES[j]) for i in range(6) for j in range(i+1,6))
def load_behavior(path,roles=None):
 out=defaultdict(dict)
 for raw in csv.DictReader(open(path,newline='')):
  if roles and raw['word_role'] not in roles:continue
  r=dict(raw)
  for n in ('seed','epoch','parameter_count','condition_index','length','sample_count'):r[n]=int(r[n])
  for n in METRICS:r[n]=float(r[n])
  out[r['model_id']][r['word']]=r
 return dict(out)
def clip(metric,x):return max(0.,x) if metric=='feature_drift' else min(1.,max(0.,x))
def components(model_rows,truth,metric,groups,q):
 clean=model_rows['Identity'][metric];singles={p:model_rows[p][metric] for p in PRIMITIVES};res={}
 for a in PRIMITIVES:
  for b in PRIMITIVES:
   if a!=b:res[(a,b)]=model_rows[f'{a}>{b}'][metric]-singles[a]-singles[b]+clean
 word=truth['word'].split('>');pairs=[(word[i],word[j]) for i in range(len(word)) for j in range(i+1,len(word))];scale=len(pairs)**(q-1);used=[res[p] if pair_group(*p) in groups else 0. for p in pairs];sym=[.5*(res[p]+res[(p[1],p[0])]) if pair_group(*p) in groups else 0. for p in pairs];base=clean+sum(singles[p]-clean for p in word);ordered=clip(metric,base+scale*sum(used));symmetric=clip(metric,base+scale*sum(sym))
 if metric in PERFORMANCE_METRICS:
  floor=CHANCE if metric in ('accuracy','balanced_accuracy') else 0.;den=max(clean-floor,1e-8);ret=[min(2.,max(.02,(singles[p]-floor)/den)) for p in word];independent=clip(metric,floor+(clean-floor)*float(np.prod(ret)));envelope=min(singles[p] for p in word)
 else:independent=clip(metric,base);envelope=max(singles[p] for p in word)
 return {'ordered':ordered,'symmetric':symmetric,'independent':independent,'envelope':envelope,'clean':clean}
def feature_vector(truth,metric,comp,include):
 word=truth['word'].split('>');x=[comp['clean'],math.log1p(truth['parameter_count']),truth['epoch'],len(word),*[word.count(p) for p in PRIMITIVES],*[float(truth['family']==f) for f in FAMILIES],*[float(truth['recipe']==r) for r in RECIPES]]
 if include:x += [comp[k] for k in ('ordered','symmetric','independent','envelope')]
 return x
def rows(behavior,role,groups,q_by_metric):
 out=[]
 for model in behavior.values():
  for truth in model.values():
   if truth['word_role']!=role:continue
   r={k:truth[k] for k in ('model_id','lineage_id','family','recipe','seed','epoch','parameter_count','word','reverse_group','length')};r['target']={};r['components']={};r['x0']={};r['x1']={}
   for metric in METRICS:
    c=components(model,truth,metric,set(groups),q_by_metric[metric]);r['target'][metric]=truth[metric];r['components'][metric]=c;r['x0'][metric]=feature_vector(truth,metric,c,False);r['x1'][metric]=feature_vector(truth,metric,c,True)
   out.append(r)
 return out
def fit(rows_,metric,include):
 key='x1' if include else 'x0';x=np.asarray([r[key][metric] for r in rows_]);mean=x.mean(0);scale=x.std(0);scale[scale<1e-10]=1.;d=np.column_stack((np.ones(len(x)),(x-mean)/scale));y=np.asarray([r['target'][metric] for r in rows_]);pen=np.eye(d.shape[1])*ALPHA;pen[0,0]=0.;coef=np.linalg.solve(d.T@d+pen,d.T@y);return {'metric':metric,'include_composition':include,'mean':mean.tolist(),'scale':scale.tolist(),'coefficients':coef.tolist(),'alpha':ALPHA}
def apply(r,model):
 metric=model['metric'];key='x1' if model['include_composition'] else 'x0';x=np.asarray(r[key][metric]);d=np.r_[1.,(x-np.asarray(model['mean']))/np.asarray(model['scale'])];return clip(metric,float(d@np.asarray(model['coefficients'])))
def select_q(behavior):
 out={};search={}
 for metric in METRICS:
  scores=[]
  for q in Q_GRID:
   rs=rows(behavior,'development',all_groups(),{m:(q if m==metric else 1.) for m in METRICS});mae=float(np.mean([abs(r['components'][metric]['ordered']-r['target'][metric]) for r in rs]));scores.append({'q':q,'ordered_mae':mae})
  win=min(scores,key=lambda x:(x['ordered_mae'],x['q']));out[metric]=win['q'];search[metric]=scores
 return out,search
def magnitude_design(behavior,scales):
 values=defaultdict(list)
 for model in behavior.values():
  for metric in METRICS:
   clean=model['Identity'][metric];single={p:model[p][metric] for p in PRIMITIVES}
   for g in all_groups():
    a,b=g.split('|');ab=model[f'{a}>{b}'][metric]-single[a]-single[b]+clean;ba=model[f'{b}>{a}'][metric]-single[a]-single[b]+clean;values[g].append((abs(.5*(ab+ba))+.5*abs(ab-ba))/scales[metric])
 return sorted(all_groups(),key=lambda g:(np.mean(values[g]),g),reverse=True)[:5]
