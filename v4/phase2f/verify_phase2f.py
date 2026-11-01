#!/usr/bin/env python3
"""Verify Phase-2F protocol, methods, and optional confirmation artifacts."""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from v4.phase2f import METRICS
BASE=ROOT/'v4/phase2f';RESULTS=BASE/'results'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
 c=[];m=json.loads((BASE/'PHASE2F_MANIFEST.json').read_text());assert m['metrics']==list(METRICS) and m['counts']['lineages_per_split']==18 and m['counts']['snapshots_per_split']==36 and m['classes']==100;c.append('full-vector manifest and bank counts')
 p=json.loads((BASE/'DATA_PROVENANCE.json').read_text());assert p['canonical_test_md5']=='f0ef6b0ae62326f3e7ffdfab6717acfc' and p['canonical_test_pixel_matches']==10000;c.append('canonical CIFAR-100 test provenance')
 if (BASE/'FROZEN_FULL_VECTOR_METHODS.json').exists():
  f=json.loads((BASE/'FROZEN_FULL_VECTOR_METHODS.json').read_text());assert f['status']=='FULL_VECTOR_METHODS_FROZEN_CONFIRMATION_OUTCOMES_UNOPENED' and sha(ROOT/f['development_behavior'])==f['development_behavior_sha256'] and set(f['metrics'])==set(METRICS) and len([x for x in f['designs'] if x.startswith('random_')])==20;c.append('development-only method freeze')
 if (RESULTS/'confirmation-behavior.csv').exists():
  b=list(csv.DictReader((RESULTS/'confirmation-behavior.csv').open()));e=json.loads((RESULTS/'confirmation-evaluation.json').read_text());assert len(b)==3492 and sha(RESULTS/'confirmation-behavior.csv')==e['sha256'];c.append('confirmation behavior accounting and hash')
 if (RESULTS/'confirmation-forecast.csv').exists():
  r=list(csv.DictReader((RESULTS/'confirmation-forecast.csv').open()));assert len(r)==2160
  for metric in METRICS:
   for method in ('g0','full15','magnitude5'):
    x=np.array([float(z[f'{method}_{metric}']) for z in r]);assert np.isfinite(x).all() and (x>=0).all()
  a=json.loads((RESULTS/'PHASE2F_ANALYSIS.json').read_text())
  for metric in METRICS:
   for method in ('g0','full15','magnitude5'):
    mae=np.mean([abs(float(z[f'{method}_{metric}'])-float(z[f'true_{metric}'])) for z in r]);assert abs(mae-a['methods'][method]['per_metric'][metric]['mae'])<1e-12
  c.append('all six forecast coordinates and MAEs')
 return c
if __name__=='__main__':
 c=verify();print(f'Phase-2F verification: {len(c)}/{len(c)} passed');[print('PASS',x) for x in c]
