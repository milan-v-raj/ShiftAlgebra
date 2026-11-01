#!/usr/bin/env python3
"""Verify Phase-2E frozen setup and optional outcome artifacts."""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
BASE=ROOT/'v4/phase2e'; RESULTS=BASE/'results'
def verify():
 c=[];m=json.loads((BASE/'PHASE2E_MANIFEST.json').read_text());assert m['counts']=={'families':4,'lineages':12,'snapshots':24,'calibration':37,'targets':60} and m['image_partition']=={'per_class':100,'offset':500};c.append('manifest counts and partition')
 f=json.loads((BASE/'FROZEN_BREADTH_METHODS.json').read_text());s=ROOT/f['method_source'];assert f['status']=='INHERITED_METHODS_AND_GATES_FROZEN_OUTCOMES_UNOPENED' and hashlib.sha256(s.read_bytes()).hexdigest()==f['method_source_sha256'];c.append('inherited methods hash-frozen')
 if (RESULTS/'behavior.csv').exists():
  b=list(csv.DictReader((RESULTS/'behavior.csv').open()));e=json.loads((RESULTS/'evaluation.json').read_text());assert len(b)==2328 and len({r['lineage_id'] for r in b})==12;assert hashlib.sha256((RESULTS/'behavior.csv').read_bytes()).hexdigest()==e['sha256'];c.append('behavior accounting and hash')
 if (RESULTS/'forecast.csv').exists():
  rows=list(csv.DictReader((RESULTS/'forecast.csv').open()));assert len(rows)==1440
  names=('g0_prior','g1_prior','full15','active','magnitude','variance','coverage','d_optimal',*[f'random_{i:02d}' for i in range(20)])
  for n in names:
   x=np.array([float(r[n]) for r in rows]);assert np.isfinite(x).all() and (x>=0).all() and (x<=1).all()
  a=json.loads((RESULTS/'PHASE2E_ANALYSIS.json').read_text())
  for n in ('g0_prior','g1_prior','full15','active','magnitude','variance','coverage','d_optimal'):
   mae=100*np.mean([abs(float(r[n])-float(r['true_accuracy'])) for r in rows]);assert abs(mae-a['methods'][n]['accuracy_mae_pp'])<1e-10
  c.append('forecast bounds and MAE recomputation')
 return c
if __name__=='__main__':
 c=verify();print(f'Phase-2E verification: {len(c)}/{len(c)} passed');[print('PASS',x) for x in c]
