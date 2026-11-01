#!/usr/bin/env python3
"""Freeze all architecture-holdout methods and alternative pair selectors."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from v4.phase2b.method import all_pair_groups, calibration, fit_gate, forecast_rows, load_behavior, pair_group
from v4.phase2d import PRIMITIVES, PROTOCOL_ID

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/"v4/phase2/results/development-behavior.csv"
P2B=ROOT/"v4/phase2b/FROZEN_ACHM.json"
P2C=ROOT/"v4/phase2c/FROZEN_PAIR_DESIGNS.json"
OUTPUT=ROOT/"v4/phase2d/FROZEN_HOLDOUT_METHODS.json"
L2=0.001; BUDGET=5


def magnitude_variance(behavior):
    values=defaultdict(list)
    for rows in behavior.values():
        cal=calibration(rows)
        for group in all_pair_groups():
            a,b=group.split("|")
            sym=.5*(cal.residuals[(a,b)]+cal.residuals[(b,a)])
            asym=.5*abs(cal.residuals[(a,b)]-cal.residuals[(b,a)])
            values[group].append(abs(sym)+asym)
    magnitude=sorted(all_pair_groups(),key=lambda g:(np.mean(values[g]),g),reverse=True)[:BUDGET]
    variance=sorted(all_pair_groups(),key=lambda g:(np.std(values[g]),g),reverse=True)[:BUDGET]
    return magnitude,variance


def coverage_design():
    groups=all_pair_groups(); selected=[]; degree={p:0 for p in PRIMITIVES}
    while len(selected)<BUDGET:
        def score(g):
            a,b=g.split("|"); new=int(degree[a]==0)+int(degree[b]==0)
            balance=-(degree[a]+degree[b]); tie=hashlib.sha256(f"coverage|{g}".encode()).hexdigest()
            return (new,balance,tie)
        winner=max((g for g in groups if g not in selected),key=score)
        selected.append(winner)
        for p in winner.split("|"): degree[p]+=1
    return selected


def d_optimal_design():
    groups=all_pair_groups(); selected=[]; dim=1+len(PRIMITIVES); info=np.eye(dim)*1e-3
    vectors={}
    for g in groups:
        x=np.zeros(dim); x[0]=1.;
        for p in g.split("|"): x[1+PRIMITIVES.index(p)]=1.
        vectors[g]=x
    while len(selected)<BUDGET:
        def gain(g):
            sign,logdet=np.linalg.slogdet(info+np.outer(vectors[g],vectors[g]))
            return (logdet,hashlib.sha256(f"dopt|{g}".encode()).hexdigest())
        winner=max((g for g in groups if g not in selected),key=gain)
        selected.append(winner); info+=np.outer(vectors[winner],vectors[winner])
    return selected


def freeze():
    behavior=load_behavior(SOURCE,{"calibration","method_development"})
    p2b=json.loads(P2B.read_text()); p2c=json.loads(P2C.read_text())
    magnitude,variance=magnitude_variance(behavior)
    groups={
        "active":p2c["designs"]["active"]["pair_groups"],
        "magnitude":magnitude,"variance":variance,
        "coverage":coverage_design(),"d_optimal":d_optimal_design(),
    }
    selectors={}
    for name,selected in groups.items():
        rows=forecast_rows(behavior,selected,"method_development")
        selectors[name]={"pair_groups":selected,"gate":fit_gate(rows,L2),"directed_measurements":10}
    payload={"protocol":PROTOCOL_ID,"status":"ALL_METHODS_FROZEN_ARCHITECTURE_OUTCOMES_UNOPENED",
             "source_roles":["calibration","method_development"],"forbidden_confirmations":["phase2","phase2b","phase2c"],
             "l2":L2,"selectors":selectors,"full15":p2b["budgets"]["15"],
             "random_designs":{k:v for k,v in p2c["designs"].items() if k.startswith("random_")},
             "prior_regressions":"v4/phase2/FROZEN_METHOD.json"}
    OUTPUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); return payload

if __name__=="__main__":
    p=freeze(); print(OUTPUT)
    for k,v in p["selectors"].items(): print(k,v["pair_groups"])
