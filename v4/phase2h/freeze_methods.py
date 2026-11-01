#!/usr/bin/env python3
"""Fit and freeze nested order-ablation methods on Phase-2F development only."""
import json
from pathlib import Path

import numpy as np

from v4.phase2.train_bank import sha256
from v4.phase2f import METRICS
from v4.phase2f.method import all_groups, load_behavior
from v4.phase2h import PROTOCOL_ID
from v4.phase2h.method import apply, fit, make_rows

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "v4/phase2h"
SOURCE = ROOT / "v4/phase2f/results/development-behavior.csv"
PHASE2F_METHOD = ROOT / "v4/phase2f/FROZEN_FULL_VECTOR_METHODS.json"
OUTPUT = BASE / "FROZEN_ORDER_ABLATION_METHODS.json"
EXPECTED_DEVELOPMENT_SHA256 = "8c3289ac47c01267751a2f3d8104d1940f04e20b8fa0cec4dea0c2baef7b90a7"


def score(rows, models, scales):
    raw = {
        metric: float(np.mean([
            abs(apply(row, models[metric]) - row["target"][metric]) for row in rows
        ]))
        for metric in METRICS
    }
    return {
        "raw_mae": raw,
        "macro_normalized_mae": float(np.mean([
            raw[metric] / scales[metric] for metric in METRICS
        ])),
    }


def freeze():
    if sha256(SOURCE) != EXPECTED_DEVELOPMENT_SHA256:
        raise ValueError("Phase-2F development behavior changed")
    phase2f = json.loads(PHASE2F_METHOD.read_text())
    if phase2f["development_behavior_sha256"] != EXPECTED_DEVELOPMENT_SHA256:
        raise ValueError("Phase-2F method points to a different development artifact")
    behavior = load_behavior(SOURCE, {"calibration", "development"})
    groups = list(all_groups())
    rows = make_rows(behavior, "development", groups, phase2f["q_by_metric"])
    models = {
        mode: {metric: fit(rows, metric, mode) for metric in METRICS}
        for mode in ("g0", "symmetric", "order_aware")
    }
    payload = {
        "protocol": PROTOCOL_ID,
        "status": "ORDER_ABLATION_METHODS_FROZEN_CONFIRMATION_OUTCOMES_UNOPENED",
        "development_source": "v4/phase2f/results/development-behavior.csv",
        "development_sha256": EXPECTED_DEVELOPMENT_SHA256,
        "phase2f_method_sha256": sha256(PHASE2F_METHOD),
        "metrics": list(METRICS),
        "target_scales": phase2f["target_scales"],
        "q_by_metric": phase2f["q_by_metric"],
        "pair_groups": groups,
        "directed_pair_measurements": 30,
        "equal_budget": (
            "both composition methods receive all thirty directed pair measurements; "
            "the symmetric method averages directions before feature construction"
        ),
        "models": models,
        "development": {
            mode: score(rows, models[mode], phase2f["target_scales"])
            for mode in models
        },
        "information_boundary": (
            "Phase-2F development calibration/targets only; Phase-2F confirmation and "
            "all Phase-2G outcomes are context but not numerical fit inputs; Phase-2H forbidden"
        ),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(OUTPUT)
    print(json.dumps(payload["development"], indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    freeze()
