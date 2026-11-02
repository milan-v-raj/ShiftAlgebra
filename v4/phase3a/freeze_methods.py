#!/usr/bin/env python3
"""Develop and freeze order-blind realistic-corruption methods."""
import json
from pathlib import Path

import numpy as np

from v4.phase2.train_bank import sha256
from v4.phase3a import METRICS, PROTOCOL_ID
from v4.phase3a.method import all_groups, apply, fit, load_behavior, make_rows, select_q

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "v4/phase3a"
RESULTS = BASE / "results"
OUTPUT = BASE / "FROZEN_REALISTIC_METHODS.json"


def model_score(rows, models, scales):
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


def formula_score(rows, name, scales):
    raw = {
        metric: float(np.mean([
            abs(row["components"][metric][name] - row["target"][metric]) for row in rows
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
    source = RESULTS / "development-behavior.csv"
    behavior = load_behavior(source, {"calibration", "development"})
    targets = [
        row for model in behavior.values() for row in model.values()
        if row["word_role"] == "development"
    ]
    scales = {
        metric: max(float(np.std([row[metric] for row in targets])), 1e-6)
        for metric in METRICS
    }
    q_by_metric, q_search = select_q(behavior)
    rows = make_rows(behavior, "development", all_groups(), q_by_metric)
    models = {
        mode: {metric: fit(rows, metric, mode) for metric in METRICS}
        for mode in ("g0", "symmetric")
    }
    payload = {
        "protocol": PROTOCOL_ID,
        "status": "REALISTIC_METHODS_FROZEN_CONFIRMATION_OUTCOMES_UNOPENED",
        "development_behavior": "v4/phase3a/results/development-behavior.csv",
        "development_behavior_sha256": sha256(source),
        "metrics": list(METRICS),
        "target_scales": scales,
        "q_by_metric": q_by_metric,
        "q_search": q_search,
        "pair_groups": list(all_groups()),
        "directed_pair_measurements": 30,
        "models": models,
        "development": {
            "g0": model_score(rows, models["g0"], scales),
            "symmetric": model_score(rows, models["symmetric"], scales),
            "independent": formula_score(rows, "independent", scales),
            "envelope": formula_score(rows, "envelope", scales),
            "symmetric_formula": formula_score(rows, "symmetric_formula", scales),
        },
        "information_boundary": (
            "Phase-3A development calibration and target outcomes only; confirmation "
            "behavior/outcomes forbidden (the prospective roster and word manifest are public)"
        ),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(OUTPUT)
    print(json.dumps(payload["development"], indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    freeze()
