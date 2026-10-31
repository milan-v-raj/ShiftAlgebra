#!/usr/bin/env python3
"""Freeze active and random five-pair designs before Phase-2C outcomes."""

from __future__ import annotations

import json
import random
from pathlib import Path

from v4.phase2b.method import (
    all_pair_groups,
    fit_gate,
    forecast_rows,
    load_behavior,
    mae_pp,
)
from v4.phase2c import PROTOCOL_ID

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "v4/phase2/results/development-behavior.csv"
PHASE2B_FROZEN = ROOT / "v4/phase2b/FROZEN_ACHM.json"
OUTPUT = ROOT / "v4/phase2c/FROZEN_PAIR_DESIGNS.json"
RANDOM_SEED = 22_403
RANDOM_DESIGNS = 20
BUDGET = 5
FIXED_L2 = 0.001  # Phase-2B development-selected value for the five-group gate


def random_designs(active: tuple[str, ...]) -> list[tuple[str, ...]]:
    rng = random.Random(RANDOM_SEED)
    groups = list(all_pair_groups())
    selected = []
    seen = {tuple(sorted(active))}
    while len(selected) < RANDOM_DESIGNS:
        candidate = tuple(sorted(rng.sample(groups, BUDGET)))
        if candidate not in seen:
            seen.add(candidate)
            selected.append(candidate)
    return selected


def freeze() -> dict:
    source = load_behavior(SOURCE, {"calibration", "method_development"})
    if len(source) != 30:
        raise ValueError(f"expected 30 development snapshots, got {len(source)}")
    p2b = json.loads(PHASE2B_FROZEN.read_text())
    active_groups = tuple(p2b["budgets"]["5"]["pair_groups"])
    designs = {"active": active_groups}
    designs.update({f"random_{i:02d}": groups for i, groups in enumerate(random_designs(active_groups))})

    frozen = {}
    for name, groups in designs.items():
        rows = forecast_rows(source, groups, "method_development")
        gate = fit_gate(rows, FIXED_L2)
        frozen[name] = {
            "pair_groups": list(groups),
            "directed_measurements": 2 * BUDGET,
            "gate": gate,
            "development_mae_pp": mae_pp(rows, gate),
        }
    payload = {
        "protocol": PROTOCOL_ID,
        "status": "PAIR_DESIGNS_FROZEN_FRESH_OUTCOMES_UNOPENED",
        "question": "Does a development-selected five-group design beat expected uniform-random five-group calibration?",
        "source_path": str(SOURCE.relative_to(ROOT)),
        "source_roles_loaded": ["calibration", "method_development"],
        "forbidden_sources": [
            "v4/phase2/results/confirmation-behavior.csv",
            "v4/phase2b/results/fresh-behavior.csv",
        ],
        "budget_unordered_groups": BUDGET,
        "directed_measurements": 2 * BUDGET,
        "random_seed": RANDOM_SEED,
        "random_design_count": RANDOM_DESIGNS,
        "fixed_l2": FIXED_L2,
        "designs": frozen,
        "full_budget_reference": p2b["budgets"]["15"],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


if __name__ == "__main__":
    result = freeze()
    print(OUTPUT)
    print("active", result["designs"]["active"]["pair_groups"])
    print("random designs", result["random_design_count"])
