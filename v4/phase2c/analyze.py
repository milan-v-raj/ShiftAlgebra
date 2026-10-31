#!/usr/bin/env python3
"""One-shot analysis for active versus random pair calibration."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from v4.phase2b.analyze import method_metrics
from v4.phase2c import PROTOCOL_ID

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "v4/phase2c/results"
RANDOM_METHODS = tuple(f"random_{index:02d}" for index in range(20))
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 22_404


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            for name in ("seed", "epoch", "parameter_count", "length"):
                row[name] = int(row[name])
            for name in (
                "true_accuracy", "clean_accuracy", "active5", "active5_disagreement",
                "full15", "g1_prior", *RANDOM_METHODS,
            ):
                row[name] = float(row[name])
            rows.append(row)
    if len(rows) != 1920:
        raise ValueError(f"expected 1920 fresh rows, got {len(rows)}")
    return rows


def bootstrap_expected_random(rows: list[dict]) -> dict:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        random_error = np.mean([abs(row[name] - row["true_accuracy"]) for name in RANDOM_METHODS])
        active_error = abs(row["active5"] - row["true_accuracy"])
        grouped[row["reverse_group"]][row["lineage_id"]].append(random_error - active_error)
    groups = sorted(grouped)
    lineages = sorted({row["lineage_id"] for row in rows})

    def estimate(sample_groups, sample_lineages):
        values = []
        for group in sample_groups:
            for lineage in sample_lineages:
                if lineage in grouped[group]:
                    values.append(np.mean(grouped[group][lineage]))
        return 100.0 * float(np.mean(values))

    point = estimate(groups, lineages)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_DRAWS)
    for index in range(BOOTSTRAP_DRAWS):
        sampled_groups = [groups[i] for i in rng.integers(0, len(groups), len(groups))]
        sampled_lineages = [lineages[i] for i in rng.integers(0, len(lineages), len(lineages))]
        draws[index] = estimate(sampled_groups, sampled_lineages)
    return {
        "estimand": "expected uniform-random-design absolute error minus active-design absolute error (pp)",
        "point_pp": point,
        "ci95_pp": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "draws": BOOTSTRAP_DRAWS,
        "clusters": "fresh reversal group and training lineage",
    }


def per_length(rows: list[dict], methods: tuple[str, ...]) -> dict:
    output = {}
    for length in (3, 4, 5, 6):
        subset = [row for row in rows if row["length"] == length]
        output[str(length)] = {
            method: 100.0 * float(np.mean([abs(row[method] - row["true_accuracy"]) for row in subset]))
            for method in methods
        }
    return output


def analyze(
    forecast_path: Path = RESULTS / "fresh-forecast.csv",
    output_path: Path = RESULTS / "PHASE2C_ANALYSIS.json",
) -> dict:
    rows = load_rows(forecast_path)
    primary = {name: method_metrics(rows, name) for name in ("active5", "full15", "g1_prior")}
    random_metrics = {name: method_metrics(rows, name) for name in RANDOM_METHODS}
    random_maes = np.asarray([random_metrics[name]["accuracy_mae_pp"] for name in RANDOM_METHODS])
    random_controlled = np.asarray([
        random_metrics[name]["controlled_model_rank_spearman"] for name in RANDOM_METHODS
    ])
    random_regret = np.asarray([random_metrics[name]["top1_selection_regret_pp"] for name in RANDOM_METHODS])
    bootstrap = bootstrap_expected_random(rows)
    active = primary["active5"]
    gates = {
        "active_mae_below_expected_random": active["accuracy_mae_pp"] < float(random_maes.mean()),
        "clustered_ci_excludes_zero": bootstrap["ci95_pp"][0] > 0,
        "active_controlled_rank_at_least_random_median": (
            active["controlled_model_rank_spearman"] >= float(np.median(random_controlled))
        ),
        "active_top1_regret_no_worse_than_random_median": (
            active["top1_selection_regret_pp"] <= float(np.median(random_regret))
        ),
    }
    passed = all(gates.values())
    result = {
        "protocol": PROTOCOL_ID,
        "status": "ACTIVE_PAIR_PRIMARY_GATE_PASSED" if passed else "ACTIVE_PAIR_PRIMARY_GATE_FAILED",
        "scope": "CPU-only fresh pair-selection confirmation",
        "fresh_rows": len(rows), "fresh_models": 30, "fresh_lineages": 15, "fresh_words": 64,
        "budget": {"unordered_groups": 5, "directed_measurements": 10, "full_directed_measurements": 30},
        "methods": primary,
        "random_strategy": {
            "design_count": len(RANDOM_METHODS),
            "mae_pp": {
                "mean": float(random_maes.mean()), "median": float(np.median(random_maes)),
                "min": float(random_maes.min()), "max": float(random_maes.max()),
                "per_design": {name: random_metrics[name]["accuracy_mae_pp"] for name in RANDOM_METHODS},
            },
            "controlled_spearman_median": float(np.median(random_controlled)),
            "top1_regret_pp_median": float(np.median(random_regret)),
            "active_percentile_by_mae": float(np.mean(random_maes >= active["accuracy_mae_pp"])),
            "all_metrics": random_metrics,
        },
        "primary_bootstrap": bootstrap,
        "gates": {**gates, "primary_gate_passed": passed},
        "per_length_mae_pp": per_length(rows, ("active5", "full15", "g1_prior")),
    }
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return result


if __name__ == "__main__":
    analyze()
