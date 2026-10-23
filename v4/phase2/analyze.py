#!/usr/bin/env python3
"""Frozen across-model ranking, regret, and incremental-value analysis."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from v4.analysis.metrics import (
    accuracy_mae_pp,
    accuracy_rmse_pp,
    model_selection_regret,
    near_oracle_selection_rate,
    partial_spearman,
    rank_environments_per_model,
    rank_models_per_environment,
)
from v4.phase2 import PROTOCOL_ID
from v4.phase2.train_bank import RESULTS

METHODS = (
    "primitive_mean",
    "max_damage",
    "product_retention",
    "pair_accuracy_q",
    "hazard_symmetric_q",
    "hazard_ordered_q",
    "g0_source_word",
    "g1_plus_hazard",
    "atc",
    "doc",
)
REGIMES = {
    "primitive_mean": "R1",
    "max_damage": "R1",
    "product_retention": "R1",
    "pair_accuracy_q": "R2",
    "hazard_symmetric_q": "R2",
    "hazard_ordered_q": "R2",
    "g0_source_word": "R0+word-metadata",
    "g1_plus_hazard": "R2 meta-calibrated",
    "atc": "R3 target-images",
    "doc": "R3 target-images",
}
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 22_202


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            for name in ("seed", "epoch", "parameter_count", "length"):
                row[name] = int(row[name])
            for name in ("true_accuracy", "clean_accuracy", *METHODS):
                row[name] = float(row[name])
            if row["word_role"] == "confirmation":
                rows.append(row)
    return rows


def matrices(rows: list[dict], method: str) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    words = sorted({row["word"] for row in rows})
    models = sorted({row["model_id"] for row in rows})
    lookup = {(row["word"], row["model_id"]): row for row in rows}
    predicted = np.asarray([[lookup[(word, model)][method] for model in models] for word in words])
    target = np.asarray([[lookup[(word, model)]["true_accuracy"] for model in models] for word in words])
    return predicted, target, words, models


def controls_for_models(rows: list[dict], models: list[str]) -> np.ndarray:
    metadata = {row["model_id"]: row for row in rows}
    families = sorted({row["family"] for row in rows})
    recipes = sorted({row["recipe"] for row in rows})
    return np.asarray([
        [
            metadata[model]["clean_accuracy"],
            np.log1p(metadata[model]["parameter_count"]),
            metadata[model]["epoch"],
            *[float(metadata[model]["family"] == value) for value in families],
            *[float(metadata[model]["recipe"] == value) for value in recipes],
        ]
        for model in models
    ])


def method_metrics(rows: list[dict], method: str) -> dict:
    predicted, target, words, models = matrices(rows, method)
    controls = controls_for_models(rows, models)
    controlled = [partial_spearman(predicted[index], target[index], controls) for index in range(len(words))]
    finite = [value for value in controlled if np.isfinite(value)]
    return {
        "information_regime": REGIMES[method],
        "accuracy_mae_pp": accuracy_mae_pp(predicted, target),
        "accuracy_rmse_pp": accuracy_rmse_pp(predicted, target),
        "rank_models_per_environment": rank_models_per_environment(predicted, target),
        "rank_environments_per_model": rank_environments_per_model(predicted, target),
        "source_family_controlled_spearman": float(np.mean(finite)) if finite else None,
        "source_family_controlled_valid_environments": len(finite),
        "top1_selection_regret_pp": model_selection_regret(predicted, target, top_k=1),
        "top3_selection_regret_pp": model_selection_regret(predicted, target, top_k=3),
        "near_oracle_top1_rate": near_oracle_selection_rate(predicted, target, tolerance_pp=1.0),
    }


def bootstrap_increment(rows: list[dict]) -> dict:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["reverse_group"]][row["lineage_id"]].append(
            abs(row["g0_source_word"] - row["true_accuracy"])
            - abs(row["g1_plus_hazard"] - row["true_accuracy"])
        )
    word_groups = sorted(grouped)
    lineages = sorted({row["lineage_id"] for row in rows})

    def estimate(sample_groups, sample_lineages) -> float:
        values = []
        for group in sample_groups:
            available = grouped[group]
            for lineage in sample_lineages:
                if lineage in available:
                    values.append(float(np.mean(available[lineage])))
        return 100.0 * float(np.mean(values))

    point = estimate(word_groups, lineages)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_DRAWS)
    for index in range(BOOTSTRAP_DRAWS):
        sampled_groups = [word_groups[i] for i in rng.integers(0, len(word_groups), len(word_groups))]
        sampled_lineages = [lineages[i] for i in rng.integers(0, len(lineages), len(lineages))]
        draws[index] = estimate(sampled_groups, sampled_lineages)
    return {
        "estimand": "g0 absolute error minus g1 absolute error in percentage points",
        "point_pp": point,
        "ci95_pp": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "clusters": "reverse-word group and training lineage; epoch snapshots retained within lineage",
    }


def per_environment_regret_fraction(rows: list[dict]) -> dict:
    g0, target, words, _ = matrices(rows, "g0_source_word")
    g1, _, _, _ = matrices(rows, "g1_plus_hazard")
    oracle = target.max(axis=1)
    g0_regret = oracle - target[np.arange(len(words)), g0.argmax(axis=1)]
    g1_regret = oracle - target[np.arange(len(words)), g1.argmax(axis=1)]
    return {
        "strictly_improved_fraction": float(np.mean(g1_regret < g0_regret)),
        "not_worse_fraction": float(np.mean(g1_regret <= g0_regret)),
    }


def order_diagnostic(rows: list[dict]) -> dict:
    lookup = {(row["model_id"], row["word"]): row for row in rows}
    records = []
    for row in rows:
        reverse = ">".join(reversed(row["word"].split(">")))
        key = (row["model_id"], reverse)
        if key not in lookup or row["word"] >= reverse:
            continue
        other = lookup[key]
        truth = row["true_accuracy"] - other["true_accuracy"]
        records.append({
            "ordered_error": abs(
                (row["hazard_ordered_q"] - other["hazard_ordered_q"]) - truth
            ),
            "symmetric_error": abs(
                (row["hazard_symmetric_q"] - other["hazard_symmetric_q"]) - truth
            ),
            "ordered_sign": np.sign(row["hazard_ordered_q"] - other["hazard_ordered_q"])
            == np.sign(truth),
            "measurable": abs(truth) >= 0.01,
        })
    measurable = [row for row in records if row["measurable"]]
    return {
        "model_reverse_pairs": len(records),
        "measurable_pairs": len(measurable),
        "ordered_reversal_mae_pp": 100.0 * float(np.mean([row["ordered_error"] for row in records])),
        "symmetric_reversal_mae_pp": 100.0 * float(np.mean([row["symmetric_error"] for row in records])),
        "ordered_measurable_sign_accuracy": (
            float(np.mean([row["ordered_sign"] for row in measurable])) if measurable else None
        ),
    }


def analyze(
    forecast_path: Path = RESULTS / "confirmation-forecast.csv",
    output_path: Path = RESULTS / "PHASE2_PILOT_ANALYSIS.json",
) -> dict:
    rows = load_rows(forecast_path)
    if len(rows) != 90 * 30:
        raise ValueError(f"expected 2700 sealed rows, got {len(rows)}")
    metrics = {method: method_metrics(rows, method) for method in METHODS}
    bootstrap = bootstrap_increment(rows)
    regret = per_environment_regret_fraction(rows)
    g0, g1 = metrics["g0_source_word"], metrics["g1_plus_hazard"]
    incremental_gate = (
        g1["accuracy_mae_pp"] < g0["accuracy_mae_pp"]
        and bootstrap["ci95_pp"][0] > 0
        and (g1["source_family_controlled_spearman"] or -1)
        > (g0["source_family_controlled_spearman"] or -1)
        and g1["top1_selection_regret_pp"] <= g0["top1_selection_regret_pp"]
    )
    result = {
        "protocol": PROTOCOL_ID,
        "status": "PILOT_INCREMENTAL_GATE_PASSED" if incremental_gate else "PILOT_INCREMENTAL_GATE_FAILED",
        "scope": "CPU development pilot; not the roadmap's 75-model final Phase-2 bank",
        "sealed_rows": len(rows),
        "confirmation_models": 30,
        "confirmation_lineages": 15,
        "confirmation_words": 90,
        "methods": metrics,
        "incremental_value_bootstrap": bootstrap,
        "selection_regret_comparison": regret,
        "order_diagnostic": order_diagnostic(rows),
        "gates": {
            "g1_mae_better_than_g0": g1["accuracy_mae_pp"] < g0["accuracy_mae_pp"],
            "clustered_ci_excludes_zero": bootstrap["ci95_pp"][0] > 0,
            "g1_controlled_rank_better_than_g0": (
                (g1["source_family_controlled_spearman"] or -1)
                > (g0["source_family_controlled_spearman"] or -1)
            ),
            "g1_top1_regret_not_worse": (
                g1["top1_selection_regret_pp"] <= g0["top1_selection_regret_pp"]
            ),
            "pilot_incremental_gate_passed": incremental_gate,
        },
    }
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return result


if __name__ == "__main__":
    analyze()
