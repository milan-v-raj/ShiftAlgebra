#!/usr/bin/env python3
"""Analyze the one-shot fresh Phase-2B confirmation."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from v4.analysis.metrics import (
    accuracy_mae_pp,
    accuracy_rmse_pp,
    model_selection_regret,
    near_oracle_selection_rate,
    partial_spearman,
    rank_environments_per_model,
    rank_models_per_environment,
)
from v4.phase2b import PROTOCOL_ID

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "v4/phase2b/results"
METHODS = (
    "expert_dominant", "expert_independent", "expert_symmetric", "expert_ordered",
    "g0_prior", "g1_prior", "achm",
    "achm_budget_0", "achm_budget_5", "achm_budget_10", "achm_budget_15",
)
PRIMARY_COMPARATOR = "expert_ordered"
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 22_303


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            for name in ("seed", "epoch", "parameter_count", "length"):
                row[name] = int(row[name])
            numeric = [
                "true_accuracy", "clean_accuracy", "achm_disagreement", "achm_support_distance",
                *METHODS,
                *[f"achm_weight_{name}" for name in ("dominant", "independent", "symmetric", "ordered")],
            ]
            for name in numeric:
                row[name] = float(row[name])
            rows.append(row)
    if len(rows) != 64 * 30:
        raise ValueError(f"expected 1920 fresh rows, got {len(rows)}")
    return rows


def matrices(rows: list[dict], method: str):
    words = sorted({row["word"] for row in rows})
    models = sorted({row["model_id"] for row in rows})
    lookup = {(row["word"], row["model_id"]): row for row in rows}
    predicted = np.asarray([[lookup[(word, model)][method] for model in models] for word in words])
    target = np.asarray([[lookup[(word, model)]["true_accuracy"] for model in models] for word in words])
    return predicted, target, words, models


def controls(rows: list[dict], models: list[str]) -> np.ndarray:
    metadata = {row["model_id"]: row for row in rows}
    families = sorted({row["family"] for row in rows})
    recipes = sorted({row["recipe"] for row in rows})
    return np.asarray([
        [metadata[m]["clean_accuracy"], np.log1p(metadata[m]["parameter_count"]), metadata[m]["epoch"],
         *[float(metadata[m]["family"] == x) for x in families],
         *[float(metadata[m]["recipe"] == x) for x in recipes]]
        for m in models
    ])


def method_metrics(rows: list[dict], method: str) -> dict:
    predicted, target, words, models = matrices(rows, method)
    model_controls = controls(rows, models)
    partial = [partial_spearman(predicted[i], target[i], model_controls) for i in range(len(words))]
    finite = [value for value in partial if np.isfinite(value)]
    return {
        "accuracy_mae_pp": accuracy_mae_pp(predicted, target),
        "accuracy_rmse_pp": accuracy_rmse_pp(predicted, target),
        "rank_models_per_environment": rank_models_per_environment(predicted, target)["macro_mean"],
        "rank_environments_per_model": rank_environments_per_model(predicted, target)["macro_mean"],
        "controlled_model_rank_spearman": float(np.mean(finite)) if finite else None,
        "top1_selection_regret_pp": model_selection_regret(predicted, target, top_k=1),
        "top3_selection_regret_pp": model_selection_regret(predicted, target, top_k=3),
        "within_1pp_selection_rate": near_oracle_selection_rate(predicted, target, tolerance_pp=1.0),
    }


def clustered_bootstrap(rows: list[dict], method: str, comparator: str) -> dict:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["reverse_group"]][row["lineage_id"]].append(
            abs(row[comparator] - row["true_accuracy"])
            - abs(row[method] - row["true_accuracy"])
        )
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
        sample_groups = [groups[i] for i in rng.integers(0, len(groups), len(groups))]
        sample_lineages = [lineages[i] for i in rng.integers(0, len(lineages), len(lineages))]
        draws[index] = estimate(sample_groups, sample_lineages)
    return {
        "estimand": f"{comparator} absolute error minus {method} absolute error (pp)",
        "point_pp": point,
        "ci95_pp": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "draws": BOOTSTRAP_DRAWS,
        "clusters": "fresh reverse-word group and training lineage",
    }


def uncertainty_metrics(rows: list[dict]) -> dict:
    error = np.asarray([abs(row["achm"] - row["true_accuracy"]) for row in rows])
    uncertainty = np.asarray([row["achm_disagreement"] for row in rows])
    order = np.argsort(uncertainty)
    selective = {}
    for coverage in (1.0, 0.8, 0.6, 0.4):
        count = max(1, int(round(coverage * len(rows))))
        selective[str(coverage)] = 100.0 * float(error[order[:count]].mean())
    correlation = spearmanr(uncertainty, error).statistic
    return {
        "expert_disagreement_error_spearman": float(correlation),
        "selective_mae_pp_by_coverage": selective,
        "monotone_selective_improvement": all(
            selective[str(lower)] <= selective[str(higher)]
            for higher, lower in ((1.0, 0.8), (0.8, 0.6), (0.6, 0.4))
        ),
    }


def weight_summary(rows: list[dict]) -> dict:
    names = ("dominant", "independent", "symmetric", "ordered")
    output = {"overall": {}, "by_length": {}}
    for name in names:
        output["overall"][name] = float(np.mean([row[f"achm_weight_{name}"] for row in rows]))
    for length in (3, 4, 5, 6):
        subset = [row for row in rows if row["length"] == length]
        output["by_length"][str(length)] = {
            name: float(np.mean([row[f"achm_weight_{name}"] for row in subset])) for name in names
        }
    return output


def order_diagnostic(rows: list[dict]) -> dict:
    lookup = {(row["model_id"], row["word"]): row for row in rows}
    achm_error = []
    symmetric_error = []
    signs = []
    measurable = []
    for row in rows:
        reverse = ">".join(reversed(row["word"].split(">")))
        if row["word"] >= reverse:
            continue
        other = lookup[(row["model_id"], reverse)]
        truth = row["true_accuracy"] - other["true_accuracy"]
        achm_difference = row["achm"] - other["achm"]
        symmetric_difference = row["expert_symmetric"] - other["expert_symmetric"]
        achm_error.append(abs(achm_difference - truth))
        symmetric_error.append(abs(symmetric_difference - truth))
        if abs(truth) >= 0.01:
            measurable.append(True)
            signs.append(np.sign(achm_difference) == np.sign(truth))
    return {
        "reverse_pairs": len(achm_error),
        "measurable_pairs": len(measurable),
        "achm_reversal_mae_pp": 100.0 * float(np.mean(achm_error)),
        "symmetric_reversal_mae_pp": 100.0 * float(np.mean(symmetric_error)),
        "achm_measurable_sign_accuracy": float(np.mean(signs)) if signs else None,
    }


def analyze(
    forecast_path: Path = RESULTS / "fresh-forecast.csv",
    output_path: Path = RESULTS / "PHASE2B_ANALYSIS.json",
) -> dict:
    rows = load_rows(forecast_path)
    metrics = {method: method_metrics(rows, method) for method in METHODS}
    bootstrap = clustered_bootstrap(rows, "achm", PRIMARY_COMPARATOR)
    achm = metrics["achm"]
    comparator = metrics[PRIMARY_COMPARATOR]
    gates = {
        "achm_mae_better_than_frozen_ordered_hazard": achm["accuracy_mae_pp"] < comparator["accuracy_mae_pp"],
        "clustered_ci_excludes_zero": bootstrap["ci95_pp"][0] > 0,
        "achm_controlled_rank_better": (
            (achm["controlled_model_rank_spearman"] or -1)
            > (comparator["controlled_model_rank_spearman"] or -1)
        ),
        "achm_top1_regret_not_worse": (
            achm["top1_selection_regret_pp"] <= comparator["top1_selection_regret_pp"]
        ),
    }
    passed = all(gates.values())
    result = {
        "protocol": PROTOCOL_ID,
        "status": "ACHM_PRIMARY_GATE_PASSED" if passed else "ACHM_PRIMARY_GATE_FAILED",
        "scope": "CPU-only Phase-2B fresh confirmation; not final 75-model Phase 2",
        "fresh_rows": len(rows),
        "fresh_models": 30,
        "fresh_lineages": 15,
        "fresh_words": 64,
        "methods": metrics,
        "primary_comparator": PRIMARY_COMPARATOR,
        "primary_bootstrap": bootstrap,
        "gates": {**gates, "primary_gate_passed": passed},
        "active_pair_budget": {
            budget: metrics[f"achm_budget_{budget}"] for budget in ("0", "5", "10", "15")
        },
        "uncertainty": uncertainty_metrics(rows),
        "weights": weight_summary(rows),
        "order_diagnostic": order_diagnostic(rows),
    }
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return result


if __name__ == "__main__":
    analyze()
