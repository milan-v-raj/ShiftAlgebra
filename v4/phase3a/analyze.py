#!/usr/bin/env python3
"""Preregistered realistic-corruption full-vector confirmation analysis."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from v4.analysis.metrics import model_selection_regret, partial_spearman, rank_models_per_environment
from v4.phase3a import METRICS, PROTOCOL_ID

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "v4/phase3a"
RESULTS = BASE / "results"
METHOD = BASE / "FROZEN_REALISTIC_METHODS.json"
METHODS = ("g0", "symmetric", "independent", "envelope", "symmetric_formula")
DRAWS = 10_000
SEED = 23702


def load_rows():
    output = []
    for raw in csv.DictReader((RESULTS / "confirmation-forecast.csv").open()):
        row = dict(raw)
        for name in ("seed", "epoch", "parameter_count", "length"):
            row[name] = int(row[name])
        for name in list(row):
            if name.startswith("true_") or any(name.startswith(f"{method}_") for method in METHODS):
                row[name] = float(row[name])
        output.append(row)
    if len(output) != 2160:
        raise ValueError(len(output))
    return output


def matrices(rows, method, metric):
    words = sorted({row["word"] for row in rows})
    models = sorted({row["model_id"] for row in rows})
    lookup = {(row["word"], row["model_id"]): row for row in rows}
    predictions = np.asarray([
        [lookup[word, model][f"{method}_{metric}"] for model in models] for word in words
    ])
    targets = np.asarray([
        [lookup[word, model][f"true_{metric}"] for model in models] for word in words
    ])
    return predictions, targets, models


def controls(rows, models):
    lookup = {row["model_id"]: row for row in rows}
    families = sorted({row["family"] for row in rows})
    recipes = sorted({row["recipe"] for row in rows})
    return np.asarray([
        [
            np.log1p(lookup[model]["parameter_count"]), lookup[model]["epoch"],
            *[float(lookup[model]["family"] == family) for family in families],
            *[float(lookup[model]["recipe"] == recipe) for recipe in recipes],
        ]
        for model in models
    ])


def metric_summary(rows, method, metric):
    predictions, targets, models = matrices(rows, method, metric)
    covariates = controls(rows, models)
    partial = [
        partial_spearman(predictions[index], targets[index], covariates)
        for index in range(len(predictions))
    ]
    finite = [value for value in partial if np.isfinite(value)]
    output = {
        "mae": float(np.mean(abs(predictions - targets))),
        "rank_models_per_environment": float(
            rank_models_per_environment(predictions, targets)["macro_mean"]
        ),
        "controlled_model_rank_spearman": float(np.mean(finite)) if finite else None,
    }
    if metric == "accuracy":
        output["top1_selection_regret_pp"] = model_selection_regret(predictions, targets, 1)
    return output


def summary(rows, method, scales):
    per_metric = {metric: metric_summary(rows, method, metric) for metric in METRICS}
    return {
        "per_metric": per_metric,
        "macro_normalized_mae": float(np.mean([
            per_metric[metric]["mae"] / scales[metric] for metric in METRICS
        ])),
        "macro_controlled_rank": float(np.mean([
            per_metric[metric]["controlled_model_rank_spearman"] for metric in METRICS
        ])),
        "accuracy_top1_regret_pp": per_metric["accuracy"]["top1_selection_regret_pp"],
    }


def row_error(row, method, scales, metrics=METRICS):
    return float(np.mean([
        abs(row[f"{method}_{metric}"] - row[f"true_{metric}"]) / scales[metric]
        for metric in metrics
    ]))


def bootstrap(rows, method_a, method_b, scales, metrics=METRICS, seed=SEED):
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        grouped[row["family"]][row["reverse_group"]][row["lineage_id"]].append(np.asarray([
            (
                abs(row[f"{method_b}_{metric}"] - row[f"true_{metric}"])
                - abs(row[f"{method_a}_{metric}"] - row[f"true_{metric}"])
            ) / scales[metric]
            for metric in metrics
        ]))
    families = sorted(grouped)
    groups = sorted({row["reverse_group"] for row in rows})
    lineages = {
        family: sorted({row["lineage_id"] for row in rows if row["family"] == family})
        for family in families
    }

    def estimate(sampled_families, sampled_groups, sampled_lineages):
        return np.mean([
            np.mean(grouped[family][group][lineage], axis=0)
            for family_index, family in enumerate(sampled_families)
            for group in sampled_groups
            for lineage in sampled_lineages[family_index]
        ], axis=0)

    point_vector = estimate(families, groups, [lineages[family] for family in families])
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(DRAWS):
        sampled_families = [families[index] for index in rng.integers(0, len(families), len(families))]
        sampled_groups = [groups[index] for index in rng.integers(0, len(groups), len(groups))]
        sampled_lineages = [
            [lineages[family][index] for index in rng.integers(0, len(lineages[family]), len(lineages[family]))]
            for family in sampled_families
        ]
        draws.append(estimate(sampled_families, sampled_groups, sampled_lineages))
    draws = np.asarray(draws)
    macro_draws = draws.mean(axis=1)
    return {
        "point": float(point_vector.mean()),
        "ci95": [float(np.quantile(macro_draws, 0.025)), float(np.quantile(macro_draws, 0.975))],
        "per_metric": {
            metric: {
                "point": float(point_vector[index]),
                "ci95": [
                    float(np.quantile(draws[:, index], 0.025)),
                    float(np.quantile(draws[:, index], 0.975)),
                ],
            }
            for index, metric in enumerate(metrics)
        },
        "draws": DRAWS,
        "seed": seed,
        "hierarchy": "architecture family -> lineage; crossed with reversal group",
    }


def reversal_behavior(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model_id"], row["reverse_group"])].append(row)
    if len(grouped) != 1080 or any(len(pair) != 2 for pair in grouped.values()):
        raise ValueError("invalid reversal pairing")
    output = {}
    for metric in METRICS:
        actual = []
        predicted = {method: [] for method in METHODS}
        for pair in grouped.values():
            actual.append(abs(pair[0][f"true_{metric}"] - pair[1][f"true_{metric}"]))
            for method in METHODS:
                predicted[method].append(
                    abs(pair[0][f"{method}_{metric}"] - pair[1][f"{method}_{metric}"])
                )
        output[metric] = {
            "mean_absolute_true_reversal_gap": float(np.mean(actual)),
            "max_absolute_true_reversal_gap": float(np.max(actual)),
            "mean_absolute_predicted_reversal_gap": {
                method: float(np.mean(values)) for method, values in predicted.items()
            },
            "max_absolute_predicted_reversal_gap": {
                method: float(np.max(values)) for method, values in predicted.items()
            },
        }
    return output


def analyze():
    rows = load_rows()
    scales = json.loads(METHOD.read_text())["target_scales"]
    summaries = {method: summary(rows, method, scales) for method in METHODS}
    increment = bootstrap(rows, "symmetric", "g0", scales)
    metrics_improved = sum(
        summaries["symmetric"]["per_metric"][metric]["mae"]
        < summaries["g0"]["per_metric"][metric]["mae"]
        for metric in METRICS
    )
    family_improvements = {}
    for family in sorted({row["family"] for row in rows}):
        selected = [row for row in rows if row["family"] == family]
        family_improvements[family] = float(np.mean([
            row_error(row, "g0", scales) - row_error(row, "symmetric", scales)
            for row in selected
        ]))
    families_positive = sum(value > 0 for value in family_improvements.values())
    gates = {
        "macro_normalized_mae_better_than_g0": (
            summaries["symmetric"]["macro_normalized_mae"] < summaries["g0"]["macro_normalized_mae"]
        ),
        "hierarchical_ci_excludes_zero": increment["ci95"][0] > 0,
        "at_least_four_metrics_better": metrics_improved >= 4,
        "better_than_independent_and_envelope": (
            summaries["symmetric"]["macro_normalized_mae"]
            < min(summaries["independent"]["macro_normalized_mae"], summaries["envelope"]["macro_normalized_mae"])
        ),
        "macro_controlled_rank_better": (
            summaries["symmetric"]["macro_controlled_rank"] > summaries["g0"]["macro_controlled_rank"]
        ),
        "accuracy_regret_not_worse": (
            summaries["symmetric"]["accuracy_top1_regret_pp"] <= summaries["g0"]["accuracy_top1_regret_pp"]
        ),
        "at_least_five_families_positive": families_positive >= 5,
    }
    gates = {name: bool(value) for name, value in gates.items()}
    gates["realistic_composition_gate_passed"] = all(gates.values())
    output = {
        "protocol": PROTOCOL_ID,
        "status": (
            "REALISTIC_COMPOSITION_GATE_PASSED"
            if gates["realistic_composition_gate_passed"] else "REALISTIC_COMPOSITION_GATE_FAILED"
        ),
        "rows": len(rows), "models": 36, "lineages": 18, "families": 6,
        "words": 60, "reversal_groups": 30, "metrics": list(METRICS),
        "methods": summaries,
        "primary_increment": increment,
        "primary_gates": gates,
        "metrics_improved": metrics_improved,
        "family_improvements": family_improvements,
        "families_positive": families_positive,
        "reversal_behavior": reversal_behavior(rows),
        "interpretation_boundary": (
            "clean-room ImageNet-C-style corruptions on CIFAR-100; not official ImageNet-C or ImageNet"
        ),
    }
    (RESULTS / "PHASE3A_ANALYSIS.json").write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return output


if __name__ == "__main__":
    analyze()
