#!/usr/bin/env python3
"""Preregistered strict no-refit architecture-transfer analysis."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from v4.analysis.metrics import model_selection_regret, partial_spearman, rank_models_per_environment
from v4.phase2g import METRICS, PROTOCOL_ID

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "v4/phase2g"
RESULTS = BASE / "results"
FROZEN = ROOT / "v4/phase2f/FROZEN_FULL_VECTOR_METHODS.json"
RANDOM = tuple(f"random_{index:02d}" for index in range(20))
DRAWS = 10_000
SEED = 22712


def load_rows():
    output = []
    for raw in csv.DictReader((RESULTS / "confirmation-forecast.csv").open()):
        row = dict(raw)
        for name in ("seed", "epoch", "parameter_count", "length"):
            row[name] = int(row[name])
        for name in list(row):
            if name.startswith("true_") or any(
                name.startswith(f"{method}_")
                for method in ("g0", "full15", "magnitude5", *RANDOM)
            ):
                row[name] = float(row[name])
        output.append(row)
    if len(output) != 2160:
        raise ValueError(f"expected 2160 rows, observed {len(output)}")
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
            np.log1p(lookup[model]["parameter_count"]),
            lookup[model]["epoch"],
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
    per_metric = {
        metric: metric_summary(rows, method, metric) for metric in METRICS
    }
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


def row_error(row, method, scales):
    return float(np.mean([
        abs(row[f"{method}_{metric}"] - row[f"true_{metric}"]) / scales[metric]
        for metric in METRICS
    ]))


def bootstrap(rows, method_a, method_b, scales, random_expected=False, seed=SEED):
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        error_a = row_error(row, method_a, scales)
        error_b = (
            np.mean([row_error(row, method, scales) for method in RANDOM])
            if random_expected else row_error(row, method_b, scales)
        )
        grouped[row["family"]][row["reverse_group"]][row["lineage_id"]].append(
            error_b - error_a
        )
    families = sorted(grouped)
    groups = sorted({row["reverse_group"] for row in rows})
    lineages = {
        family: sorted({row["lineage_id"] for row in rows if row["family"] == family})
        for family in families
    }

    def estimate(sampled_families, sampled_groups, sampled_lineages):
        return float(np.mean([
            np.mean(grouped[family][group][lineage])
            for family_index, family in enumerate(sampled_families)
            for group in sampled_groups
            for lineage in sampled_lineages[family_index]
        ]))

    point = estimate(families, groups, [lineages[family] for family in families])
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
    return {
        "point_normalized_mae": point,
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "draws": DRAWS,
        "seed": seed,
        "hierarchy": "architecture family -> lineage; crossed with reversal group",
    }


def analyze():
    rows = load_rows()
    scales = json.loads(FROZEN.read_text())["target_scales"]
    names = ("g0", "full15", "magnitude5", *RANDOM)
    summaries = {name: summary(rows, name, scales) for name in names}
    primary_increment = bootstrap(rows, "full15", "g0", scales)
    metrics_improved = sum(
        summaries["full15"]["per_metric"][metric]["mae"]
        < summaries["g0"]["per_metric"][metric]["mae"]
        for metric in METRICS
    )
    family_improvements = {}
    for family in sorted({row["family"] for row in rows}):
        selected = [row for row in rows if row["family"] == family]
        family_improvements[family] = float(np.mean([
            row_error(row, "g0", scales) - row_error(row, "full15", scales)
            for row in selected
        ]))
    families_positive = sum(value > 0 for value in family_improvements.values())
    primary_gates = {
        "macro_normalized_mae_better": summaries["full15"]["macro_normalized_mae"] < summaries["g0"]["macro_normalized_mae"],
        "hierarchical_ci_excludes_zero": primary_increment["ci95"][0] > 0,
        "at_least_four_metrics_better": metrics_improved >= 4,
        "macro_controlled_rank_better": summaries["full15"]["macro_controlled_rank"] > summaries["g0"]["macro_controlled_rank"],
        "accuracy_regret_not_worse": summaries["full15"]["accuracy_top1_regret_pp"] <= summaries["g0"]["accuracy_top1_regret_pp"],
        "at_least_five_families_positive": families_positive >= 5,
    }
    primary_gates = {name: bool(value) for name, value in primary_gates.items()}
    primary_gates["full_vector_gate_passed"] = all(primary_gates.values())

    random_mae = np.asarray([summaries[name]["macro_normalized_mae"] for name in RANDOM])
    random_rank = np.asarray([summaries[name]["macro_controlled_rank"] for name in RANDOM])
    random_regret = np.asarray([summaries[name]["accuracy_top1_regret_pp"] for name in RANDOM])
    magnitude = summaries["magnitude5"]
    magnitude_increment = bootstrap(
        rows, "magnitude5", "", scales, random_expected=True, seed=SEED + 1
    )
    magnitude_gates = {
        "mae_below_random_mean": magnitude["macro_normalized_mae"] < random_mae.mean(),
        "hierarchical_ci_excludes_zero": magnitude_increment["ci95"][0] > 0,
        "macro_controlled_rank_at_least_random_median": magnitude["macro_controlled_rank"] >= np.median(random_rank),
        "accuracy_regret_no_worse_random_median": magnitude["accuracy_top1_regret_pp"] <= np.median(random_regret),
    }
    magnitude_gates = {name: bool(value) for name, value in magnitude_gates.items()}
    magnitude_gates["magnitude_gate_passed"] = all(magnitude_gates.values())

    output = {
        "protocol": PROTOCOL_ID,
        "status": (
            "STRICT_NOREFIT_FULL_VECTOR_GATE_PASSED"
            if primary_gates["full_vector_gate_passed"]
            else "STRICT_NOREFIT_FULL_VECTOR_GATE_FAILED"
        ),
        "rows": len(rows),
        "models": 36,
        "lineages": 18,
        "families": 6,
        "words": 60,
        "metrics": list(METRICS),
        "methods": {
            name: summaries[name] for name in ("g0", "full15", "magnitude5")
        },
        "primary_increment": primary_increment,
        "primary_gates": primary_gates,
        "metrics_improved": metrics_improved,
        "family_improvements": family_improvements,
        "families_positive": families_positive,
        "magnitude_confirmation": {
            "gates": magnitude_gates,
            "bootstrap": magnitude_increment,
            "random_macro_normalized_mae_mean": float(random_mae.mean()),
            "random_macro_normalized_mae_median": float(np.median(random_mae)),
            "random_range": [float(random_mae.min()), float(random_mae.max())],
        },
        "random_methods": {name: summaries[name] for name in RANDOM},
        "interpretation_boundary": "strict no-refit architecture transfer within CIFAR-100; not independent-dataset transfer",
    }
    (RESULTS / "PHASE2G_ANALYSIS.json").write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return output


if __name__ == "__main__":
    analyze()
