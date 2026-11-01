#!/usr/bin/env python3
"""Preregistered Phase-2H order-awareness analysis."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from v4.analysis.metrics import model_selection_regret, partial_spearman, rank_models_per_environment
from v4.phase2h import METRICS, PROTOCOL_ID

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "v4/phase2h"
RESULTS = BASE / "results"
METHOD = BASE / "FROZEN_ORDER_ABLATION_METHODS.json"
METHODS = ("g0", "symmetric", "order_aware")
DRAWS = 10_000
SEED = 22722


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


def hierarchical_bootstrap(rows, value_function, seed):
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        grouped[row["family"]][row["reverse_group"]][row["lineage_id"]].append(
            value_function(row)
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
        "point": point,
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "draws": DRAWS,
        "seed": seed,
        "hierarchy": "architecture family -> lineage; crossed with reversal group",
    }


def contrast_records(rows, scales, metrics=METRICS):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model_id"], row["reverse_group"])].append(row)
    output = []
    for (model_id, reverse_group), pair in grouped.items():
        if len(pair) != 2:
            raise ValueError((model_id, reverse_group, len(pair)))
        left, right = sorted(pair, key=lambda row: row["word"])
        record = {
            key: left[key]
            for key in ("model_id", "lineage_id", "family", "recipe", "epoch", "reverse_group")
        }
        record["true"] = {}
        record["errors"] = {method: {} for method in ("symmetric", "order_aware")}
        record["predicted"] = {method: {} for method in ("symmetric", "order_aware")}
        for metric in metrics:
            true_difference = left[f"true_{metric}"] - right[f"true_{metric}"]
            record["true"][metric] = true_difference
            for method in ("symmetric", "order_aware"):
                predicted_difference = (
                    left[f"{method}_{metric}"] - right[f"{method}_{metric}"]
                )
                record["predicted"][method][metric] = predicted_difference
                record["errors"][method][metric] = (
                    abs(predicted_difference - true_difference) / scales[metric]
                )
        output.append(record)
    if len(output) != 1080:
        raise ValueError(f"expected 1080 model/group records, observed {len(output)}")
    return output


def contrast_summary(records, metrics=METRICS):
    output = {}
    for method in ("symmetric", "order_aware"):
        per_metric = {
            metric: float(np.mean([record["errors"][method][metric] for record in records]))
            for metric in metrics
        }
        output[method] = {
            "per_metric_normalized_mae": per_metric,
            "macro_normalized_mae": float(np.mean(list(per_metric.values()))),
        }
    true_effect = {
        metric: float(np.mean([abs(record["true"][metric]) for record in records]))
        for metric in metrics
    }
    correlations = {}
    for metric in metrics:
        truth = [record["true"][metric] for record in records]
        predicted = [record["predicted"]["order_aware"][metric] for record in records]
        correlation = spearmanr(truth, predicted).statistic
        correlations[metric] = float(correlation) if np.isfinite(correlation) else None
    output["true_mean_absolute_contrast"] = true_effect
    output["order_aware_contrast_spearman"] = correlations
    return output


def contrast_bootstrap(records, metrics=METRICS):
    pseudo_rows = []
    for record in records:
        symmetric_error = float(np.mean([
            record["errors"]["symmetric"][metric] for metric in metrics
        ]))
        order_error = float(np.mean([
            record["errors"]["order_aware"][metric] for metric in metrics
        ]))
        pseudo_rows.append({
            "family": record["family"],
            "lineage_id": record["lineage_id"],
            "reverse_group": record["reverse_group"],
            "improvement": symmetric_error - order_error,
        })
    return hierarchical_bootstrap(
        pseudo_rows, lambda row: row["improvement"], SEED + 1
    )


def analyze():
    rows = load_rows()
    frozen = json.loads(METHOD.read_text())
    scales = frozen["target_scales"]
    summaries = {method: summary(rows, method, scales) for method in METHODS}
    primary_increment = hierarchical_bootstrap(
        rows,
        lambda row: row_error(row, "symmetric", scales) - row_error(row, "order_aware", scales),
        SEED,
    )
    metrics_improved = sum(
        summaries["order_aware"]["per_metric"][metric]["mae"]
        < summaries["symmetric"]["per_metric"][metric]["mae"]
        for metric in METRICS
    )
    contrast = contrast_records(rows, scales)
    contrast_results = contrast_summary(contrast)
    contrast_increment = contrast_bootstrap(contrast)
    symmetric_contrast_max = max(
        abs(record["predicted"]["symmetric"][metric])
        for record in contrast for metric in METRICS
    )
    family_improvements = {}
    for family in sorted({row["family"] for row in rows}):
        selected = [row for row in rows if row["family"] == family]
        family_improvements[family] = float(np.mean([
            row_error(row, "symmetric", scales) - row_error(row, "order_aware", scales)
            for row in selected
        ]))
    gates = {
        "macro_normalized_mae_better": (
            summaries["order_aware"]["macro_normalized_mae"]
            < summaries["symmetric"]["macro_normalized_mae"]
        ),
        "hierarchical_ci_excludes_zero": primary_increment["ci95"][0] > 0,
        "at_least_four_metrics_better": metrics_improved >= 4,
        "reversal_contrast_mae_better": (
            contrast_results["order_aware"]["macro_normalized_mae"]
            < contrast_results["symmetric"]["macro_normalized_mae"]
        ),
        "macro_controlled_rank_not_worse": (
            summaries["order_aware"]["macro_controlled_rank"]
            >= summaries["symmetric"]["macro_controlled_rank"]
        ),
        "accuracy_regret_not_worse": (
            summaries["order_aware"]["accuracy_top1_regret_pp"]
            <= summaries["symmetric"]["accuracy_top1_regret_pp"]
        ),
    }
    gates = {name: bool(value) for name, value in gates.items()}
    gates["order_awareness_gate_passed"] = all(gates.values())
    output = {
        "protocol": PROTOCOL_ID,
        "status": (
            "ORDER_AWARENESS_GATE_PASSED"
            if gates["order_awareness_gate_passed"] else "ORDER_AWARENESS_GATE_FAILED"
        ),
        "rows": len(rows),
        "models": 36,
        "lineages": 18,
        "families": 6,
        "words": 60,
        "reversal_groups": 30,
        "metrics": list(METRICS),
        "methods": summaries,
        "primary_increment": primary_increment,
        "primary_gates": gates,
        "metrics_improved": metrics_improved,
        "family_improvements": family_improvements,
        "families_positive": sum(value > 0 for value in family_improvements.values()),
        "reversal_contrast": contrast_results,
        "reversal_contrast_increment": contrast_increment,
        "symmetric_max_absolute_predicted_reversal_contrast": symmetric_contrast_max,
        "interpretation_rule": (
            "retain order as an established contribution only if all six primary criteria pass"
        ),
    }
    (RESULTS / "PHASE2H_ANALYSIS.json").write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return output


if __name__ == "__main__":
    analyze()
