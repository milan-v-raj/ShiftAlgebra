#!/usr/bin/env python3
"""Post-outcome sensitivity excluding redundant balanced accuracy."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "v4/phase2g"
RESULTS = BASE / "results"
METRICS = ("margin", "feature_drift", "ece", "accuracy", "mean_confidence")
RANDOM = tuple(f"random_{index:02d}" for index in range(20))
DRAWS = 10_000
SEED = 22714


def run():
    rows = list(csv.DictReader((RESULTS / "confirmation-forecast.csv").open()))
    scales = json.loads((ROOT / "v4/phase2f/FROZEN_FULL_VECTOR_METHODS.json").read_text())[
        "target_scales"
    ]
    primary = json.loads((RESULTS / "PHASE2G_ANALYSIS.json").read_text())

    def error(row, method):
        return float(np.mean([
            abs(float(row[f"{method}_{metric}"]) - float(row[f"true_{metric}"])) / scales[metric]
            for metric in METRICS
        ]))

    methods = ("g0", "full15", "magnitude5", *RANDOM)
    means = {
        method: float(np.mean([error(row, method) for row in rows])) for method in methods
    }

    def bootstrap(method_a, method_b="", random_expected=False):
        grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for row in rows:
            comparator = (
                np.mean([error(row, method) for method in RANDOM])
                if random_expected else error(row, method_b)
            )
            grouped[row["family"]][row["reverse_group"]][row["lineage_id"]].append(
                comparator - error(row, method_a)
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
        rng = np.random.default_rng(SEED + int(random_expected))
        draws = []
        for _ in range(DRAWS):
            sampled_families = [families[i] for i in rng.integers(0, len(families), len(families))]
            sampled_groups = [groups[i] for i in rng.integers(0, len(groups), len(groups))]
            sampled_lineages = [
                [lineages[family][i] for i in rng.integers(0, len(lineages[family]), len(lineages[family]))]
                for family in sampled_families
            ]
            draws.append(estimate(sampled_families, sampled_groups, sampled_lineages))
        return {
            "point": point,
            "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
            "draws": DRAWS,
        }

    def method_record(method):
        key = "methods" if method in ("g0", "full15", "magnitude5") else "random_methods"
        return primary[key][method]

    def rank(method):
        return float(np.mean([
            method_record(method)["per_metric"][metric]["controlled_model_rank_spearman"]
            for metric in METRICS
        ]))

    full_improvement = bootstrap("full15", "g0")
    magnitude_improvement = bootstrap("magnitude5", random_expected=True)
    metric_improvements = sum(
        primary["methods"]["full15"]["per_metric"][metric]["mae"]
        < primary["methods"]["g0"]["per_metric"][metric]["mae"]
        for metric in METRICS
    )
    family_improvements = {}
    for family in sorted({row["family"] for row in rows}):
        selected = [row for row in rows if row["family"] == family]
        family_improvements[family] = float(np.mean([
            error(row, "g0") - error(row, "full15") for row in selected
        ]))
    families_positive = sum(value > 0 for value in family_improvements.values())
    full_gates = {
        "macro_normalized_mae_better": means["full15"] < means["g0"],
        "hierarchical_ci_excludes_zero": full_improvement["ci95"][0] > 0,
        "at_least_four_metrics_better": metric_improvements >= 4,
        "macro_controlled_rank_better": rank("full15") > rank("g0"),
        "accuracy_regret_not_worse": (
            primary["methods"]["full15"]["accuracy_top1_regret_pp"]
            <= primary["methods"]["g0"]["accuracy_top1_regret_pp"]
        ),
        "at_least_five_families_positive": families_positive >= 5,
    }
    full_gates["full_vector_gate_passed"] = all(full_gates.values())

    random_ranks = np.asarray([rank(method) for method in RANDOM])
    random_regrets = np.asarray([
        primary["random_methods"][method]["accuracy_top1_regret_pp"] for method in RANDOM
    ])
    magnitude_gates = {
        "mae_below_random_mean": bool(
            means["magnitude5"] < np.mean([means[method] for method in RANDOM])
        ),
        "hierarchical_ci_excludes_zero": magnitude_improvement["ci95"][0] > 0,
        "macro_controlled_rank_at_least_random_median": bool(
            rank("magnitude5") >= np.median(random_ranks)
        ),
        "accuracy_regret_no_worse_random_median": bool(
            primary["methods"]["magnitude5"]["accuracy_top1_regret_pp"]
            <= np.median(random_regrets)
        ),
    }
    magnitude_gates["magnitude_gate_passed"] = all(magnitude_gates.values())

    output = {
        "status": "POST_OUTCOME_UNIQUE_METRIC_SENSITIVITY_NO_GATE_CHANGE",
        "metrics": list(METRICS),
        "reason": "balanced accuracy duplicates accuracy on the balanced partition",
        "macro_normalized_mae": {
            method: means[method] for method in ("g0", "full15", "magnitude5")
        },
        "macro_controlled_rank": {
            method: rank(method) for method in ("g0", "full15", "magnitude5")
        },
        "full_improvement": full_improvement,
        "full_metrics_improved": metric_improvements,
        "full_gates": full_gates,
        "magnitude_vs_expected_random": magnitude_improvement,
        "magnitude_random_macro_normalized_mae_mean": float(np.mean([
            means[method] for method in RANDOM
        ])),
        "magnitude_random_macro_controlled_rank_median": float(np.median(random_ranks)),
        "magnitude_gates": magnitude_gates,
        "families": family_improvements,
        "families_positive": families_positive,
        "primary_gate_remains": "PASSED" if full_gates["full_vector_gate_passed"] else "FAILED",
        "magnitude_gate_remains": "PASSED" if magnitude_gates["magnitude_gate_passed"] else "FAILED",
    }
    (RESULTS / "POST_OUTCOME_ROBUSTNESS.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    return output


if __name__ == "__main__":
    run()
