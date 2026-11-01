#!/usr/bin/env python3
"""Post-outcome order-ablation sensitivity excluding redundant balanced accuracy."""
import json
from pathlib import Path

import numpy as np

from v4.phase2h import analyze as primary_analysis

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "v4/phase2h"
RESULTS = BASE / "results"
METRICS = ("margin", "feature_drift", "ece", "accuracy", "mean_confidence")


def run():
    rows = primary_analysis.load_rows()
    frozen = json.loads((BASE / "FROZEN_ORDER_ABLATION_METHODS.json").read_text())
    scales = frozen["target_scales"]

    def row_error(row, method):
        return float(np.mean([
            abs(row[f"{method}_{metric}"] - row[f"true_{metric}"]) / scales[metric]
            for metric in METRICS
        ]))

    summaries = {}
    for method in primary_analysis.METHODS:
        per_metric = {
            metric: primary_analysis.metric_summary(rows, method, metric) for metric in METRICS
        }
        summaries[method] = {
            "per_metric": per_metric,
            "macro_normalized_mae": float(np.mean([
                per_metric[metric]["mae"] / scales[metric] for metric in METRICS
            ])),
            "macro_controlled_rank": float(np.mean([
                per_metric[metric]["controlled_model_rank_spearman"] for metric in METRICS
            ])),
            "accuracy_top1_regret_pp": per_metric["accuracy"]["top1_selection_regret_pp"],
        }

    increment = primary_analysis.hierarchical_bootstrap(
        rows,
        lambda row: row_error(row, "symmetric") - row_error(row, "order_aware"),
        22724,
    )
    metrics_improved = sum(
        summaries["order_aware"]["per_metric"][metric]["mae"]
        < summaries["symmetric"]["per_metric"][metric]["mae"]
        for metric in METRICS
    )
    contrast_records = primary_analysis.contrast_records(rows, scales, METRICS)
    contrast = primary_analysis.contrast_summary(contrast_records, METRICS)
    contrast_increment = primary_analysis.contrast_bootstrap(contrast_records, METRICS)
    family_improvements = {}
    for family in sorted({row["family"] for row in rows}):
        selected = [row for row in rows if row["family"] == family]
        family_improvements[family] = float(np.mean([
            row_error(row, "symmetric") - row_error(row, "order_aware")
            for row in selected
        ]))
    gates = {
        "macro_normalized_mae_better": (
            summaries["order_aware"]["macro_normalized_mae"]
            < summaries["symmetric"]["macro_normalized_mae"]
        ),
        "hierarchical_ci_excludes_zero": increment["ci95"][0] > 0,
        "at_least_four_metrics_better": metrics_improved >= 4,
        "reversal_contrast_mae_better": (
            contrast["order_aware"]["macro_normalized_mae"]
            < contrast["symmetric"]["macro_normalized_mae"]
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
        "status": "POST_OUTCOME_UNIQUE_METRIC_SENSITIVITY_GATE_STILL_FAILED",
        "reason": "balanced accuracy duplicates accuracy on the balanced partition",
        "metrics": list(METRICS),
        "methods": summaries,
        "primary_increment": increment,
        "primary_gates": gates,
        "metrics_improved": metrics_improved,
        "reversal_contrast": contrast,
        "reversal_contrast_increment": contrast_increment,
        "family_improvements": family_improvements,
        "families_positive": sum(value > 0 for value in family_improvements.values()),
    }
    (RESULTS / "POST_OUTCOME_ROBUSTNESS.json").write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return output


if __name__ == "__main__":
    run()
