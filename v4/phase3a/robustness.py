#!/usr/bin/env python3
"""Labeled post-outcome robustness summaries; never replaces the primary gate."""
import json
from pathlib import Path

import numpy as np

from v4.phase3a import METRICS, PROTOCOL_ID
from v4.phase3a.analyze import METHODS, RESULTS, bootstrap, load_rows, row_error

ROOT = Path(__file__).resolve().parents[2]
METHOD = ROOT / "v4/phase3a/FROZEN_REALISTIC_METHODS.json"


def mean_error(rows, method, scales, metrics=METRICS):
    return float(np.mean([row_error(row, method, scales, metrics) for row in rows]))


def paired_slice(rows, scales, metrics=METRICS):
    errors = {method: mean_error(rows, method, scales, metrics) for method in METHODS}
    return {
        "rows": len(rows),
        "macro_normalized_mae": errors,
        "symmetric_improvement_over_g0": errors["g0"] - errors["symmetric"],
    }


def run():
    rows = load_rows()
    scales = json.loads(METHOD.read_text())["target_scales"]
    unique_metrics = tuple(metric for metric in METRICS if metric != "balanced_accuracy")
    output = {
        "protocol": PROTOCOL_ID,
        "status": "POST_OUTCOME_ROBUSTNESS_RETAINED_NO_RETUNING",
        "primary_decision_replaced": False,
        "note": (
            "These declared sensitivities retain the frozen predictions and cannot replace "
            "the six-metric primary decision."
        ),
        "remove_redundant_balanced_accuracy": {
            **paired_slice(rows, scales, unique_metrics),
            "hierarchical_improvement": bootstrap(
                rows, "symmetric", "g0", scales, unique_metrics, seed=23703
            ),
        },
        "by_epoch": {
            str(epoch): paired_slice([row for row in rows if row["epoch"] == epoch], scales)
            for epoch in (1, 2)
        },
        "by_word_length": {
            str(length): paired_slice([row for row in rows if row["length"] == length], scales)
            for length in (4, 5, 6)
        },
        "by_family": {
            family: paired_slice([row for row in rows if row["family"] == family], scales)
            for family in sorted({row["family"] for row in rows})
        },
        "by_recipe": {
            recipe: paired_slice([row for row in rows if row["recipe"] == recipe], scales)
            for recipe in sorted({row["recipe"] for row in rows})
        },
    }
    path = RESULTS / "POST_OUTCOME_ROBUSTNESS.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return output


if __name__ == "__main__":
    run()
