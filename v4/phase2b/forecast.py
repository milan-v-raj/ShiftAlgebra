#!/usr/bin/env python3
"""Apply the frozen ACHM and prior Phase-2 regressions to fresh outcomes."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from v4.phase2.forecast import apply_ridge
from v4.phase2b import PROTOCOL_ID
from v4.phase2b.method import EXPERTS, apply_gate, forecast_rows, load_behavior

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "v4/phase2b/results"


def run(
    behavior_path: Path = RESULTS / "fresh-behavior.csv",
    frozen_path: Path = ROOT / "v4/phase2b/FROZEN_ACHM.json",
    prior_path: Path = ROOT / "v4/phase2/FROZEN_METHOD.json",
    output_path: Path = RESULTS / "fresh-forecast.csv",
) -> list[dict]:
    frozen = json.loads(frozen_path.read_text())
    if frozen["protocol"] != PROTOCOL_ID or frozen["status"] != "ACHM_FROZEN_FRESH_CONFIRMATION_UNOPENED":
        raise ValueError("ACHM is not in the required frozen state")
    prior = json.loads(prior_path.read_text())
    behavior = load_behavior(behavior_path, {"calibration", "fresh_confirmation"})
    if len(behavior) != 30:
        raise ValueError(f"expected 30 fresh snapshots, found {len(behavior)}")

    by_budget = {}
    for budget, details in frozen["budgets"].items():
        rows = forecast_rows(behavior, details["pair_groups"], "fresh_confirmation")
        by_budget[budget] = {(row["model_id"], row["word"]): row for row in rows}
    keys = sorted(by_budget["15"])
    if len(keys) != 30 * 64:
        raise ValueError(f"expected 1920 fresh rows, found {len(keys)}")

    output = []
    for key in keys:
        primary = by_budget["15"][key]
        primary_result = apply_gate(primary, frozen["budgets"]["15"]["gate"])
        experts = dict(zip(EXPERTS, primary["expert_accuracies"]))
        prior_row = {
            **primary,
            "hazard_ordered_q": experts["ordered"],
            "hazard_symmetric_q": experts["symmetric"],
            "product_retention": experts["independent"],
            "max_damage": experts["dominant"],
        }
        row = {
            "protocol": PROTOCOL_ID,
            "bank_split": primary["bank_split"],
            "model_id": primary["model_id"],
            "lineage_id": primary["lineage_id"],
            "family": primary["family"],
            "recipe": primary["recipe"],
            "seed": primary["seed"],
            "epoch": primary["epoch"],
            "parameter_count": primary["parameter_count"],
            "word": primary["word"],
            "word_role": primary["word_role"],
            "reverse_group": primary["reverse_group"],
            "length": primary["length"],
            "true_accuracy": primary["true_accuracy"],
            "clean_accuracy": primary["clean_accuracy"],
            **{f"expert_{name}": experts[name] for name in EXPERTS},
            "g0_prior": apply_ridge(prior_row, prior["ridge_g0"]),
            "g1_prior": apply_ridge(prior_row, prior["ridge_g1"]),
            "achm": primary_result["prediction"],
            "achm_disagreement": primary_result["disagreement"],
            "achm_support_distance": primary_result["support_distance"],
            **{
                f"achm_weight_{name}": value
                for name, value in zip(EXPERTS, primary_result["weights"])
            },
        }
        for budget, details in frozen["budgets"].items():
            candidate = by_budget[budget][key]
            result = apply_gate(candidate, details["gate"])
            row[f"achm_budget_{budget}"] = result["prediction"]
            row[f"disagreement_budget_{budget}"] = result["disagreement"]
        output.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)
    return output


if __name__ == "__main__":
    rows = run()
    print(f"wrote {len(rows)} fresh forecast rows")
