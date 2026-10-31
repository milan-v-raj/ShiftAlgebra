#!/usr/bin/env python3
"""Apply frozen active/random pair designs to fresh Phase-2C targets."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from v4.phase2.forecast import apply_ridge
from v4.phase2b.method import EXPERTS, apply_gate, forecast_rows, load_behavior
from v4.phase2c import PROTOCOL_ID

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "v4/phase2c/results"


def run(
    behavior_path: Path = RESULTS / "fresh-behavior.csv",
    frozen_path: Path = ROOT / "v4/phase2c/FROZEN_PAIR_DESIGNS.json",
    prior_path: Path = ROOT / "v4/phase2/FROZEN_METHOD.json",
    output_path: Path = RESULTS / "fresh-forecast.csv",
) -> list[dict]:
    frozen = json.loads(frozen_path.read_text())
    if frozen["protocol"] != PROTOCOL_ID or frozen["status"] != "PAIR_DESIGNS_FROZEN_FRESH_OUTCOMES_UNOPENED":
        raise ValueError("pair designs are not frozen")
    prior = json.loads(prior_path.read_text())
    behavior = load_behavior(behavior_path, {"calibration", "fresh_pair_confirmation"})
    if len(behavior) != 30:
        raise ValueError(f"expected 30 fresh models, got {len(behavior)}")

    design_rows = {}
    for name, design in frozen["designs"].items():
        rows = forecast_rows(behavior, design["pair_groups"], "fresh_pair_confirmation")
        design_rows[name] = {(row["model_id"], row["word"]): row for row in rows}
    full = frozen["full_budget_reference"]
    full_rows_list = forecast_rows(behavior, full["pair_groups"], "fresh_pair_confirmation")
    full_rows = {(row["model_id"], row["word"]): row for row in full_rows_list}
    keys = sorted(full_rows)
    if len(keys) != 1920:
        raise ValueError(f"expected 1920 rows, got {len(keys)}")

    output = []
    for key in keys:
        active_row = design_rows["active"][key]
        active = apply_gate(active_row, frozen["designs"]["active"]["gate"])
        full_row = full_rows[key]
        full_result = apply_gate(full_row, full["gate"])
        expert = dict(zip(EXPERTS, full_row["expert_accuracies"]))
        prior_row = {
            **full_row,
            "hazard_ordered_q": expert["ordered"],
            "hazard_symmetric_q": expert["symmetric"],
            "product_retention": expert["independent"],
            "max_damage": expert["dominant"],
        }
        row = {
            "protocol": PROTOCOL_ID,
            "model_id": active_row["model_id"],
            "lineage_id": active_row["lineage_id"],
            "family": active_row["family"],
            "recipe": active_row["recipe"],
            "seed": active_row["seed"],
            "epoch": active_row["epoch"],
            "parameter_count": active_row["parameter_count"],
            "word": active_row["word"],
            "reverse_group": active_row["reverse_group"],
            "length": active_row["length"],
            "true_accuracy": active_row["true_accuracy"],
            "clean_accuracy": active_row["clean_accuracy"],
            "active5": active["prediction"],
            "active5_disagreement": active["disagreement"],
            "full15": full_result["prediction"],
            "g1_prior": apply_ridge(prior_row, prior["ridge_g1"]),
        }
        for name, design in frozen["designs"].items():
            if name == "active":
                continue
            candidate = design_rows[name][key]
            row[name] = apply_gate(candidate, design["gate"])["prediction"]
        output.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)
    return output


if __name__ == "__main__":
    rows = run()
    print(f"wrote {len(rows)} Phase-2C forecast rows")
