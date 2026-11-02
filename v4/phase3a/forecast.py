#!/usr/bin/env python3
"""Apply frozen order-blind methods to sealed realistic-corruption behavior."""
import csv
import json
from pathlib import Path

from v4.phase2.train_bank import sha256
from v4.phase3a import METRICS, PROTOCOL_ID
from v4.phase3a.method import apply, load_behavior, make_rows

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "v4/phase3a"
RESULTS = BASE / "results"
METHOD = BASE / "FROZEN_REALISTIC_METHODS.json"


def run():
    frozen = json.loads(METHOD.read_text())
    if frozen["status"] != "REALISTIC_METHODS_FROZEN_CONFIRMATION_OUTCOMES_UNOPENED":
        raise ValueError("method not frozen")
    development = ROOT / frozen["development_behavior"]
    if sha256(development) != frozen["development_behavior_sha256"]:
        raise ValueError("development behavior changed after method freeze")
    behavior = load_behavior(
        RESULTS / "confirmation-behavior.csv", {"calibration", "confirmation"}
    )
    rows = make_rows(
        behavior, "confirmation", frozen["pair_groups"], frozen["q_by_metric"]
    )
    output = []
    for row in sorted(rows, key=lambda value: (value["model_id"], value["word"])):
        result = {
            field: row[field]
            for field in (
                "model_id", "lineage_id", "family", "recipe", "seed", "epoch",
                "parameter_count", "word", "reverse_group", "length",
            )
        }
        result["protocol"] = PROTOCOL_ID
        for metric in METRICS:
            result[f"true_{metric}"] = row["target"][metric]
            result[f"g0_{metric}"] = apply(row, frozen["models"]["g0"][metric])
            result[f"symmetric_{metric}"] = apply(
                row, frozen["models"]["symmetric"][metric]
            )
            for formula in ("independent", "envelope", "symmetric_formula"):
                result[f"{formula}_{metric}"] = row["components"][metric][formula]
        output.append(result)
    path = RESULTS / "confirmation-forecast.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)
    metadata = {
        "protocol": PROTOCOL_ID,
        "status": "REALISTIC_CONFIRMATION_FORECAST",
        "rows": len(output),
        "metrics": list(METRICS),
        "frozen_method_sha256": sha256(METHOD),
        "sha256": sha256(path),
    }
    (RESULTS / "forecast.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {len(output)} realistic-corruption forecast rows")
    return output


if __name__ == "__main__":
    run()
