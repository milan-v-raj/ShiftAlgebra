#!/usr/bin/env python3
"""Apply frozen order-ablation methods to sealed Phase-2H behavior."""
import csv
import json
from pathlib import Path

from v4.phase2.train_bank import sha256
from v4.phase2f.method import load_behavior
from v4.phase2h import METRICS, PROTOCOL_ID
from v4.phase2h.manifest import EXPECTED_METHOD_SHA256
from v4.phase2h.method import apply, make_rows

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "v4/phase2h"
RESULTS = BASE / "results"
METHOD_PATH = BASE / "FROZEN_ORDER_ABLATION_METHODS.json"


def run():
    if sha256(METHOD_PATH) != EXPECTED_METHOD_SHA256:
        raise ValueError("order-ablation method changed after freeze")
    frozen = json.loads(METHOD_PATH.read_text())
    behavior = load_behavior(
        RESULTS / "confirmation-behavior.csv", {"calibration", "confirmation"}
    )
    rows = make_rows(
        behavior,
        "confirmation",
        frozen["pair_groups"],
        frozen["q_by_metric"],
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
            for mode in ("g0", "symmetric", "order_aware"):
                result[f"{mode}_{metric}"] = apply(
                    row, frozen["models"][mode][metric]
                )
        output.append(result)
    path = RESULTS / "confirmation-forecast.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)
    metadata = {
        "protocol": PROTOCOL_ID,
        "status": "FROZEN_ORDER_ABLATION_FORECAST",
        "rows": len(output),
        "metrics": list(METRICS),
        "frozen_method_sha256": EXPECTED_METHOD_SHA256,
        "sha256": sha256(path),
    }
    (RESULTS / "forecast.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {len(output)} frozen order-ablation rows")
    return output


if __name__ == "__main__":
    run()
