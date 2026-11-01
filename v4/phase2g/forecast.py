#!/usr/bin/env python3
"""Apply the byte-frozen Phase-2F methods to Phase-2G without refitting."""
import csv
import json
from pathlib import Path

from v4.phase2.train_bank import sha256
from v4.phase2f.method import apply, load_behavior, rows
from v4.phase2g import METRICS, PROTOCOL_ID
from v4.phase2g.manifest import EXPECTED_DEVELOPMENT_SHA256, EXPECTED_METHOD_SHA256

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "v4/phase2g"
RESULTS = BASE / "results"
FROZEN_PATH = ROOT / "v4/phase2f/FROZEN_FULL_VECTOR_METHODS.json"


def run():
    if sha256(FROZEN_PATH) != EXPECTED_METHOD_SHA256:
        raise ValueError("Phase-2F method changed after transfer freeze")
    frozen = json.loads(FROZEN_PATH.read_text())
    if frozen["development_behavior_sha256"] != EXPECTED_DEVELOPMENT_SHA256:
        raise ValueError("Phase-2F development boundary changed")
    behavior = load_behavior(
        RESULTS / "confirmation-behavior.csv", {"calibration", "confirmation"}
    )
    lookups = {}
    for name, design in frozen["designs"].items():
        transfer_rows = rows(
            behavior, "confirmation", design["pair_groups"], frozen["q_by_metric"]
        )
        lookups[name] = {
            (row["model_id"], row["word"]): row for row in transfer_rows
        }
    baseline = lookups["full15"]
    output = []
    for key in sorted(baseline):
        transfer_row = baseline[key]
        result = {
            field: transfer_row[field]
            for field in (
                "model_id", "lineage_id", "family", "recipe", "seed", "epoch",
                "parameter_count", "word", "reverse_group", "length",
            )
        }
        result["protocol"] = PROTOCOL_ID
        for metric in METRICS:
            result[f"true_{metric}"] = transfer_row["target"][metric]
            result[f"g0_{metric}"] = apply(transfer_row, frozen["g0_models"][metric])
            for name, design in frozen["designs"].items():
                result[f"{name}_{metric}"] = apply(
                    lookups[name][key], design["models"][metric]
                )
        output.append(result)
    path = RESULTS / "confirmation-forecast.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)
    metadata = {
        "protocol": PROTOCOL_ID,
        "status": "STRICT_NOREFIT_FULL_VECTOR_FORECAST",
        "rows": len(output),
        "metrics": list(METRICS),
        "frozen_method_sha256": EXPECTED_METHOD_SHA256,
        "sha256": sha256(path),
    }
    (RESULTS / "forecast.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {len(output)} strict no-refit forecast rows")
    return output


if __name__ == "__main__":
    run()
