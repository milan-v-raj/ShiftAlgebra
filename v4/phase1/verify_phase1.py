#!/usr/bin/env python3
"""Verify Phase-1 infrastructure and any completed development runs."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v4.phase1.corruptions import PRIMITIVES


def validate_result(behavior_path: Path) -> dict:
    with behavior_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    counts = Counter(int(row["length"]) for row in rows)
    expected = {0: 1, 1: 6, 2: 30, 3: 40, 4: 30, 5: 30, 6: 30}
    if counts != expected or len({row["condition_id"] for row in rows}) != 167:
        raise ValueError(f"invalid behavior protocol in {behavior_path}: {counts}")
    prefix = behavior_path.name.removesuffix("-behavior.csv")
    fingerprint_path = behavior_path.with_name(prefix + "-fingerprints.json")
    forecast_path = behavior_path.with_name(prefix + "-forecast.json")
    fingerprints = json.loads(fingerprint_path.read_text())
    if set(fingerprints) != set(PRIMITIVES) or any(len(value) != 301 for value in fingerprints.values()):
        raise ValueError(f"invalid primitive fingerprints in {fingerprint_path}")
    forecast = json.loads(forecast_path.read_text())
    if forecast["calibration_pairs"] != 30 or forecast["target_words"] != 130 \
            or forecast["target_access_during_fit"]:
        raise ValueError(f"invalid forecast access/split in {forecast_path}")
    return {
        "dataset": rows[0]["dataset"],
        "model_id": rows[0]["model_id"],
        "split": "confirm" if "-confirm-behavior.csv" in behavior_path.name else "dev",
        "samples": int(rows[0]["sample_count"]),
        "conditions": len(rows),
        "best_accuracy_method": min(
            forecast["methods"], key=lambda name: forecast["methods"][name]["accuracy_mae_pp"]
        ),
        "accuracy_mae_pp": {
            name: values["accuracy_mae_pp"] for name, values in forecast["methods"].items()
        },
    }


def main() -> None:
    tests = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "v4/phase1/tests", "-v"],
        cwd=ROOT,
        check=False,
    )
    if tests.returncode:
        raise SystemExit(tests.returncode)
    completed = [
        validate_result(path)
        for path in sorted((ROOT / "v4/phase1/results").glob("*-behavior.csv"))
    ]
    confirmatory = [run for run in completed if run["split"] == "confirm"]
    gate_passed = bool(confirmatory) and all(
        run["best_accuracy_method"] == "potential_algebra" for run in confirmatory
    )
    summary = {
        "phase": 1,
        "status": (
            "COMPLETE_GATE_PASSED" if gate_passed
            else "COMPLETE_METHOD_GATE_FAILED" if len(confirmatory) == 2
            else "DEVELOPMENT_ACTIVE"
        ),
        "tests": "PASS",
        "completed_runs": completed,
        "confirmatory_run_opened": bool(confirmatory),
        "method_gate_passed": gate_passed,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
