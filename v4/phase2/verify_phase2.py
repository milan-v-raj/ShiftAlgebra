#!/usr/bin/env python3
"""Stage-aware verifier for the Phase-2 CIFAR model-bank pilot."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v4.phase2 import PROTOCOL_ID
from v4.phase2.manifest import model_records, word_records

PHASE = ROOT / "v4/phase2"
RESULTS = PHASE / "results"


def validate_behavior(path: Path, split: str) -> dict:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 30 * 167:
        raise ValueError(f"wrong behavior count in {path}: {len(rows)}")
    if {row["protocol"] for row in rows} != {PROTOCOL_ID}:
        raise ValueError(f"protocol mismatch in {path}")
    per_model = Counter(row["model_id"] for row in rows)
    if set(per_model.values()) != {167} or len(per_model) != 30:
        raise ValueError(f"model/condition mismatch in {path}")
    samples = 500 if split == "development" else 1000
    if {int(row["sample_count"]) for row in rows} != {samples}:
        raise ValueError(f"sample count mismatch in {path}")
    return {"split": split, "rows": len(rows), "models": len(per_model), "samples": samples}


def main() -> None:
    tests = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "v4/phase2/tests", "-v"],
        cwd=ROOT,
        check=False,
    )
    if tests.returncode:
        raise SystemExit(tests.returncode)
    manifest_path = PHASE / "PHASE2_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["protocol"] != PROTOCOL_ID or len(manifest["models"]) != 60 \
            or len(manifest["words"]) != 167:
        raise ValueError("invalid Phase-2 manifest")
    if Counter(row["role"] for row in manifest["words"]) != {
        "calibration": 37,
        "method_development": 40,
        "confirmation": 90,
    }:
        raise ValueError("invalid word-role split")

    completed = []
    for split in ("development", "confirmation"):
        path = RESULTS / f"{split}-behavior.csv"
        if path.is_file():
            completed.append(validate_behavior(path, split))
    frozen_path = PHASE / "FROZEN_METHOD.json"
    frozen = json.loads(frozen_path.read_text()) if frozen_path.is_file() else None
    if frozen is not None and (
        frozen["protocol"] != PROTOCOL_ID
        or frozen["status"] != "METHOD_FROZEN_CONFIRMATION_UNOPENED"
        or frozen["development_fit_rows"] != 1200
    ):
        raise ValueError("invalid frozen Phase-2 method")
    analysis_path = RESULTS / "PHASE2_PILOT_ANALYSIS.json"
    analysis = json.loads(analysis_path.read_text()) if analysis_path.is_file() else None
    confirmation_forecast = RESULTS / "confirmation-forecast.csv"
    if confirmation_forecast.is_file():
        with confirmation_forecast.open(newline="") as handle:
            forecast_rows = list(csv.DictReader(handle))
        if len(forecast_rows) != 30 * 130:
            raise ValueError("confirmation forecast must contain 30 x 130 target rows")
        sealed = [row for row in forecast_rows if row["word_role"] == "confirmation"]
        if len(sealed) != 30 * 90:
            raise ValueError("confirmation forecast must contain 30 x 90 sealed rows")
        if analysis is not None:
            for method in analysis["methods"]:
                recomputed = 100 * sum(
                    abs(float(row[method]) - float(row["true_accuracy"])) for row in sealed
                ) / len(sealed)
                reported = analysis["methods"][method]["accuracy_mae_pp"]
                if not math.isclose(recomputed, reported, rel_tol=0, abs_tol=1e-10):
                    raise ValueError(f"analysis MAE mismatch for {method}")
    if any(row["split"] == "confirmation" for row in completed) and analysis is None:
        if confirmation_forecast.is_file():
            raise ValueError("confirmation forecast exists without final analysis")
    status = (
        analysis["status"] if analysis is not None
        else "METHOD_FROZEN_CONFIRMATION_UNOPENED" if frozen is not None
        else "DEVELOPMENT_OUTCOMES_OPENED" if completed
        else "PREREGISTERED_NO_OUTCOMES"
    )
    print(json.dumps({
        "phase": 2,
        "scope": "CPU_MODEL_BANK_PILOT",
        "protocol": PROTOCOL_ID,
        "status": status,
        "tests": "PASS",
        "manifest_models": len(model_records()),
        "manifest_words": len(word_records()),
        "completed": completed,
        "method_frozen": frozen is not None,
        "gates": analysis["gates"] if analysis is not None else None,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
