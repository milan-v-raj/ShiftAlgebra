#!/usr/bin/env python3
"""Verify Phase-1B preregistration, calibration freeze, results, and gates."""

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
from v4.phase1b import PROTOCOL_ID

PHASE = ROOT / "v4/phase1b"
RESULTS = PHASE / "results"


def validate_complete_run(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload["protocol"] != f"{PROTOCOL_ID}-confirm":
        raise ValueError(f"wrong protocol in {path}")
    if payload["target_access_during_fit"] or len(payload.get("target_rows", [])) != 130:
        raise ValueError(f"target access/row violation in {path}")
    behavior = Path(payload["behavior_csv"])
    with behavior.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {0: 1, 1: 6, 2: 30, 3: 40, 4: 30, 5: 30, 6: 30}
    if Counter(int(row["length"]) for row in rows) != expected:
        raise ValueError(f"condition count violation in {behavior}")
    fingerprints = json.loads(
        behavior.with_name(behavior.name.replace("-behavior.csv", "-fingerprints.json")).read_text()
    )
    if set(fingerprints) != set(PRIMITIVES) or any(len(value) != 301 for value in fingerprints.values()):
        raise ValueError(f"fingerprint violation for {behavior}")
    return {
        "dataset": payload["dataset"],
        "samples": int(rows[0]["sample_count"]),
        "conditions": len(rows),
        "best_method": min(
            payload["methods"], key=lambda name: payload["methods"][name]["accuracy_mae_pp"]
        ),
    }


def main() -> None:
    tests = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "v4/phase1b/tests", "-v"],
        cwd=ROOT,
        check=False,
    )
    if tests.returncode:
        raise SystemExit(tests.returncode)
    protocol = ROOT / "v4/protocol/phase1b_preregistration.md"
    if not protocol.is_file() or PROTOCOL_ID not in protocol.read_text():
        raise ValueError("missing Phase-1B preregistration")

    frozen_path = PHASE / "FROZEN_STRENGTHS.json"
    frozen = None
    if frozen_path.is_file():
        frozen = json.loads(frozen_path.read_text())
        if frozen["protocol"] != PROTOCOL_ID or set(frozen["datasets"]) != {"cifar10", "emnist"}:
            raise ValueError("invalid frozen strengths")
        for entry in frozen["datasets"].values():
            if set(entry["strengths"]) != set(PRIMITIVES):
                raise ValueError("incomplete frozen strengths")

    forecasts = sorted(RESULTS.glob("*-p1b-confirm-forecast.json"))
    completed = [validate_complete_run(path) for path in forecasts]
    analysis_path = RESULTS / "PHASE1B_ANALYSIS.json"
    analysis = json.loads(analysis_path.read_text()) if analysis_path.is_file() else None
    if len(completed) == 2 and analysis is None:
        raise ValueError("two completed runs require frozen aggregate analysis")
    status = (
        analysis["status"] if analysis is not None
        else "STRENGTHS_FROZEN_TARGETS_UNOPENED" if frozen is not None
        else "PREREGISTERED_CALIBRATION_NOT_RUN"
    )
    summary = {
        "phase": "1B",
        "protocol": PROTOCOL_ID,
        "status": status,
        "tests": "PASS",
        "strengths_frozen": frozen is not None,
        "completed_runs": completed,
        "analysis_gates": analysis["gates"] if analysis is not None else None,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
