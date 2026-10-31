#!/usr/bin/env python3
"""Verify frozen and completed Phase-2C artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v4.phase2c import PROTOCOL_ID
from v4.phase2c.manifest import fresh_words

BASE = ROOT / "v4/phase2c"
RESULTS = BASE / "results"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def verify() -> list[str]:
    checks = []
    manifest = json.loads((BASE / "PHASE2C_MANIFEST.json").read_text())
    assert manifest["protocol"] == PROTOCOL_ID
    source = ROOT / manifest["source_development"]["path"]
    assert sha256(source) == manifest["source_development"]["sha256"]
    assert len(set(fresh_words())) == 64
    checks.append("manifest, source hash, and fresh-word split")

    frozen = json.loads((BASE / "FROZEN_PAIR_DESIGNS.json").read_text())
    assert frozen["status"] == "PAIR_DESIGNS_FROZEN_FRESH_OUTCOMES_UNOPENED"
    assert len(frozen["designs"]) == 21
    groups = [tuple(sorted(value["pair_groups"])) for value in frozen["designs"].values()]
    assert len(set(groups)) == 21 and all(len(group) == 5 for group in groups)
    checks.append("one active and twenty unique equal-budget random designs")

    behavior_path = RESULTS / "fresh-behavior.csv"
    if behavior_path.exists():
        behavior = list(csv.DictReader(behavior_path.open(newline="")))
        assert len(behavior) == 3030
        summary = json.loads((RESULTS / "fresh-evaluation.json").read_text())
        assert sha256(behavior_path) == summary["behavior_csv_sha256"]
        checks.append("fresh behavior counts and hash")

    forecast_path = RESULTS / "fresh-forecast.csv"
    if forecast_path.exists():
        rows = list(csv.DictReader(forecast_path.open(newline="")))
        assert len(rows) == 1920
        methods = ["active5", "full15", "g1_prior", *[f"random_{i:02d}" for i in range(20)]]
        for method in methods:
            values = np.asarray([float(row[method]) for row in rows])
            assert np.isfinite(values).all() and (values >= 0).all() and (values <= 1).all()
        analysis = json.loads((RESULTS / "PHASE2C_ANALYSIS.json").read_text())
        for method in ("active5", "full15", "g1_prior"):
            recomputed = 100.0 * np.mean([
                abs(float(row[method]) - float(row["true_accuracy"])) for row in rows
            ])
            assert abs(recomputed - analysis["methods"][method]["accuracy_mae_pp"]) < 1e-10
        checks.append("bounded forecasts and primary MAEs recomputed")
    return checks


if __name__ == "__main__":
    checks = verify()
    print(f"Phase-2C verification: {len(checks)}/{len(checks)} passed")
    for check in checks:
        print(f"PASS {check}")
