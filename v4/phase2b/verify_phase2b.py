#!/usr/bin/env python3
"""Independent structural and numerical verification for Phase 2B artifacts."""

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

from v4.phase2b import PROTOCOL_ID
from v4.phase2b.manifest import fresh_words
BASE = ROOT / "v4/phase2b"
RESULTS = BASE / "results"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def verify() -> list[str]:
    checks = []
    manifest = json.loads((BASE / "PHASE2B_MANIFEST.json").read_text())
    assert manifest["protocol"] == PROTOCOL_ID
    source = ROOT / manifest["source_development"]["path"]
    assert sha256(source) == manifest["source_development"]["sha256"]
    assert manifest["word_design"]["fresh_confirmation_words"] == 64
    assert len(set(fresh_words())) == 64
    checks.append("manifest/source hash and 64 fresh words")

    frozen = json.loads((BASE / "FROZEN_ACHM.json").read_text())
    assert frozen["status"] == "ACHM_FROZEN_FRESH_CONFIRMATION_UNOPENED"
    assert frozen["source_roles_loaded"] == ["calibration", "method_development"]
    assert set(frozen["budgets"]) == {"0", "5", "10", "15"}
    checks.append("frozen method and information boundary")

    behavior_path = RESULTS / "fresh-behavior.csv"
    if behavior_path.exists():
        with behavior_path.open(newline="") as handle:
            behavior = list(csv.DictReader(handle))
        assert len(behavior) == 30 * 101
        assert sum(row["word_role"] == "fresh_confirmation" for row in behavior) == 30 * 64
        assert len({row["model_id"] for row in behavior}) == 30
        assert len({row["lineage_id"] for row in behavior}) == 15
        summary = json.loads((RESULTS / "fresh-evaluation.json").read_text())
        assert sha256(behavior_path) == summary["behavior_csv_sha256"]
        checks.append("fresh behavior counts and hash")

    forecast_path = RESULTS / "fresh-forecast.csv"
    if forecast_path.exists():
        with forecast_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 30 * 64
        for method in ("achm", "expert_ordered", "g0_prior", "g1_prior"):
            values = np.asarray([float(row[method]) for row in rows])
            assert np.isfinite(values).all() and (values >= 0).all() and (values <= 1).all()
        analysis = json.loads((RESULTS / "PHASE2B_ANALYSIS.json").read_text())
        for method in analysis["methods"]:
            recomputed = 100.0 * np.mean([
                abs(float(row[method]) - float(row["true_accuracy"])) for row in rows
            ])
            assert abs(recomputed - analysis["methods"][method]["accuracy_mae_pp"]) < 1e-10
        checks.append("fresh forecasts bounded and every reported MAE recomputed")

    return checks


if __name__ == "__main__":
    completed = verify()
    print(f"Phase-2B verification: {len(completed)}/{len(completed)} passed")
    for check in completed:
        print(f"PASS {check}")
