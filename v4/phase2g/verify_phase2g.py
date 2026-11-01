#!/usr/bin/env python3
"""Verify Phase-2G freeze, strict transfer boundary, and optional outcomes."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v4.phase2g import METRICS
from v4.phase2g.manifest import EXPECTED_DEVELOPMENT_SHA256, EXPECTED_METHOD_SHA256

BASE = ROOT / "v4/phase2g"
RESULTS = BASE / "results"
FROZEN = ROOT / "v4/phase2f/FROZEN_FULL_VECTOR_METHODS.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    checks = []
    manifest = json.loads((BASE / "PHASE2G_MANIFEST.json").read_text())
    assert manifest["metrics"] == list(METRICS)
    assert manifest["counts"] == {
        "calibration_conditions": 37,
        "families": 6,
        "lineages": 18,
        "recipes": 3,
        "snapshots": 36,
        "target_words": 60,
    }
    assert not manifest["architecture_boundary"]["family_overlap"]
    target_words = {
        record["word_string"] for record in manifest["words"]
        if record["role"] == "confirmation"
    }
    assert len(target_words) == 60
    checks.append("manifest, held-out families, and target counts")

    frozen = json.loads(FROZEN.read_text())
    assert sha256(FROZEN) == EXPECTED_METHOD_SHA256
    assert frozen["development_behavior_sha256"] == EXPECTED_DEVELOPMENT_SHA256
    assert manifest["frozen_method"]["allowed_change"].startswith("none")
    assert len([name for name in frozen["designs"] if name.startswith("random_")]) == 20
    checks.append("byte-frozen no-refit Phase-2F method boundary")

    partition = manifest["image_partition"]
    assert partition["offset"] == 40 and partition["per_class"] == 20
    provenance = manifest["data_provenance"]
    assert provenance["canonical_test_md5"] == "f0ef6b0ae62326f3e7ffdfab6717acfc"
    assert provenance["canonical_test_pixel_matches"] == 10_000
    checks.append("fresh image partition and canonical test provenance")

    training_path = RESULTS / "confirmation-training.json"
    if training_path.exists():
        training = json.loads(training_path.read_text())
        assert training["status"] == "CONFIRMATION_ARCHITECTURE_BANK_TRAINED_BEHAVIOR_UNOPENED"
        assert training["lineages"] == 18 and training["models"] == 36
        histories = [history for summary in training["summaries"] for history in summary["history"]]
        hashes = [history["sha256"] for history in histories]
        assert len(hashes) == 36 and len(set(hashes)) == 36
        for history in histories:
            assert sha256(ROOT / history["checkpoint"]) == history["sha256"]
        checks.append("training record and checkpoint hashes")

    behavior_path = RESULTS / "confirmation-behavior.csv"
    if behavior_path.exists():
        behavior = list(csv.DictReader(behavior_path.open()))
        metadata = json.loads((RESULTS / "confirmation-evaluation.json").read_text())
        assert len(behavior) == 3492
        assert sha256(behavior_path) == metadata["sha256"]
        assert len({row["lineage_id"] for row in behavior}) == 18
        assert len({row["family"] for row in behavior}) == 6
        checks.append("confirmation behavior accounting and hash")

    forecast_path = RESULTS / "confirmation-forecast.csv"
    if forecast_path.exists():
        rows = list(csv.DictReader(forecast_path.open()))
        metadata = json.loads((RESULTS / "forecast.json").read_text())
        assert len(rows) == 2160 and sha256(forecast_path) == metadata["sha256"]
        assert metadata["frozen_method_sha256"] == EXPECTED_METHOD_SHA256
        for metric in METRICS:
            for method in ("g0", "full15", "magnitude5"):
                values = np.asarray([float(row[f"{method}_{metric}"]) for row in rows])
                assert np.isfinite(values).all() and (values >= 0).all()
                assert metric == "feature_drift" or (values <= 1).all()
        analysis = json.loads((RESULTS / "PHASE2G_ANALYSIS.json").read_text())
        for metric in METRICS:
            for method in ("g0", "full15", "magnitude5"):
                mae = np.mean([
                    abs(float(row[f"{method}_{metric}"]) - float(row[f"true_{metric}"]))
                    for row in rows
                ])
                assert abs(mae - analysis["methods"][method]["per_metric"][metric]["mae"]) < 1e-12
        assert analysis["metrics"] == list(METRICS)
        checks.append("all six forecast coordinates and analysis arithmetic")
    return checks


if __name__ == "__main__":
    completed = verify()
    print(f"Phase-2G verification: {len(completed)}/{len(completed)} passed")
    for check in completed:
        print("PASS", check)
