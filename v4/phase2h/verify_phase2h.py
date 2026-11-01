#!/usr/bin/env python3
"""Verify Phase-2H freeze, equal-budget nesting, and optional outcomes."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v4.phase2h import METRICS
from v4.phase2h.manifest import EXPECTED_METHOD_SHA256

BASE = ROOT / "v4/phase2h"
RESULTS = BASE / "results"
METHOD = BASE / "FROZEN_ORDER_ABLATION_METHODS.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    checks = []
    manifest = json.loads((BASE / "PHASE2H_MANIFEST.json").read_text())
    assert manifest["metrics"] == list(METRICS)
    assert manifest["counts"] == {
        "calibration_conditions": 37,
        "families": 6,
        "lineages": 18,
        "recipes": 3,
        "reversal_groups": 30,
        "snapshots": 36,
        "target_words": 60,
    }
    target = [row for row in manifest["words"] if row["role"] == "confirmation"]
    assert len(target) == 60 and len({row["reverse_group"] for row in target}) == 30
    assert all(sum(other["reverse_group"] == row["reverse_group"] for other in target) == 2 for row in target)
    checks.append("manifest, fresh bank, and reversal-pair counts")

    frozen = json.loads(METHOD.read_text())
    assert sha256(METHOD) == EXPECTED_METHOD_SHA256
    assert frozen["status"] == "ORDER_ABLATION_METHODS_FROZEN_CONFIRMATION_OUTCOMES_UNOPENED"
    assert frozen["directed_pair_measurements"] == 30
    assert set(frozen["models"]) == {"g0", "symmetric", "order_aware"}
    for metric in METRICS:
        symmetric = frozen["models"]["symmetric"][metric]
        order = frozen["models"]["order_aware"][metric]
        assert len(order["coefficients"]) == len(symmetric["coefficients"]) + 1
    checks.append("frozen nested methods and equal calibration budget")

    partition = manifest["image_partition"]
    assert partition == {"offset": 60, "per_class": 20}
    provenance = manifest["data_provenance"]
    assert provenance["canonical_test_md5"] == "f0ef6b0ae62326f3e7ffdfab6717acfc"
    assert provenance["canonical_test_pixel_matches"] == 10_000
    checks.append("fresh image partition and canonical provenance")

    training_path = RESULTS / "confirmation-training.json"
    if training_path.exists():
        training = json.loads(training_path.read_text())
        assert training["status"] == "CONFIRMATION_BANK_TRAINED_BEHAVIOR_UNOPENED"
        assert training["lineages"] == 18 and training["models"] == 36
        histories = [history for summary in training["summaries"] for history in summary["history"]]
        assert len(histories) == 36 and len({history["sha256"] for history in histories}) == 36
        paths = [ROOT / history["checkpoint"] for history in histories]
        assert sum(path.exists() for path in paths) in (0, 36)
        if all(path.exists() for path in paths):
            for history, path in zip(histories, paths):
                assert sha256(path) == history["sha256"]
        checks.append("training record and checkpoint hash ledger")

    behavior_path = RESULTS / "confirmation-behavior.csv"
    if behavior_path.exists():
        behavior = list(csv.DictReader(behavior_path.open()))
        metadata = json.loads((RESULTS / "confirmation-evaluation.json").read_text())
        assert len(behavior) == 3492 and sha256(behavior_path) == metadata["sha256"]
        assert len({row["lineage_id"] for row in behavior}) == 18
        checks.append("confirmation behavior accounting and hash")

    forecast_path = RESULTS / "confirmation-forecast.csv"
    if forecast_path.exists():
        rows = list(csv.DictReader(forecast_path.open()))
        metadata = json.loads((RESULTS / "forecast.json").read_text())
        assert len(rows) == 2160 and sha256(forecast_path) == metadata["sha256"]
        assert metadata["frozen_method_sha256"] == EXPECTED_METHOD_SHA256
        for metric in METRICS:
            for method in ("g0", "symmetric", "order_aware"):
                values = np.asarray([float(row[f"{method}_{metric}"]) for row in rows])
                assert np.isfinite(values).all() and (values >= 0).all()
                assert metric == "feature_drift" or (values <= 1).all()
        grouped = {}
        for row in rows:
            grouped.setdefault((row["model_id"], row["reverse_group"]), []).append(row)
        assert len(grouped) == 1080 and all(len(pair) == 2 for pair in grouped.values())
        for pair in grouped.values():
            for metric in METRICS:
                assert abs(float(pair[0][f"symmetric_{metric}"]) - float(pair[1][f"symmetric_{metric}"])) < 1e-12
        analysis = json.loads((RESULTS / "PHASE2H_ANALYSIS.json").read_text())
        for metric in METRICS:
            for method in ("g0", "symmetric", "order_aware"):
                mae = np.mean([
                    abs(float(row[f"{method}_{metric}"]) - float(row[f"true_{metric}"]))
                    for row in rows
                ])
                assert abs(mae - analysis["methods"][method]["per_metric"][metric]["mae"]) < 1e-12
        checks.append("order blindness, all six coordinates, and analysis arithmetic")
    return checks


if __name__ == "__main__":
    completed = verify()
    print(f"Phase-2H verification: {len(completed)}/{len(completed)} passed")
    for check in completed:
        print("PASS", check)
