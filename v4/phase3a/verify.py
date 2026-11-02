#!/usr/bin/env python3
"""Independent fail-closed Phase-3A chronology and arithmetic audit."""
import csv
import hashlib
import json
import math
from pathlib import Path

from v4.phase3a import METRICS, PROTOCOL_ID
from v4.phase3a.manifest import calibration_words, model_records, target_words

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "v4/phase3a"
RESULTS = BASE / "results"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(condition, name, checks):
    checks[name] = bool(condition)


def verify():
    checks = {}
    manifest = json.loads((BASE / "PHASE3A_MANIFEST.json").read_text())
    check(manifest["protocol"] == PROTOCOL_ID, "protocol_identity", checks)
    check(manifest["metrics"] == list(METRICS), "six_metric_contract", checks)
    partitions = manifest["image_partitions"]
    development_offsets = range(
        partitions["development"]["offset"],
        partitions["development"]["offset"] + partitions["development"]["per_class"],
    )
    confirmation_offsets = range(
        partitions["confirmation"]["offset"],
        partitions["confirmation"]["offset"] + partitions["confirmation"]["per_class"],
    )
    check(set(development_offsets).isdisjoint(confirmation_offsets), "image_partition_disjoint", checks)
    words = target_words()
    check(set(words["development"]).isdisjoint(words["confirmation"]),
          "word_partition_disjoint", checks)
    check(len(words["development"]) == len(words["confirmation"]) == 60,
          "word_counts", checks)
    check(len(calibration_words()) == 37, "calibration_count", checks)
    check(len(manifest["corruptions"]["primitives"]) == 6, "corruption_count", checks)
    check("OUTCOMES_UNOPENED" in manifest["status"], "manifest_was_prospective", checks)

    method = json.loads((BASE / "FROZEN_REALISTIC_METHODS.json").read_text())
    check(method["protocol"] == PROTOCOL_ID and "OUTCOMES_UNOPENED" in method["status"],
          "method_was_frozen_prospectively", checks)
    check(method["metrics"] == list(METRICS), "all_metrics_fitted", checks)
    check(set(method["models"]) == {"g0", "symmetric"}, "primary_and_baseline_frozen", checks)
    check(all(set(method["models"][name]) == set(METRICS) for name in method["models"]),
          "six_coordinate_models", checks)

    ledger = json.loads((BASE / "CONFIRMATION_CHECKPOINT_LEDGER.json").read_text())
    expected_records = [record for record in model_records() if record.bank_split == "confirmation"]
    check(ledger["protocol"] == PROTOCOL_ID and len(ledger["checkpoints"]) == 36,
          "ledger_complete", checks)
    check(ledger["behavior_opened"] is False, "ledger_was_pre_outcome", checks)
    hash_ok = True
    for item in ledger["checkpoints"]:
        path = ROOT / item["path"]
        hash_ok &= path.exists() and sha256(path) == item["sha256"]
    check(hash_ok, "checkpoint_hashes", checks)
    check({item["model_id"] for item in ledger["checkpoints"]} ==
          {record.model_id for record in expected_records}, "checkpoint_roster", checks)

    behavior_rows = list(csv.DictReader((RESULTS / "confirmation-behavior.csv").open()))
    forecast_rows = list(csv.DictReader((RESULTS / "confirmation-forecast.csv").open()))
    check(len(behavior_rows) == 3492, "behavior_row_count", checks)
    check(len(forecast_rows) == 2160, "forecast_row_count", checks)
    check(len({(row["model_id"], row["word"]) for row in behavior_rows}) == 3492,
          "behavior_unique_keys", checks)
    check(len({(row["model_id"], row["word"]) for row in forecast_rows}) == 2160,
          "forecast_unique_keys", checks)
    check(all(all(math.isfinite(float(row[metric])) for metric in METRICS)
              for row in behavior_rows), "behavior_finite", checks)
    methods = ("g0", "symmetric", "independent", "envelope", "symmetric_formula")
    check(all(all(math.isfinite(float(row[f"{prefix}_{metric}"]))
                        for metric in METRICS for prefix in ("true", *methods))
              for row in forecast_rows), "forecasts_finite", checks)

    behavior = {(row["model_id"], row["word"]): row for row in behavior_rows}
    target_match = True
    for row in forecast_rows:
        observed = behavior[(row["model_id"], row["word"])]
        target_match &= all(abs(float(row[f"true_{metric}"]) - float(observed[metric])) < 5e-12
                            for metric in METRICS)
    check(target_match, "forecast_targets_match_behavior", checks)

    all_records = model_records()
    development_models = {row.model_id for row in all_records if row.bank_split == "development"}
    confirmation_models = {row.model_id for row in all_records if row.bank_split == "confirmation"}
    check(development_models.isdisjoint(confirmation_models), "model_rosters_disjoint", checks)
    check({row["model_id"] for row in forecast_rows} == confirmation_models,
          "forecast_roster", checks)

    analysis = json.loads((RESULTS / "PHASE3A_ANALYSIS.json").read_text())
    check(analysis["protocol"] == PROTOCOL_ID and analysis["rows"] == 2160,
          "analysis_identity", checks)
    check(set(analysis["methods"]) == set(methods), "all_comparators_reported", checks)
    check(len(analysis["methods"]["symmetric"]["per_metric"]) == 6,
          "analysis_six_metrics", checks)

    report = {
        "protocol": PROTOCOL_ID,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "passed": sum(checks.values()), "total": len(checks), "checks": checks,
    }
    (RESULTS / "VERIFICATION.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)
    return report


if __name__ == "__main__":
    verify()
