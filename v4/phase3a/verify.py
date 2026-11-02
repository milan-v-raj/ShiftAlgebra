#!/usr/bin/env python3
"""Independent fail-closed Phase-3A chronology and arithmetic audit."""
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v4.phase3a import METRICS, PROTOCOL_ID
from v4.phase3a.manifest import calibration_words, model_records, target_words

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
    checkpoint_paths = [ROOT / item["path"] for item in ledger["checkpoints"]]
    present = sum(path.exists() for path in checkpoint_paths)
    hash_ok = present in (0, len(checkpoint_paths))
    if present == len(checkpoint_paths):
        hash_ok &= all(
            sha256(path) == item["sha256"]
            for path, item in zip(checkpoint_paths, ledger["checkpoints"])
        )
    check(hash_ok, "checkpoint_ledger_and_optional_hashes", checks)
    check({item["model_id"] for item in ledger["checkpoints"]} ==
          {record.model_id for record in expected_records}, "checkpoint_roster", checks)

    behavior_path = RESULTS / "confirmation-behavior.csv"
    forecast_path = RESULTS / "confirmation-forecast.csv"
    evaluation_metadata = json.loads((RESULTS / "confirmation-evaluation.json").read_text())
    forecast_metadata = json.loads((RESULTS / "forecast.json").read_text())
    check(evaluation_metadata["sha256"] == sha256(behavior_path),
          "behavior_hash_matches_evaluation_record", checks)
    check(forecast_metadata["sha256"] == sha256(forecast_path),
          "forecast_hash_matches_record", checks)
    check(forecast_metadata["frozen_method_sha256"] == sha256(BASE / "FROZEN_REALISTIC_METHODS.json"),
          "forecast_method_hash_matches", checks)
    check(method["development_behavior_sha256"] == sha256(RESULTS / "development-behavior.csv"),
          "frozen_development_hash_matches", checks)

    behavior_rows = list(csv.DictReader(behavior_path.open()))
    forecast_rows = list(csv.DictReader(forecast_path.open()))
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

    arithmetic_ok = True
    for name in methods:
        normalized = []
        for metric in METRICS:
            observed = float(np.mean([
                abs(float(row[f"{name}_{metric}"]) - float(row[f"true_{metric}"]))
                for row in forecast_rows
            ]))
            recorded = analysis["methods"][name]["per_metric"][metric]["mae"]
            arithmetic_ok &= abs(observed - recorded) < 1e-12
            normalized.append(observed / method["target_scales"][metric])
        arithmetic_ok &= abs(
            float(np.mean(normalized)) - analysis["methods"][name]["macro_normalized_mae"]
        ) < 1e-12
    check(arithmetic_ok, "independent_mae_recalculation", checks)

    reversal_ok = True
    reversal_pairs = {}
    for row in forecast_rows:
        reversal_pairs.setdefault((row["model_id"], row["reverse_group"]), []).append(row)
    reversal_ok &= len(reversal_pairs) == 1080 and all(
        len(pair) == 2 for pair in reversal_pairs.values()
    )
    reversal_ok &= all(
        float(pair[0][f"symmetric_{metric}"]) == float(pair[1][f"symmetric_{metric}"])
        for pair in reversal_pairs.values() for metric in METRICS
    )
    check(reversal_ok, "symmetric_predictions_exactly_order_blind", checks)

    expected_gates = {
        "macro_normalized_mae_better_than_g0": (
            analysis["methods"]["symmetric"]["macro_normalized_mae"]
            < analysis["methods"]["g0"]["macro_normalized_mae"]
        ),
        "hierarchical_ci_excludes_zero": analysis["primary_increment"]["ci95"][0] > 0,
        "at_least_four_metrics_better": sum(
            analysis["methods"]["symmetric"]["per_metric"][metric]["mae"]
            < analysis["methods"]["g0"]["per_metric"][metric]["mae"]
            for metric in METRICS
        ) >= 4,
        "better_than_independent_and_envelope": (
            analysis["methods"]["symmetric"]["macro_normalized_mae"]
            < min(
                analysis["methods"]["independent"]["macro_normalized_mae"],
                analysis["methods"]["envelope"]["macro_normalized_mae"],
            )
        ),
        "macro_controlled_rank_better": (
            analysis["methods"]["symmetric"]["macro_controlled_rank"]
            > analysis["methods"]["g0"]["macro_controlled_rank"]
        ),
        "accuracy_regret_not_worse": (
            analysis["methods"]["symmetric"]["accuracy_top1_regret_pp"]
            <= analysis["methods"]["g0"]["accuracy_top1_regret_pp"]
        ),
        "at_least_five_families_positive": analysis["families_positive"] >= 5,
    }
    recorded_gates = dict(analysis["primary_gates"])
    recorded_overall = recorded_gates.pop("realistic_composition_gate_passed")
    check(recorded_gates == expected_gates and recorded_overall == all(expected_gates.values()),
          "independent_gate_recalculation", checks)

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
