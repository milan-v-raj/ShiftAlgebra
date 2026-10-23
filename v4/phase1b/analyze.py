#!/usr/bin/env python3
"""Frozen aggregate analysis and hierarchical paired bootstrap for Phase 1B."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from v4.phase1b import PROTOCOL_ID

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "v4/phase1b/results"
DEFAULT_OUTPUT = RESULTS / "PHASE1B_ANALYSIS.json"
POTENTIAL = "potential_algebra"
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 15_101


def load_forecast(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if not payload["protocol"].startswith(PROTOCOL_ID) or len(payload.get("target_rows", [])) != 130:
        raise ValueError(f"invalid Phase-1B forecast: {path}")
    return payload


def clustered_difference(
    rows: list[dict], comparator: str, rng: np.random.Generator | None = None
) -> float:
    """Comparator minus potential error, macroed over length and word family."""

    by_length: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        difference = (
            row["methods"][comparator]["absolute_error_pp"]
            - row["methods"][POTENTIAL]["absolute_error_pp"]
        )
        by_length[int(row["length"])][row["word_family"]].append(difference)
    length_means = []
    for length in sorted(by_length):
        families = list(by_length[length].values())
        if rng is None:
            sampled = families
        else:
            sampled = [families[index] for index in rng.integers(0, len(families), len(families))]
        length_means.append(float(np.mean([np.mean(values) for values in sampled])))
    return float(np.mean(length_means))


def bootstrap_ci(forecasts: dict[str, dict], comparator: str) -> dict:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    keys = sorted(forecasts)
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for draw in range(BOOTSTRAP_DRAWS):
        sampled_keys = [keys[index] for index in rng.integers(0, len(keys), len(keys))]
        draws[draw] = np.mean(
            [clustered_difference(forecasts[key]["target_rows"], comparator, rng) for key in sampled_keys]
        )
    point = float(np.mean([
        clustered_difference(forecasts[key]["target_rows"], comparator) for key in keys
    ]))
    return {
        "estimand": "comparator MAE minus potential MAE in percentage points",
        "word_clustered_point_pp": point,
        "ci95_pp": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "hierarchy": "dataset -> length -> unordered word family; all available orders retained",
    }


def dynamic_range(forecast: dict) -> dict:
    clean = float(forecast["clean_accuracy"])
    class_count = 10 if forecast["dataset"] == "CIFAR-10" else 47
    chance = 1.0 / class_count
    floor = chance + 0.15 * (clean - chance)
    by_length = {}
    all_target_accuracy = []
    for length in (3, 4, 5, 6):
        values = [
            float(row["true_accuracy"])
            for row in forecast["target_rows"]
            if int(row["length"]) == length
        ]
        all_target_accuracy.extend(values)
        by_length[str(length)] = {
            "mean_accuracy": float(np.mean(values)),
            "minimum_mean_required": floor,
            "passes": float(np.mean(values)) > floor,
        }
    near_chance_fraction = float(np.mean(np.asarray(all_target_accuracy) <= chance + 0.02))
    passed = all(row["passes"] for row in by_length.values()) and near_chance_fraction <= 0.10
    return {
        "clean_accuracy": clean,
        "chance_accuracy": chance,
        "length_mean_floor": floor,
        "by_length": by_length,
        "near_chance_fraction": near_chance_fraction,
        "maximum_near_chance_fraction": 0.10,
        "passed": passed,
    }


def order_diagnostics(forecasts: dict[str, dict]) -> dict:
    records = []
    sign_threshold = {"CIFAR-10": 0.01, "EMNIST": 0.005}
    for forecast in forecasts.values():
        lookup = {tuple(row["word"].split(">")): row for row in forecast["target_rows"]}
        for word, row in lookup.items():
            reverse = tuple(reversed(word))
            if reverse not in lookup or word >= reverse:
                continue
            other = lookup[reverse]
            true_difference = row["true_accuracy"] - other["true_accuracy"]
            entry = {
                "dataset": forecast["dataset"],
                "word": row["word"],
                "reverse": other["word"],
                "true_reversal_difference": true_difference,
                "methods": {},
            }
            for method in ("lp", POTENTIAL):
                predicted_difference = (
                    row["methods"][method]["predicted_accuracy"]
                    - other["methods"][method]["predicted_accuracy"]
                )
                entry["methods"][method] = {
                    "predicted_reversal_difference": predicted_difference,
                    "absolute_difference_error_pp": 100.0 * abs(
                        predicted_difference - true_difference
                    ),
                }
            entry["sign_is_measurable"] = abs(true_difference) >= sign_threshold[forecast["dataset"]]
            records.append(entry)
    summary = {}
    for method in ("lp", POTENTIAL):
        errors = [row["methods"][method]["absolute_difference_error_pp"] for row in records]
        measurable = [row for row in records if row["sign_is_measurable"]]
        sign_correct = []
        for row in measurable:
            truth = np.sign(row["true_reversal_difference"])
            prediction = np.sign(row["methods"][method]["predicted_reversal_difference"])
            sign_correct.append(truth == prediction)
        summary[method] = {
            "reversal_difference_mae_pp": float(np.mean(errors)) if errors else None,
            "measurable_sign_accuracy": float(np.mean(sign_correct)) if sign_correct else None,
            "measurable_pairs": len(measurable),
        }
    passed = bool(records) and (
        summary[POTENTIAL]["reversal_difference_mae_pp"]
        < summary["lp"]["reversal_difference_mae_pp"]
    )
    return {
        "available_reverse_pairs": len(records),
        "summary": summary,
        "potential_beats_lp_on_reversal_difference": passed,
        "records": records,
    }


def analyze(paths: list[Path], output: Path) -> dict:
    forecasts = {}
    for path in paths:
        payload = load_forecast(path)
        forecasts[payload["dataset"]] = payload
    if set(forecasts) != {"CIFAR-10", "EMNIST"}:
        raise ValueError("analysis requires exactly CIFAR-10 and EMNIST forecasts")

    methods = sorted(next(iter(forecasts.values()))["methods"])
    per_dataset = {
        dataset: {
            method: forecast["methods"][method]["accuracy_mae_pp"] for method in methods
        }
        for dataset, forecast in forecasts.items()
    }
    macro = {
        method: float(np.mean([per_dataset[dataset][method] for dataset in forecasts]))
        for method in methods
    }
    baselines = [method for method in methods if method != POTENTIAL]
    comparator = min(baselines, key=lambda method: macro[method])
    improvement = macro[comparator] - macro[POTENTIAL]
    relative_improvement = improvement / macro[comparator]
    bootstrap = bootstrap_ci(forecasts, comparator)
    range_checks = {dataset: dynamic_range(forecast) for dataset, forecast in forecasts.items()}
    range_passed = all(check["passed"] for check in range_checks.values())
    no_material_harm = all(
        per_dataset[dataset][POTENTIAL]
        <= 1.05 * min(per_dataset[dataset][method] for method in baselines)
        for dataset in forecasts
    )
    directional = (
        range_passed
        and improvement > 0
        and bootstrap["ci95_pp"][0] > 0
        and no_material_harm
    )
    strong = directional and relative_improvement >= 0.15
    order = order_diagnostics(forecasts)
    overall = strong and order["potential_beats_lp_on_reversal_difference"]

    result = {
        "protocol": PROTOCOL_ID,
        "status": (
            "COMPLETE_METHOD_GATE_PASSED" if overall
            else "COMPLETE_DIRECTIONAL_ONLY" if directional
            else "COMPLETE_METHOD_GATE_FAILED"
        ),
        "primary_endpoint": "dataset-macro accuracy MAE over 130 length-3--6 words",
        "per_dataset_accuracy_mae_pp": per_dataset,
        "macro_accuracy_mae_pp": macro,
        "best_non_algebraic_r2_baseline": comparator,
        "potential_improvement_pp": improvement,
        "potential_relative_improvement": relative_improvement,
        "bootstrap": bootstrap,
        "dynamic_range": range_checks,
        "order_diagnostics": order,
        "gates": {
            "dynamic_range_passed": range_passed,
            "potential_beats_best_baseline": improvement > 0,
            "clustered_ci_excludes_zero": bootstrap["ci95_pp"][0] > 0,
            "no_dataset_harm_over_5_percent": no_material_harm,
            "directional_gate_passed": directional,
            "relative_improvement_at_least_15_percent": relative_improvement >= 0.15,
            "strong_gate_passed": strong,
            "order_channel_gate_passed": order["potential_beats_lp_on_reversal_difference"],
            "overall_method_gate_passed": overall,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[
            RESULTS / "cifar10-smallcnn-w24-s1301-p1b-confirm-forecast.json",
            RESULTS / "emnist-smallcnn-w16-s1301-p1b-confirm-forecast.json",
        ],
    )
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return result


def main() -> None:
    args = parser().parse_args()
    analyze(args.paths, args.output)


if __name__ == "__main__":
    main()
