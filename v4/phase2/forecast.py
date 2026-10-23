"""Phase-2 accuracy forecasters, extensive hazard algebra, and method freeze."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from v4.phase2 import PRIMITIVES, PROTOCOL_ID
from v4.phase2.train_bank import RESULTS

Q_GRID = tuple(round(0.1 * index, 1) for index in range(21))
RIDGE_ALPHA = 1.0
CHANCE = 0.1
EPSILON_RETENTION = 0.02
PRIMARY = "hazard_ordered_q"


def load_behavior(path: Path) -> dict[str, dict[str, dict]]:
    output: dict[str, dict[str, dict]] = defaultdict(dict)
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            for name in (
                "seed", "epoch", "parameter_count", "condition_index", "length", "sample_count"
            ):
                row[name] = int(row[name])
            for name in (
                "clean_accuracy", "accuracy", "mean_confidence", "mean_entropy",
                "atc_accuracy", "doc_accuracy",
            ):
                row[name] = float(row[name])
            output[row["model_id"]][row["word"]] = row
    return dict(output)


def _retention(accuracy: float, clean: float) -> float:
    value = (accuracy - CHANCE) / max(clean - CHANCE, 1e-8)
    return min(2.0, max(EPSILON_RETENTION, value))


def _hazard(accuracy: float, clean: float) -> float:
    return -math.log(_retention(accuracy, clean))


def _accuracy_from_hazard(hazard: float, clean: float) -> float:
    value = CHANCE + (clean - CHANCE) * math.exp(-hazard)
    return min(1.0, max(0.0, value))


def forecast_one_model(model_rows: dict[str, dict], q: float) -> list[dict]:
    clean_row = model_rows["Identity"]
    clean = clean_row["accuracy"]
    singles = {primitive: model_rows[primitive]["accuracy"] for primitive in PRIMITIVES}
    single_delta = {primitive: singles[primitive] - clean for primitive in PRIMITIVES}
    single_hazard = {primitive: _hazard(singles[primitive], clean) for primitive in PRIMITIVES}
    directed_accuracy_residual = {}
    directed_hazard_residual = {}
    for first in PRIMITIVES:
        for second in PRIMITIVES:
            if first == second:
                continue
            pair_accuracy = model_rows[f"{first}>{second}"]["accuracy"]
            directed_accuracy_residual[(first, second)] = (
                pair_accuracy - clean - single_delta[first] - single_delta[second]
            )
            directed_hazard_residual[(first, second)] = (
                _hazard(pair_accuracy, clean) - single_hazard[first] - single_hazard[second]
            )

    output = []
    for word_string, truth_row in model_rows.items():
        word = tuple(word_string.split(">"))
        if len(word) < 3:
            continue
        pairs = [(word[i], word[j]) for i in range(len(word)) for j in range(i + 1, len(word))]
        pair_count = len(pairs)
        scale = pair_count ** (q - 1.0)
        additive = clean + sum(single_delta[item] for item in word)
        raw_residual = sum(directed_accuracy_residual[pair] for pair in pairs)
        pair_accuracy_q = min(1.0, max(0.0, additive + scale * raw_residual))

        base_hazard = sum(single_hazard[item] for item in word)
        ordered_residual = sum(directed_hazard_residual[pair] for pair in pairs)
        symmetric_residual = sum(
            0.5 * (
                directed_hazard_residual[pair]
                + directed_hazard_residual[(pair[1], pair[0])]
            )
            for pair in pairs
        )
        product = _accuracy_from_hazard(base_hazard, clean)
        ordered = _accuracy_from_hazard(base_hazard + scale * ordered_residual, clean)
        symmetric = _accuracy_from_hazard(base_hazard + scale * symmetric_residual, clean)
        max_damage = min(singles[item] for item in word)
        row = {
            "protocol": PROTOCOL_ID,
            "bank_split": truth_row["bank_split"],
            "model_id": truth_row["model_id"],
            "lineage_id": truth_row["lineage_id"],
            "family": truth_row["family"],
            "recipe": truth_row["recipe"],
            "seed": truth_row["seed"],
            "epoch": truth_row["epoch"],
            "parameter_count": truth_row["parameter_count"],
            "word": word_string,
            "word_role": truth_row["word_role"],
            "reverse_group": truth_row["reverse_group"],
            "length": truth_row["length"],
            "true_accuracy": truth_row["accuracy"],
            "clean_accuracy": clean,
            "primitive_mean": float(np.mean([singles[item] for item in word])),
            "max_damage": max_damage,
            "product_retention": product,
            "pair_accuracy_q": pair_accuracy_q,
            "hazard_symmetric_q": symmetric,
            "hazard_ordered_q": ordered,
            "atc": truth_row["atc_accuracy"],
            "doc": truth_row["doc_accuracy"],
        }
        output.append(row)
    return output


def all_forecasts(behavior: dict[str, dict[str, dict]], q: float) -> list[dict]:
    rows = []
    for model_rows in behavior.values():
        rows.extend(forecast_one_model(model_rows, q))
    return rows


def mae(rows: Iterable[dict], method: str) -> float:
    rows = list(rows)
    return 100.0 * float(np.mean([abs(row[method] - row["true_accuracy"]) for row in rows]))


def select_q(behavior: dict[str, dict[str, dict]]) -> tuple[float, list[dict]]:
    scores = []
    for q in Q_GRID:
        rows = [
            row for row in all_forecasts(behavior, q)
            if row["word_role"] == "method_development"
        ]
        scores.append({
            "q": q,
            "ordered_hazard_mae_pp": mae(rows, "hazard_ordered_q"),
            "symmetric_hazard_mae_pp": mae(rows, "hazard_symmetric_q"),
            "pair_accuracy_mae_pp": mae(rows, "pair_accuracy_q"),
        })
    winner = min(scores, key=lambda row: (row["ordered_hazard_mae_pp"], row["q"]))
    return winner["q"], scores


def _numeric_features(row: dict, include_composition: bool) -> list[float]:
    word = row["word"].split(">")
    counts = [word.count(primitive) for primitive in PRIMITIVES]
    values = [
        row["clean_accuracy"],
        math.log1p(row["parameter_count"]),
        row["epoch"],
        row["length"],
        *counts,
    ]
    if include_composition:
        values.extend([
            row["hazard_ordered_q"],
            row["hazard_symmetric_q"],
            row["product_retention"],
            row["max_damage"],
        ])
    return values


def fit_ridge(rows: list[dict], include_composition: bool) -> dict:
    family_levels = sorted({row["family"] for row in rows})
    recipe_levels = sorted({row["recipe"] for row in rows})
    numeric = np.asarray([_numeric_features(row, include_composition) for row in rows])
    mean = numeric.mean(axis=0)
    scale = numeric.std(axis=0)
    scale[scale < 1e-12] = 1.0
    categories = np.asarray([
        [float(row["family"] == level) for level in family_levels]
        + [float(row["recipe"] == level) for level in recipe_levels]
        for row in rows
    ])
    design = np.column_stack([np.ones(len(rows)), (numeric - mean) / scale, categories])
    target = np.asarray([row["true_accuracy"] for row in rows])
    penalty = np.eye(design.shape[1]) * RIDGE_ALPHA
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    return {
        "include_composition": include_composition,
        "numeric_mean": mean.tolist(),
        "numeric_scale": scale.tolist(),
        "family_levels": family_levels,
        "recipe_levels": recipe_levels,
        "coefficients": coefficients.tolist(),
        "alpha": RIDGE_ALPHA,
    }


def apply_ridge(row: dict, model: dict) -> float:
    numeric = np.asarray(_numeric_features(row, model["include_composition"]))
    standardized = (numeric - np.asarray(model["numeric_mean"])) / np.asarray(model["numeric_scale"])
    categories = np.asarray(
        [float(row["family"] == level) for level in model["family_levels"]]
        + [float(row["recipe"] == level) for level in model["recipe_levels"]]
    )
    design = np.concatenate([[1.0], standardized, categories])
    return min(1.0, max(0.0, float(design @ np.asarray(model["coefficients"]))))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def freeze_development(
    behavior_path: Path = RESULTS / "development-behavior.csv",
    output_path: Path = Path(__file__).resolve().parents[0] / "FROZEN_METHOD.json",
) -> dict:
    behavior = load_behavior(behavior_path)
    q, search = select_q(behavior)
    rows = all_forecasts(behavior, q)
    fit_rows = [row for row in rows if row["word_role"] == "method_development"]
    g0 = fit_ridge(fit_rows, include_composition=False)
    g1 = fit_ridge(fit_rows, include_composition=True)
    for row in rows:
        row["g0_source_word"] = apply_ridge(row, g0)
        row["g1_plus_hazard"] = apply_ridge(row, g1)
    write_csv(RESULTS / "development-forecast.csv", rows)
    frozen = {
        "protocol": PROTOCOL_ID,
        "status": "METHOD_FROZEN_CONFIRMATION_UNOPENED",
        "primary_method": "g1_plus_hazard",
        "primary_comparator": "g0_source_word",
        "hazard_pair_scaling": "n_pairs ** (q - 1) times directed pair-residual sum",
        "selected_q": q,
        "q_grid": list(Q_GRID),
        "q_search": search,
        "ridge_g0": g0,
        "ridge_g1": g1,
        "development_fit_rows": len(fit_rows),
        "development_models": len(behavior),
        "target_access": "method_development words on development models only",
    }
    output_path.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")
    return frozen


def forecast_confirmation(
    behavior_path: Path = RESULTS / "confirmation-behavior.csv",
    frozen_path: Path = Path(__file__).resolve().parents[0] / "FROZEN_METHOD.json",
) -> list[dict]:
    frozen = json.loads(frozen_path.read_text())
    if frozen["protocol"] != PROTOCOL_ID \
            or frozen["status"] != "METHOD_FROZEN_CONFIRMATION_UNOPENED":
        raise ValueError("method is not frozen for confirmation")
    rows = all_forecasts(load_behavior(behavior_path), frozen["selected_q"])
    for row in rows:
        row["g0_source_word"] = apply_ridge(row, frozen["ridge_g0"])
        row["g1_plus_hazard"] = apply_ridge(row, frozen["ridge_g1"])
    write_csv(RESULTS / "confirmation-forecast.csv", rows)
    return rows
