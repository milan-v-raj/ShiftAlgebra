"""Adaptive Composition Hazard Mixture (ACHM), fit only on Phase-2 development data."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import minimize

from v4.phase2b import CHANCE, PRIMITIVES, PROTOCOL_ID, Q

EPSILON_RETENTION = 0.02
EXPERTS = ("dominant", "independent", "symmetric", "ordered")
BUDGETS = (0, 5, 10, 15)  # unordered groups; each group costs two directed measurements
L2_GRID = (0.0, 1e-4, 1e-3, 1e-2)
CV_FOLDS = 5
HUBER_DELTA = 0.02


def retention(accuracy: float, clean: float) -> float:
    value = (accuracy - CHANCE) / max(clean - CHANCE, 1e-8)
    return min(2.0, max(EPSILON_RETENTION, value))


def hazard(accuracy: float, clean: float) -> float:
    return -math.log(retention(accuracy, clean))


def accuracy_from_hazard(value: float | np.ndarray, clean: float | np.ndarray):
    output = CHANCE + (np.asarray(clean) - CHANCE) * np.exp(-np.asarray(value))
    return np.clip(output, 0.0, 1.0)


def pair_group(first: str, second: str) -> str:
    return "|".join(sorted((first, second)))


def all_pair_groups() -> tuple[str, ...]:
    return tuple(
        pair_group(PRIMITIVES[i], PRIMITIVES[j])
        for i in range(len(PRIMITIVES)) for j in range(i + 1, len(PRIMITIVES))
    )


def load_behavior(path: Path, allowed_roles: set[str] | None = None) -> dict[str, dict[str, dict]]:
    output: dict[str, dict[str, dict]] = defaultdict(dict)
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            if allowed_roles is not None and raw["word_role"] not in allowed_roles:
                continue
            row = dict(raw)
            for name in ("seed", "epoch", "parameter_count", "condition_index", "length", "sample_count"):
                row[name] = int(row[name])
            for name in ("clean_accuracy", "accuracy", "mean_confidence", "mean_entropy", "atc_accuracy", "doc_accuracy"):
                row[name] = float(row[name])
            output[row["model_id"]][row["word"]] = row
    return dict(output)


@dataclass
class Calibration:
    clean: float
    singles: dict[str, float]
    single_hazards: dict[str, float]
    residuals: dict[tuple[str, str], float]


def calibration(model_rows: dict[str, dict]) -> Calibration:
    clean = model_rows["Identity"]["accuracy"]
    singles = {primitive: model_rows[primitive]["accuracy"] for primitive in PRIMITIVES}
    single_hazards = {primitive: hazard(value, clean) for primitive, value in singles.items()}
    residuals = {}
    for first in PRIMITIVES:
        for second in PRIMITIVES:
            if first == second:
                continue
            pair_hazard = hazard(model_rows[f"{first}>{second}"]["accuracy"], clean)
            residuals[(first, second)] = pair_hazard - single_hazards[first] - single_hazards[second]
    return Calibration(clean, singles, single_hazards, residuals)


def select_pair_groups(behavior: dict[str, dict[str, dict]], budget: int) -> list[str]:
    """Greedy development-only selection balancing effect size and primitive coverage."""
    if budget == 0:
        return []
    groups = all_pair_groups()
    residual_values = defaultdict(list)
    frequency = defaultdict(int)
    for model_rows in behavior.values():
        cal = calibration(model_rows)
        for group in groups:
            first, second = group.split("|")
            symmetric = 0.5 * (cal.residuals[(first, second)] + cal.residuals[(second, first)])
            asymmetric = 0.5 * abs(cal.residuals[(first, second)] - cal.residuals[(second, first)])
            residual_values[group].append(abs(symmetric) + asymmetric)
        for word_string, row in model_rows.items():
            if row["word_role"] != "method_development":
                continue
            word = word_string.split(">")
            for i in range(len(word)):
                for j in range(i + 1, len(word)):
                    frequency[pair_group(word[i], word[j])] += 1
    base = {
        group: float(np.mean(residual_values[group])) * math.sqrt(max(frequency[group], 1))
        for group in groups
    }
    maximum = max(base.values())
    selected: list[str] = []
    covered: set[str] = set()
    while len(selected) < budget:
        def score(group: str) -> tuple[float, str]:
            nodes = set(group.split("|"))
            # Before all primitives are covered, a new endpoint dominates raw
            # effect size. This prevents a small budget from ignoring a whole
            # primitive simply because its development residuals are modest.
            coverage_bonus = 2.0 * maximum * len(nodes - covered)
            return base[group] + coverage_bonus, group
        winner = max((group for group in groups if group not in selected), key=score)
        selected.append(winner)
        covered.update(winner.split("|"))
    return selected


def row_from_word(
    truth_row: dict,
    model_rows: dict[str, dict],
    selected_groups: set[str],
) -> dict:
    cal = calibration(model_rows)
    word = truth_row["word"].split(">")
    pairs = [(word[i], word[j]) for i in range(len(word)) for j in range(i + 1, len(word))]
    n_pairs = len(pairs)
    scale = n_pairs ** (Q - 1.0)
    primitive_hazards = [cal.single_hazards[item] for item in word]
    used_ordered = []
    used_symmetric = []
    asymmetries = []
    available = 0
    for first, second in pairs:
        if pair_group(first, second) in selected_groups:
            ordered = cal.residuals[(first, second)]
            reverse = cal.residuals[(second, first)]
            available += 1
        else:
            ordered = reverse = 0.0
        used_ordered.append(ordered)
        used_symmetric.append(0.5 * (ordered + reverse))
        asymmetries.append(abs(ordered - reverse))
    base = sum(primitive_hazards)
    expert_hazards = np.asarray([
        max(primitive_hazards),
        base,
        base + scale * sum(used_symmetric),
        base + scale * sum(used_ordered),
    ], dtype=float)
    expert_accuracies = accuracy_from_hazard(expert_hazards, cal.clean)
    pair_array = np.asarray(used_ordered or [0.0])
    feature_values = {
        "clean_accuracy": cal.clean,
        "log_parameter_count": math.log1p(truth_row["parameter_count"]),
        "epoch": truth_row["epoch"],
        "length_scaled": len(word) / 6.0,
        "single_hazard_mean": float(np.mean(primitive_hazards)),
        "single_hazard_max": float(np.max(primitive_hazards)),
        "single_hazard_std": float(np.std(primitive_hazards)),
        "pair_residual_mean": float(pair_array.mean()),
        "pair_residual_abs_mean": float(np.abs(pair_array).mean()),
        "pair_residual_std": float(pair_array.std()),
        "pair_asymmetry_mean": float(np.mean(asymmetries or [0.0])),
        "pair_coverage": available / max(n_pairs, 1),
        "expert_disagreement": float(np.std(expert_accuracies)),
    }
    for family in ("shallow", "resnet", "depthwise", "patchconv", "pointwise"):
        feature_values[f"family={family}"] = float(truth_row["family"] == family)
    for recipe in ("erm_flip", "label_smooth", "noise_aug"):
        feature_values[f"recipe={recipe}"] = float(truth_row["recipe"] == recipe)
    return {
        "protocol": PROTOCOL_ID,
        "bank_split": truth_row["bank_split"],
        "model_id": truth_row["model_id"],
        "lineage_id": truth_row["lineage_id"],
        "family": truth_row["family"],
        "recipe": truth_row["recipe"],
        "seed": truth_row["seed"],
        "epoch": truth_row["epoch"],
        "parameter_count": truth_row["parameter_count"],
        "word": truth_row["word"],
        "word_role": truth_row["word_role"],
        "reverse_group": truth_row["reverse_group"],
        "length": truth_row["length"],
        "true_accuracy": truth_row["accuracy"],
        "clean_accuracy": cal.clean,
        "expert_hazards": expert_hazards,
        "expert_accuracies": np.asarray(expert_accuracies),
        "features": feature_values,
    }


def forecast_rows(
    behavior: dict[str, dict[str, dict]], selected_groups: Iterable[str], target_role: str
) -> list[dict]:
    selected = set(selected_groups)
    output = []
    for model_rows in behavior.values():
        for truth_row in model_rows.values():
            if truth_row["word_role"] == target_role:
                output.append(row_from_word(truth_row, model_rows, selected))
    return output


def _schema(rows: list[dict]) -> dict:
    names = list(rows[0]["features"])
    values = np.asarray([[row["features"][name] for name in names] for row in rows], dtype=float)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return {"feature_names": names, "mean": mean, "scale": scale}


def _design(rows: list[dict], schema: dict) -> np.ndarray:
    values = np.asarray([
        [row["features"][name] for name in schema["feature_names"]] for row in rows
    ], dtype=float)
    standardized = (values - schema["mean"]) / schema["scale"]
    return np.column_stack([np.ones(len(rows)), standardized])


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_gate(rows: list[dict], l2: float) -> dict:
    schema = _schema(rows)
    x = _design(rows, schema)
    experts = np.asarray([row["expert_hazards"] for row in rows])
    clean = np.asarray([row["clean_accuracy"] for row in rows])
    target = np.asarray([row["true_accuracy"] for row in rows])
    # Use the ordered expert as a zero-logit reference.  This removes the
    # additive softmax non-identifiability that otherwise makes L-BFGS wander.
    shape = (x.shape[1], len(EXPERTS) - 1)

    def objective(flat: np.ndarray):
        weights = flat.reshape(shape)
        logits = np.column_stack([x @ weights, np.zeros(len(rows))])
        mixture = _softmax(logits)
        predicted_hazard = np.sum(mixture * experts, axis=1)
        predicted = accuracy_from_hazard(predicted_hazard, clean)
        residual = predicted - target
        root = np.sqrt(1.0 + (residual / HUBER_DELTA) ** 2)
        loss = float(np.mean(HUBER_DELTA ** 2 * (root - 1.0)))
        dloss_da = residual / root / len(rows)
        dloss_dh = dloss_da * -(predicted - CHANCE)
        dloss_dlogit = dloss_dh[:, None] * mixture * (experts - predicted_hazard[:, None])
        gradient = x.T @ dloss_dlogit[:, :-1]
        if l2:
            loss += l2 * float(np.mean(weights[1:] ** 2))
            gradient[1:] += (2.0 * l2 / weights[1:].size) * weights[1:]
        return loss, gradient.ravel()

    result = minimize(
        objective,
        np.zeros(np.prod(shape), dtype=float),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-6},
    )
    if not result.success:
        raise RuntimeError(f"ACHM optimization failed: {result.message}")
    return {
        "experts": list(EXPERTS),
        "feature_names": schema["feature_names"],
        "feature_mean": schema["mean"].tolist(),
        "feature_scale": schema["scale"].tolist(),
        "coefficients": result.x.reshape(shape).tolist(),
        "l2": l2,
        "optimization": {"iterations": int(result.nit), "objective": float(result.fun)},
    }


def apply_gate(row: dict, gate: dict) -> dict:
    schema = {
        "feature_names": gate["feature_names"],
        "mean": np.asarray(gate["feature_mean"]),
        "scale": np.asarray(gate["feature_scale"]),
    }
    x = _design([row], schema)
    coefficients = np.asarray(gate["coefficients"])
    logits = np.column_stack([x @ coefficients, np.zeros(1)])
    weights = _softmax(logits)[0]
    predicted_hazard = float(weights @ row["expert_hazards"])
    predicted = float(accuracy_from_hazard(predicted_hazard, row["clean_accuracy"]))
    expert_predictions = row["expert_accuracies"]
    disagreement = float(np.sqrt(weights @ (expert_predictions - predicted) ** 2))
    z = np.abs(x[0, 1:])
    return {
        "prediction": predicted,
        "hazard": predicted_hazard,
        "weights": weights,
        "disagreement": disagreement,
        "support_distance": float(np.max(z)),
    }


def mae_pp(rows: list[dict], gate: dict) -> float:
    return 100.0 * float(np.mean([
        abs(apply_gate(row, gate)["prediction"] - row["true_accuracy"]) for row in rows
    ]))


def cross_validate(rows: list[dict]) -> dict:
    lineages = sorted({row["lineage_id"] for row in rows})
    fold_for = {lineage: index % CV_FOLDS for index, lineage in enumerate(lineages)}
    scores = []
    for l2 in L2_GRID:
        errors = []
        for fold in range(CV_FOLDS):
            train = [row for row in rows if fold_for[row["lineage_id"]] != fold]
            valid = [row for row in rows if fold_for[row["lineage_id"]] == fold]
            gate = fit_gate(train, l2)
            errors.extend(
                abs(apply_gate(row, gate)["prediction"] - row["true_accuracy"]) for row in valid
            )
        scores.append({"l2": l2, "lineage_cv_mae_pp": 100.0 * float(np.mean(errors))})
    winner = min(scores, key=lambda item: (item["lineage_cv_mae_pp"], -item["l2"]))
    return {"folds": CV_FOLDS, "scores": scores, "selected_l2": winner["l2"]}


def freeze_method(
    behavior_path: Path,
    output_path: Path,
    development_forecast_path: Path,
) -> dict:
    # The role filter is a hard information boundary: opened Phase-2 confirmation rows are never loaded.
    behavior = load_behavior(behavior_path, {"calibration", "method_development"})
    if len(behavior) != 30:
        raise ValueError(f"expected 30 development snapshots, found {len(behavior)}")
    frozen_budgets = {}
    all_development_rows = []
    for budget in BUDGETS:
        selected = select_pair_groups(behavior, budget)
        rows = forecast_rows(behavior, selected, "method_development")
        if len(rows) != 1200:
            raise ValueError(f"expected 1200 development rows at budget {budget}, got {len(rows)}")
        cv = cross_validate(rows)
        gate = fit_gate(rows, cv["selected_l2"])
        frozen_budgets[str(budget)] = {
            "pair_groups": selected,
            "directed_pair_measurements": 2 * budget,
            "cv": cv,
            "gate": gate,
            "development_mae_pp": mae_pp(rows, gate),
        }
        if budget == 15:
            all_development_rows = rows
    payload = {
        "protocol": PROTOCOL_ID,
        "status": "ACHM_FROZEN_FRESH_CONFIRMATION_UNOPENED",
        "source_behavior_path": str(behavior_path),
        "source_roles_loaded": ["calibration", "method_development"],
        "forbidden_source_role": "confirmation",
        "q": Q,
        "chance": CHANCE,
        "experts": list(EXPERTS),
        "primary_budget_unordered_groups": 15,
        "budgets": frozen_budgets,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    # Compact human-auditable development predictions for the full-budget primary method.
    gate = frozen_budgets["15"]["gate"]
    development_forecast_path.parent.mkdir(parents=True, exist_ok=True)
    with development_forecast_path.open("w", newline="") as handle:
        fieldnames = [
            "protocol", "model_id", "lineage_id", "word", "true_accuracy",
            *[f"expert_{name}" for name in EXPERTS], "achm", "disagreement", "support_distance",
            *[f"weight_{name}" for name in EXPERTS],
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in all_development_rows:
            result = apply_gate(row, gate)
            writer.writerow({
                "protocol": PROTOCOL_ID,
                "model_id": row["model_id"],
                "lineage_id": row["lineage_id"],
                "word": row["word"],
                "true_accuracy": row["true_accuracy"],
                **{f"expert_{name}": value for name, value in zip(EXPERTS, row["expert_accuracies"])},
                "achm": result["prediction"],
                "disagreement": result["disagreement"],
                "support_distance": result["support_distance"],
                **{f"weight_{name}": value for name, value in zip(EXPERTS, result["weights"])},
            })
    return payload
