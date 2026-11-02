"""Order-blind composition method for realistic-corruption words."""
from __future__ import annotations

import csv
import math
from collections import defaultdict

import numpy as np

from v4.phase2f import PERFORMANCE_METRICS
from v4.phase2f.method import RECIPES
from v4.phase2g.models import FAMILIES
from v4.phase3a import METRICS
from v4.phase3a.corruptions import PRIMITIVES

ALPHA = 1.0
Q_GRID = tuple(round(0.1 * index, 1) for index in range(21))
CHANCE = 0.01


def pair_group(left, right):
    return "|".join(sorted((left, right)))


def all_groups():
    return tuple(
        pair_group(PRIMITIVES[left], PRIMITIVES[right])
        for left in range(len(PRIMITIVES)) for right in range(left + 1, len(PRIMITIVES))
    )


def load_behavior(path, roles=None):
    output = defaultdict(dict)
    for raw in csv.DictReader(open(path, newline="")):
        if roles and raw["word_role"] not in roles:
            continue
        row = dict(raw)
        for name in ("seed", "epoch", "parameter_count", "condition_index", "length", "sample_count"):
            row[name] = int(row[name])
        for name in METRICS:
            row[name] = float(row[name])
        output[row["model_id"]][row["word"]] = row
    return dict(output)


def clip(metric, value):
    return max(0.0, value) if metric == "feature_drift" else min(1.0, max(0.0, value))


def components(model_rows, truth, metric, groups, q):
    clean = model_rows["Identity"][metric]
    singles = {primitive: model_rows[primitive][metric] for primitive in PRIMITIVES}
    residuals = {}
    for left in PRIMITIVES:
        for right in PRIMITIVES:
            if left != right:
                residuals[left, right] = (
                    model_rows[f"{left}>{right}"][metric]
                    - singles[left] - singles[right] + clean
                )
    # Canonical ordering makes the primary method exactly order-blind, including
    # floating-point accumulation order, rather than merely equal in theory.
    word = sorted(truth["word"].split(">"))
    pairs = [(word[left], word[right]) for left in range(len(word)) for right in range(left + 1, len(word))]
    scale = len(pairs) ** (q - 1)
    symmetric_residuals = [
        0.5 * (residuals[pair] + residuals[pair[1], pair[0]])
        if pair_group(*pair) in groups else 0.0
        for pair in pairs
    ]
    base = clean + sum(singles[primitive] - clean for primitive in word)
    symmetric = clip(metric, base + scale * sum(symmetric_residuals))
    if metric in PERFORMANCE_METRICS:
        floor = CHANCE if metric in ("accuracy", "balanced_accuracy") else 0.0
        denominator = max(clean - floor, 1e-8)
        retentions = [
            min(2.0, max(0.02, (singles[primitive] - floor) / denominator))
            for primitive in word
        ]
        independent = clip(metric, floor + (clean - floor) * float(np.prod(retentions)))
        envelope = min(singles[primitive] for primitive in word)
    else:
        independent = clip(metric, base)
        envelope = max(singles[primitive] for primitive in word)
    return {
        "clean": clean,
        "symmetric_formula": symmetric,
        "independent": independent,
        "envelope": envelope,
    }


def feature_vector(truth, component, include_composition):
    word = truth["word"].split(">")
    features = [
        component["clean"],
        math.log1p(truth["parameter_count"]),
        truth["epoch"],
        len(word),
        *[word.count(primitive) for primitive in PRIMITIVES],
        *[float(truth["family"] == family) for family in FAMILIES],
        *[float(truth["recipe"] == recipe) for recipe in RECIPES],
    ]
    if include_composition:
        features.extend([
            component["symmetric_formula"],
            component["independent"],
            component["envelope"],
        ])
    return features


def make_rows(behavior, role, groups, q_by_metric):
    output = []
    selected = set(groups)
    for model_rows in behavior.values():
        for truth in model_rows.values():
            if truth["word_role"] != role:
                continue
            row = {
                key: truth[key]
                for key in (
                    "model_id", "lineage_id", "family", "recipe", "seed", "epoch",
                    "parameter_count", "word", "reverse_group", "length",
                )
            }
            row["target"] = {}
            row["components"] = {}
            row["features"] = {"g0": {}, "symmetric": {}}
            for metric in METRICS:
                component = components(
                    model_rows, truth, metric, selected, q_by_metric[metric]
                )
                row["target"][metric] = truth[metric]
                row["components"][metric] = component
                row["features"]["g0"][metric] = feature_vector(truth, component, False)
                row["features"]["symmetric"][metric] = feature_vector(truth, component, True)
            output.append(row)
    return output


def fit(rows, metric, mode):
    inputs = np.asarray([row["features"][mode][metric] for row in rows])
    mean = inputs.mean(0)
    scale = inputs.std(0)
    scale[scale < 1e-10] = 1.0
    design = np.column_stack((np.ones(len(inputs)), (inputs - mean) / scale))
    targets = np.asarray([row["target"][metric] for row in rows])
    penalty = np.eye(design.shape[1]) * ALPHA
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
    return {
        "metric": metric,
        "mode": mode,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficients": coefficients.tolist(),
        "alpha": ALPHA,
    }


def apply(row, model):
    metric = model["metric"]
    values = np.asarray(row["features"][model["mode"]][metric])
    design = np.r_[1.0, (values - np.asarray(model["mean"])) / np.asarray(model["scale"])]
    return clip(metric, float(design @ np.asarray(model["coefficients"])))


def select_q(behavior):
    selected = {}
    search = {}
    for metric in METRICS:
        scores = []
        for q in Q_GRID:
            q_values = {name: (q if name == metric else 1.0) for name in METRICS}
            rows = make_rows(behavior, "development", all_groups(), q_values)
            mae = float(np.mean([
                abs(row["components"][metric]["symmetric_formula"] - row["target"][metric])
                for row in rows
            ]))
            scores.append({"q": q, "symmetric_formula_mae": mae})
        winner = min(scores, key=lambda value: (value["symmetric_formula_mae"], value["q"]))
        selected[metric] = winner["q"]
        search[metric] = scores
    return selected, search
