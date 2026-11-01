"""Nested order-blind and order-aware full-vector methods for Phase 2H."""
from __future__ import annotations

import math

import numpy as np

from v4.phase2f import METRICS, PRIMITIVES
from v4.phase2f.method import FAMILIES, RECIPES, clip, components

ALPHA = 1.0


def base_features(truth, clean):
    word = truth["word"].split(">")
    return [
        clean,
        math.log1p(truth["parameter_count"]),
        truth["epoch"],
        len(word),
        *[word.count(primitive) for primitive in PRIMITIVES],
        *[float(truth["family"] == family) for family in FAMILIES],
        *[float(truth["recipe"] == recipe) for recipe in RECIPES],
    ]


def feature_vector(truth, component, mode):
    features = base_features(truth, component["clean"])
    if mode in ("symmetric", "order_aware"):
        features.extend([
            component["symmetric"],
            component["independent"],
            component["envelope"],
        ])
    if mode == "order_aware":
        # The only additional information is the directed-minus-symmetric channel.
        features.append(component["ordered"] - component["symmetric"])
    return features


def make_rows(behavior, role, groups, q_by_metric):
    output = []
    selected_groups = set(groups)
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
            row["features"] = {mode: {} for mode in ("g0", "symmetric", "order_aware")}
            row["components"] = {}
            for metric in METRICS:
                component = components(
                    model_rows, truth, metric, selected_groups, q_by_metric[metric]
                )
                row["target"][metric] = truth[metric]
                row["components"][metric] = component
                for mode in row["features"]:
                    row["features"][mode][metric] = feature_vector(truth, component, mode)
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
    coefficients = np.linalg.solve(
        design.T @ design + penalty, design.T @ targets
    )
    return {
        "metric": metric,
        "mode": mode,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficients": coefficients.tolist(),
        "alpha": ALPHA,
        "feature_contract": (
            "metadata + symmetric/independent/envelope"
            if mode == "symmetric"
            else "metadata + symmetric/independent/envelope + ordered-minus-symmetric"
            if mode == "order_aware"
            else "metadata only"
        ),
    }


def apply(row, model):
    metric = model["metric"]
    values = np.asarray(row["features"][model["mode"]][metric])
    design = np.r_[
        1.0,
        (values - np.asarray(model["mean"])) / np.asarray(model["scale"]),
    ]
    return clip(metric, float(design @ np.asarray(model["coefficients"])))
