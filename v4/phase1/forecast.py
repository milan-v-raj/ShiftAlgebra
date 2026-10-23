"""Target-free R2 forecasting models and Phase-1 evaluation."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from v4.src.algebra import signed_power_inverse, signed_power_transform

from .behavior import BEHAVIOR_NAMES

Tensor = torch.Tensor
ACC_INDEX = BEHAVIOR_NAMES.index("accuracy")


def _raw_for_p(initial: float = 2.0) -> float:
    return math.log(math.expm1(initial - 1.0))


def _lp(behaviors: Tensor, p: Tensor) -> Tensor:
    return signed_power_inverse(signed_power_transform(behaviors, p).sum(dim=0), p)


class LearnedLp(nn.Module):
    def __init__(self, behavior_dim: int) -> None:
        super().__init__()
        self.raw_p = nn.Parameter(torch.full((behavior_dim,), _raw_for_p(), dtype=torch.float64))

    @property
    def p(self) -> Tensor:
        return 1.0 + F.softplus(self.raw_p)

    def forward(self, behaviors: Tensor) -> Tensor:
        return _lp(behaviors, self.p)


class PotentialAlgebra(nn.Module):
    """Small transitive V4 algebra for six known Phase-1 primitives."""

    def __init__(
        self,
        primitive_count: int,
        behavior_dim: int,
        seed: int,
        initial_p: Tensor | None = None,
    ) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        if initial_p is None:
            raw_p = torch.full((behavior_dim,), _raw_for_p(), dtype=torch.float64)
        else:
            raw_p = torch.log(torch.expm1((initial_p.detach().to(torch.float64) - 1.0).clamp_min(1e-4)))
        # Pair interactions can otherwise compensate for an additive p and make
        # long-word predictions explode.  Stage 0 estimates p first; Stage 1
        # freezes it while learning symmetric/order residuals.
        self.raw_p = nn.Parameter(raw_p, requires_grad=False)
        self.raw_kappa = nn.Parameter(
            0.05 * torch.randn(primitive_count, behavior_dim, generator=generator, dtype=torch.float64)
        )
        self.symmetric_scale = nn.Parameter(torch.zeros(behavior_dim, dtype=torch.float64))
        self.ordered_scale = nn.Parameter(torch.full((behavior_dim,), 0.01, dtype=torch.float64))

    @property
    def p(self) -> Tensor:
        return 1.0 + F.softplus(self.raw_p)

    @property
    def kappa(self) -> Tensor:
        centered = self.raw_kappa - self.raw_kappa.mean(dim=0, keepdim=True)
        return centered / centered.square().mean(dim=0, keepdim=True).sqrt().clamp_min(1e-3)

    def forward(self, primitive_ids: Tensor, primitive_behaviors: Tensor) -> Tensor:
        base = _lp(primitive_behaviors, self.p)
        if primitive_ids.numel() < 2:
            return base
        symmetric = torch.zeros_like(base)
        ordered = torch.zeros_like(base)
        kappa = self.kappa[primitive_ids]
        pi = primitive_behaviors.abs() + 0.02
        pair_count = 0
        for i in range(primitive_ids.numel()):
            for j in range(i + 1, primitive_ids.numel()):
                symmetric = symmetric + primitive_behaviors[i] * primitive_behaviors[j]
                ordered = ordered + 0.5 * pi[i] * pi[j] * (kappa[j] - kappa[i])
                pair_count += 1
        return base + self.symmetric_scale * symmetric / pair_count \
            + self.ordered_scale * ordered / pair_count


def load_behavior_csv(path: str | Path) -> tuple[dict[tuple[str, ...], Tensor], dict]:
    targets = {}
    metadata = {}
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            word = tuple(row["word"].split(">"))
            targets[word] = torch.tensor(
                [float(row[f"delta_{name}"]) for name in BEHAVIOR_NAMES], dtype=torch.float64
            )
            metadata[word] = row
    return targets, metadata


def fit_lp(
    primitive_behavior: Tensor,
    pair_words: Sequence[tuple[int, int]],
    pair_targets: Tensor,
    *,
    steps: int = 1500,
) -> LearnedLp:
    model = LearnedLp(primitive_behavior.shape[1])
    scale = pair_targets.std(dim=0, unbiased=False).clamp_min(0.02)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    for _ in range(steps):
        predictions = torch.stack([model(primitive_behavior[list(word)]) for word in pair_words])
        loss = ((predictions - pair_targets) / scale).square().mean() \
            + 1e-4 * (model.p - 2.0).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model


def fit_potential_ensemble(
    primitive_behavior: Tensor,
    pair_words: Sequence[tuple[int, int]],
    pair_targets: Tensor,
    initial_p: Tensor,
    *,
    seeds: Sequence[int] = (4101, 4102, 4103, 4104, 4105),
    steps: int = 2500,
) -> list[PotentialAlgebra]:
    scale = pair_targets.std(dim=0, unbiased=False).clamp_min(0.02)
    models = []
    for seed in seeds:
        model = PotentialAlgebra(
            primitive_behavior.shape[0], primitive_behavior.shape[1], seed, initial_p=initial_p
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.025)
        for _ in range(steps):
            predictions = torch.stack(
                [model(torch.tensor(word), primitive_behavior[list(word)]) for word in pair_words]
            )
            fit = ((predictions - pair_targets) / scale).square().mean()
            regularization = 2e-4 * (
                model.symmetric_scale.square().mean() + model.ordered_scale.square().mean()
            )
            loss = fit + regularization
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        models.append(model)
    return models


@dataclass
class PairResidualForecast:
    p: Tensor
    residuals: Mapping[tuple[int, int], Tensor]

    def predict(self, ids: Sequence[int], behaviors: Tensor) -> Tensor:
        base = _lp(behaviors, self.p)
        if len(ids) < 2:
            return base
        residual = torch.zeros_like(base)
        count = 0
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                residual = residual + self.residuals[(ids[i], ids[j])]
                count += 1
        return base + residual / count


@dataclass
class PairFeatureRidge:
    p: Tensor
    coefficients: Tensor

    @staticmethod
    def features(behaviors: Tensor) -> Tensor:
        symmetric, ordered, count = torch.zeros_like(behaviors[0]), torch.zeros_like(behaviors[0]), 0
        for i in range(behaviors.shape[0]):
            for j in range(i + 1, behaviors.shape[0]):
                symmetric = symmetric + behaviors[i] * behaviors[j]
                ordered = ordered + behaviors[j] - behaviors[i]
                count += 1
        return torch.cat((torch.ones(1, dtype=behaviors.dtype), symmetric / count, ordered / count))

    def predict(self, behaviors: Tensor) -> Tensor:
        return _lp(behaviors, self.p) + self.features(behaviors) @ self.coefficients


def fit_pair_feature_ridge(
    primitive_behavior: Tensor,
    pair_words: Sequence[tuple[int, int]],
    pair_targets: Tensor,
    p: Tensor,
    ridge: float = 0.1,
) -> PairFeatureRidge:
    features, residuals = [], []
    for word, target in zip(pair_words, pair_targets):
        behavior = primitive_behavior[list(word)]
        features.append(PairFeatureRidge.features(behavior))
        residuals.append(target - _lp(behavior, p))
    x, y = torch.stack(features), torch.stack(residuals)
    penalty = torch.eye(x.shape[1], dtype=x.dtype) * ridge
    penalty[0, 0] = 0
    coefficients = torch.linalg.solve(x.T @ x + penalty, x.T @ y)
    return PairFeatureRidge(p.detach().clone(), coefficients)


def _max_abs_pool(behaviors: Tensor) -> Tensor:
    indices = behaviors.abs().argmax(dim=0)
    return behaviors[indices, torch.arange(behaviors.shape[1])]


def run_forecast_evaluation(
    behavior_csv: str | Path,
    primitives: Sequence[str],
    output_json: str | Path,
    *,
    lp_steps: int = 1500,
    potential_steps: int = 2500,
    split_name: str = "development",
    protocol_id: str = "V4-P1-MASON-001",
    include_target_rows: bool = False,
) -> dict:
    targets, metadata = load_behavior_csv(behavior_csv)
    primitive_to_id = {primitive: index for index, primitive in enumerate(primitives)}
    primitive_behavior = torch.stack([targets[(primitive,)] for primitive in primitives])
    pair_word_names = [word for word in targets if len(word) == 2]
    pair_words = [tuple(primitive_to_id[value] for value in word) for word in pair_word_names]
    pair_targets = torch.stack([targets[word] for word in pair_word_names])
    target_words = [word for word in targets if len(word) >= 3]
    if len(pair_words) != 30 or len(target_words) != 130:
        raise ValueError("Phase-1 protocol requires 30 calibration pairs and 130 target words")

    lp_model = fit_lp(primitive_behavior, pair_words, pair_targets, steps=lp_steps)
    potential_models = fit_potential_ensemble(
        primitive_behavior,
        pair_words,
        pair_targets,
        lp_model.p.detach().clone(),
        steps=potential_steps,
    )
    residual_lookup = {}
    for ids, target in zip(pair_words, pair_targets):
        residual_lookup[ids] = target - lp_model(primitive_behavior[list(ids)]).detach()
    residual_model = PairResidualForecast(lp_model.p.detach().clone(), residual_lookup)
    ridge_model = fit_pair_feature_ridge(
        primitive_behavior, pair_words, pair_targets, lp_model.p.detach().clone()
    )

    predictions: dict[str, list[Tensor]] = {
        "additive": [],
        "mean": [],
        "max_abs": [],
        "lp": [],
        "pair_feature_ridge": [],
        "pair_residual": [],
        "potential_algebra": [],
    }
    truth = []
    lengths = []
    with torch.no_grad():
        for word in target_words:
            ids = [primitive_to_id[value] for value in word]
            behavior = primitive_behavior[ids]
            truth.append(targets[word])
            lengths.append(len(word))
            predictions["additive"].append(behavior.sum(dim=0))
            predictions["mean"].append(behavior.mean(dim=0))
            predictions["max_abs"].append(_max_abs_pool(behavior))
            predictions["lp"].append(lp_model(behavior))
            predictions["pair_feature_ridge"].append(ridge_model.predict(behavior))
            predictions["pair_residual"].append(residual_model.predict(ids, behavior))
            potential = torch.stack(
                [model(torch.tensor(ids), behavior) for model in potential_models]
            ).mean(dim=0)
            predictions["potential_algebra"].append(potential)

    truth_tensor = torch.stack(truth)
    scale = pair_targets.std(dim=0, unbiased=False).clamp_min(0.02)
    clean_accuracy = float(metadata[("Identity",)]["accuracy"])
    accuracy_lower, accuracy_upper = -clean_accuracy, 1.0 - clean_accuracy
    result_methods = {}
    scored_predictions: dict[str, Tensor] = {}
    clipped_masks: dict[str, Tensor] = {}
    for name, values in predictions.items():
        raw_prediction = torch.stack(values)
        invalid_accuracy = (
            (raw_prediction[:, ACC_INDEX] < accuracy_lower)
            | (raw_prediction[:, ACC_INDEX] > accuracy_upper)
        )
        prediction = raw_prediction.clone()
        prediction[:, ACC_INDEX].clamp_(accuracy_lower, accuracy_upper)
        scored_predictions[name] = prediction
        clipped_masks[name] = invalid_accuracy
        absolute = (prediction - truth_tensor).abs()
        by_length = {}
        for length in sorted(set(lengths)):
            mask = torch.tensor([value == length for value in lengths])
            by_length[str(length)] = {
                "accuracy_mae_pp": float(100 * absolute[mask, ACC_INDEX].mean()),
                "normalized_behavior_mae": float((absolute[mask] / scale).mean()),
                "conditions": int(mask.sum()),
            }
        result_methods[name] = {
            "accuracy_mae_pp": float(100 * absolute[:, ACC_INDEX].mean()),
            "accuracy_rmse_pp": float(100 * (prediction[:, ACC_INDEX] - truth_tensor[:, ACC_INDEX]).square().mean().sqrt()),
            "normalized_behavior_mae": float((absolute / scale).mean()),
            "accuracy_clipped_fraction": float(invalid_accuracy.double().mean()),
            "worst_decile_accuracy_mae_pp": float(
                100 * torch.topk(absolute[:, ACC_INDEX], max(1, math.ceil(0.1 * len(target_words)))).values.mean()
            ),
            "by_length": by_length,
        }

    target_rows = []
    if include_target_rows:
        for index, word in enumerate(target_words):
            row = {
                "word": ">".join(word),
                "length": len(word),
                "word_family": ">".join(sorted(word)),
                "true_accuracy": float(metadata[word]["accuracy"]),
                "true_delta_accuracy": float(truth_tensor[index, ACC_INDEX]),
                "methods": {},
            }
            for name, prediction in scored_predictions.items():
                predicted_delta = float(prediction[index, ACC_INDEX])
                predicted_accuracy = clean_accuracy + predicted_delta
                row["methods"][name] = {
                    "predicted_accuracy": predicted_accuracy,
                    "predicted_delta_accuracy": predicted_delta,
                    "absolute_error_pp": 100.0 * abs(
                        predicted_delta - float(truth_tensor[index, ACC_INDEX])
                    ),
                    "accuracy_was_clipped": bool(clipped_masks[name][index]),
                }
            target_rows.append(row)

    result = {
        "protocol": f"{protocol_id}-{split_name}",
        "behavior_csv": str(behavior_csv),
        "calibration_pairs": len(pair_words),
        "target_words": len(target_words),
        "primitive_order": list(primitives),
        "behavior_order": list(BEHAVIOR_NAMES),
        "lp_exponents": lp_model.p.detach().tolist(),
        "clean_accuracy": clean_accuracy,
        "accuracy_constraint": [accuracy_lower, accuracy_upper],
        "potential_ensemble_size": len(potential_models),
        "methods": result_methods,
        "target_access_during_fit": False,
    }
    if include_target_rows:
        result["target_rows"] = target_rows
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
