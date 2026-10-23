"""Source/model-metadata baseline required by the ODP comparison.

This baseline intentionally receives no composition fingerprint or target-side
sample.  It estimates whether Shift Algebra adds value beyond source behavior,
architecture family, and training recipe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass
class SourceFamilyRidge:
    alpha: float = 1.0

    def __post_init__(self) -> None:
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.family_levels: list[str] = []
        self.recipe_levels: list[str] = []
        self.source_mean: np.ndarray | None = None
        self.source_scale: np.ndarray | None = None
        self.coef_: np.ndarray | None = None

    @staticmethod
    def _source_array(source_features) -> np.ndarray:
        x = np.asarray(source_features, dtype=float)
        if x.ndim == 1:
            x = x[:, None]
        if x.ndim != 2 or x.shape[0] == 0 or not np.isfinite(x).all():
            raise ValueError("source_features must be finite [samples, features]")
        return x

    @staticmethod
    def _validate_categories(values: Sequence[str], n: int, name: str) -> list[str]:
        output = [str(v) for v in values]
        if len(output) != n or any(not value for value in output):
            raise ValueError(f"{name} must contain one non-empty value per sample")
        return output

    @staticmethod
    def _levels(values: Sequence[str]) -> list[str]:
        return sorted(set(values)) + ["__UNK__"]

    @staticmethod
    def _one_hot(values: Sequence[str], levels: Sequence[str]) -> np.ndarray:
        index = {value: i for i, value in enumerate(levels)}
        unknown = index["__UNK__"]
        output = np.zeros((len(values), len(levels)), dtype=float)
        for row, value in enumerate(values):
            output[row, index.get(value, unknown)] = 1.0
        return output

    def _design(
        self,
        source_features,
        model_families: Sequence[str],
        training_recipes: Sequence[str],
        *,
        fitting: bool,
    ) -> np.ndarray:
        source = self._source_array(source_features)
        families = self._validate_categories(model_families, source.shape[0], "model_families")
        recipes = self._validate_categories(training_recipes, source.shape[0], "training_recipes")
        if fitting:
            self.family_levels = self._levels(families)
            self.recipe_levels = self._levels(recipes)
            self.source_mean = source.mean(axis=0)
            self.source_scale = source.std(axis=0)
            self.source_scale[self.source_scale < 1e-12] = 1.0
        if self.source_mean is None or self.source_scale is None:
            raise RuntimeError("baseline has not been fitted")
        if source.shape[1] != self.source_mean.size:
            raise ValueError("source feature dimension differs from fitted baseline")
        source_z = (source - self.source_mean) / self.source_scale
        return np.column_stack(
            [
                np.ones(source.shape[0]),
                source_z,
                self._one_hot(families, self.family_levels),
                self._one_hot(recipes, self.recipe_levels),
            ]
        )

    def fit(
        self,
        source_features,
        model_families: Sequence[str],
        training_recipes: Sequence[str],
        targets,
    ) -> "SourceFamilyRidge":
        design = self._design(source_features, model_families, training_recipes, fitting=True)
        y = np.asarray(targets, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        if y.ndim != 2 or y.shape[0] != design.shape[0] or not np.isfinite(y).all():
            raise ValueError("targets must be finite [samples, outputs]")
        penalty = np.eye(design.shape[1]) * self.alpha
        penalty[0, 0] = 0.0  # never penalize the intercept
        self.coef_ = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        return self

    def predict(
        self,
        source_features,
        model_families: Sequence[str],
        training_recipes: Sequence[str],
    ) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("baseline has not been fitted")
        design = self._design(source_features, model_families, training_recipes, fitting=False)
        output = design @ self.coef_
        return output[:, 0] if output.shape[1] == 1 else output

    def describe_information(self) -> dict:
        return {
            "regime": "R0",
            "uses_source_behavior": True,
            "uses_model_family": True,
            "uses_training_recipe": True,
            "uses_primitive_behavior": False,
            "uses_pair_labels": False,
            "uses_composed_target_images": False,
            "uses_composed_target_labels": False,
        }
