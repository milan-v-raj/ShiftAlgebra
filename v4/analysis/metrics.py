"""Standard prediction, ranking, regret, and interval metrics for V4.

Unlike historical BPE, accuracy errors are returned in percentage points and
ranking axes are explicitly named.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata, spearmanr

Array = np.ndarray


def _arrays(predicted, target) -> tuple[Array, Array]:
    p = np.asarray(predicted, dtype=float)
    y = np.asarray(target, dtype=float)
    if p.shape != y.shape or p.size == 0:
        raise ValueError("predicted and target must be non-empty arrays with identical shapes")
    if not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("metrics require finite values")
    return p, y


def accuracy_mae_pp(predicted, target) -> float:
    """Accuracy mean absolute error in percentage points (fractional inputs)."""

    p, y = _arrays(predicted, target)
    return float(100.0 * np.mean(np.abs(p - y)))


def accuracy_rmse_pp(predicted, target) -> float:
    p, y = _arrays(predicted, target)
    return float(100.0 * np.sqrt(np.mean((p - y) ** 2)))


def accuracy_median_ae_pp(predicted, target) -> float:
    p, y = _arrays(predicted, target)
    return float(100.0 * np.median(np.abs(p - y)))


def accuracy_worst_decile_mae_pp(predicted, target) -> float:
    p, y = _arrays(predicted, target)
    errors = np.abs(p - y).reshape(-1)
    n = max(1, int(np.ceil(0.1 * errors.size)))
    return float(100.0 * np.mean(np.partition(errors, errors.size - n)[-n:]))


def safe_spearman(predicted, target) -> float:
    p, y = _arrays(predicted, target)
    if p.size < 3 or np.std(p) == 0 or np.std(y) == 0:
        return float("nan")
    return float(spearmanr(p, y).statistic)


def rank_models_per_environment(predicted, target) -> Dict[str, object]:
    """ODP-style rank: models are ranked separately inside each environment.

    Arrays have shape ``[n_environments, n_models]``.
    """

    p, y = _arrays(predicted, target)
    if p.ndim != 2:
        raise ValueError("rank arrays must have shape [environment, model]")
    per_environment = [safe_spearman(p[i], y[i]) for i in range(p.shape[0])]
    finite = np.asarray([x for x in per_environment if np.isfinite(x)])
    return {
        "axis": "rho_model_given_environment",
        "per_environment": per_environment,
        "macro_mean": float(finite.mean()) if finite.size else float("nan"),
        "n_valid": int(finite.size),
    }


def rank_environments_per_model(predicted, target) -> Dict[str, object]:
    """Current-project rank: environments are ranked inside each model."""

    p, y = _arrays(predicted, target)
    if p.ndim != 2:
        raise ValueError("rank arrays must have shape [environment, model]")
    per_model = [safe_spearman(p[:, j], y[:, j]) for j in range(p.shape[1])]
    finite = np.asarray([x for x in per_model if np.isfinite(x)])
    return {
        "axis": "rho_environment_given_model",
        "per_model": per_model,
        "macro_mean": float(finite.mean()) if finite.size else float("nan"),
        "n_valid": int(finite.size),
    }


def partial_spearman(predicted, target, controls) -> float:
    """Spearman correlation after linear residualization of ranked controls.

    Categorical controls must be one-hot encoded by the caller.  An intercept
    is added here.  This is a descriptive controlled-rank metric; hierarchical
    uncertainty is computed separately at the benchmark-family level.
    """

    p, y = _arrays(predicted, target)
    if p.ndim != 1:
        raise ValueError("partial_spearman expects one-dimensional scores")
    x = np.asarray(controls, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[0] != p.shape[0] or not np.isfinite(x).all():
        raise ValueError("controls must be finite [samples, controls]")
    design = np.column_stack([np.ones(p.shape[0]), np.apply_along_axis(rankdata, 0, x)])
    rp, ry = rankdata(p), rankdata(y)
    p_resid = rp - design @ np.linalg.lstsq(design, rp, rcond=None)[0]
    y_resid = ry - design @ np.linalg.lstsq(design, ry, rcond=None)[0]
    if np.std(p_resid) == 0 or np.std(y_resid) == 0:
        return float("nan")
    return float(np.corrcoef(p_resid, y_resid)[0, 1])


def model_selection_regret(predicted, target, top_k: int = 1) -> float:
    """Mean accuracy regret in percentage points across environments.

    For each environment, select the ``top_k`` models by predicted accuracy and
    credit the best true accuracy among those selected.
    """

    p, y = _arrays(predicted, target)
    if p.ndim != 2:
        raise ValueError("regret arrays must have shape [environment, model]")
    if top_k < 1 or top_k > p.shape[1]:
        raise ValueError("top_k must be between one and the model count")
    selected = np.argpartition(p, -top_k, axis=1)[:, -top_k:]
    chosen_true = np.take_along_axis(y, selected, axis=1).max(axis=1)
    oracle_true = y.max(axis=1)
    return float(100.0 * np.mean(oracle_true - chosen_true))


def near_oracle_selection_rate(predicted, target, tolerance_pp: float = 1.0, top_k: int = 1) -> float:
    p, y = _arrays(predicted, target)
    if p.ndim != 2:
        raise ValueError("selection arrays must have shape [environment, model]")
    selected = np.argpartition(p, -top_k, axis=1)[:, -top_k:]
    chosen_true = np.take_along_axis(y, selected, axis=1).max(axis=1)
    gap_pp = 100.0 * (y.max(axis=1) - chosen_true)
    return float(np.mean(gap_pp <= tolerance_pp))


def macro_group_mae_pp(predicted, target, groups: Sequence[str]) -> float:
    """Macro-average accuracy MAE over declared family labels."""

    p, y = _arrays(predicted, target)
    if p.ndim != 1 or len(groups) != p.size:
        raise ValueError("groups must align with one-dimensional inputs")
    labels = sorted(set(groups))
    if not labels:
        raise ValueError("at least one group is required")
    values = [accuracy_mae_pp(p[np.asarray(groups) == g], y[np.asarray(groups) == g]) for g in labels]
    return float(np.mean(values))


@dataclass(frozen=True)
class BehaviorScale:
    """Training-only scale for comparable multivariate behavior errors."""

    center: Array
    scale: Array

    @classmethod
    def fit_iqr(cls, training_targets, floor: float = 1e-6) -> "BehaviorScale":
        values = np.asarray(training_targets, dtype=float)
        if values.ndim != 2 or values.shape[0] < 2 or not np.isfinite(values).all():
            raise ValueError("training targets must be finite [samples, behavior]")
        center = np.median(values, axis=0)
        iqr = np.percentile(values, 75, axis=0) - np.percentile(values, 25, axis=0)
        return cls(center=center, scale=np.maximum(iqr, floor))

    def normalized_mae(self, predicted, target) -> dict:
        p, y = _arrays(predicted, target)
        if p.ndim != 2 or p.shape[1] != self.scale.size:
            raise ValueError("behavior arrays do not match fitted scale")
        per_coordinate = np.mean(np.abs(p - y) / self.scale, axis=0)
        return {
            "per_coordinate": per_coordinate.tolist(),
            "macro": float(per_coordinate.mean()),
        }


def interval_metrics(lower, upper, target) -> dict:
    lo, y = _arrays(lower, target)
    hi, _ = _arrays(upper, target)
    if np.any(lo > hi):
        raise ValueError("lower interval endpoint exceeds upper endpoint")
    return {
        "coverage": float(np.mean((y >= lo) & (y <= hi))),
        "mean_width": float(np.mean(hi - lo)),
    }
