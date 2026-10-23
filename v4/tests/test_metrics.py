import unittest

import numpy as np

from v4.analysis.metrics import (
    BehaviorScale,
    accuracy_mae_pp,
    accuracy_rmse_pp,
    interval_metrics,
    macro_group_mae_pp,
    model_selection_regret,
    near_oracle_selection_rate,
    partial_spearman,
    rank_environments_per_model,
    rank_models_per_environment,
)


class MetricTests(unittest.TestCase):
    def test_accuracy_errors_are_percentage_points(self):
        pred = np.array([0.7, 0.8])
        true = np.array([0.8, 0.6])
        self.assertAlmostEqual(accuracy_mae_pp(pred, true), 15.0)
        self.assertAlmostEqual(accuracy_rmse_pp(pred, true), np.sqrt(0.025) * 100)

    def test_ranking_axes_are_not_conflated(self):
        # Perfect model ranking within both environments, but the first model's
        # environment order is reversed.
        pred = np.array([[0.5, 0.6, 0.7], [0.9, 0.8, 0.7], [0.3, 0.4, 0.5]])
        true = np.array([[0.4, 0.5, 0.6], [0.6, 0.5, 0.4], [0.6, 0.7, 0.8]])
        by_env = rank_models_per_environment(pred, true)
        by_model = rank_environments_per_model(pred, true)
        self.assertEqual(by_env["axis"], "rho_model_given_environment")
        self.assertEqual(by_model["axis"], "rho_environment_given_model")
        self.assertAlmostEqual(by_env["macro_mean"], 1.0)
        self.assertNotEqual(by_env["macro_mean"], by_model["macro_mean"])

    def test_selection_regret(self):
        pred = np.array([[0.7, 0.9, 0.6], [0.6, 0.5, 0.4]])
        true = np.array([[0.95, 0.8, 0.7], [0.4, 0.6, 0.5]])
        self.assertAlmostEqual(model_selection_regret(pred, true, top_k=1), 17.5)
        self.assertAlmostEqual(model_selection_regret(pred, true, top_k=2), 0.0)
        self.assertEqual(near_oracle_selection_rate(pred, true, tolerance_pp=1, top_k=2), 1.0)

    def test_partial_spearman_removes_linear_rank_control(self):
        rng = np.random.default_rng(4)
        control = np.arange(100.0)
        predicted = control + rng.normal(0, 8, size=control.size)
        target = control + rng.normal(0, 8, size=control.size)
        raw = rank_models_per_environment(predicted[None, :], target[None, :])["macro_mean"]
        controlled = partial_spearman(predicted, target, control)
        self.assertGreater(raw, 0.85)
        self.assertLess(abs(controlled), 0.3)

    def test_training_only_behavior_scale(self):
        train = np.array([[0.0, 0.0], [1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        scale = BehaviorScale.fit_iqr(train)
        result = scale.normalized_mae(np.array([[1.0, 10.0]]), np.array([[2.0, 20.0]]))
        self.assertAlmostEqual(result["per_coordinate"][0], result["per_coordinate"][1])

    def test_interval_and_macro_group_metrics(self):
        target = np.array([0.5, 0.6, 0.7, 0.8])
        result = interval_metrics(target - 0.05, target + 0.05, target)
        self.assertEqual(result["coverage"], 1.0)
        self.assertAlmostEqual(result["mean_width"], 0.1)
        macro = macro_group_mae_pp(
            np.array([0.4, 0.6, 0.6, 0.6]), target, ["a", "a", "b", "b"]
        )
        self.assertAlmostEqual(macro, 10.0)


if __name__ == "__main__":
    unittest.main()
