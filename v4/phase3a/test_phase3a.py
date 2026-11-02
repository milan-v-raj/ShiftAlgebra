import unittest

import numpy as np
import torch

from v4.phase3a import METRICS
from v4.phase3a.corruptions import PARAMETERS, PRIMITIVES, apply_primitive, apply_word
from v4.phase3a.manifest import calibration_words, model_records, target_words
from v4.phase3a.method import (
    apply, clip, components, feature_vector, fit, pair_group,
)


class CorruptionTests(unittest.TestCase):
    def setUp(self):
        self.images = torch.linspace(0.0, 1.0, 12 * 3 * 32 * 32).reshape(12, 3, 32, 32)
        self.ids = torch.arange(100, 112, dtype=torch.long)

    def test_all_corruptions_are_deterministic_finite_and_bounded(self):
        for primitive in PRIMITIVES:
            first = apply_primitive(self.images, primitive, self.ids)
            second = apply_primitive(self.images, primitive, self.ids)
            self.assertTrue(torch.equal(first, second), primitive)
            self.assertTrue(torch.isfinite(first).all(), primitive)
            self.assertGreaterEqual(float(first.min()), 0.0)
            self.assertLessEqual(float(first.max()), 1.0)

    def test_stochastic_fields_depend_on_canonical_id(self):
        for primitive in ("GaussianNoise", "ShotNoise"):
            first = apply_primitive(self.images, primitive, self.ids)
            second = apply_primitive(self.images, primitive, self.ids + 1)
            self.assertFalse(torch.equal(first, second))

    def test_word_executes_left_to_right(self):
        word = ("Brightness", "Pixelate")
        actual = apply_word(self.images, word, self.ids)
        expected = apply_primitive(
            apply_primitive(self.images, "Brightness", self.ids), "Pixelate", self.ids
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_parameters_cover_exact_vocabulary(self):
        self.assertEqual(set(PARAMETERS), set(PRIMITIVES))


class ManifestTests(unittest.TestCase):
    def test_word_sets_are_exact_and_disjoint(self):
        words = target_words()
        development = {">".join(word) for word in words["development"]}
        confirmation = {">".join(word) for word in words["confirmation"]}
        self.assertEqual(len(development), 60)
        self.assertEqual(len(confirmation), 60)
        self.assertTrue(development.isdisjoint(confirmation))
        self.assertEqual(len(calibration_words()), 37)

    def test_each_target_set_has_thirty_reversal_pairs(self):
        for words in target_words().values():
            word_set = set(words)
            self.assertEqual({tuple(reversed(word)) for word in words}, word_set)
            self.assertEqual(len(word_set) // 2, 30)

    def test_model_rosters_are_disjoint_and_balanced(self):
        records = model_records()
        development = [row for row in records if row.bank_split == "development"]
        confirmation = [row for row in records if row.bank_split == "confirmation"]
        self.assertEqual(len(development), 36)
        self.assertEqual(len(confirmation), 36)
        self.assertTrue({row.model_id for row in development}.isdisjoint(
            row.model_id for row in confirmation
        ))
        self.assertEqual(len({row.lineage_id for row in confirmation}), 18)
        self.assertEqual(len({row.family for row in confirmation}), 6)

    def test_full_metric_vector_is_frozen(self):
        self.assertEqual(METRICS, (
            "margin", "feature_drift", "ece", "balanced_accuracy", "accuracy", "mean_confidence"
        ))


class MethodTests(unittest.TestCase):
    def _rows(self):
        clean = {metric: 0.8 for metric in METRICS}
        rows = {"Identity": dict(clean)}
        for index, primitive in enumerate(PRIMITIVES):
            rows[primitive] = {metric: 0.75 - 0.01 * index for metric in METRICS}
        for left in PRIMITIVES:
            for right in PRIMITIVES:
                if left != right:
                    rows[f"{left}>{right}"] = {metric: 0.65 for metric in METRICS}
        return rows

    def test_pair_group_is_symmetric(self):
        self.assertEqual(pair_group("A", "B"), pair_group("B", "A"))

    def test_components_are_order_blind_under_reversal(self):
        rows = self._rows()
        groups = {pair_group(left, right) for left in PRIMITIVES for right in PRIMITIVES if left != right}
        forward = {"word": ">".join(PRIMITIVES[:4])}
        reverse = {"word": ">".join(reversed(PRIMITIVES[:4]))}
        for metric in METRICS:
            first = components(rows, forward, metric, groups, 1.0)
            second = components(rows, reverse, metric, groups, 1.0)
            self.assertEqual(first, second)

    def test_ridge_fit_and_apply(self):
        rows = []
        for value in (0.2, 0.4, 0.6, 0.8):
            rows.append({
                "features": {"g0": {"accuracy": [value]}},
                "target": {"accuracy": value},
            })
        model = fit(rows, "accuracy", "g0")
        self.assertTrue(np.isfinite(model["coefficients"]).all())
        self.assertAlmostEqual(apply(rows[2], model), 0.58, places=12)

    def test_feature_vector_adds_three_composition_coordinates(self):
        truth = {
            "word": ">".join(PRIMITIVES[:4]), "parameter_count": 100,
            "epoch": 1, "family": "residual_holdout", "recipe": "erm_flip",
        }
        component = {"clean": 0.8, "symmetric_formula": 0.7, "independent": 0.6, "envelope": 0.5}
        self.assertEqual(
            len(feature_vector(truth, component, True)),
            len(feature_vector(truth, component, False)) + 3,
        )

    def test_clipping(self):
        self.assertEqual(clip("accuracy", 2.0), 1.0)
        self.assertEqual(clip("margin", -4.0), 0.0)
        self.assertEqual(clip("feature_drift", -1.0), 0.0)


if __name__ == "__main__":
    unittest.main()
