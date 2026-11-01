import json
import unittest
from pathlib import Path

import torch

from v4.phase2.train_bank import sha256
from v4.phase2f import METRICS
from v4.phase2g.manifest import target_words as phase2g_target_words
from v4.phase2g.models import FAMILIES, logits_and_features, make_model
from v4.phase2h.manifest import EXPECTED_METHOD_SHA256, model_records, payload, target_words
from v4.phase2h.method import feature_vector

ROOT = Path(__file__).resolve().parents[3]


class Phase2HTests(unittest.TestCase):
    def test_manifest_and_method_freeze(self):
        manifest = payload()
        self.assertEqual(manifest["metrics"], list(METRICS))
        self.assertEqual(manifest["counts"]["lineages"], 18)
        self.assertEqual(manifest["counts"]["snapshots"], 36)
        self.assertEqual(manifest["counts"]["reversal_groups"], 30)
        self.assertEqual(manifest["image_partition"], {"per_class": 20, "offset": 60})
        method = ROOT / manifest["frozen_method"]["path"]
        self.assertEqual(sha256(method), EXPECTED_METHOD_SHA256)

    def test_model_bank_counts(self):
        records = model_records()
        self.assertEqual(len(records), 36)
        self.assertEqual(len({record.lineage_id for record in records}), 18)
        self.assertEqual({record.family for record in records}, set(FAMILIES))
        self.assertEqual(len({record.seed for record in records}), 18)

    def test_words_are_fresh_and_reversal_paired(self):
        words = target_words()
        self.assertEqual(len(words), 60)
        self.assertEqual(len(set(words)), 60)
        self.assertFalse(set(words) & set(phase2g_target_words()))
        self.assertTrue(all(tuple(reversed(word)) in words for word in words))
        self.assertEqual({len(word) for word in words}, {4, 5, 6})

    def test_methods_are_nested_by_one_order_feature(self):
        truth = {
            "word": "Contrast>Invert>Swirl>Rotate90",
            "parameter_count": 10_000,
            "epoch": 2,
            "family": "residual_holdout",
            "recipe": "erm_flip",
        }
        component = {
            "clean": 0.1,
            "ordered": 0.3,
            "symmetric": 0.2,
            "independent": 0.15,
            "envelope": 0.12,
        }
        g0 = feature_vector(truth, component, "g0")
        symmetric = feature_vector(truth, component, "symmetric")
        order = feature_vector(truth, component, "order_aware")
        self.assertEqual(symmetric[:len(g0)], g0)
        self.assertEqual(order[:len(symmetric)], symmetric)
        self.assertEqual(len(order), len(symmetric) + 1)
        self.assertAlmostEqual(order[-1], component["ordered"] - component["symmetric"])

    def test_symmetric_features_are_reversal_invariant(self):
        component = {
            "clean": 0.1,
            "ordered": 0.3,
            "symmetric": 0.2,
            "independent": 0.15,
            "envelope": 0.12,
        }
        base = {
            "parameter_count": 10_000,
            "epoch": 2,
            "family": "residual_holdout",
            "recipe": "erm_flip",
        }
        forward = feature_vector(
            {**base, "word": "Contrast>Invert>Swirl>Rotate90"}, component, "symmetric"
        )
        reverse = feature_vector(
            {**base, "word": "Rotate90>Swirl>Invert>Contrast"}, component, "symmetric"
        )
        self.assertEqual(forward, reverse)

    def test_models_expose_full_vector_inputs(self):
        images = torch.randn(2, 3, 32, 32)
        for family in FAMILIES:
            logits, features = logits_and_features(make_model(family), family, images)
            self.assertEqual(logits.shape, (2, 100))
            self.assertEqual(features.ndim, 2)
            self.assertTrue(torch.isfinite(logits).all())

    def test_frozen_file_contains_all_six_metric_models(self):
        frozen = json.loads(
            (ROOT / "v4/phase2h/FROZEN_ORDER_ABLATION_METHODS.json").read_text()
        )
        for mode in ("g0", "symmetric", "order_aware"):
            self.assertEqual(set(frozen["models"][mode]), set(METRICS))


if __name__ == "__main__":
    unittest.main()
