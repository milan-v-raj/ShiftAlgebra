import json
import unittest
from pathlib import Path

import torch

from v4.phase2f.manifest import target_words as phase2f_target_words
from v4.phase2f.method import feature_vector
from v4.phase2g import METRICS, PROTOCOL_ID
from v4.phase2g.manifest import EXPECTED_METHOD_SHA256, model_records, payload, target_words
from v4.phase2g.models import FAMILIES, logits_and_features, make_model

ROOT = Path(__file__).resolve().parents[3]


class Phase2GTests(unittest.TestCase):
    def test_manifest_contract(self):
        manifest = payload()
        self.assertEqual(manifest["protocol"], PROTOCOL_ID)
        self.assertEqual(manifest["metrics"], list(METRICS))
        self.assertEqual(manifest["counts"]["lineages"], 18)
        self.assertEqual(manifest["counts"]["snapshots"], 36)
        self.assertEqual(manifest["image_partition"], {"per_class": 20, "offset": 40})
        self.assertEqual(manifest["frozen_method"]["sha256"], EXPECTED_METHOD_SHA256)
        self.assertEqual(manifest["architecture_boundary"]["family_overlap"], [])

    def test_model_bank_counts_and_families(self):
        records = model_records()
        self.assertEqual(len(records), 36)
        self.assertEqual(len({record.lineage_id for record in records}), 18)
        self.assertEqual({record.family for record in records}, set(FAMILIES))
        self.assertTrue(all(record.parameter_count > 0 for record in records))

    def test_words_are_fresh_paired_and_complete(self):
        words = target_words()
        prior = phase2f_target_words()
        self.assertEqual(len(words), 60)
        self.assertEqual(len(set(words)), 60)
        self.assertFalse(set(words) & (set(prior["development"]) | set(prior["confirmation"])))
        self.assertTrue(all(tuple(reversed(word)) in words for word in words))
        self.assertEqual({len(word) for word in words}, {4, 5, 6})

    def test_new_models_expose_logits_and_features(self):
        images = torch.randn(2, 3, 32, 32)
        for family in FAMILIES:
            model = make_model(family)
            logits, features = logits_and_features(model, family, images)
            self.assertEqual(logits.shape, (2, 100))
            self.assertEqual(features.ndim, 2)
            self.assertTrue(torch.isfinite(logits).all())
            self.assertTrue(torch.isfinite(features).all())

    def test_new_families_use_frozen_unknown_encoding(self):
        components = {"clean": 0.1, "ordered": 0.2, "symmetric": 0.2, "independent": 0.1, "envelope": 0.1}
        for family in FAMILIES:
            truth = {
                "word": "Contrast>Invert>Swirl>Rotate90",
                "parameter_count": 10_000,
                "epoch": 2,
                "family": family,
                "recipe": "erm_flip",
            }
            vector = feature_vector(truth, "accuracy", components, False)
            self.assertEqual(vector[10:16], [0.0] * 6)

    def test_phase2f_method_status_is_frozen(self):
        frozen = json.loads((ROOT / "v4/phase2f/FROZEN_FULL_VECTOR_METHODS.json").read_text())
        self.assertEqual(
            frozen["status"], "FULL_VECTOR_METHODS_FROZEN_CONFIRMATION_OUTCOMES_UNOPENED"
        )
        self.assertEqual(len([name for name in frozen["designs"] if name.startswith("random_")]), 20)


if __name__ == "__main__":
    unittest.main()
