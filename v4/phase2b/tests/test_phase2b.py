import json
import unittest
from pathlib import Path

import numpy as np

from v4.benchmark.generate_mason_manifest import exact_words
from v4.phase2b.manifest import fresh_words, model_records, word_records
from v4.phase2b.method import EXPERTS, accuracy_from_hazard, apply_gate, load_behavior

ROOT = Path(__file__).resolve().parents[3]


class Phase2BManifestTests(unittest.TestCase):
    def test_fresh_words_are_new_unique_and_reversal_paired(self):
        fresh = fresh_words()
        self.assertEqual(len(fresh), 64)
        self.assertEqual(len(set(fresh)), 64)
        self.assertTrue(set(fresh).isdisjoint(exact_words()))
        for word in fresh:
            self.assertIn(tuple(reversed(word)), fresh)
        self.assertEqual({length: sum(len(w) == length for w in fresh) for length in (3, 4, 5, 6)},
                         {3: 16, 4: 16, 5: 16, 6: 16})

    def test_bank_has_15_independent_lineages_and_30_snapshots(self):
        records = model_records()
        self.assertEqual(len(records), 30)
        self.assertEqual(len({row.lineage_id for row in records}), 15)
        self.assertEqual(len({row.seed for row in records}), 15)

    def test_manifest_counts(self):
        records = word_records()
        self.assertEqual(sum(row["role"] == "calibration" for row in records), 37)
        self.assertEqual(sum(row["role"] == "fresh_confirmation" for row in records), 64)


class ACHMTests(unittest.TestCase):
    def test_hazard_conversion_is_bounded(self):
        values = accuracy_from_hazard(np.asarray([-3.0, 0.0, 2.0, 20.0]), 0.8)
        self.assertTrue(np.all(values >= 0.0))
        self.assertTrue(np.all(values <= 1.0))

    def test_gate_is_convex_and_bounded(self):
        row = {
            "features": {"one": 1.0},
            "expert_hazards": np.asarray([0.1, 0.2, 0.3, 0.4]),
            "expert_accuracies": np.asarray([0.73, 0.67, 0.61, 0.56]),
            "clean_accuracy": 0.8,
        }
        gate = {
            "feature_names": ["one"],
            "feature_mean": [0.0],
            "feature_scale": [1.0],
            "coefficients": [[0.0, 0.0, 0.0], [1.0, -1.0, 0.5]],
        }
        result = apply_gate(row, gate)
        self.assertAlmostEqual(float(result["weights"].sum()), 1.0)
        self.assertTrue(np.all(result["weights"] >= 0.0))
        self.assertGreaterEqual(result["prediction"], 0.0)
        self.assertLessEqual(result["prediction"], 1.0)

    def test_development_loader_never_loads_confirmation_role(self):
        behavior = load_behavior(
            ROOT / "v4/phase2/results/development-behavior.csv",
            {"calibration", "method_development"},
        )
        roles = {row["word_role"] for model in behavior.values() for row in model.values()}
        self.assertEqual(roles, {"calibration", "method_development"})
        self.assertNotIn("confirmation", roles)

    def test_frozen_gate_has_four_experts_and_all_budgets(self):
        frozen = json.loads((ROOT / "v4/phase2b/FROZEN_ACHM.json").read_text())
        self.assertEqual(frozen["experts"], list(EXPERTS))
        self.assertEqual(set(frozen["budgets"]), {"0", "5", "10", "15"})
        self.assertEqual(frozen["status"], "ACHM_FROZEN_FRESH_CONFIRMATION_UNOPENED")


if __name__ == "__main__":
    unittest.main()
