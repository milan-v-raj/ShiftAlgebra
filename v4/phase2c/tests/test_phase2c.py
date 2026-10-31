import json
import unittest
from pathlib import Path

from v4.phase2c.freeze_designs import BUDGET, RANDOM_DESIGNS
from v4.phase2c.manifest import excluded_words, fresh_words, model_records

ROOT = Path(__file__).resolve().parents[3]


class Phase2CTests(unittest.TestCase):
    def test_fresh_words_are_new_and_reversal_paired(self):
        words = fresh_words()
        self.assertEqual(len(words), 64)
        self.assertEqual(len(set(words)), 64)
        self.assertTrue(set(words).isdisjoint(excluded_words()))
        for word in words:
            self.assertIn(tuple(reversed(word)), words)
        for length in (3, 4, 5, 6):
            self.assertEqual(sum(len(word) == length for word in words), 16)

    def test_fresh_models_have_independent_seeds(self):
        records = model_records()
        self.assertEqual(len(records), 30)
        self.assertEqual(len({row.lineage_id for row in records}), 15)
        self.assertEqual(len({row.seed for row in records}), 15)

    def test_active_and_random_designs_are_unique_equal_budget(self):
        frozen = json.loads((ROOT / "v4/phase2c/FROZEN_PAIR_DESIGNS.json").read_text())
        self.assertEqual(len(frozen["designs"]), RANDOM_DESIGNS + 1)
        groups = [tuple(sorted(value["pair_groups"])) for value in frozen["designs"].values()]
        self.assertEqual(len(set(groups)), len(groups))
        self.assertTrue(all(len(value) == BUDGET for value in groups))
        self.assertTrue(all(value["directed_measurements"] == 10 for value in frozen["designs"].values()))

    def test_fitting_contract_forbids_previous_confirmations(self):
        frozen = json.loads((ROOT / "v4/phase2c/FROZEN_PAIR_DESIGNS.json").read_text())
        self.assertEqual(frozen["source_roles_loaded"], ["calibration", "method_development"])
        self.assertEqual(len(frozen["forbidden_sources"]), 2)
        self.assertEqual(frozen["status"], "PAIR_DESIGNS_FROZEN_FRESH_OUTCOMES_UNOPENED")


if __name__ == "__main__":
    unittest.main()
