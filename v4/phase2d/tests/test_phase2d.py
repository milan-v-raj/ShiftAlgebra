import json, unittest
from pathlib import Path
import torch
from v4.phase2.models import FAMILY_WIDTHS
from v4.phase2d.manifest import excluded_words,fresh_words,model_records
from v4.phase2d.models import FAMILIES,make_model
ROOT=Path(__file__).resolve().parents[3]
class Phase2DTests(unittest.TestCase):
 def test_holdout_families_are_unseen_and_executable(self):
  self.assertTrue(set(FAMILIES).isdisjoint(FAMILY_WIDTHS))
  for family in FAMILIES: self.assertEqual(make_model(family)(torch.randn(2,3,32,32)).shape,(2,10))
 def test_bank_counts(self):
  rows=model_records(); self.assertEqual(len(rows),12); self.assertEqual(len({r.lineage_id for r in rows}),6)
 def test_words_are_fresh_and_reversed(self):
  words=fresh_words(); self.assertEqual(len(words),64); self.assertTrue(set(words).isdisjoint(excluded_words()))
  for word in words: self.assertIn(tuple(reversed(word)),words)
 def test_all_methods_frozen(self):
  p=json.loads((ROOT/"v4/phase2d/FROZEN_HOLDOUT_METHODS.json").read_text()); self.assertEqual(p["status"],"ALL_METHODS_FROZEN_ARCHITECTURE_OUTCOMES_UNOPENED"); self.assertEqual(set(p["selectors"]),{"active","magnitude","variance","coverage","d_optimal"}); self.assertEqual(len(p["random_designs"]),20)
if __name__=="__main__": unittest.main()
