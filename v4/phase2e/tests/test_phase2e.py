import hashlib,json,unittest
from pathlib import Path
import torch
from v4.phase2.models import FAMILY_WIDTHS
from v4.phase2d.models import FAMILIES as D_FAMILIES
from v4.phase2e.manifest import excluded_words,fresh_words,model_records
from v4.phase2e.models import FAMILIES,make_model
ROOT=Path(__file__).resolve().parents[3]
class Tests(unittest.TestCase):
 def test_families_new_and_executable(self):
  self.assertTrue(set(FAMILIES).isdisjoint(set(FAMILY_WIDTHS)|set(D_FAMILIES)))
  for f in FAMILIES:self.assertEqual(make_model(f)(torch.randn(2,3,32,32)).shape,(2,10))
 def test_bank(self):
  r=model_records();self.assertEqual((len(r),len({x.lineage_id for x in r})),(24,12))
 def test_words(self):
  w=fresh_words();self.assertEqual(len(w),60);self.assertTrue(set(w).isdisjoint(excluded_words()));self.assertEqual({len(x) for x in w},{4,5,6})
  for x in w:self.assertIn(tuple(reversed(x)),w)
 def test_frozen_dependency_hash(self):
  p=json.loads((ROOT/'v4/phase2e/FROZEN_BREADTH_METHODS.json').read_text());s=ROOT/p['method_source'];self.assertEqual(hashlib.sha256(s.read_bytes()).hexdigest(),p['method_source_sha256'])
if __name__=='__main__':unittest.main()
