import json,unittest
from pathlib import Path
import torch
from v4.phase2f import METRICS
from v4.phase2f.manifest import excluded_words,model_records,target_words
from v4.phase2f.models import FAMILIES,logits_and_features,make_model
ROOT=Path(__file__).resolve().parents[3]
class Tests(unittest.TestCase):
 def test_complete_metric_contract(self):self.assertEqual(METRICS,('margin','feature_drift','ece','balanced_accuracy','accuracy','mean_confidence'))
 def test_bank_counts(self):
  r=model_records();self.assertEqual(len(r),72);self.assertEqual(len({x.lineage_id for x in r}),36);self.assertEqual({x.bank_split for x in r},{'development','confirmation'})
 def test_words_fresh_and_paired(self):
  w=target_words();self.assertEqual((len(w['development']),len(w['confirmation'])),(60,60));self.assertFalse(set(w['development'])&set(w['confirmation']));self.assertTrue((set(w['development'])|set(w['confirmation'])).isdisjoint(excluded_words()))
  for split in w:
   for x in w[split]:self.assertIn(tuple(reversed(x)),w[split])
 def test_models_expose_logits_and_features(self):
  x=torch.randn(2,3,32,32)
  for f in FAMILIES:
   m=make_model(f);y,z=logits_and_features(m,f,x);self.assertEqual(y.shape,(2,100));self.assertEqual(z.shape[0],2)
 def test_data_provenance(self):
  p=json.loads((ROOT/'v4/phase2f/DATA_PROVENANCE.json').read_text());self.assertEqual(p['canonical_test_md5'],'f0ef6b0ae62326f3e7ffdfab6717acfc');self.assertEqual(p['canonical_test_pixel_matches'],10000)
if __name__=='__main__':unittest.main()
