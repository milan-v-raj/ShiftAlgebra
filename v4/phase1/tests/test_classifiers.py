import unittest

import torch

from v4.phase1.classifiers import SmallConvNet


class ClassifierTests(unittest.TestCase):
    def test_forward_exposes_multidepth_fixed_batches(self):
        model = SmallConvNet(3, 10, width=8)
        logits, layers = model.forward_with_layers(torch.randn(5, 3, 32, 32))
        self.assertEqual(logits.shape, (5, 10))
        self.assertEqual([layer.shape[0] for layer in layers], [5, 5, 5])
        self.assertEqual([layer.shape[1] for layer in layers], [8, 16, 32])


if __name__ == "__main__":
    unittest.main()
