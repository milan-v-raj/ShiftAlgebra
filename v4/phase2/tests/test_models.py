import unittest

import torch

from v4.phase2.models import FAMILY_WIDTHS, make_model


class ModelTests(unittest.TestCase):
    def test_all_families_expose_three_layers(self):
        inputs = torch.randn(3, 3, 32, 32)
        for family in FAMILY_WIDTHS:
            model = make_model(family)
            logits, layers = model.forward_with_layers(inputs)
            self.assertEqual(logits.shape, (3, 10))
            self.assertEqual(len(layers), 3)
            self.assertTrue(all(layer.shape[0] == 3 and layer.ndim == 2 for layer in layers))


if __name__ == "__main__":
    unittest.main()
