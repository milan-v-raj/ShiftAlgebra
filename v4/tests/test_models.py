import unittest

import torch

from v4.src.models import LpOnly, MBCore


class ModelTests(unittest.TestCase):
    def test_mb_core_constrains_p_and_propagates_gradients(self):
        torch.manual_seed(9)
        model = MBCore(behavior_dim=5, symmetric_dim=7, order_dim=4, hidden_dim=12)
        behavior = torch.randn(3, 5)
        z = torch.randn(3, 7)
        pi = torch.nn.functional.softplus(torch.randn(3, 4))
        kappa = torch.randn(3, 4)
        prediction = model(behavior, z, pi, kappa)
        prediction.square().mean().backward()
        self.assertEqual(tuple(prediction.shape), (5,))
        self.assertTrue(torch.all(model.p >= 1.0))
        self.assertIsNotNone(model.raw_p.grad)
        self.assertTrue(torch.isfinite(model.raw_p.grad).all())
        self.assertGreater(float(model.residual[-1].weight.detach().abs().sum()), 0.0)

    def test_lp_only_has_one_exponent_per_behavior(self):
        model = LpOnly(5)
        output = model(torch.randn(4, 5))
        self.assertEqual(tuple(output.shape), (5,))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 5)


if __name__ == "__main__":
    unittest.main()
