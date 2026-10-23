import unittest

import torch

from v4.src.order_models import (
    AntisymmetrizedPairNetwork,
    SkewBilinearOrder,
    potential_cycle_residual,
    potential_order_effect,
    rock_paper_scissors_target,
)


class OrderModelTests(unittest.TestCase):
    def test_potential_reversal_is_exactly_antisymmetric(self):
        pi_a = torch.tensor([1.2, 0.7])
        pi_b = torch.tensor([0.8, 1.1])
        kappa_a = torch.tensor([-0.4, 0.3])
        kappa_b = torch.tensor([0.9, -0.2])
        ab = potential_order_effect(pi_a, kappa_a, pi_b, kappa_b)
        ba = potential_order_effect(pi_b, kappa_b, pi_a, kappa_a)
        torch.testing.assert_close(ab, -ba)

    def test_transitive_potential_cannot_represent_three_cycle(self):
        result = potential_cycle_residual(rock_paper_scissors_target())
        self.assertAlmostEqual(result["target_circulation"], 3.0)
        self.assertAlmostEqual(result["predicted_circulation"], 0.0)
        self.assertAlmostEqual(result["minimum_mse"], 1.0)

    def test_skew_bilinear_can_represent_three_cycle(self):
        target = rock_paper_scissors_target(dtype=torch.float32)
        model = SkewBilinearOrder(input_dim=3, output_dim=1)
        with torch.no_grad():
            model.raw[0].copy_(target)
        basis = torch.eye(3)
        for i, j in ((0, 1), (1, 2), (2, 0)):
            self.assertAlmostEqual(float(model(basis[i], basis[j])[0].detach()), 1.0)
            self.assertAlmostEqual(float(model(basis[j], basis[i])[0].detach()), -1.0)

    def test_pair_network_is_antisymmetric_by_construction(self):
        torch.manual_seed(5)
        model = AntisymmetrizedPairNetwork(4, 3)
        a, b = torch.randn(7, 4), torch.randn(7, 4)
        torch.testing.assert_close(model(a, b), -model(b, a))
        torch.testing.assert_close(model(a, a), torch.zeros(7, 3))


if __name__ == "__main__":
    unittest.main()
