"""Property tests for the V4 exact algebraic core."""

import unittest

import torch

from v4.src.algebra import (
    AlgebraState,
    compose_closed_form,
    compose_streaming,
    compose_via_matrices,
    signed_power_inverse,
    signed_power_transform,
    state_from_sequence,
)


class AlgebraPropertyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2401)
        self.dtype = torch.float64
        self.p = torch.tensor([1.0, 1.4, 2.3, 4.0, 6.5], dtype=self.dtype)

    def random_word(self, length=6):
        behavior = 0.3 * torch.randn(length, 5, dtype=self.dtype)
        z = torch.randn(length, 7, dtype=self.dtype)
        pi = torch.nn.functional.softplus(torch.randn(length, 4, dtype=self.dtype))
        kappa = torch.randn(length, 4, dtype=self.dtype)
        return behavior, z, pi, kappa

    def assert_output_close(self, left, right, atol=2e-11, rtol=2e-10):
        self.assertEqual(left.length, right.length)
        torch.testing.assert_close(left.base, right.base, atol=atol, rtol=rtol)
        torch.testing.assert_close(left.symmetric, right.symmetric, atol=atol, rtol=rtol)
        torch.testing.assert_close(left.ordered, right.ordered, atol=atol, rtol=rtol)

    def test_transform_inverse_is_exact_for_signed_values_and_zero(self):
        values = torch.tensor(
            [-10.0, -1.0, -1e-8, 0.0, 1e-8, 0.5, 9.0], dtype=self.dtype
        )
        p = torch.tensor([1.0, 1.1, 1.8, 2.0, 3.2, 5.0, 8.0], dtype=self.dtype)
        recovered = signed_power_inverse(signed_power_transform(values, p), p)
        torch.testing.assert_close(recovered, values, atol=5e-13, rtol=2e-10)
        self.assertEqual(float(signed_power_transform(values, p)[3]), 0.0)

    def test_rejects_exponents_below_one(self):
        with self.assertRaisesRegex(ValueError, "p >= 1"):
            signed_power_transform(torch.ones(2), torch.tensor([0.9, 1.0]))

    def test_closed_form_streaming_and_literal_matrices_match_lengths_1_to_6(self):
        for length in range(1, 7):
            word = self.random_word(length)
            expected = compose_closed_form(*word, self.p)
            self.assert_output_close(compose_streaming(*word, self.p), expected)
            self.assert_output_close(compose_via_matrices(*word, self.p), expected)

    def test_reversal_preserves_symmetric_and_negates_ordered(self):
        word = self.random_word(6)
        forward = compose_closed_form(*word, self.p)
        reverse = compose_closed_form(*(x.flip(0) for x in word), self.p)
        torch.testing.assert_close(reverse.base, forward.base)
        torch.testing.assert_close(reverse.symmetric, forward.symmetric)
        torch.testing.assert_close(reverse.ordered, -forward.ordered)

    def test_state_combine_is_associative(self):
        behavior, z, pi, kappa = self.random_word(6)
        a = state_from_sequence(behavior[:2], z[:2], pi[:2], kappa[:2], self.p)
        b = state_from_sequence(behavior[2:4], z[2:4], pi[2:4], kappa[2:4], self.p)
        c = state_from_sequence(behavior[4:], z[4:], pi[4:], kappa[4:], self.p)
        left = a.then(b).then(c)
        right = a.then(b.then(c))
        self.assert_output_close(left.extract(self.p), right.extract(self.p), atol=1e-12, rtol=1e-12)

    def test_identity_state_is_two_sided(self):
        behavior, z, pi, kappa = self.random_word(4)
        state = state_from_sequence(behavior, z, pi, kappa, self.p)
        identity = AlgebraState.identity(5, 7, 4, dtype=self.dtype)
        self.assert_output_close(identity.then(state).extract(self.p), state.extract(self.p))
        self.assert_output_close(state.then(identity).extract(self.p), state.extract(self.p))

    def test_repeated_identical_primitive_has_zero_order_channel(self):
        behavior, z, pi, kappa = self.random_word(1)
        repeated = tuple(x.repeat(5, 1) for x in (behavior, z, pi, kappa))
        output = compose_streaming(*repeated, self.p)
        torch.testing.assert_close(output.ordered, torch.zeros_like(output.ordered), atol=1e-12, rtol=0)

    def test_gradients_are_finite_near_zero(self):
        x = torch.tensor([-1e-10, 0.0, 1e-10, 0.2, -0.2], dtype=self.dtype, requires_grad=True)
        y = signed_power_inverse(signed_power_transform(x, self.p), self.p).square().sum()
        y.backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_input_shapes_are_validated(self):
        behavior, z, pi, kappa = self.random_word(2)
        with self.assertRaisesRegex(ValueError, "share a positive length"):
            compose_streaming(behavior[:1], z, pi, kappa, self.p)


if __name__ == "__main__":
    unittest.main()
