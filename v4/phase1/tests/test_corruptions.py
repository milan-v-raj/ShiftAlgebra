import unittest

import torch

from v4.phase1.corruptions import PRIMITIVES, apply_primitive, apply_word


class CorruptionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.images = torch.rand(8, 3, 32, 32)
        self.ids = torch.arange(8)

    def test_all_primitives_are_finite_and_bounded(self):
        for primitive in PRIMITIVES:
            output = apply_primitive(self.images, primitive, self.ids)
            self.assertEqual(output.shape, self.images.shape)
            self.assertTrue(torch.isfinite(output).all())
            self.assertGreaterEqual(float(output.min()), 0.0)
            self.assertLessEqual(float(output.max()), 1.0)

    def test_impulse_field_is_batching_independent(self):
        complete = apply_primitive(self.images, "ImpulseNoise", self.ids)
        split = torch.cat(
            [
                apply_primitive(self.images[:3], "ImpulseNoise", self.ids[:3]),
                apply_primitive(self.images[3:], "ImpulseNoise", self.ids[3:]),
            ]
        )
        torch.testing.assert_close(complete, split, atol=0, rtol=0)

    def test_word_chronology_and_noncommutativity(self):
        first = apply_word(self.images, ("ImpulseNoise", "GaussianBlur"), self.ids)
        reverse = apply_word(self.images, ("GaussianBlur", "ImpulseNoise"), self.ids)
        self.assertFalse(torch.allclose(first, reverse))
        manual = apply_primitive(
            apply_primitive(self.images, "ImpulseNoise", self.ids),
            "GaussianBlur",
            self.ids,
        )
        torch.testing.assert_close(first, manual)

    def test_invert_is_involution(self):
        twice = apply_word(self.images, ("Invert", "Invert"), self.ids)
        torch.testing.assert_close(twice, self.images)


if __name__ == "__main__":
    unittest.main()
