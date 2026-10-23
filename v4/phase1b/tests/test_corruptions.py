import unittest

import torch

from v4.phase1.corruptions import PRIMITIVES, apply_primitive
from v4.phase1b.corruptions import (
    STRENGTH_GRID,
    apply_attenuated_primitive,
    apply_calibrated_word,
    validate_strengths,
)


class AttenuatedCorruptionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.images = torch.rand(5, 3, 32, 32)
        self.ids = torch.arange(5)
        self.strengths = {name: 0.2 for name in PRIMITIVES}

    def test_grid_is_fixed_sorted_and_bounded(self):
        self.assertEqual(tuple(sorted(set(STRENGTH_GRID))), STRENGTH_GRID)
        self.assertEqual(STRENGTH_GRID[0], 0.0)
        self.assertEqual(STRENGTH_GRID[-1], 1.0)

    def test_homotopy_has_exact_endpoints(self):
        for primitive in PRIMITIVES:
            zero = apply_attenuated_primitive(self.images, primitive, self.ids, 0.0)
            one = apply_attenuated_primitive(self.images, primitive, self.ids, 1.0)
            torch.testing.assert_close(zero, self.images, atol=0, rtol=0)
            torch.testing.assert_close(
                one, apply_primitive(self.images, primitive, self.ids), atol=0, rtol=0
            )

    def test_word_is_chronological(self):
        word = ("ImpulseNoise", "GaussianBlur")
        complete = apply_calibrated_word(self.images, word, self.ids, self.strengths)
        manual = apply_attenuated_primitive(
            apply_attenuated_primitive(
                self.images, "ImpulseNoise", self.ids, self.strengths["ImpulseNoise"]
            ),
            "GaussianBlur",
            self.ids,
            self.strengths["GaussianBlur"],
        )
        torch.testing.assert_close(complete, manual)

    def test_strength_map_must_be_complete(self):
        with self.assertRaises(ValueError):
            validate_strengths({"Contrast": 0.2})
        invalid = dict(self.strengths)
        invalid["Contrast"] = 1.1
        with self.assertRaises(ValueError):
            validate_strengths(invalid)


if __name__ == "__main__":
    unittest.main()
