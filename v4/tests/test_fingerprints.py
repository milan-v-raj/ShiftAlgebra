import unittest

import torch

from v4.src.fingerprints import FingerprintConfig, FixedFingerprint


class FingerprintTests(unittest.TestCase):
    def make_inputs(self, widths, classes=10, channels=3):
        torch.manual_seed(sum(widths) + classes + channels)
        n = 12
        clean_layers = [torch.randn(n, width, dtype=torch.float64) for width in widths]
        shifted_layers = [value + 0.1 * torch.randn_like(value) for value in clean_layers]
        clean_logits = torch.randn(n, classes, dtype=torch.float64)
        shifted_logits = clean_logits + 0.2 * torch.randn_like(clean_logits)
        clean_pixels = torch.rand(n, channels, 12, 10, dtype=torch.float64)
        shifted_pixels = (clean_pixels + 0.05 * torch.randn_like(clean_pixels)).clamp(0, 1)
        return clean_layers, shifted_layers, clean_logits, shifted_logits, clean_pixels, shifted_pixels

    def test_output_dimension_is_architecture_and_class_independent(self):
        extractor = FixedFingerprint(FingerprintConfig(depth_slots=4, projection_dim=8))
        first = extractor(*self.make_inputs([32, 64, 128], classes=10, channels=3))
        second = extractor(*self.make_inputs([96, 384, 768, 1024, 1280], classes=100, channels=1))
        self.assertEqual(first.numel(), extractor.output_dim)
        self.assertEqual(second.numel(), extractor.output_dim)
        self.assertEqual(first.shape, second.shape)

    def test_deterministic_for_same_input_and_seed(self):
        inputs = self.make_inputs([24, 48], classes=7)
        a = FixedFingerprint(FingerprintConfig(seed=44))(*inputs)
        b = FixedFingerprint(FingerprintConfig(seed=44))(*inputs)
        torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_paired_displacement_changes_when_pairing_changes(self):
        inputs = list(self.make_inputs([20, 40], classes=5))
        extractor = FixedFingerprint()
        aligned = extractor(*inputs)
        inputs[1] = [layer.flip(0) for layer in inputs[1]]
        inputs[3] = inputs[3].flip(0)
        inputs[5] = inputs[5].flip(0)
        shuffled = extractor(*inputs)
        self.assertFalse(torch.allclose(aligned, shuffled))

    def test_rejects_unpaired_sample_counts(self):
        inputs = list(self.make_inputs([16], classes=3))
        inputs[3] = inputs[3][:-1]
        with self.assertRaisesRegex(ValueError, "share"):
            FixedFingerprint()(*inputs)


if __name__ == "__main__":
    unittest.main()
