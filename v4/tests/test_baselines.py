import unittest

import numpy as np

from v4.baselines.source_family import SourceFamilyRidge


class SourceBaselineTests(unittest.TestCase):
    def setUp(self):
        self.source = np.array(
            [[0.50, 0.60], [0.60, 0.70], [0.70, 0.80], [0.80, 0.90], [0.90, 1.00]]
        )
        self.family = ["resnet", "resnet", "vit", "vit", "convnext"]
        self.recipe = ["erm", "augmix", "erm", "augmix", "erm"]
        self.target = np.array([0.40, 0.52, 0.63, 0.71, 0.78])

    def test_fit_predict_and_information_contract(self):
        baseline = SourceFamilyRidge(alpha=0.1).fit(
            self.source, self.family, self.recipe, self.target
        )
        prediction = baseline.predict(self.source, self.family, self.recipe)
        self.assertEqual(prediction.shape, self.target.shape)
        self.assertTrue(np.isfinite(prediction).all())
        contract = baseline.describe_information()
        self.assertEqual(contract["regime"], "R0")
        self.assertFalse(contract["uses_composed_target_images"])
        self.assertFalse(contract["uses_composed_target_labels"])

    def test_unknown_family_and_recipe_use_explicit_unknown_columns(self):
        baseline = SourceFamilyRidge(alpha=1.0).fit(
            self.source, self.family, self.recipe, self.target
        )
        prediction = baseline.predict([[0.75, 0.85]], ["new_arch"], ["new_recipe"])
        self.assertEqual(prediction.shape, (1,))
        self.assertTrue(np.isfinite(prediction).all())

    def test_rejects_misaligned_metadata(self):
        with self.assertRaisesRegex(ValueError, "one non-empty"):
            SourceFamilyRidge().fit(self.source, self.family[:-1], self.recipe, self.target)


if __name__ == "__main__":
    unittest.main()
