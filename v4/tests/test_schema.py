import json
import tempfile
import unittest
from pathlib import Path

from v4.benchmark.schema import (
    BenchmarkManifest,
    Condition,
    ConditionRole,
    InformationAccess,
    InformationRegime,
)


class SchemaTests(unittest.TestCase):
    def condition(self, condition_id="c1"):
        return Condition(
            condition_id=condition_id,
            dataset="CIFAR-10",
            word=("Blur", "Noise"),
            severities=(1.0, 2.0),
            role=ConditionRole.VALIDATION,
            word_family="Blur|Noise",
        )

    def test_manifest_roundtrip_and_hash(self):
        manifest = BenchmarkManifest(
            benchmark_name="unit",
            conditions=[self.condition()],
            source={"url": "https://example.invalid"},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            manifest.write(path)
            recovered = BenchmarkManifest.read(path)
            self.assertEqual(recovered.sha256(), manifest.sha256())
            self.assertEqual(recovered.conditions[0].word, ("Blur", "Noise"))

    def test_duplicate_ids_are_rejected(self):
        manifest = BenchmarkManifest("unit", [self.condition(), self.condition()], source={})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            manifest.validate()

    def test_r2_forbids_composed_target_images(self):
        invalid = InformationAccess(
            regime=InformationRegime.R2_PAIR_CALIBRATED,
            source_labels=True,
            primitive_images=True,
            primitive_behavior_labels=True,
            pair_behavior_labels=True,
            composed_target_images=True,
            composed_target_labels=False,
        )
        with self.assertRaisesRegex(ValueError, "cannot access"):
            invalid.validate()

    def test_valid_r2_contract(self):
        access = InformationAccess(
            regime=InformationRegime.R2_PAIR_CALIBRATED,
            source_labels=True,
            primitive_images=True,
            primitive_behavior_labels=True,
            pair_behavior_labels=True,
            composed_target_images=False,
            composed_target_labels=False,
        )
        access.validate()

    def test_r2_requires_primitive_information(self):
        invalid = InformationAccess(
            regime=InformationRegime.R2_PAIR_CALIBRATED,
            source_labels=True,
            primitive_images=False,
            primitive_behavior_labels=False,
            pair_behavior_labels=True,
            composed_target_images=False,
            composed_target_labels=False,
        )
        with self.assertRaisesRegex(ValueError, "primitive"):
            invalid.validate()

    def test_r4_requires_target_images_and_labels(self):
        valid = InformationAccess(
            regime=InformationRegime.R4_LABELED_TARGET,
            source_labels=True,
            primitive_images=False,
            primitive_behavior_labels=False,
            pair_behavior_labels=False,
            composed_target_images=True,
            composed_target_labels=True,
        )
        valid.validate()


if __name__ == "__main__":
    unittest.main()
