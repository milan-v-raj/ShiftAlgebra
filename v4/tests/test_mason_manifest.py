import unittest
from collections import Counter

from v4.benchmark.generate_mason_manifest import (
    DATASETS,
    DEFAULT_SEVERITY,
    PRIMITIVES,
    build_manifest,
    exact_words,
)


class MasonManifestTests(unittest.TestCase):
    def test_exact_upstream_counts(self):
        words = exact_words()
        counts = Counter(0 if word == ("Identity",) else len(word) for word in words)
        self.assertEqual(counts, {0: 1, 1: 6, 2: 30, 3: 40, 4: 30, 5: 30, 6: 30})
        self.assertEqual(len([word for word in words if len(word) == 2]), 30)
        self.assertEqual(len(set(words)), 167)

    def test_all_ordered_distinct_pairs_are_present(self):
        pairs = {word for word in exact_words() if len(word) == 2}
        expected = {(a, b) for a in PRIMITIVES for b in PRIMITIVES if a != b}
        self.assertEqual(pairs, expected)

    def test_manifest_has_167_conditions_per_dataset(self):
        manifest = build_manifest()
        manifest.validate()
        counts = Counter(condition.dataset for condition in manifest.conditions)
        self.assertEqual(counts, {dataset: 167 for dataset in DATASETS})
        self.assertEqual(len(manifest.conditions), 501)

    def test_manifest_pins_constructor_default_severities(self):
        manifest = build_manifest()
        first_dataset = manifest.conditions[:167]
        for condition in first_dataset:
            self.assertEqual(
                condition.severities,
                tuple(DEFAULT_SEVERITY[primitive] for primitive in condition.word),
            )


if __name__ == "__main__":
    unittest.main()
