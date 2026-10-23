import unittest
from collections import Counter

from v4.phase2.manifest import model_records, word_records


class ManifestTests(unittest.TestCase):
    def test_model_bank_counts_and_clusters(self):
        records = model_records()
        self.assertEqual(Counter(row.split for row in records), {"development": 30, "confirmation": 30})
        self.assertEqual(len({row.lineage_id for row in records if row.split == "development"}), 15)
        self.assertEqual(len({row.lineage_id for row in records if row.split == "confirmation"}), 15)
        self.assertEqual(len({row.family for row in records}), 5)
        self.assertEqual(len({row.recipe for row in records}), 3)

    def test_word_roles_and_reverses(self):
        rows = word_records()
        self.assertEqual(len(rows), 167)
        self.assertEqual(
            Counter(row["role"] for row in rows),
            {"calibration": 37, "method_development": 40, "confirmation": 90},
        )
        lookup = {row["word_string"]: row for row in rows}
        for row in rows:
            reverse = ">".join(reversed(row["word_string"].split(">")))
            if reverse in lookup:
                self.assertEqual(row["role"], lookup[reverse]["role"])


if __name__ == "__main__":
    unittest.main()
