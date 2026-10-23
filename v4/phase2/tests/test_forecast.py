import unittest

from v4.benchmark.generate_mason_manifest import exact_words
from v4.phase2 import PRIMITIVES
from v4.phase2.forecast import forecast_one_model


class ForecastTests(unittest.TestCase):
    def synthetic_rows(self):
        ids = {name: index for index, name in enumerate(PRIMITIVES)}
        rows = {}
        for word in exact_words():
            name = ">".join(word)
            if word == ("Identity",):
                accuracy = 0.8
            elif len(word) == 1:
                accuracy = 0.72
            elif len(word) == 2:
                accuracy = 0.61 + 0.003 * (ids[word[1]] - ids[word[0]])
            else:
                accuracy = max(0.11, 0.8 - 0.08 * len(word))
            rows[name] = {
                "protocol": "test",
                "bank_split": "development",
                "model_id": "model",
                "lineage_id": "lineage",
                "family": "shallow",
                "recipe": "erm_flip",
                "seed": 1,
                "epoch": 1,
                "parameter_count": 100,
                "word": name,
                "word_role": "method_development" if len(word) >= 3 else "calibration",
                "reverse_group": min(name, ">".join(reversed(word))),
                "length": 0 if word == ("Identity",) else len(word),
                "accuracy": accuracy,
                "atc_accuracy": accuracy + 0.01,
                "doc_accuracy": accuracy + 0.02,
            }
        return rows

    def test_forecasts_are_bounded_and_ordered_channel_is_distinct(self):
        predictions = forecast_one_model(self.synthetic_rows(), q=1.0)
        self.assertEqual(len(predictions), 130)
        for row in predictions:
            for name in (
                "primitive_mean", "max_damage", "product_retention", "pair_accuracy_q",
                "hazard_symmetric_q", "hazard_ordered_q",
            ):
                self.assertGreaterEqual(row[name], 0.0)
                self.assertLessEqual(row[name], 1.0)
        lookup = {row["word"]: row for row in predictions}
        checked = 0
        for word, row in lookup.items():
            reverse = ">".join(reversed(word.split(">")))
            if reverse in lookup and word < reverse:
                self.assertAlmostEqual(row["hazard_symmetric_q"], lookup[reverse]["hazard_symmetric_q"])
                if abs(row["hazard_ordered_q"] - lookup[reverse]["hazard_ordered_q"]) > 1e-12:
                    checked += 1
        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main()
