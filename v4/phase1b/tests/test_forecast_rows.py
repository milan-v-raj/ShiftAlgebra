import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from v4.benchmark.generate_mason_manifest import PRIMITIVES, exact_words
from v4.phase1.behavior import BEHAVIOR_NAMES
from v4.phase1.forecast import run_forecast_evaluation
from v4.phase1b import PROTOCOL_ID


class ForecastRowTests(unittest.TestCase):
    def test_target_rows_are_scored_but_not_fit(self):
        primitive = {
            name: np.array([0.01 * (i + 1), -0.01, 0.003, 0.02, -0.005 * (i + 1)])
            for i, name in enumerate(PRIMITIVES)
        }
        with tempfile.TemporaryDirectory() as directory:
            behavior = Path(directory) / "behavior.csv"
            fields = ["word", "accuracy"] + [f"delta_{name}" for name in BEHAVIOR_NAMES]
            with behavior.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for word in exact_words():
                    delta = np.zeros(5) if word == ("Identity",) else sum(primitive[item] for item in word)
                    writer.writerow({
                        "word": ">".join(word),
                        "accuracy": 0.8 + delta[-1],
                        **{f"delta_{name}": value for name, value in zip(BEHAVIOR_NAMES, delta)},
                    })
            output = Path(directory) / "forecast.json"
            result = run_forecast_evaluation(
                behavior,
                PRIMITIVES,
                output,
                lp_steps=20,
                potential_steps=20,
                protocol_id=PROTOCOL_ID,
                include_target_rows=True,
            )
        self.assertEqual(len(result["target_rows"]), 130)
        self.assertFalse(result["target_access_during_fit"])
        self.assertTrue(result["protocol"].startswith(PROTOCOL_ID))
        self.assertEqual(set(result["target_rows"][0]["methods"]), set(result["methods"]))


if __name__ == "__main__":
    unittest.main()
