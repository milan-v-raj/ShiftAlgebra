import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from v4.benchmark.generate_mason_manifest import PRIMITIVES, exact_words
from v4.phase1.behavior import BEHAVIOR_NAMES
from v4.phase1.forecast import run_forecast_evaluation


class ForecastTests(unittest.TestCase):
    def test_protocol_split_and_target_access_contract(self):
        primitive = {
            name: np.array([0.01 * (i + 1), -0.02 * (i + 1), 0.005, 0.03, -0.01 * (i + 1)])
            for i, name in enumerate(PRIMITIVES)
        }
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "behavior.csv"
            fieldnames = ["word", "accuracy"] + [f"delta_{name}" for name in BEHAVIOR_NAMES]
            with csv_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for word in exact_words():
                    value = np.zeros(5) if word == ("Identity",) else sum(primitive[x] for x in word)
                    writer.writerow(
                        {"word": ">".join(word), "accuracy": 0.8, **{
                            f"delta_{name}": component
                            for name, component in zip(BEHAVIOR_NAMES, value)
                        }}
                    )
            output = Path(directory) / "result.json"
            result = run_forecast_evaluation(
                csv_path, PRIMITIVES, output, lp_steps=30, potential_steps=30
            )
        self.assertEqual(result["calibration_pairs"], 30)
        self.assertEqual(result["target_words"], 130)
        self.assertFalse(result["target_access_during_fit"])
        self.assertLess(result["methods"]["additive"]["accuracy_mae_pp"], 1e-9)


if __name__ == "__main__":
    unittest.main()
