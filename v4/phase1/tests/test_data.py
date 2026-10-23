import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from v4.phase1.data import _decode_emnist_sheet, balanced_subset


class DataTests(unittest.TestCase):
    def test_balanced_subset_is_stable_and_disjoint_by_offset(self):
        labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        first = balanced_subset(labels, 2, offset=0)
        second = balanced_subset(labels, 2, offset=2)
        self.assertEqual(first.tolist(), [0, 2, 1, 3])
        self.assertTrue(set(first.tolist()).isdisjoint(second.tolist()))

    def test_emnist_sheet_decoding_is_row_major(self):
        tiles = np.stack(
            [np.full((28, 28), value, dtype=np.uint8) for value in (10, 20, 30, 40)]
        )
        sheet = tiles.reshape(2, 2, 28, 28).transpose(0, 2, 1, 3).reshape(56, 56)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sheet.webp"
            Image.fromarray(sheet).save(path, lossless=True)
            recovered = _decode_emnist_sheet(path, 4)
        self.assertEqual([int(tile.mean()) for tile in recovered], [10, 20, 30, 40])


if __name__ == "__main__":
    unittest.main()
