#!/usr/bin/env python3
"""Generate the exact public Mason et al. composition-word protocol.

Source mirrored in logic (not code copied verbatim):
https://github.com/ianxmason/compositional-robustness/blob/
fc62a92ccd11f64eec26c7a019e9e0957fd9bd33/src/data/sample_corruptions.py

The upstream seed is 2778.  It creates 167 words per dataset:
identity + 6 singles + 30 ordered pairs + 40 triples + 30 quadruples +
30 quintuples + 30 length-six permutations.
"""

from __future__ import annotations

import argparse
import math
import random
from collections import Counter
from itertools import combinations, permutations
from pathlib import Path

from v4.benchmark.schema import BenchmarkManifest, Condition, ConditionRole

SOURCE_COMMIT = "fc62a92ccd11f64eec26c7a019e9e0957fd9bd33"
SOURCE_URL = "https://github.com/ianxmason/compositional-robustness"
SEED = 2778
DATASETS = ("EMNIST", "CIFAR-10", "FaceScrub")
PRIMITIVES = (
    "Contrast",
    "GaussianBlur",
    "ImpulseNoise",
    "Invert",
    "Rotate90",
    "Swirl",
)
# Constructor defaults in the pinned upstream data/data_transforms.py.
DEFAULT_SEVERITY = {
    "Contrast": 2,
    "GaussianBlur": 2,
    "ImpulseNoise": 5,
    "Invert": "fixed",
    "Rotate90": "fixed",
    "Swirl": 3,
    "Identity": "identity",
}


def exact_words() -> list[tuple[str, ...]]:
    rng = random.Random(SEED)
    words: list[tuple[str, ...]] = [("Identity",)]
    words.extend((primitive,) for primitive in PRIMITIVES)
    words.extend(permutations(PRIMITIVES, 2))
    for length in range(3, len(PRIMITIVES) + 1):
        groups = list(combinations(PRIMITIVES, length))
        sample_count = math.ceil(30 / len(groups))
        for group in groups:
            words.extend(rng.sample(list(permutations(group)), sample_count))
    if len(words) != 167 or len(set(words)) != 167:
        raise AssertionError("upstream protocol must contain 167 unique words")
    return words


def build_manifest() -> BenchmarkManifest:
    words = exact_words()
    conditions: list[Condition] = []
    for dataset in DATASETS:
        for index, word in enumerate(words):
            identity = word == ("Identity",)
            conditions.append(
                Condition(
                    condition_id=f"mason-{dataset.lower().replace('-', '')}-{index:03d}",
                    dataset=dataset,
                    word=word,
                    severities=tuple(DEFAULT_SEVERITY[primitive] for primitive in word),
                    role=ConditionRole.UNASSIGNED,
                    word_family="|".join(sorted(word)),
                    random_field_id="upstream-deterministic-or-runtime-rng",
                    metadata={
                        "template_index": index,
                        "length": 0 if identity else len(word),
                        "is_identity": identity,
                        "upstream_order": index,
                    },
                )
            )
    length_counts = Counter(0 if word == ("Identity",) else len(word) for word in words)
    return BenchmarkManifest(
        benchmark_name="Mason-2023-exact-compositional-corruptions",
        conditions=conditions,
        source={
            "repository": SOURCE_URL,
            "commit": SOURCE_COMMIT,
            "path": "src/data/sample_corruptions.py",
            "seed": SEED,
        },
        metadata={
            "datasets": list(DATASETS),
            "primitives": list(PRIMITIVES),
            "conditions_per_dataset": len(words),
            "total_dataset_conditions": len(conditions),
            "word_count_by_length": {str(k): v for k, v in sorted(length_counts.items())},
            "constructor_default_severity": DEFAULT_SEVERITY,
            "severity_note": "Pinned to constructor defaults in upstream data/data_transforms.py.",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="v4/benchmark/manifests/mason_exact.json",
        help="destination JSON path",
    )
    args = parser.parse_args()
    manifest = build_manifest()
    manifest.write(args.output)
    print(f"wrote {len(manifest.conditions)} dataset-conditions to {args.output}")
    print(f"manifest sha256: {manifest.sha256()}")


if __name__ == "__main__":
    main()
