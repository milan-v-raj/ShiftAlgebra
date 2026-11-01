#!/usr/bin/env python3
"""Freeze the strict no-refit Phase-2G architecture-transfer bank."""
from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from v4.phase2f import PRIMITIVES
from v4.phase2f.data import digest
from v4.phase2f.manifest import target_words as phase2f_target_words
from v4.phase2g import METRICS, PROTOCOL_ID
from v4.phase2g.models import FAMILIES, make_model

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "v4/phase2g/PHASE2G_MANIFEST.json"
RECIPES = ("erm_flip", "label_smooth", "noise_aug")
EPOCHS = (1, 2)
WORD_SEED = 22711
FROZEN_METHOD = ROOT / "v4/phase2f/FROZEN_FULL_VECTOR_METHODS.json"
EXPECTED_METHOD_SHA256 = "243761ddd3749dcc0149e83cca7e40241d43a3800320d8f777bac624b1f5827e"
EXPECTED_DEVELOPMENT_SHA256 = "8c3289ac47c01267751a2f3d8104d1940f04e20b8fa0cec4dea0c2baef7b90a7"


@dataclass(frozen=True)
class ModelRecord:
    model_id: str
    lineage_id: str
    family: str
    recipe: str
    seed: int
    epoch: int
    parameter_count: int


def model_records():
    output = []
    for family_index, family in enumerate(FAMILIES):
        parameters = sum(parameter.numel() for parameter in make_model(family).parameters())
        for recipe_index, recipe in enumerate(RECIPES):
            seed = 67100 + 100 * family_index + recipe_index
            lineage = f"p2g-{family}-{recipe}-s{seed}"
            for epoch in EPOCHS:
                output.append(ModelRecord(
                    f"{lineage}-e{epoch}", lineage, family, recipe, seed, epoch, parameters
                ))
    return output


def reverse_group(word):
    forward, reverse = ">".join(word), ">".join(reversed(word))
    return min(forward, reverse)


def excluded_words():
    from v4.benchmark.generate_mason_manifest import exact_words
    from v4.phase2b.manifest import fresh_words as phase2b_words
    from v4.phase2c.manifest import fresh_words as phase2c_words
    from v4.phase2d.manifest import fresh_words as phase2d_words
    from v4.phase2e.manifest import fresh_words as phase2e_words

    prior = set(exact_words())
    prior.update(phase2b_words())
    prior.update(phase2c_words())
    prior.update(phase2d_words())
    prior.update(phase2e_words())
    phase2f = phase2f_target_words()
    prior.update(phase2f["development"])
    prior.update(phase2f["confirmation"])
    return prior


def target_words():
    excluded = excluded_words()
    output = []
    for length in (4, 5, 6):
        candidates = {}
        for word in itertools.permutations(PRIMITIVES, length):
            reverse = tuple(reversed(word))
            if word not in excluded and reverse not in excluded:
                candidates[reverse_group(word)] = (word, reverse)
        score = lambda group: hashlib.sha256(f"{WORD_SEED}|{group}".encode()).hexdigest()
        for group in sorted(candidates, key=score)[:10]:
            output.extend(sorted(candidates[group]))
    if len(output) != 60 or set(output) & excluded:
        raise RuntimeError("invalid fresh target-word design")
    return output


def calibration_words():
    return [("Identity",)] + [(primitive,) for primitive in PRIMITIVES] + [
        (left, right) for left in PRIMITIVES for right in PRIMITIVES if left != right
    ]


def words():
    return calibration_words() + target_words()


def word_records():
    calibration = set(calibration_words())
    return [
        {
            "condition_index": index,
            "word": list(word),
            "word_string": ">".join(word),
            "length": 0 if word == ("Identity",) else len(word),
            "role": "calibration" if word in calibration else "confirmation",
            "reverse_group": reverse_group(word),
        }
        for index, word in enumerate(words())
    ]


def payload():
    if digest(FROZEN_METHOD) != EXPECTED_METHOD_SHA256:
        raise ValueError("Phase-2F frozen method hash changed")
    frozen = json.loads(FROZEN_METHOD.read_text())
    if frozen["development_behavior_sha256"] != EXPECTED_DEVELOPMENT_SHA256:
        raise ValueError("Phase-2F development boundary changed")
    provenance = json.loads((ROOT / "v4/phase2f/DATA_PROVENANCE.json").read_text())
    return {
        "protocol": PROTOCOL_ID,
        "status": "STRICT_NOREFIT_ARCHITECTURE_PROTOCOL_FROZEN_OUTCOMES_UNOPENED",
        "dataset": "CIFAR-100 fine labels",
        "classes": 100,
        "chance": 0.01,
        "metrics": list(METRICS),
        "models": [asdict(record) for record in model_records()],
        "words": word_records(),
        "image_partition": {"per_class": 20, "offset": 40},
        "counts": {
            "families": 6,
            "recipes": 3,
            "lineages": 18,
            "snapshots": 36,
            "calibration_conditions": 37,
            "target_words": 60,
        },
        "word_seed": WORD_SEED,
        "frozen_method": {
            "path": "v4/phase2f/FROZEN_FULL_VECTOR_METHODS.json",
            "sha256": EXPECTED_METHOD_SHA256,
            "development_behavior_sha256": EXPECTED_DEVELOPMENT_SHA256,
            "allowed_change": "none: coefficients, q, scales, full/magnitude/random designs all transfer verbatim",
        },
        "architecture_boundary": {
            "development_families": [
                "vgg_holdout", "dense_holdout", "inception_holdout",
                "squeeze_holdout", "mixer_holdout", "attention_holdout",
            ],
            "confirmation_families": list(FAMILIES),
            "family_overlap": [],
            "unknown_family_encoding": "all six frozen family indicators equal zero",
        },
        "gates": {
            "full_vector": [
                "macro normalized MAE full15 < g0",
                "family/lineage/reversal hierarchical 95% CI for improvement excludes zero",
                "at least four of six coordinate MAEs improve",
                "macro controlled model-rank Spearman improves",
                "accuracy top-1 selection regret is no worse",
                "at least five of six held-out family mean improvements are positive",
            ],
            "magnitude": [
                "macro normalized MAE below mean of twenty frozen random five-pair designs",
                "hierarchical 95% CI versus expected random excludes zero",
                "macro controlled rank at least random-design median",
                "accuracy regret no worse than random-design median",
            ],
        },
        "data_provenance": provenance,
    }


def main():
    OUTPUT.write_text(json.dumps(payload(), indent=2, sort_keys=True) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
