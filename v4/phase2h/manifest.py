#!/usr/bin/env python3
"""Freeze the prospective Phase-2H order-ablation confirmation bank."""
from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from v4.phase2.train_bank import sha256
from v4.phase2f import PRIMITIVES
from v4.phase2f.manifest import target_words as phase2f_target_words
from v4.phase2g.manifest import target_words as phase2g_target_words
from v4.phase2g.models import FAMILIES, make_model
from v4.phase2h import METRICS, PROTOCOL_ID

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "v4/phase2h/PHASE2H_MANIFEST.json"
METHOD = ROOT / "v4/phase2h/FROZEN_ORDER_ABLATION_METHODS.json"
EXPECTED_METHOD_SHA256 = "9c4eef64aa2e1812306c6df3ba7c90e5f8a7bbf02f97eeece6603509d31e375e"
RECIPES = ("erm_flip", "label_smooth", "noise_aug")
EPOCHS = (1, 2)
WORD_SEED = 22721


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
            seed = 68100 + 100 * family_index + recipe_index
            lineage = f"p2h-{family}-{recipe}-s{seed}"
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
    prior.update(phase2g_target_words())
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
    if sha256(METHOD) != EXPECTED_METHOD_SHA256:
        raise ValueError("frozen order-ablation method hash changed")
    method = json.loads(METHOD.read_text())
    provenance = json.loads((ROOT / "v4/phase2f/DATA_PROVENANCE.json").read_text())
    return {
        "protocol": PROTOCOL_ID,
        "status": "ORDER_ABLATION_PROTOCOL_FROZEN_CONFIRMATION_OUTCOMES_UNOPENED",
        "dataset": "CIFAR-100 fine labels",
        "classes": 100,
        "chance": 0.01,
        "metrics": list(METRICS),
        "models": [asdict(record) for record in model_records()],
        "words": word_records(),
        "image_partition": {"per_class": 20, "offset": 60},
        "counts": {
            "families": 6,
            "recipes": 3,
            "lineages": 18,
            "snapshots": 36,
            "calibration_conditions": 37,
            "target_words": 60,
            "reversal_groups": 30,
        },
        "word_seed": WORD_SEED,
        "frozen_method": {
            "path": "v4/phase2h/FROZEN_ORDER_ABLATION_METHODS.json",
            "sha256": EXPECTED_METHOD_SHA256,
            "development_source": method["development_source"],
            "development_sha256": method["development_sha256"],
        },
        "comparison": {
            "symmetric": "metadata plus symmetric, independent, and envelope composition features",
            "order_aware": "same features plus one ordered-minus-symmetric channel",
            "measurement_budget": "equal: identity, six singles, thirty directed pairs",
            "nesting": "order-aware differs by exactly one directional feature",
        },
        "primary_gates": [
            "order-aware macro normalized MAE below symmetric",
            "family/lineage/reversal hierarchical 95% improvement CI excludes zero",
            "at least four of six raw coordinate MAEs improve",
            "order-aware reversal-contrast macro normalized MAE below symmetric",
            "order-aware macro controlled model-rank Spearman is no worse",
            "order-aware accuracy top-1 selection regret is no worse",
        ],
        "data_provenance": provenance,
    }


def main():
    OUTPUT.write_text(json.dumps(payload(), indent=2, sort_keys=True) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
