#!/usr/bin/env python3
"""Freeze Phase-3A realistic-corruption development and confirmation banks."""
from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from v4.phase2g.models import FAMILIES, make_model
from v4.phase2h.manifest import model_records as phase2h_model_records
from v4.phase3a import METRICS, PROTOCOL_ID
from v4.phase3a.corruptions import PARAMETERS, PRIMITIVES

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "v4/phase3a/PHASE3A_MANIFEST.json"
RECIPES = ("erm_flip", "label_smooth", "noise_aug")
EPOCHS = (1, 2)
WORD_SEED = 23701


@dataclass(frozen=True)
class ModelRecord:
    model_id: str
    lineage_id: str
    bank_split: str
    family: str
    recipe: str
    seed: int
    epoch: int
    parameter_count: int


def model_records():
    output = [
        ModelRecord(
            record.model_id,
            record.lineage_id,
            "development",
            record.family,
            record.recipe,
            record.seed,
            record.epoch,
            record.parameter_count,
        )
        for record in phase2h_model_records()
    ]
    for family_index, family in enumerate(FAMILIES):
        parameters = sum(parameter.numel() for parameter in make_model(family).parameters())
        for recipe_index, recipe in enumerate(RECIPES):
            seed = 69100 + 100 * family_index + recipe_index
            lineage = f"p3a-confirmation-{family}-{recipe}-s{seed}"
            for epoch in EPOCHS:
                output.append(ModelRecord(
                    f"{lineage}-e{epoch}", lineage, "confirmation", family, recipe,
                    seed, epoch, parameters,
                ))
    return output


def reverse_group(word):
    forward, reverse = ">".join(word), ">".join(reversed(word))
    return min(forward, reverse)


def target_words():
    output = {"development": [], "confirmation": []}
    used_groups = set()
    for length in (4, 5, 6):
        candidates = {}
        for word in itertools.permutations(PRIMITIVES, length):
            candidates[reverse_group(word)] = (word, tuple(reversed(word)))
        score = lambda group: hashlib.sha256(f"{WORD_SEED}|{length}|{group}".encode()).hexdigest()
        chosen = [group for group in sorted(candidates, key=score) if group not in used_groups][:20]
        used_groups.update(chosen)
        for split, groups in (("development", chosen[:10]), ("confirmation", chosen[10:])):
            for group in groups:
                output[split].extend(sorted(candidates[group]))
    if any(len(output[split]) != 60 for split in output):
        raise RuntimeError("invalid target count")
    if set(output["development"]) & set(output["confirmation"]):
        raise RuntimeError("development/confirmation target overlap")
    return output


def calibration_words():
    return [("Identity",)] + [(primitive,) for primitive in PRIMITIVES] + [
        (left, right) for left in PRIMITIVES for right in PRIMITIVES if left != right
    ]


def words_for(split):
    return calibration_words() + target_words()[split]


def word_records(split):
    calibration = set(calibration_words())
    return [
        {
            "condition_index": index,
            "word": list(word),
            "word_string": ">".join(word),
            "length": 0 if word == ("Identity",) else len(word),
            "role": "calibration" if word in calibration else split,
            "reverse_group": reverse_group(word),
        }
        for index, word in enumerate(words_for(split))
    ]


def payload():
    provenance = json.loads((ROOT / "v4/phase2f/DATA_PROVENANCE.json").read_text())
    return {
        "protocol": PROTOCOL_ID,
        "status": "REALISTIC_CORRUPTION_PROTOCOL_FROZEN_CONFIRMATION_OUTCOMES_UNOPENED",
        "dataset": "CIFAR-100 fine labels",
        "classes": 100,
        "chance": 0.01,
        "metrics": list(METRICS),
        "corruptions": {
            "primitives": list(PRIMITIVES),
            "parameters": PARAMETERS,
            "scope": (
                "deterministic clean-room ImageNet-C-style categories; not bitwise or "
                "severity-equivalent to the official ImageNet-C benchmark"
            ),
        },
        "models": [asdict(record) for record in model_records()],
        "words": {split: word_records(split) for split in ("development", "confirmation")},
        "image_partitions": {
            "development": {"per_class": 10, "offset": 80},
            "confirmation": {"per_class": 10, "offset": 90},
        },
        "counts": {
            "families": 6,
            "recipes": 3,
            "lineages_per_split": 18,
            "snapshots_per_split": 36,
            "calibration_conditions": 37,
            "target_words_per_split": 60,
            "reversal_groups_per_split": 30,
        },
        "word_seed": WORD_SEED,
        "method_development": {
            "primary": "order-blind symmetric ridge using all thirty directed pairs",
            "comparators": [
                "metadata-only ridge", "independent-retention formula",
                "envelope formula", "unfitted symmetric-pair formula",
            ],
            "fit_access": "development calibration and target outcomes only",
            "confirmation_behavior_access": "forbidden until fitted method file is committed",
        },
        "primary_gates": [
            "symmetric macro normalized MAE below metadata-only ridge",
            "family/lineage/reversal hierarchical 95% improvement CI excludes zero",
            "at least four of six coordinate MAEs improve",
            "symmetric macro normalized MAE below both independent and envelope formulas",
            "symmetric macro controlled model-rank Spearman improves",
            "symmetric accuracy top-1 selection regret is no worse",
            "at least five of six family mean improvements are positive",
        ],
        "data_provenance": provenance,
    }


def main():
    OUTPUT.write_text(json.dumps(payload(), indent=2, sort_keys=True) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
