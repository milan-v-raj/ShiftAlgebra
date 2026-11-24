#!/usr/bin/env python3
"""Freeze the Phase-3B independent CIFAR-10 replication protocol."""
from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from v4.phase1.data import CIFAR_BATCH_MD5
from v4.phase2g.models import FAMILIES, make_model
from v4.phase3a.manifest import target_words as phase3a_target_words
from v4.phase3b import METRICS, PROTOCOL_ID
from v4.phase3b.corruptions import PARAMETERS, PRIMITIVES

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "v4/phase3b/PHASE3B_MANIFEST.json"
RECIPES = ("erm_flip", "label_smooth", "noise_aug")
EPOCHS = (1, 2)
WORD_SEED = 23801
SPLIT_SEEDS = {"development": 73_100, "confirmation": 74_100}


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
    output = []
    for split in ("development", "confirmation"):
        for family_index, family in enumerate(FAMILIES):
            parameters = sum(
                parameter.numel() for parameter in make_model(family, classes=10).parameters()
            )
            for recipe_index, recipe in enumerate(RECIPES):
                seed = SPLIT_SEEDS[split] + 100 * family_index + recipe_index
                lineage = f"p3b-{split}-{family}-{recipe}-s{seed}"
                for epoch in EPOCHS:
                    output.append(ModelRecord(
                        f"{lineage}-e{epoch}", lineage, split, family, recipe,
                        seed, epoch, parameters,
                    ))
    return output


def reverse_group(word):
    forward, reverse = ">".join(word), ">".join(reversed(word))
    return min(forward, reverse)


def target_words():
    prior = {
        reverse_group(word)
        for words in phase3a_target_words().values()
        for word in words
    }
    output = {"development": [], "confirmation": []}
    used_groups = set(prior)
    for length in (4, 5, 6):
        candidates = {}
        for word in itertools.permutations(PRIMITIVES, length):
            candidates[reverse_group(word)] = (word, tuple(reversed(word)))
        score = lambda group: hashlib.sha256(
            f"{WORD_SEED}|{length}|{group}".encode()
        ).hexdigest()
        chosen = [
            group for group in sorted(candidates, key=score)
            if group not in used_groups
        ][:20]
        used_groups.update(chosen)
        for split, groups in (("development", chosen[:10]), ("confirmation", chosen[10:])):
            for group in groups:
                output[split].extend(sorted(candidates[group]))
    if any(len(output[split]) != 60 for split in output):
        raise RuntimeError("invalid target count")
    if set(output["development"]) & set(output["confirmation"]):
        raise RuntimeError("development/confirmation target overlap")
    prior_words = {word for words in phase3a_target_words().values() for word in words}
    if any(set(output[split]) & prior_words for split in output):
        raise RuntimeError("Phase-3A/3B target overlap")
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
    return {
        "protocol": PROTOCOL_ID,
        "status": "CIFAR10_REPLICATION_PROTOCOL_FROZEN_CONFIRMATION_OUTCOMES_UNOPENED",
        "dataset": "CIFAR-10",
        "classes": 10,
        "chance": 0.1,
        "metrics": list(METRICS),
        "replication_of": "V4-P3A-CIFAR100-REALISTIC-CORRUPTIONS-001",
        "corruptions": {
            "primitives": list(PRIMITIVES),
            "parameters": PARAMETERS,
            "implementation": "byte-identical operator code copied from prospective Phase 3A",
            "phase3a_corruption_source_sha256": hashlib.sha256(
                (ROOT / "v4/phase3a/corruptions.py").read_bytes()
            ).hexdigest(),
            "phase3b_corruption_source_sha256": hashlib.sha256(
                (ROOT / "v4/phase3b/corruptions.py").read_bytes()
            ).hexdigest(),
            "scope": (
                "deterministic clean-room ImageNet-C-style categories; not bitwise or "
                "severity-equivalent to official CIFAR-C or ImageNet-C"
            ),
        },
        "models": [asdict(record) for record in model_records()],
        "words": {split: word_records(split) for split in ("development", "confirmation")},
        "image_partitions": {
            "development": {"per_class": 200, "offset": 600},
            "confirmation": {"per_class": 200, "offset": 800},
            "note": (
                "disjoint within Phase 3B and unused by Phase-2 through 2E banks; some images "
                "were included in historical Phase-1 broad confirmation and are not claimed globally untouched"
            ),
        },
        "counts": {
            "families": 6,
            "recipes": 3,
            "lineages_per_split": 18,
            "snapshots_per_split": 36,
            "calibration_conditions": 37,
            "target_words_per_split": 60,
            "reversal_groups_per_split": 30,
            "samples_per_condition": 2000,
        },
        "word_seed": WORD_SEED,
        "phase3a_target_word_overlap": 0,
        "method_development": {
            "primary": "order-blind symmetric ridge using all thirty directed pairs",
            "comparators": [
                "metadata/clean ridge", "independent-retention formula",
                "envelope formula", "unfitted symmetric-pair formula",
            ],
            "fit_access": "Phase-3B development calibration and target outcomes only",
            "confirmation_behavior_access": "forbidden until fitted method file is committed",
            "structural_change_from_phase3a": (
                "CIFAR-10 chance floor 0.1 replaces CIFAR-100 chance floor 0.01; "
                "all other method structure and seven gates are retained"
            ),
        },
        "primary_gates": [
            "symmetric macro normalized MAE below metadata/clean ridge",
            "family/lineage/reversal hierarchical 95% improvement CI excludes zero",
            "at least four of six coordinate MAEs improve",
            "symmetric macro normalized MAE below both independent and envelope formulas",
            "symmetric macro controlled model-rank Spearman improves",
            "symmetric accuracy top-1 selection regret is no worse",
            "at least five of six family mean improvements are positive",
        ],
        "replication_reporting": (
            "report direction agreement with Phase 3A for every gate and all six coordinate MAEs; "
            "this descriptive comparison cannot alter the primary gate"
        ),
        "data_provenance": {
            "transport": "https://github.com/wikiabhi/Cifar-10",
            "transport_commit": "fad27e50dfca620330fa5f824432abcf1781c161",
            "transport_role": "transport only",
            "canonical_batch_md5": CIFAR_BATCH_MD5,
            "training_split": "class-stable first 4000 of 5000 training images per class",
            "test_count": 10000,
        },
    }


def main():
    OUTPUT.write_text(json.dumps(payload(), indent=2, sort_keys=True) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
