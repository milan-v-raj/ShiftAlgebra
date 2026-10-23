#!/usr/bin/env python3
"""Generate the fixed Phase-2 model-bank and word-role manifests."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from v4.benchmark.generate_mason_manifest import exact_words
from v4.phase2 import PRIMITIVES, PROTOCOL_ID
from v4.phase2.models import FAMILY_WIDTHS, make_model

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "v4/phase2/PHASE2_MANIFEST.json"
FAMILIES = tuple(FAMILY_WIDTHS)
RECIPES = ("erm_flip", "label_smooth", "noise_aug")
SNAPSHOT_EPOCHS = (1, 2)
WORD_SPLIT_SEED = 2202


@dataclass(frozen=True)
class ModelRecord:
    model_id: str
    lineage_id: str
    split: str
    family: str
    recipe: str
    seed: int
    epoch: int
    width: int
    parameter_count: int


def model_records() -> list[ModelRecord]:
    records = []
    for split, seed_base in (("development", 22_100), ("confirmation", 32_100)):
        for family_index, family in enumerate(FAMILIES):
            parameters = sum(value.numel() for value in make_model(family).parameters())
            for recipe_index, recipe in enumerate(RECIPES):
                seed = seed_base + 100 * family_index + recipe_index
                lineage = f"{split[:3]}-{family}-{recipe}-s{seed}"
                for epoch in SNAPSHOT_EPOCHS:
                    records.append(ModelRecord(
                        model_id=f"{lineage}-e{epoch}",
                        lineage_id=lineage,
                        split=split,
                        family=family,
                        recipe=recipe,
                        seed=seed,
                        epoch=epoch,
                        width=FAMILY_WIDTHS[family],
                        parameter_count=parameters,
                    ))
    return records


def _reverse_group(word: tuple[str, ...]) -> str:
    forward = ">".join(word)
    reverse = ">".join(reversed(word))
    return min(forward, reverse)


def _hash(value: str) -> str:
    return hashlib.sha256(f"{WORD_SPLIT_SEED}|{value}".encode()).hexdigest()


def word_records() -> list[dict]:
    words = exact_words()
    roles = {}
    for word in words:
        if word == ("Identity",) or len(word) <= 2:
            roles[word] = "calibration"
    for length in (3, 4, 5, 6):
        groups: dict[str, list[tuple[str, ...]]] = defaultdict(list)
        for word in words:
            if len(word) == length:
                groups[_reverse_group(word)].append(word)
        development_groups = set()
        count = 0
        for group, members in sorted(groups.items(), key=lambda item: _hash(item[0])):
            if count + len(members) <= 10:
                development_groups.add(group)
                count += len(members)
        if count != 10:
            raise RuntimeError(f"could not assign exactly 10 length-{length} development words")
        for group, members in groups.items():
            role = "method_development" if group in development_groups else "confirmation"
            for word in members:
                roles[word] = role
    output = []
    for index, word in enumerate(words):
        output.append({
            "condition_index": index,
            "word": list(word),
            "word_string": ">".join(word),
            "length": 0 if word == ("Identity",) else len(word),
            "role": roles[word],
            "reverse_group": _reverse_group(word),
            "primitive_multiset": sorted(word) if word != ("Identity",) else [],
        })
    return output


def payload() -> dict:
    models = [asdict(record) for record in model_records()]
    words = word_records()
    return {
        "protocol": PROTOCOL_ID,
        "status": "FROZEN_BEFORE_PHASE2_OUTCOMES",
        "dataset": "CIFAR-10",
        "model_design": {
            "development_models": sum(row["split"] == "development" for row in models),
            "confirmation_models": sum(row["split"] == "confirmation" for row in models),
            "development_lineages": 15,
            "confirmation_lineages": 15,
            "architecture_families": list(FAMILIES),
            "training_recipes": list(RECIPES),
            "snapshot_epochs": list(SNAPSHOT_EPOCHS),
            "note": "CPU pilot; snapshots share training lineages and are clustered accordingly",
        },
        "word_design": {
            "split_seed": WORD_SPLIT_SEED,
            "calibration": sum(row["role"] == "calibration" for row in words),
            "method_development": sum(row["role"] == "method_development" for row in words),
            "confirmation": sum(row["role"] == "confirmation" for row in words),
            "reverse_pairs_kept_together": True,
        },
        "models": models,
        "words": words,
    }


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload(), indent=2, sort_keys=True) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
