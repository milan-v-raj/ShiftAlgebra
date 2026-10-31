#!/usr/bin/env python3
"""Outcome-blind fresh model/word manifest for Phase-2C pair selection."""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from v4.phase2.models import FAMILY_WIDTHS, make_model
from v4.phase2b.manifest import fresh_words as phase2b_words
from v4.phase2c import PRIMITIVES, PROTOCOL_ID

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "v4/phase2c/PHASE2C_MANIFEST.json"
SOURCE_DEVELOPMENT = ROOT / "v4/phase2/results/development-behavior.csv"
FAMILIES = tuple(FAMILY_WIDTHS)
RECIPES = ("erm_flip", "label_smooth", "noise_aug")
SNAPSHOT_EPOCHS = (1, 2)
SEED_BASE = 52_100
WORD_SEED = 22_402
FRESH_PER_LENGTH = 16


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def model_records() -> list[ModelRecord]:
    records = []
    for family_index, family in enumerate(FAMILIES):
        parameter_count = sum(value.numel() for value in make_model(family).parameters())
        for recipe_index, recipe in enumerate(RECIPES):
            seed = SEED_BASE + 100 * family_index + recipe_index
            lineage = f"p2c-{family}-{recipe}-s{seed}"
            for epoch in SNAPSHOT_EPOCHS:
                records.append(ModelRecord(
                    model_id=f"{lineage}-e{epoch}",
                    lineage_id=lineage,
                    split="fresh_pair_confirmation",
                    family=family,
                    recipe=recipe,
                    seed=seed,
                    epoch=epoch,
                    width=FAMILY_WIDTHS[family],
                    parameter_count=parameter_count,
                ))
    return records


def reverse_group(word: tuple[str, ...]) -> str:
    forward = ">".join(word)
    reverse = ">".join(reversed(word))
    return min(forward, reverse)


def _score(value: str) -> str:
    return hashlib.sha256(f"{WORD_SEED}|{value}".encode()).hexdigest()


def excluded_words() -> set[tuple[str, ...]]:
    from v4.benchmark.generate_mason_manifest import exact_words
    return set(exact_words()) | set(phase2b_words())


def fresh_words() -> list[tuple[str, ...]]:
    excluded = excluded_words()
    selected = []
    for length in (3, 4, 5, 6):
        candidates = {}
        for word in itertools.permutations(PRIMITIVES, length):
            reverse = tuple(reversed(word))
            if word in excluded or reverse in excluded:
                continue
            candidates[reverse_group(word)] = (word, reverse)
        groups = sorted(candidates, key=_score)[: FRESH_PER_LENGTH // 2]
        for group in groups:
            selected.extend(sorted(candidates[group]))
    if len(selected) != 64 or len(set(selected)) != 64:
        raise RuntimeError("fresh Phase-2C word selection is invalid")
    if not set(selected).isdisjoint(excluded):
        raise RuntimeError("Phase-2C words overlap earlier targets")
    return selected


def calibration_words() -> list[tuple[str, ...]]:
    words = [("Identity",)]
    words.extend((primitive,) for primitive in PRIMITIVES)
    words.extend((a, b) for a in PRIMITIVES for b in PRIMITIVES if a != b)
    return words


def evaluation_words() -> list[tuple[str, ...]]:
    return calibration_words() + fresh_words()


def word_records() -> list[dict]:
    calibration = set(calibration_words())
    return [{
        "condition_index": index,
        "word": list(word),
        "word_string": ">".join(word),
        "length": 0 if word == ("Identity",) else len(word),
        "role": "calibration" if word in calibration else "fresh_pair_confirmation",
        "reverse_group": reverse_group(word),
    } for index, word in enumerate(evaluation_words())]


def payload() -> dict:
    models = [asdict(record) for record in model_records()]
    words = word_records()
    return {
        "protocol": PROTOCOL_ID,
        "status": "OUTCOME_BLIND_ACTIVE_PAIR_MANIFEST_FROZEN",
        "scope": "CPU-only active pair-selection pilot",
        "dataset": "CIFAR-10",
        "source_development": {
            "path": str(SOURCE_DEVELOPMENT.relative_to(ROOT)),
            "sha256": sha256(SOURCE_DEVELOPMENT),
            "permitted_roles": ["calibration", "method_development"],
        },
        "fresh_image_partition": {"per_class": 100, "offset_per_class": 300},
        "model_design": {
            "fresh_models": len(models),
            "fresh_lineages": len({row["lineage_id"] for row in models}),
            "architecture_families": list(FAMILIES),
            "training_recipes": list(RECIPES),
            "snapshot_epochs": list(SNAPSHOT_EPOCHS),
        },
        "word_design": {
            "selection_seed": WORD_SEED,
            "fresh_confirmation_words": 64,
            "fresh_words_per_length": FRESH_PER_LENGTH,
            "overlap_with_phase2_or_phase2b_words": 0,
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
