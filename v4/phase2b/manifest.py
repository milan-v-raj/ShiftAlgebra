#!/usr/bin/env python3
"""Create the outcome-blind Phase-2B model and fresh-word manifest."""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from v4.phase2.models import FAMILY_WIDTHS, make_model
from v4.phase2b import PRIMITIVES, PROTOCOL_ID, SOURCE_DEVELOPMENT_PROTOCOL

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "v4/phase2b/PHASE2B_MANIFEST.json"
SOURCE_DEVELOPMENT = ROOT / "v4/phase2/results/development-behavior.csv"
FAMILIES = tuple(FAMILY_WIDTHS)
RECIPES = ("erm_flip", "label_smooth", "noise_aug")
SNAPSHOT_EPOCHS = (1, 2)
SEED_BASE = 42_100
WORD_SEED = 22_302
FRESH_PER_LENGTH = 16  # eight reversal groups, lengths 3--6


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
        parameters = sum(value.numel() for value in make_model(family).parameters())
        for recipe_index, recipe in enumerate(RECIPES):
            seed = SEED_BASE + 100 * family_index + recipe_index
            lineage = f"p2b-{family}-{recipe}-s{seed}"
            for epoch in SNAPSHOT_EPOCHS:
                records.append(ModelRecord(
                    model_id=f"{lineage}-e{epoch}",
                    lineage_id=lineage,
                    split="fresh_confirmation",
                    family=family,
                    recipe=recipe,
                    seed=seed,
                    epoch=epoch,
                    width=FAMILY_WIDTHS[family],
                    parameter_count=parameters,
                ))
    return records


def reverse_group(word: tuple[str, ...]) -> str:
    forward = ">".join(word)
    reverse = ">".join(reversed(word))
    return min(forward, reverse)


def _score(value: str) -> str:
    return hashlib.sha256(f"{WORD_SEED}|{value}".encode()).hexdigest()


def _old_words() -> set[tuple[str, ...]]:
    from v4.benchmark.generate_mason_manifest import exact_words
    return set(exact_words())


def fresh_words() -> list[tuple[str, ...]]:
    """Select deterministic, unseen, reversal-paired permutations.

    Selection is based only on string hashes. No behavior outcome is read.
    """
    old = _old_words()
    selected = []
    for length in (3, 4, 5, 6):
        candidates: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
        for word in itertools.permutations(PRIMITIVES, length):
            reverse = tuple(reversed(word))
            if word in old or reverse in old:
                continue
            group = reverse_group(word)
            candidates[group] = (word, reverse)
        groups = sorted(candidates, key=_score)[: FRESH_PER_LENGTH // 2]
        for group in groups:
            pair = sorted(candidates[group])
            selected.extend(pair)
    if len(selected) != 4 * FRESH_PER_LENGTH or len(set(selected)) != len(selected):
        raise RuntimeError("fresh-word selection has the wrong size or duplicates")
    if any(word in old for word in selected):
        raise RuntimeError("fresh confirmation overlaps the opened Phase-2 manifest")
    return selected


def calibration_words() -> list[tuple[str, ...]]:
    output = [("Identity",)]
    output.extend((primitive,) for primitive in PRIMITIVES)
    output.extend((first, second) for first in PRIMITIVES for second in PRIMITIVES if first != second)
    return output


def evaluation_words() -> list[tuple[str, ...]]:
    return calibration_words() + fresh_words()


def word_records() -> list[dict]:
    calibration = set(calibration_words())
    return [
        {
            "condition_index": index,
            "word": list(word),
            "word_string": ">".join(word),
            "length": 0 if word == ("Identity",) else len(word),
            "role": "calibration" if word in calibration else "fresh_confirmation",
            "reverse_group": reverse_group(word),
            "primitive_multiset": sorted(word) if word != ("Identity",) else [],
        }
        for index, word in enumerate(evaluation_words())
    ]


def payload() -> dict:
    models = [asdict(record) for record in model_records()]
    words = word_records()
    source_hash = sha256(SOURCE_DEVELOPMENT)
    return {
        "protocol": PROTOCOL_ID,
        "status": "OUTCOME_BLIND_MANIFEST_FROZEN",
        "scope": "CPU-only Phase-2B pilot; not the 75-model roadmap benchmark",
        "dataset": "CIFAR-10",
        "source_development": {
            "protocol": SOURCE_DEVELOPMENT_PROTOCOL,
            "path": str(SOURCE_DEVELOPMENT.relative_to(ROOT)),
            "sha256": source_hash,
            "permitted_roles": ["calibration", "method_development"],
            "forbidden_role": "confirmation",
        },
        "fresh_image_partition": {"per_class": 100, "offset_per_class": 200},
        "model_design": {
            "fresh_models": len(models),
            "fresh_lineages": len({row["lineage_id"] for row in models}),
            "architecture_families": list(FAMILIES),
            "training_recipes": list(RECIPES),
            "snapshot_epochs": list(SNAPSHOT_EPOCHS),
            "seeds": sorted({row["seed"] for row in models}),
        },
        "word_design": {
            "selection_seed": WORD_SEED,
            "calibration_conditions": sum(row["role"] == "calibration" for row in words),
            "fresh_confirmation_words": sum(row["role"] == "fresh_confirmation" for row in words),
            "fresh_words_per_length": FRESH_PER_LENGTH,
            "reverse_pairs_kept_together": True,
            "overlap_with_phase2_exact_words": 0,
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
