#!/usr/bin/env python3
"""Freeze Phase-2D unseen-family models and fresh words."""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from v4.phase2d import PRIMITIVES, PROTOCOL_ID
from v4.phase2d.models import FAMILIES, make_model

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "v4/phase2d/PHASE2D_MANIFEST.json"
RECIPES = ("erm_flip", "label_smooth", "noise_aug")
EPOCHS = (1, 2)
SEED_BASE = 62_100
WORD_SEED = 22_502


@dataclass(frozen=True)
class ModelRecord:
    model_id: str; lineage_id: str; family: str; recipe: str; seed: int; epoch: int; parameter_count: int


def model_records() -> list[ModelRecord]:
    records = []
    for fi, family in enumerate(FAMILIES):
        count = sum(p.numel() for p in make_model(family).parameters())
        for ri, recipe in enumerate(RECIPES):
            seed = SEED_BASE + 100 * fi + ri
            lineage = f"p2d-{family}-{recipe}-s{seed}"
            for epoch in EPOCHS:
                records.append(ModelRecord(f"{lineage}-e{epoch}", lineage, family, recipe, seed, epoch, count))
    return records


def reverse_group(word: tuple[str, ...]) -> str:
    a, b = ">".join(word), ">".join(reversed(word))
    return min(a, b)


def excluded_words() -> set[tuple[str, ...]]:
    from v4.benchmark.generate_mason_manifest import exact_words
    from v4.phase2b.manifest import fresh_words as b_words
    from v4.phase2c.manifest import fresh_words as c_words
    return set(exact_words()) | set(b_words()) | set(c_words())


def fresh_words() -> list[tuple[str, ...]]:
    excluded = excluded_words(); selected = []
    for length in (3, 4, 5, 6):
        candidates = {}
        for word in itertools.permutations(PRIMITIVES, length):
            rev = tuple(reversed(word))
            if word not in excluded and rev not in excluded:
                candidates[reverse_group(word)] = (word, rev)
        score = lambda group: hashlib.sha256(f"{WORD_SEED}|{group}".encode()).hexdigest()
        for group in sorted(candidates, key=score)[:8]:
            selected.extend(sorted(candidates[group]))
    if len(selected) != 64 or not set(selected).isdisjoint(excluded):
        raise RuntimeError("invalid Phase-2D fresh words")
    return selected


def calibration_words():
    return [("Identity",)] + [(p,) for p in PRIMITIVES] + [(a,b) for a in PRIMITIVES for b in PRIMITIVES if a != b]


def evaluation_words(): return calibration_words() + fresh_words()


def word_records():
    calibration = set(calibration_words())
    return [{"condition_index":i,"word":list(w),"word_string":">".join(w),"length":0 if w==("Identity",) else len(w),
             "role":"calibration" if w in calibration else "architecture_holdout","reverse_group":reverse_group(w)}
            for i,w in enumerate(evaluation_words())]


def payload():
    models=[asdict(r) for r in model_records()]; words=word_records()
    return {"protocol":PROTOCOL_ID,"status":"ARCHITECTURE_HOLDOUT_FROZEN_OUTCOMES_UNOPENED","dataset":"CIFAR-10",
            "image_partition":{"per_class":100,"offset":400},
            "unseen_families":list(FAMILIES),"recipes":list(RECIPES),"models":models,"words":words,
            "counts":{"lineages":6,"snapshots":12,"calibration":37,"targets":64},
            "word_overlap_with_prior_phases":0,"word_seed":WORD_SEED}


def main():
    OUTPUT.parent.mkdir(parents=True,exist_ok=True); OUTPUT.write_text(json.dumps(payload(),indent=2,sort_keys=True)+"\n"); print(OUTPUT)

if __name__=="__main__": main()
