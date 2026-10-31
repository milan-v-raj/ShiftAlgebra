#!/usr/bin/env python3
"""Evaluate fresh Phase-2B models on calibration and previously unseen words."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from v4.phase1.classifiers import normalization
from v4.phase1.data import balanced_subset, load_cifar10
from v4.phase1.pipeline import DEFAULT_CIFAR
from v4.phase1b.corruptions import apply_calibrated_word
from v4.phase2.evaluate_bank import clean_statistics, evaluate_condition
from v4.phase2.models import make_model
from v4.phase2.train_bank import sha256
from v4.phase2b import PROTOCOL_ID
from v4.phase2b.manifest import evaluation_words, model_records, word_records
from v4.phase2b.train_bank import CHECKPOINTS, RESULTS

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STRENGTHS = ROOT / "v4/phase1b/FROZEN_STRENGTHS.json"
PER_CLASS = 100
OFFSET = 200


def load_models(checkpoint_dir: Path):
    records = model_records()
    models = []
    for record in records:
        checkpoint = checkpoint_dir / f"{record.model_id}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model = make_model(record.family)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        model.eval()
        models.append(model)
    return records, models


def run(args: argparse.Namespace) -> dict:
    torch.set_num_threads(min(2, torch.get_num_threads()))
    started = time.time()
    bundle = load_cifar10(args.cifar_root)
    mean, std = normalization(bundle)
    subset = balanced_subset(bundle.test_labels, PER_CLASS, offset=OFFSET)
    images, labels = bundle.test_images[subset], bundle.test_labels[subset]
    records, models = load_models(args.checkpoint_dir)
    clean = clean_statistics(models, images, labels, mean, std, args.batch_size)
    strengths = json.loads(args.strengths.read_text())["datasets"]["cifar10"]["strengths"]
    role_lookup = {row["word_string"]: row for row in word_records()}

    rows = []
    words = evaluation_words()
    for condition_index, word in enumerate(words):
        values = evaluate_condition(
            models, clean, images, labels, subset, word, strengths, mean, std, args.batch_size
        )
        word_string = ">".join(word)
        role = role_lookup[word_string]
        for model_index, (record, metrics) in enumerate(zip(records, values)):
            rows.append({
                "protocol": PROTOCOL_ID,
                "bank_split": "fresh_confirmation",
                "model_id": record.model_id,
                "lineage_id": record.lineage_id,
                "family": record.family,
                "recipe": record.recipe,
                "seed": record.seed,
                "epoch": record.epoch,
                "parameter_count": record.parameter_count,
                "condition_index": condition_index,
                "word": word_string,
                "word_role": role["role"],
                "reverse_group": role["reverse_group"],
                "length": role["length"],
                "sample_count": labels.numel(),
                "clean_accuracy": clean[model_index]["accuracy"],
                **metrics,
            })
        if condition_index % 10 == 0 or condition_index + 1 == len(words):
            print(
                f"[fresh] condition {condition_index + 1}/{len(words)} word={word_string}",
                flush=True,
            )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.results_dir / "fresh-behavior.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protocol": PROTOCOL_ID,
        "status": "FRESH_CONFIRMATION_EVALUATED",
        "models": len(records),
        "lineages": len({record.lineage_id for record in records}),
        "conditions": len(words),
        "calibration_conditions": 37,
        "fresh_confirmation_words": 64,
        "samples_per_condition": int(labels.numel()),
        "partition": {"per_class": PER_CLASS, "offset": OFFSET},
        "behavior_csv": str(csv_path),
        "behavior_csv_sha256": sha256(csv_path),
        "elapsed_seconds": time.time() - started,
    }
    (args.results_dir / "fresh-evaluation.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--cifar-root", type=Path, default=DEFAULT_CIFAR)
    result.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINTS)
    result.add_argument("--results-dir", type=Path, default=RESULTS)
    result.add_argument("--strengths", type=Path, default=DEFAULT_STRENGTHS)
    result.add_argument("--batch-size", type=int, default=256)
    return result


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
