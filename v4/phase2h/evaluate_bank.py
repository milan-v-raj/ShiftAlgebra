#!/usr/bin/env python3
"""Evaluate the sealed Phase-2H full-vector confirmation bank exactly once."""
import csv
import json
import time
from pathlib import Path

import torch

from v4.phase2.train_bank import sha256
from v4.phase2f.data import balanced_indices, load
from v4.phase2g.models import logits_and_features, make_model
from v4.phase2h import METRICS, PROTOCOL_ID
from v4.phase2h.manifest import model_records, word_records, words

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "v4/phase2h/results"
CHECKPOINTS = ROOT / "checkpoints_v4/phase2h"
STRENGTHS = ROOT / "v4/phase1b/FROZEN_STRENGTHS.json"


def load_models():
    records = model_records()
    training = json.loads((RESULTS / "confirmation-training.json").read_text())
    expected = {
        history["model_id"]: history["sha256"]
        for summary in training["summaries"] for history in summary["history"]
    }
    if set(expected) != {record.model_id for record in records}:
        raise ValueError("training record mismatch")
    models = []
    for record in records:
        path = CHECKPOINTS / f"{record.model_id}.pt"
        if sha256(path) != expected[record.model_id]:
            raise ValueError(f"checkpoint hash mismatch: {record.model_id}")
        model = make_model(record.family)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()
        models.append(model)
    mean = torch.tensor(training["normalization_mean"]).view(1, 3, 1, 1)
    std = torch.tensor(training["normalization_std"]).view(1, 3, 1, 1)
    return records, models, mean, std


def run():
    import v4.phase2f.evaluate_bank as evaluator
    evaluator.logits_and_features = logits_and_features

    torch.set_num_threads(min(2, torch.get_num_threads()))
    start = time.time()
    data = load()
    indices = balanced_indices(data["test_labels"], per_class=20, offset=60)
    images = data["test_images"][indices]
    labels = data["test_labels"][indices]
    records, models, mean, std = load_models()
    strengths = json.loads(STRENGTHS.read_text())["datasets"]["cifar10"]["strengths"]
    roles = {record["word_string"]: record for record in word_records()}
    output_rows = []
    clean_centroids = None
    for condition_index, word in enumerate(words()):
        values, centroids = evaluator.evaluate(
            models, records, images, labels, indices, word, strengths, mean, std, clean_centroids
        )
        if condition_index == 0:
            clean_centroids = centroids
        word_string = ">".join(word)
        role = roles[word_string]
        for record, value in zip(records, values):
            output_rows.append({
                "protocol": PROTOCOL_ID,
                "bank_split": "confirmation",
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
                "sample_count": len(labels),
                **value,
            })
        if condition_index % 10 == 0 or condition_index + 1 == len(words()):
            print(f"[p2h confirmation] {condition_index + 1}/{len(words())} {word_string}", flush=True)
    path = RESULTS / "confirmation-behavior.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    result = {
        "protocol": PROTOCOL_ID,
        "status": "CONFIRMATION_FULL_VECTOR_EVALUATED_ONCE",
        "models": 36,
        "lineages": 18,
        "conditions": 97,
        "samples_per_condition": len(labels),
        "partition": {"per_class": 20, "offset": 60},
        "metrics": list(METRICS),
        "behavior_csv": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "elapsed_seconds": time.time() - start,
    }
    (RESULTS / "confirmation-evaluation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


if __name__ == "__main__":
    run()
