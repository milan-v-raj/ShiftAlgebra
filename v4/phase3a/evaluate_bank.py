#!/usr/bin/env python3
"""Evaluate realistic-corruption behavior vectors on development or confirmation."""
import argparse
import csv
import json
import time
from pathlib import Path

import torch

from v4.phase2.train_bank import sha256
from v4.phase2f.data import balanced_indices, load
from v4.phase2g.models import logits_and_features, make_model
from v4.phase3a import METRICS, PROTOCOL_ID
from v4.phase3a.corruptions import apply_word
from v4.phase3a.manifest import model_records, word_records, words_for

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "v4/phase3a/results"


def locations(split):
    if split == "development":
        return (
            ROOT / "v4/phase2h/results/confirmation-training.json",
            ROOT / "checkpoints_v4/phase2h",
        )
    return (
        RESULTS / "confirmation-training.json",
        ROOT / "checkpoints_v4/phase3a",
    )


def load_models(split):
    records = [record for record in model_records() if record.bank_split == split]
    training_path, checkpoint_root = locations(split)
    training = json.loads(training_path.read_text())
    expected = {
        history["model_id"]: history["sha256"]
        for summary in training["summaries"] for history in summary["history"]
    }
    if set(expected) != {record.model_id for record in records}:
        raise ValueError("training record mismatch")
    models = []
    for record in records:
        path = checkpoint_root / f"{record.model_id}.pt"
        if sha256(path) != expected[record.model_id]:
            raise ValueError(f"checkpoint hash mismatch: {record.model_id}")
        model = make_model(record.family)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()
        models.append(model)
    mean = torch.tensor(training["normalization_mean"]).view(1, 3, 1, 1)
    std = torch.tensor(training["normalization_std"]).view(1, 3, 1, 1)
    return records, models, mean, std


@torch.inference_mode()
def evaluate(models, records, images, labels, sample_ids, word, mean, std, clean_centroids=None, batch=256):
    count = len(labels)
    correct = [torch.zeros(100, dtype=torch.long) for _ in models]
    class_counts = torch.bincount(labels, minlength=100)
    confidence = [0.0] * len(models)
    margins = [0.0] * len(models)
    bin_count = [torch.zeros(10) for _ in models]
    bin_confidence = [torch.zeros(10) for _ in models]
    bin_correct = [torch.zeros(10) for _ in models]
    feature_sum = [None] * len(models)
    for start in range(0, count, batch):
        stop = min(start + batch, count)
        pixels = images[start:stop].float().div(255)
        shifted = apply_word(pixels, word, sample_ids[start:stop])
        inputs = (shifted - mean) / std
        targets = labels[start:stop]
        for index, (model, record) in enumerate(zip(models, records)):
            logits, features = logits_and_features(model, record.family, inputs)
            probabilities = logits.softmax(1)
            top = probabilities.topk(2, dim=1).values
            maximum, predictions = probabilities.max(1)
            good = predictions == targets
            correct[index] += torch.bincount(targets[good], minlength=100)
            confidence[index] += float(maximum.sum())
            margins[index] += float((top[:, 0] - top[:, 1]).sum())
            feature_sum[index] = (
                features.double().sum(0)
                if feature_sum[index] is None
                else feature_sum[index] + features.double().sum(0)
            )
            bins = torch.clamp((maximum * 10).long(), max=9)
            for bin_index in range(10):
                mask = bins == bin_index
                bin_count[index][bin_index] += int(mask.sum())
                bin_confidence[index][bin_index] += float(maximum[mask].sum())
                bin_correct[index][bin_index] += int(good[mask].sum())
    output = []
    centroids = []
    for index in range(len(models)):
        centroid = feature_sum[index] / count
        centroids.append(centroid)
        ece = 0.0
        for bin_index in range(10):
            if bin_count[index][bin_index] > 0:
                ece += float(
                    bin_count[index][bin_index] / count
                    * abs(
                        bin_correct[index][bin_index] / bin_count[index][bin_index]
                        - bin_confidence[index][bin_index] / bin_count[index][bin_index]
                    )
                )
        total_correct = int(correct[index].sum())
        drift = (
            0.0 if clean_centroids is None
            else float(torch.linalg.vector_norm(centroid - clean_centroids[index]))
        )
        output.append({
            "margin": margins[index] / count,
            "feature_drift": drift,
            "ece": ece,
            "balanced_accuracy": float((correct[index] / class_counts).double().mean()),
            "accuracy": total_correct / count,
            "mean_confidence": confidence[index] / count,
        })
    return output, centroids


def run(split):
    torch.set_num_threads(min(2, torch.get_num_threads()))
    start = time.time()
    data = load()
    partition = {"development": (10, 80), "confirmation": (10, 90)}[split]
    per_class, offset = partition
    indices = balanced_indices(data["test_labels"], per_class, offset)
    images = data["test_images"][indices]
    labels = data["test_labels"][indices]
    records, models, mean, std = load_models(split)
    roles = {record["word_string"]: record for record in word_records(split)}
    rows = []
    clean_centroids = None
    words = words_for(split)
    for condition_index, word in enumerate(words):
        values, centroids = evaluate(
            models, records, images, labels, indices, word, mean, std, clean_centroids
        )
        if condition_index == 0:
            clean_centroids = centroids
        word_string = ">".join(word)
        role = roles[word_string]
        for record, value in zip(records, values):
            rows.append({
                "protocol": PROTOCOL_ID,
                "bank_split": split,
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
        if condition_index % 10 == 0 or condition_index + 1 == len(words):
            print(f"[p3a {split}] {condition_index + 1}/{len(words)} {word_string}", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{split}-behavior.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "protocol": PROTOCOL_ID,
        "status": f"{split.upper()}_REALISTIC_FULL_VECTOR_EVALUATED",
        "split": split,
        "models": 36,
        "lineages": 18,
        "conditions": 97,
        "samples_per_condition": len(labels),
        "partition": {"per_class": per_class, "offset": offset},
        "metrics": list(METRICS),
        "behavior_csv": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "elapsed_seconds": time.time() - start,
    }
    (RESULTS / f"{split}-evaluation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("development", "confirmation"), required=True)
    run(parser.parse_args().split)
