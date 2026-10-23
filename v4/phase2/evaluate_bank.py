#!/usr/bin/env python3
"""Evaluate a complete model-bank split on the fixed 167 CIFAR conditions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch

from v4.benchmark.generate_mason_manifest import exact_words
from v4.phase1.classifiers import normalization
from v4.phase1.data import balanced_subset, load_cifar10
from v4.phase1.pipeline import DEFAULT_CIFAR
from v4.phase1b.corruptions import apply_calibrated_word
from v4.phase2 import PROTOCOL_ID
from v4.phase2.manifest import model_records, word_records
from v4.phase2.models import make_model
from v4.phase2.train_bank import CHECKPOINTS, RESULTS, sha256

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STRENGTHS = ROOT / "v4/phase1b/FROZEN_STRENGTHS.json"


def partition(split: str) -> tuple[int, int]:
    return (50, 0) if split == "development" else (100, 100)


def load_models(split: str, checkpoint_dir: Path) -> tuple[list, list[torch.nn.Module]]:
    records = [record for record in model_records() if record.split == split]
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


@torch.inference_mode()
def clean_statistics(models, images, labels, mean, std, batch_size: int) -> list[dict]:
    confidences = [[] for _ in models]
    correct = [0 for _ in models]
    for start in range(0, labels.numel(), batch_size):
        stop = min(start + batch_size, labels.numel())
        batch = (images[start:stop].float().div(255.0) - mean) / std
        target = labels[start:stop]
        for index, model in enumerate(models):
            probabilities = model(batch).softmax(-1)
            confidence, prediction = probabilities.max(-1)
            confidences[index].append(confidence.cpu())
            correct[index] += int((prediction == target).sum())
    output = []
    for parts, count in zip(confidences, correct):
        values = torch.cat(parts)
        accuracy = count / labels.numel()
        k = min(max(count, 1), labels.numel())
        threshold = float(torch.sort(values, descending=True).values[k - 1])
        output.append({
            "accuracy": accuracy,
            "mean_confidence": float(values.mean()),
            "atc_threshold": threshold,
        })
    return output


@torch.inference_mode()
def evaluate_condition(
    models,
    clean,
    images,
    labels,
    sample_ids,
    word,
    strengths,
    mean,
    std,
    batch_size: int,
) -> list[dict]:
    correct = [0 for _ in models]
    confidence_sum = [0.0 for _ in models]
    entropy_sum = [0.0 for _ in models]
    atc_count = [0 for _ in models]
    for start in range(0, labels.numel(), batch_size):
        stop = min(start + batch_size, labels.numel())
        pixels = images[start:stop].float().div(255.0)
        shifted = apply_calibrated_word(
            pixels, word, sample_ids[start:stop], strengths
        )
        batch = (shifted - mean) / std
        target = labels[start:stop]
        for index, model in enumerate(models):
            probabilities = model(batch).softmax(-1)
            confidence, prediction = probabilities.max(-1)
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
            correct[index] += int((prediction == target).sum())
            confidence_sum[index] += float(confidence.sum())
            entropy_sum[index] += float(entropy.sum())
            atc_count[index] += int((confidence >= clean[index]["atc_threshold"]).sum())
    output = []
    n = labels.numel()
    for index in range(len(models)):
        mean_confidence = confidence_sum[index] / n
        output.append({
            "accuracy": correct[index] / n,
            "mean_confidence": mean_confidence,
            "mean_entropy": entropy_sum[index] / n,
            "atc_accuracy": atc_count[index] / n,
            "doc_accuracy": min(1.0, max(
                0.0,
                clean[index]["accuracy"]
                + mean_confidence
                - clean[index]["mean_confidence"],
            )),
        })
    return output


def run(args: argparse.Namespace) -> dict:
    torch.set_num_threads(min(2, torch.get_num_threads()))
    started = time.time()
    bundle = load_cifar10(args.cifar_root)
    mean, std = normalization(bundle)
    per_class, offset = partition(args.split)
    subset = balanced_subset(bundle.test_labels, per_class, offset=offset)
    images, labels = bundle.test_images[subset], bundle.test_labels[subset]
    records, models = load_models(args.split, args.checkpoint_dir)
    clean = clean_statistics(models, images, labels, mean, std, args.batch_size)
    strengths_payload = json.loads(args.strengths.read_text())
    strengths = strengths_payload["datasets"]["cifar10"]["strengths"]
    roles = {row["word_string"]: row for row in word_records()}

    rows = []
    words = exact_words()
    for condition_index, word in enumerate(words):
        metrics = evaluate_condition(
            models, clean, images, labels, subset, word, strengths, mean, std, args.batch_size
        )
        word_string = ">".join(word)
        role = roles[word_string]
        for record, values in zip(records, metrics):
            rows.append({
                "protocol": PROTOCOL_ID,
                "bank_split": args.split,
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
                "clean_accuracy": clean[records.index(record)]["accuracy"],
                **values,
            })
        if condition_index % 10 == 0 or condition_index + 1 == len(words):
            print(
                f"[{args.split}] condition {condition_index + 1}/{len(words)} "
                f"word={word_string}",
                flush=True,
            )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.results_dir / f"{args.split}-behavior.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protocol": PROTOCOL_ID,
        "status": f"{args.split.upper()}_BANK_EVALUATED",
        "split": args.split,
        "models": len(records),
        "lineages": len(set(record.lineage_id for record in records)),
        "conditions": len(words),
        "samples_per_condition": int(labels.numel()),
        "partition": {"per_class": per_class, "offset": offset},
        "behavior_csv": str(csv_path),
        "behavior_csv_sha256": sha256(csv_path),
        "strengths": strengths,
        "elapsed_seconds": time.time() - started,
    }
    summary_path = args.results_dir / f"{args.split}-evaluation.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--split", choices=("development", "confirmation"), required=True)
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
