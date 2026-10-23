#!/usr/bin/env python3
"""Single-primitive-only severity calibration for Phase 1B.

This stage is intentionally incapable of evaluating pairs or target words.  It
selects one attenuation strength per primitive on the old Phase-1 development
partition, using only clean and single-primitive accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from v4.phase1.classifiers import ClassifierConfig, train_classifier
from v4.phase1.corruptions import PRIMITIVES
from v4.phase1.data import balanced_subset
from v4.phase1.pipeline import (
    CHECKPOINTS,
    DEFAULT_CIFAR,
    DEFAULT_EMNIST,
    load_dataset,
)
from v4.phase1b import PROTOCOL_ID
from v4.phase1b.corruptions import (
    ACCEPTANCE_BAND,
    STRENGTH_GRID,
    TARGET_NORMALIZED_DROP,
    apply_attenuated_primitive,
)

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "v4/phase1b/results"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_configuration(dataset: str) -> tuple[int, int, int, int, int]:
    """Return width, epochs, train batch, eval batch, dev images/class."""

    if dataset == "cifar10":
        return 24, 8, 256, 256, 100
    if dataset == "emnist":
        return 16, 5, 512, 512, 40
    raise ValueError(dataset)


@torch.no_grad()
def accuracy_at_strength(
    model,
    images_uint8: torch.Tensor,
    labels: torch.Tensor,
    sample_ids: torch.Tensor,
    primitive: str,
    alpha: float,
    mean: torch.Tensor,
    std: torch.Tensor,
    batch_size: int,
) -> float:
    model.eval()
    correct = 0
    for start in range(0, labels.numel(), batch_size):
        stop = min(start + batch_size, labels.numel())
        pixels = images_uint8[start:stop].float().div(255.0)
        shifted = apply_attenuated_primitive(
            pixels, primitive, sample_ids[start:stop], alpha
        )
        logits = model((shifted - mean) / std)
        correct += int((logits.argmax(dim=-1) == labels[start:stop]).sum())
    return correct / labels.numel()


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    width, epochs, train_batch, eval_batch, per_class = fixed_configuration(args.dataset)
    bundle = load_dataset(args.dataset, args.cifar_root, args.emnist_root)
    config = ClassifierConfig(
        dataset=bundle.name,
        input_channels=bundle.train_images.shape[1],
        class_count=bundle.class_count,
        width=width,
        seed=1301,
        epochs=epochs,
        batch_size=train_batch,
        learning_rate=3e-3,
    )
    model, mean, std, training = train_classifier(bundle, config, args.checkpoint_dir)
    checkpoint = args.checkpoint_dir / (
        f"{bundle.name.lower().replace('-', '')}-{config.model_id}.pt"
    )
    subset = balanced_subset(bundle.test_labels, per_class, offset=0)
    images = bundle.test_images[subset]
    labels = bundle.test_labels[subset]

    clean_accuracy = accuracy_at_strength(
        model, images, labels, subset, "Contrast", 0.0, mean, std, eval_batch
    )
    chance = 1.0 / bundle.class_count
    denominator = clean_accuracy - chance
    if denominator <= 0:
        raise ValueError("clean accuracy must exceed chance for normalized calibration")

    rows = []
    selected = {}
    for primitive in PRIMITIVES:
        candidates = []
        for alpha in STRENGTH_GRID:
            accuracy = accuracy_at_strength(
                model, images, labels, subset, primitive, alpha, mean, std, eval_batch
            )
            normalized_drop = (clean_accuracy - accuracy) / denominator
            candidate = {
                "primitive": primitive,
                "alpha": alpha,
                "accuracy": accuracy,
                "accuracy_drop_pp": 100.0 * (clean_accuracy - accuracy),
                "normalized_drop": normalized_drop,
                "distance_to_target": abs(normalized_drop - TARGET_NORMALIZED_DROP),
            }
            candidates.append(candidate)
            rows.append(candidate)
        # Frozen tie break: select the smaller alpha.
        winner = min(candidates, key=lambda row: (row["distance_to_target"], row["alpha"]))
        selected[primitive] = {
            **winner,
            "within_preregistered_band": (
                ACCEPTANCE_BAND[0] <= winner["normalized_drop"] <= ACCEPTANCE_BAND[1]
            ),
        }
        print(
            f"[{bundle.name}] {primitive}: alpha={winner['alpha']:.3f} "
            f"drop={winner['normalized_drop']:.4f} acc={winner['accuracy']:.4f}",
            flush=True,
        )

    result = {
        "protocol": PROTOCOL_ID,
        "status": "CALIBRATION_ONLY_NO_PAIRS_OR_TARGET_WORDS",
        "dataset": bundle.name,
        "model_id": config.model_id,
        "classifier_checkpoint": str(checkpoint),
        "classifier_checkpoint_sha256": sha256(checkpoint),
        "classifier_training": training,
        "calibration_partition": {
            "source": "Phase-1 development image partition",
            "per_class": per_class,
            "offset": 0,
            "samples": int(subset.numel()),
        },
        "information_access": {
            "clean": True,
            "single_primitives": True,
            "ordered_pairs": False,
            "length_3_to_6_targets": False,
        },
        "class_count": bundle.class_count,
        "chance_accuracy": chance,
        "clean_accuracy": clean_accuracy,
        "target_normalized_drop": TARGET_NORMALIZED_DROP,
        "acceptance_band": list(ACCEPTANCE_BAND),
        "strength_grid": list(STRENGTH_GRID),
        "selected": selected,
        "all_selected_within_band": all(
            row["within_preregistered_band"] for row in selected.values()
        ),
        "candidate_rows": rows,
        "elapsed_seconds": time.time() - started,
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    output = args.results_dir / f"{args.dataset}-calibration.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {output}")
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", choices=("cifar10", "emnist"), required=True)
    result.add_argument("--cifar-root", type=Path, default=DEFAULT_CIFAR)
    result.add_argument("--emnist-root", type=Path, default=DEFAULT_EMNIST)
    result.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINTS)
    result.add_argument("--results-dir", type=Path, default=RESULTS)
    return result


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
