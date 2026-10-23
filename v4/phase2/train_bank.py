#!/usr/bin/env python3
"""Train deterministic two-epoch CIFAR model-bank lineages and save snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from v4.phase1.classifiers import normalization
from v4.phase1.data import load_cifar10
from v4.phase1.pipeline import DEFAULT_CIFAR
from v4.phase2 import PROTOCOL_ID
from v4.phase2.manifest import model_records
from v4.phase2.models import make_model

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = ROOT / "checkpoints_v4/phase2"
RESULTS = ROOT / "v4/phase2/results"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return torch.Generator().manual_seed(seed)


def augment(raw: torch.Tensor, recipe: str, generator: torch.Generator) -> torch.Tensor:
    pixels = raw.float().div(255.0)
    flip = torch.rand(pixels.shape[0], generator=generator) < 0.5
    pixels[flip] = pixels[flip].flip(-1)
    if recipe == "noise_aug":
        noise = torch.randn(pixels.shape, generator=generator, dtype=pixels.dtype) * 0.05
        pixels = (pixels + noise).clamp(0.0, 1.0)
    return pixels


def train_split(split: str, cifar_root: Path, checkpoint_dir: Path, results_dir: Path) -> dict:
    torch.set_num_threads(min(2, torch.get_num_threads()))
    bundle = load_cifar10(cifar_root)
    mean, std = normalization(bundle)
    records = [record for record in model_records() if record.split == split]
    by_lineage = {}
    for record in records:
        by_lineage.setdefault(record.lineage_id, record)
    summaries = []
    started_all = time.time()
    for lineage_index, template in enumerate(by_lineage.values(), start=1):
        generator = seed_everything(template.seed)
        model = make_model(template.family)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
        loader = DataLoader(
            TensorDataset(bundle.train_images, bundle.train_labels),
            batch_size=512,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        history = []
        started = time.time()
        for epoch in (1, 2):
            model.train()
            correct = 0
            loss_sum = 0.0
            count = 0
            for raw, labels in loader:
                pixels = augment(raw, template.recipe, generator)
                images = (pixels - mean) / std
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = F.cross_entropy(
                    logits,
                    labels,
                    label_smoothing=0.1 if template.recipe == "label_smooth" else 0.0,
                )
                loss.backward()
                optimizer.step()
                correct += int((logits.argmax(-1) == labels).sum())
                loss_sum += float(loss.detach()) * labels.numel()
                count += labels.numel()
            snapshot_record = next(
                row for row in records if row.lineage_id == template.lineage_id and row.epoch == epoch
            )
            checkpoint = checkpoint_dir / f"{snapshot_record.model_id}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint)
            history.append({
                "epoch": epoch,
                "train_accuracy": correct / count,
                "train_loss": loss_sum / count,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "model_id": snapshot_record.model_id,
            })
            print(
                f"[{split} {lineage_index}/{len(by_lineage)}] {template.lineage_id} "
                f"epoch={epoch} train_acc={correct / count:.4f}",
                flush=True,
            )
        summaries.append({
            "lineage_id": template.lineage_id,
            "family": template.family,
            "recipe": template.recipe,
            "seed": template.seed,
            "parameter_count": template.parameter_count,
            "elapsed_seconds": time.time() - started,
            "history": history,
        })
    result = {
        "protocol": PROTOCOL_ID,
        "status": f"{split.upper()}_BANK_TRAINED",
        "split": split,
        "lineages": len(summaries),
        "models": len(records),
        "dataset_training_images": int(bundle.train_labels.numel()),
        "normalization_mean": mean.flatten().tolist(),
        "normalization_std": std.flatten().tolist(),
        "elapsed_seconds": time.time() - started_all,
        "summaries": summaries,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{split}-training.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--split", choices=("development", "confirmation"), required=True)
    result.add_argument("--cifar-root", type=Path, default=DEFAULT_CIFAR)
    result.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINTS)
    result.add_argument("--results-dir", type=Path, default=RESULTS)
    return result


def main() -> None:
    args = parser().parse_args()
    train_split(args.split, args.cifar_root, args.checkpoint_dir, args.results_dir)


if __name__ == "__main__":
    main()
