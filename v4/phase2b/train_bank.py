#!/usr/bin/env python3
"""Train the fresh CPU-only Phase-2B confirmation lineages."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from v4.phase1.classifiers import normalization
from v4.phase1.data import load_cifar10
from v4.phase1.pipeline import DEFAULT_CIFAR
from v4.phase2.models import make_model
from v4.phase2.train_bank import augment, seed_everything, sha256
from v4.phase2b import PROTOCOL_ID
from v4.phase2b.manifest import model_records

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = ROOT / "checkpoints_v4/phase2b"
RESULTS = ROOT / "v4/phase2b/results"


def train(cifar_root: Path, checkpoint_dir: Path, results_dir: Path) -> dict:
    torch.set_num_threads(min(2, torch.get_num_threads()))
    bundle = load_cifar10(cifar_root)
    mean, std = normalization(bundle)
    records = model_records()
    templates = {}
    for record in records:
        templates.setdefault(record.lineage_id, record)
    summaries = []
    started_all = time.time()
    for index, template in enumerate(templates.values(), start=1):
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
            correct = count = 0
            loss_sum = 0.0
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
            snapshot = next(
                row for row in records if row.lineage_id == template.lineage_id and row.epoch == epoch
            )
            checkpoint = checkpoint_dir / f"{snapshot.model_id}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint)
            history.append({
                "epoch": epoch,
                "train_accuracy": correct / count,
                "train_loss": loss_sum / count,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "model_id": snapshot.model_id,
            })
            print(
                f"[fresh {index}/{len(templates)}] {template.lineage_id} "
                f"epoch={epoch} train_acc={correct / count:.4f}", flush=True,
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
        "status": "FRESH_CONFIRMATION_BANK_TRAINED",
        "lineages": len(summaries),
        "models": len(records),
        "dataset_training_images": int(bundle.train_labels.numel()),
        "elapsed_seconds": time.time() - started_all,
        "summaries": summaries,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "fresh-training.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--cifar-root", type=Path, default=DEFAULT_CIFAR)
    result.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINTS)
    result.add_argument("--results-dir", type=Path, default=RESULTS)
    return result


def main() -> None:
    args = parser().parse_args()
    train(args.cifar_root, args.checkpoint_dir, args.results_dir)


if __name__ == "__main__":
    main()
