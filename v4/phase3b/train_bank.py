#!/usr/bin/env python3
"""Train fresh Phase-3B CIFAR-10 development or confirmation banks."""
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
from v4.phase2.train_bank import augment, seed_everything, sha256
from v4.phase2g.models import make_model
from v4.phase3b import PROTOCOL_ID
from v4.phase3b.manifest import model_records

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = ROOT / "checkpoints_v4/phase3b"
RESULTS = ROOT / "v4/phase3b/results"
LEDGER = ROOT / "v4/phase3b/CONFIRMATION_CHECKPOINT_LEDGER.json"


def train(split):
    torch.set_num_threads(min(2, torch.get_num_threads()))
    bundle = load_cifar10(DEFAULT_CIFAR)
    mean, std = normalization(bundle)
    records = [record for record in model_records() if record.bank_split == split]
    templates = {}
    for record in records:
        templates.setdefault(record.lineage_id, record)
    summaries = []
    all_start = time.time()
    for index, template in enumerate(templates.values(), 1):
        generator = seed_everything(template.seed)
        model = make_model(template.family, classes=10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
        loader = DataLoader(
            TensorDataset(bundle.train_images, bundle.train_labels),
            batch_size=512, shuffle=True, generator=generator, num_workers=0,
        )
        history = []
        start = time.time()
        for epoch in (1, 2):
            model.train()
            correct = count = 0
            loss_sum = 0.0
            for raw, labels in loader:
                images = (augment(raw, template.recipe, generator) - mean) / std
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = F.cross_entropy(
                    logits, labels,
                    label_smoothing=0.1 if template.recipe == "label_smooth" else 0.0,
                )
                loss.backward()
                optimizer.step()
                correct += int((logits.argmax(1) == labels).sum())
                count += len(labels)
                loss_sum += float(loss.detach()) * len(labels)
            record = next(
                item for item in records
                if item.lineage_id == template.lineage_id and item.epoch == epoch
            )
            checkpoint = CHECKPOINTS / f"{record.model_id}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint)
            history.append({
                "epoch": epoch, "model_id": record.model_id,
                "train_accuracy": correct / count, "train_loss": loss_sum / count,
                "checkpoint": str(checkpoint.relative_to(ROOT)), "sha256": sha256(checkpoint),
            })
            print(
                f"[p3b {split} {index}/18] {template.lineage_id} "
                f"epoch={epoch} acc={correct / count:.4f}", flush=True,
            )
        summaries.append({
            "lineage_id": template.lineage_id, "family": template.family,
            "recipe": template.recipe, "seed": template.seed,
            "parameter_count": template.parameter_count,
            "elapsed_seconds": time.time() - start, "history": history,
        })
    output = {
        "protocol": PROTOCOL_ID,
        "status": (
            "CONFIRMATION_BANK_TRAINED_BEHAVIOR_UNOPENED"
            if split == "confirmation" else "DEVELOPMENT_BANK_TRAINED"
        ),
        "split": split, "lineages": 18, "models": 36,
        "dataset_training_images": len(bundle.train_labels),
        "normalization_mean": mean.flatten().tolist(),
        "normalization_std": std.flatten().tolist(),
        "elapsed_seconds": time.time() - all_start, "summaries": summaries,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{split}-training.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )
    if split == "confirmation":
        by_model = {record.model_id: record for record in records}
        ledger = {
            "protocol": PROTOCOL_ID,
            "status": "CONFIRMATION_CHECKPOINTS_HASHED_BEHAVIOR_UNOPENED",
            "models": 36, "lineages": 18, "behavior_opened": False,
            "checkpoints": [
                {
                    "model_id": item["model_id"],
                    "lineage_id": by_model[item["model_id"]].lineage_id,
                    "family": by_model[item["model_id"]].family,
                    "recipe": by_model[item["model_id"]].recipe,
                    "seed": by_model[item["model_id"]].seed,
                    "epoch": by_model[item["model_id"]].epoch,
                    "parameter_count": by_model[item["model_id"]].parameter_count,
                    "path": item["checkpoint"], "sha256": item["sha256"],
                }
                for summary in summaries for item in summary["history"]
            ],
        }
        LEDGER.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("development", "confirmation"), required=True)
    train(parser.parse_args().split)
