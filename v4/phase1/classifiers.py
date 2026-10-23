"""Small frozen classifiers and deterministic CPU training for Phase 1."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .data import DatasetBundle

Tensor = torch.Tensor


@dataclass(frozen=True)
class ClassifierConfig:
    dataset: str
    input_channels: int
    class_count: int
    width: int = 24
    seed: int = 1301
    epochs: int = 8
    batch_size: int = 256
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4

    @property
    def model_id(self) -> str:
        return f"smallcnn-w{self.width}-s{self.seed}"


class SmallConvNet(nn.Module):
    def __init__(self, input_channels: int, class_count: int, width: int = 24) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.class_count = class_count
        self.width = width
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(width, 2 * width, 3, padding=1, bias=False),
            nn.BatchNorm2d(2 * width),
            nn.SiLU(),
            nn.Conv2d(2 * width, 2 * width, 3, padding=1, bias=False),
            nn.BatchNorm2d(2 * width),
            nn.SiLU(),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(2 * width, 4 * width, 3, padding=1, bias=False),
            nn.BatchNorm2d(4 * width),
            nn.SiLU(),
        )
        self.classifier = nn.Linear(4 * width, class_count)

    def forward_with_layers(self, images: Tensor) -> tuple[Tensor, list[Tensor]]:
        x1 = self.block1(images)
        x2 = self.block2(F.max_pool2d(x1, 2))
        x3 = self.block3(F.max_pool2d(x2, 2))
        pooled = F.adaptive_avg_pool2d(x3, 1).flatten(1)
        logits = self.classifier(pooled)
        layers = [
            F.adaptive_avg_pool2d(x1, 1).flatten(1),
            F.adaptive_avg_pool2d(x2, 1).flatten(1),
            pooled,
        ]
        return logits, layers

    def forward(self, images: Tensor) -> Tensor:
        return self.forward_with_layers(images)[0]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _normalize(images: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    return (images.float().div(255.0) - mean) / std


def normalization(bundle: DatasetBundle) -> tuple[Tensor, Tensor]:
    sample = bundle.train_images.float().div(255.0)
    mean = sample.mean(dim=(0, 2, 3), keepdim=True)
    std = sample.std(dim=(0, 2, 3), keepdim=True, unbiased=False).clamp_min(1e-4)
    return mean, std


def evaluate_clean(
    model: SmallConvNet,
    images: Tensor,
    labels: Tensor,
    mean: Tensor,
    std: Tensor,
    batch_size: int = 512,
) -> dict:
    model.eval()
    correct, loss_sum = 0, 0.0
    with torch.no_grad():
        for start in range(0, labels.numel(), batch_size):
            stop = min(start + batch_size, labels.numel())
            batch = _normalize(images[start:stop], mean, std)
            target = labels[start:stop]
            logits = model(batch)
            loss_sum += float(F.cross_entropy(logits, target, reduction="sum"))
            correct += int((logits.argmax(dim=-1) == target).sum())
    return {"accuracy": correct / labels.numel(), "nll": loss_sum / labels.numel()}


def train_classifier(
    bundle: DatasetBundle,
    config: ClassifierConfig,
    checkpoint_dir: str | Path,
    *,
    force: bool = False,
) -> tuple[SmallConvNet, Tensor, Tensor, dict]:
    """Train or load a deterministic clean classifier.

    Checkpoints are large transient artifacts and belong in ignored
    ``checkpoints_v4/``.  A compact metadata JSON is written next to each one.
    """

    if config.dataset != bundle.name or config.class_count != bundle.class_count:
        raise ValueError("classifier config does not match dataset")
    _seed_everything(config.seed)
    torch.set_num_threads(min(2, torch.get_num_threads()))
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"{bundle.name.lower().replace('-', '')}-{config.model_id}.pt"
    metadata_path = checkpoint.with_suffix(".json")
    model = SmallConvNet(config.input_channels, config.class_count, config.width)
    mean, std = normalization(bundle)

    if checkpoint.is_file() and not force:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        metadata = json.loads(metadata_path.read_text())
        return model, mean, std, metadata

    dataset = TensorDataset(bundle.train_images, bundle.train_labels)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    history = []
    started = time.time()
    for epoch in range(config.epochs):
        model.train()
        total_loss, total_correct, total = 0.0, 0, 0
        for raw_images, labels in loader:
            images = _normalize(raw_images, mean, std)
            # Deterministic horizontal flips only for natural CIFAR images.
            if bundle.name == "CIFAR-10":
                flip_mask = torch.rand(images.shape[0], generator=generator) < 0.5
                images[flip_mask] = images[flip_mask].flip(-1)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * labels.numel()
            total_correct += int((logits.argmax(dim=-1) == labels).sum())
            total += labels.numel()
        scheduler.step()
        validation = evaluate_clean(
            model, bundle.validation_images, bundle.validation_labels, mean, std, config.batch_size
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total_loss / total,
                "train_accuracy": total_correct / total,
                "validation_accuracy": validation["accuracy"],
                "validation_nll": validation["nll"],
            }
        )
        print(
            f"[{bundle.name}/{config.model_id}] epoch {epoch + 1}/{config.epochs} "
            f"train_acc={history[-1]['train_accuracy']:.4f} "
            f"val_acc={validation['accuracy']:.4f}",
            flush=True,
        )

    torch.save(model.state_dict(), checkpoint)
    test = evaluate_clean(model, bundle.test_images, bundle.test_labels, mean, std, config.batch_size)
    metadata = {
        "config": asdict(config),
        "model_id": config.model_id,
        "normalization_mean": mean.flatten().tolist(),
        "normalization_std": std.flatten().tolist(),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "history": history,
        "test_clean": test,
        "elapsed_seconds": time.time() - started,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return model, mean, std, metadata
