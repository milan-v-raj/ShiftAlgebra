"""Behavior measurement for frozen classifiers under composed words."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch
import torch.nn.functional as F

from v4.src.fingerprints import FixedFingerprint

from .classifiers import SmallConvNet
from .corruptions import apply_word

Tensor = torch.Tensor

BEHAVIOR_NAMES = ("margin", "drift", "ece", "nll", "accuracy")


@dataclass(frozen=True)
class Behavior:
    accuracy: float
    margin: float
    nll: float
    ece: float
    drift: float
    sample_count: int

    def delta(self, clean: "Behavior") -> list[float]:
        return [
            self.margin - clean.margin,
            self.drift,
            self.ece - clean.ece,
            self.nll - clean.nll,
            self.accuracy - clean.accuracy,
        ]


def _ece(confidence: Tensor, correct: Tensor, bins: int = 15) -> float:
    boundaries = torch.linspace(0, 1, bins + 1, dtype=confidence.dtype)
    result = torch.tensor(0.0, dtype=confidence.dtype)
    for index in range(bins):
        if index == bins - 1:
            mask = (confidence >= boundaries[index]) & (confidence <= boundaries[index + 1])
        else:
            mask = (confidence >= boundaries[index]) & (confidence < boundaries[index + 1])
        if mask.any():
            result += mask.float().mean() * (correct[mask].float().mean() - confidence[mask].mean()).abs()
    return float(result)


def _normalize(images: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    return (images - mean) / std


@torch.no_grad()
def _clean_cache(
    model: SmallConvNet,
    images_uint8: Tensor,
    labels: Tensor,
    mean: Tensor,
    std: Tensor,
    batch_size: int,
) -> dict:
    model.eval()
    logits_parts: list[Tensor] = []
    layer_parts: list[list[Tensor]] | None = None
    for start in range(0, labels.numel(), batch_size):
        stop = min(start + batch_size, labels.numel())
        pixels = images_uint8[start:stop].float().div(255.0)
        logits, layers = model.forward_with_layers(_normalize(pixels, mean, std))
        logits_parts.append(logits.cpu())
        if layer_parts is None:
            layer_parts = [[] for _ in layers]
        for destination, values in zip(layer_parts, layers):
            destination.append(values.cpu())
    return {
        "logits": torch.cat(logits_parts),
        "layers": [torch.cat(parts) for parts in (layer_parts or [])],
        "final": torch.cat((layer_parts or [[]])[-1]) if layer_parts else torch.empty(0),
        "pixels": images_uint8.float().div(255.0),
    }


@torch.no_grad()
def evaluate_word(
    model: SmallConvNet,
    images_uint8: Tensor,
    labels: Tensor,
    sample_ids: Tensor,
    word: Sequence[str],
    mean: Tensor,
    std: Tensor,
    clean_cache: dict,
    *,
    batch_size: int = 256,
    collect_fingerprint: bool = False,
    word_transform: Callable[[Tensor, Sequence[str], Tensor], Tensor] = apply_word,
) -> tuple[Behavior, Tensor | None]:
    model.eval()
    logits_parts, final_parts = [], []
    shifted_layers: list[list[Tensor]] | None = None
    shifted_pixels = []
    for start in range(0, labels.numel(), batch_size):
        stop = min(start + batch_size, labels.numel())
        pixels = images_uint8[start:stop].float().div(255.0)
        shifted = word_transform(pixels, word, sample_ids[start:stop])
        logits, layers = model.forward_with_layers(_normalize(shifted, mean, std))
        logits_parts.append(logits.cpu())
        final_parts.append(layers[-1].cpu())
        if collect_fingerprint:
            if shifted_layers is None:
                shifted_layers = [[] for _ in layers]
            for destination, values in zip(shifted_layers, layers):
                destination.append(values.cpu())
            shifted_pixels.append(shifted.cpu())

    logits = torch.cat(logits_parts)
    final = torch.cat(final_parts)
    probabilities = logits.softmax(dim=-1)
    confidence, prediction = probabilities.max(dim=-1)
    top2 = probabilities.topk(2, dim=-1).values
    correct = prediction == labels
    behavior = Behavior(
        accuracy=float(correct.float().mean()),
        margin=float((top2[:, 0] - top2[:, 1]).mean()),
        nll=float(F.cross_entropy(logits, labels)),
        ece=_ece(confidence, correct),
        drift=float((final - clean_cache["final"]).norm(dim=-1).mean()),
        sample_count=labels.numel(),
    )

    fingerprint = None
    if collect_fingerprint:
        extractor = FixedFingerprint()
        fingerprint = extractor(
            clean_cache["layers"],
            [torch.cat(parts) for parts in (shifted_layers or [])],
            clean_cache["logits"],
            logits,
            clean_cache["pixels"],
            torch.cat(shifted_pixels),
        )
    return behavior, fingerprint


def evaluate_protocol(
    model: SmallConvNet,
    model_id: str,
    dataset: str,
    images_uint8: Tensor,
    labels: Tensor,
    sample_ids: Tensor,
    words: Sequence[tuple[str, ...]],
    condition_ids: Sequence[str],
    mean: Tensor,
    std: Tensor,
    output_csv: str | Path,
    fingerprint_json: str | Path,
    *,
    batch_size: int = 256,
    word_transform: Callable[[Tensor, Sequence[str], Tensor], Tensor] = apply_word,
) -> dict:
    if len(words) != len(condition_ids):
        raise ValueError("word and condition id counts differ")
    started = time.time()
    cache = _clean_cache(model, images_uint8, labels, mean, std, batch_size)
    identity = words.index(("Identity",))
    clean, _ = evaluate_word(
        model, images_uint8, labels, sample_ids, words[identity], mean, std, cache,
        batch_size=batch_size, word_transform=word_transform,
    )

    rows = []
    fingerprints = {}
    for index, (condition_id, word) in enumerate(zip(condition_ids, words)):
        collect = len(word) == 1 and word != ("Identity",)
        behavior, fingerprint = evaluate_word(
            model,
            images_uint8,
            labels,
            sample_ids,
            word,
            mean,
            std,
            cache,
            batch_size=batch_size,
            collect_fingerprint=collect,
            word_transform=word_transform,
        )
        delta = behavior.delta(clean)
        row = {
            "condition_id": condition_id,
            "dataset": dataset,
            "model_id": model_id,
            "word": ">".join(word),
            "length": 0 if word == ("Identity",) else len(word),
            **asdict(behavior),
            **{f"delta_{name}": value for name, value in zip(BEHAVIOR_NAMES, delta)},
        }
        rows.append(row)
        if fingerprint is not None:
            fingerprints[word[0]] = fingerprint.tolist()
        if index % 10 == 0 or index + 1 == len(words):
            print(
                f"[{dataset}/{model_id}] condition {index + 1}/{len(words)} "
                f"word={row['word']} acc={behavior.accuracy:.4f}",
                flush=True,
            )

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    fingerprint_json = Path(fingerprint_json)
    fingerprint_json.write_text(json.dumps(fingerprints, indent=2, sort_keys=True) + "\n")
    return {
        "dataset": dataset,
        "model_id": model_id,
        "conditions": len(rows),
        "samples": labels.numel(),
        "clean": asdict(clean),
        "elapsed_seconds": time.time() - started,
        "output_csv": str(output_csv),
        "fingerprint_json": str(fingerprint_json),
    }
