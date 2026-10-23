"""Clean-room dataset loaders for the Phase-1 Mason protocol bridge.

Large data stay under ``data/`` and are never committed.  The loaders verify
known file hashes/provenance before exposing tensors.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

Tensor = torch.Tensor

CIFAR_BATCH_MD5 = {
    "data_batch_1": "c99cafc152244af753f735de768cd75f",
    "data_batch_2": "d4bba439e000b95fd0a9bffe97cbabec",
    "data_batch_3": "54ebc095f3ab1f0389bbae665268c751",
    "data_batch_4": "634d18415352ddfa80567beed471001a",
    "data_batch_5": "482c414d41f54cd18b22e5b47cb7c3cb",
    "test_batch": "40351d587109b95175f43aff81a1287e",
}

EMNIST_BALANCED_COUNTS = {"train": 112_800, "test": 18_800}


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    train_images: Tensor  # uint8 [N,C,H,W]
    train_labels: Tensor  # int64 [N]
    validation_images: Tensor
    validation_labels: Tensor
    test_images: Tensor
    test_labels: Tensor
    class_count: int
    provenance: dict


def _md5(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_cifar_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        raw = pickle.load(handle, encoding="latin1")
    images = np.asarray(raw["data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
    labels = np.asarray(raw.get("labels", raw.get("fine_labels")), dtype=np.int64)
    if images.shape[0] != labels.shape[0]:
        raise ValueError(f"CIFAR image/label mismatch in {path}")
    return images, labels


def load_cifar10(root: str | Path) -> DatasetBundle:
    root = Path(root)
    hashes = {}
    for filename, expected in CIFAR_BATCH_MD5.items():
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = _md5(path)
        if observed != expected:
            raise ValueError(f"CIFAR hash mismatch for {filename}: {observed} != {expected}")
        hashes[filename] = observed

    train_parts = [_load_cifar_batch(root / f"data_batch_{index}") for index in range(1, 6)]
    train_images = np.concatenate([part[0] for part in train_parts])
    train_labels = np.concatenate([part[1] for part in train_parts])
    test_images, test_labels = _load_cifar_batch(root / "test_batch")

    # Match Mason's sorted per-class split: 4,000 train + 1,000 validation.
    order = np.argsort(train_labels, kind="stable")
    images_sorted, labels_sorted = train_images[order], train_labels[order]
    images_by_class = images_sorted.reshape(10, 5_000, 3, 32, 32)
    labels_by_class = labels_sorted.reshape(10, 5_000)
    fit_images = images_by_class[:, :4_000].reshape(-1, 3, 32, 32)
    fit_labels = labels_by_class[:, :4_000].reshape(-1)
    val_images = images_by_class[:, 4_000:].reshape(-1, 3, 32, 32)
    val_labels = labels_by_class[:, 4_000:].reshape(-1)

    return DatasetBundle(
        name="CIFAR-10",
        train_images=torch.from_numpy(fit_images.copy()),
        train_labels=torch.from_numpy(fit_labels.copy()),
        validation_images=torch.from_numpy(val_images.copy()),
        validation_labels=torch.from_numpy(val_labels.copy()),
        test_images=torch.from_numpy(test_images.copy()),
        test_labels=torch.from_numpy(test_labels.copy()),
        class_count=10,
        provenance={
            "mirror": "https://github.com/wikiabhi/Cifar-10",
            "mirror_role": "transport only; hashes match canonical CIFAR Python batch hashes",
            "batch_md5": hashes,
            "split": "class-stable first 4000 train / last 1000 validation",
        },
    )


def _decode_emnist_sheet(path: Path, expected_count: int) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    if image.shape[0] % 28 or image.shape[1] % 28:
        raise ValueError(f"EMNIST sheet dimensions are not multiples of 28: {path}")
    rows, columns = image.shape[0] // 28, image.shape[1] // 28
    tiles = image.reshape(rows, 28, columns, 28).transpose(0, 2, 1, 3).reshape(-1, 28, 28)
    if tiles.shape[0] != expected_count:
        raise ValueError(f"EMNIST sheet count mismatch in {path}: {tiles.shape[0]} != {expected_count}")
    return tiles


def _load_emnist_split(root: Path, split: str) -> tuple[np.ndarray, np.ndarray, dict]:
    metadata_path = root / f"{split}.json"
    metadata = json.loads(metadata_path.read_text())
    labels = np.asarray(metadata["id"], dtype=np.int64)
    images = []
    file_hashes = {}
    for filename in metadata["files"]:
        path = root / filename
        # Filename encodes an inclusive index range.
        stem = path.stem
        start, end = (int(value) for value in stem.rsplit("_", 4)[-3::2])
        count = end - start + 1
        images.append(_decode_emnist_sheet(path, count))
        file_hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    output = np.concatenate(images)[: labels.size]
    if output.shape[0] != labels.size or labels.min() != 0 or labels.max() != 46:
        raise ValueError(f"invalid EMNIST balanced {split} split")
    return output[:, None, :, :], labels, {
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "file_sha256": file_hashes,
    }


def load_emnist_balanced(root: str | Path) -> DatasetBundle:
    root = Path(root)
    train_images, train_labels, train_hashes = _load_emnist_split(root, "train")
    test_images, test_labels, test_hashes = _load_emnist_split(root, "test")
    if train_images.shape[0] != EMNIST_BALANCED_COUNTS["train"] \
            or test_images.shape[0] != EMNIST_BALANCED_COUNTS["test"]:
        raise ValueError("EMNIST balanced count mismatch")

    # Match Mason: torchvision data are transposed before sorting.  The packed
    # sheets are already display-oriented, so no additional transpose is used.
    order = np.argsort(train_labels, kind="stable")
    images_sorted, labels_sorted = train_images[order], train_labels[order]
    images_by_class = images_sorted.reshape(47, 2_400, 1, 28, 28)
    labels_by_class = labels_sorted.reshape(47, 2_400)
    fit_images = images_by_class[:, :2_000].reshape(-1, 1, 28, 28)
    fit_labels = labels_by_class[:, :2_000].reshape(-1)
    val_images = images_by_class[:, 2_000:].reshape(-1, 1, 28, 28)
    val_labels = labels_by_class[:, 2_000:].reshape(-1)

    return DatasetBundle(
        name="EMNIST",
        train_images=torch.from_numpy(fit_images.copy()),
        train_labels=torch.from_numpy(fit_labels.copy()),
        validation_images=torch.from_numpy(val_images.copy()),
        validation_labels=torch.from_numpy(val_labels.copy()),
        test_images=torch.from_numpy(test_images.copy()),
        test_labels=torch.from_numpy(test_labels.copy()),
        class_count=47,
        provenance={
            "mirror": "https://github.com/0r4nd/emnist-packed",
            "mirror_role": "transport-only lossless WebP packaging of EMNIST balanced",
            "hashes": {"train": train_hashes, "test": test_hashes},
            "split": "class-stable first 2000 train / last 400 validation",
        },
    )


def balanced_subset(labels: Tensor, per_class: int, *, offset: int = 0) -> Tensor:
    """Return stable balanced indices without consulting model outcomes."""

    if labels.ndim != 1 or per_class <= 0 or offset < 0:
        raise ValueError("invalid balanced subset request")
    selected = []
    for class_id in range(int(labels.max()) + 1):
        indices = torch.nonzero(labels == class_id, as_tuple=False).flatten()
        if offset + per_class > indices.numel():
            raise ValueError(f"class {class_id} lacks requested examples")
        selected.append(indices[offset:offset + per_class])
    return torch.cat(selected)
