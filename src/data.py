"""CIFAR-10 loading from the official extracted batches + fixed data pools.

Integrity: every batch file's md5 is verified against the official reference
values (torchvision) before use. Pools are fixed: the train sample used for
encoder inputs / behavior labels is a seeded 2000-image subset of the train
pool; ground-truth evaluation uses the full 10k test pool.
"""
import hashlib
import os
import pickle

import numpy as np
import torch

from . import config as cfg

CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)

# Official per-file md5 checksums, transcribed programmatically from the
# torchvision reference (pytorch/vision torchvision/datasets/cifar.py,
# fetched via the GitHub API) and cross-verified against two independent
# GitHub mirrors of the extracted batches.
BATCH_MD5 = {
    "data_batch_1": "c99cafc152244af753f735de768cd75f",
    "data_batch_2": "d4bba439e000b95fd0a9bffe97cbabec",
    "data_batch_3": "54ebc095f3ab1f0389bbae665268c751",
    "data_batch_4": "634d18415352ddfa80567beed471001a",
    "data_batch_5": "482c414d41f54cd18b22e5b47cb7c3cb",
    "test_batch": "40351d587109b95175f43aff81a1287e",
    "batches.meta": "5ff9c542aee3614f3951f8cda6e48888",
}


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="bytes")


def load_batches(verify=True):
    """Returns (train_images, train_labels, test_images, test_labels) as
    uint8 NCHW tensors + int64 labels. Verifies md5 + structure."""
    cache = os.path.join(cfg.DATA_DIR, "cifar10_tensors.pt")
    if os.path.exists(cache):
        d = torch.load(cache, weights_only=False)
        return d["train_images"], d["train_labels"], d["test_images"], d["test_labels"]

    if verify:
        for name, md5 in BATCH_MD5.items():
            p = os.path.join(cfg.BATCHES_DIR, name)
            assert os.path.exists(p), f"missing batch file {p}"
            actual = _md5(p)
            assert actual == md5, f"md5 mismatch for {name}: {actual} != {md5}"
        print("[data] all 7 batch-file md5 checksums verified (official CIFAR-10)")

    def read_batch(name):
        d = _load_pickle(os.path.join(cfg.BATCHES_DIR, name))
        assert set(d.keys()) >= {b"data", b"labels"}, f"bad keys in {name}"
        x = d[b"data"].reshape(-1, 3, 32, 32)          # (N,3,32,32) NCHW
        y = np.asarray(d[b"labels"], dtype=np.int64)
        return x, y

    xs, ys = [], []
    for i in range(1, 6):
        x, y = read_batch(f"data_batch_{i}")
        xs.append(x)
        ys.append(y)
    train_images = np.concatenate(xs).astype(np.uint8)
    train_labels = np.concatenate(ys)
    test_images, test_labels = read_batch("test_batch")
    test_images = test_images.astype(np.uint8)

    assert train_images.shape == (50000, 3, 32, 32)
    assert test_images.shape == (10000, 3, 32, 32)
    assert train_labels.shape == (50000,) and test_labels.shape == (10000,)
    assert train_labels.min() >= 0 and train_labels.max() <= 9

    out = dict(
        train_images=torch.from_numpy(train_images),
        train_labels=torch.from_numpy(train_labels),
        test_images=torch.from_numpy(test_images),
        test_labels=torch.from_numpy(test_labels),
    )
    os.makedirs(cfg.DATA_DIR, exist_ok=True)
    torch.save(out, cache)
    print(f"[data] cached tensors to {cache}")
    return out["train_images"], out["train_labels"], out["test_images"], out["test_labels"]


def to_float01(x_uint8):
    """uint8 NCHW -> float32 in [0,1]."""
    return x_uint8.float() / 255.0


def normalize(x01):
    """[0,1] float NCHW -> normalized model input."""
    return (x01 - CIFAR_MEAN) / CIFAR_STD


def get_pools():
    """Fixed pools + fixed train-sample indices.

    Returns dict with:
      train_images, train_labels, test_images, test_labels (uint8 tensors)
      train_sample_idx (int64, TRAIN_SAMPLE indices into the train pool)
    """
    tr_x, tr_y, te_x, te_y = load_batches()
    rng = np.random.RandomState(cfg.MASTER_SEED)
    idx = np.sort(rng.choice(len(tr_x), cfg.TRAIN_SAMPLE, replace=False))
    return dict(
        train_images=tr_x, train_labels=tr_y,
        test_images=te_x, test_labels=te_y,
        train_sample_idx=torch.from_numpy(idx.astype(np.int64)),
    )
