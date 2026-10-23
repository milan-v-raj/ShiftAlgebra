"""Vectorized clean-room implementation of the Mason composition primitives.

These operators reproduce the pinned primitive names/default severities and
chronological word semantics.  They are protocol-compatible, not claimed to be
bitwise copies of the unlicensed upstream implementation.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Sequence

import torch
import torch.nn.functional as F

Tensor = torch.Tensor

PRIMITIVES = (
    "Contrast",
    "GaussianBlur",
    "ImpulseNoise",
    "Invert",
    "Rotate90",
    "Swirl",
)

DEFAULT_SEVERITY = {
    "Contrast": 2,
    "GaussianBlur": 2,
    "ImpulseNoise": 5,
    "Invert": "fixed",
    "Rotate90": "fixed",
    "Swirl": 3,
}


def _validate_images(images: Tensor) -> None:
    if images.ndim != 4 or images.shape[-2] != images.shape[-1]:
        raise ValueError("images must have shape [N,C,H,W] with square spatial dimensions")
    if not images.is_floating_point() or not torch.isfinite(images).all():
        raise ValueError("images must be finite floating-point tensors")


def contrast(images: Tensor, factor: float = 0.3) -> Tensor:
    means = images.mean(dim=(-2, -1), keepdim=True)
    return ((images - means) * factor + means).clamp(0.0, 1.0)


@lru_cache(maxsize=16)
def _gaussian_kernel(sigma: float, dtype_name: str) -> Tensor:
    radius = max(1, int(math.ceil(4.0 * sigma)))
    dtype = getattr(torch, dtype_name)
    coordinates = torch.arange(-radius, radius + 1, dtype=dtype)
    kernel = torch.exp(-0.5 * (coordinates / sigma).square())
    return kernel / kernel.sum()


def gaussian_blur(images: Tensor, sigma: float = 2.0) -> Tensor:
    kernel = _gaussian_kernel(float(sigma), str(images.dtype).split(".")[-1]).to(images.device)
    radius = kernel.numel() // 2
    channels = images.shape[1]
    vertical = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    horizontal = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    padded = F.pad(images, (0, 0, radius, radius), mode="reflect")
    output = F.conv2d(padded, vertical, groups=channels)
    padded = F.pad(output, (radius, radius, 0, 0), mode="reflect")
    return F.conv2d(padded, horizontal, groups=channels).clamp(0.0, 1.0)


def _deterministic_uniform(images: Tensor, sample_ids: Tensor, seed: int) -> Tensor:
    """Batching-independent integer hash converted to [0,1)."""

    n, c, h, w = images.shape
    if sample_ids.shape != (n,):
        raise ValueError("sample_ids must align with image batch")
    sid = sample_ids.to(device=images.device, dtype=torch.int64).view(n, 1, 1, 1)
    channel = torch.arange(c, device=images.device, dtype=torch.int64).view(1, c, 1, 1)
    yy = torch.arange(h, device=images.device, dtype=torch.int64).view(1, 1, h, 1)
    xx = torch.arange(w, device=images.device, dtype=torch.int64).view(1, 1, 1, w)
    hashed = (
        sid * 73_856_093
        ^ channel * 19_349_663
        ^ yy * 83_492_791
        ^ xx * 2_654_435_761
        ^ int(seed)
    )
    hashed = (hashed * 6_364_136_223_846_793_005 + 1_442_695_040_888_963_407) \
        & 0x7FFF_FFFF_FFFF_FFFF
    return (hashed & 0x00FF_FFFF).to(images.dtype) / float(1 << 24)


def impulse_noise(
    images: Tensor,
    sample_ids: Tensor,
    amount: float = 0.27,
    *,
    seed: int = 2778,
) -> Tensor:
    uniform = _deterministic_uniform(images, sample_ids, seed)
    output = images.clone()
    output[uniform < amount / 2] = 0.0
    output[uniform > 1.0 - amount / 2] = 1.0
    return output


def invert(images: Tensor) -> Tensor:
    return 1.0 - images


def rotate90(images: Tensor) -> Tensor:
    return torch.rot90(images, k=1, dims=(-2, -1))


@lru_cache(maxsize=16)
def _swirl_grid(size: int, strength: float, dtype_name: str) -> Tensor:
    dtype = getattr(torch, dtype_name)
    coordinates = torch.arange(size, dtype=dtype)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    center = (size - 1) / 2.0
    dx, dy = xx - center, yy - center
    radius_value = math.sqrt(2.0) * size / 2.0
    rho = torch.sqrt(dx.square() + dy.square())
    theta = torch.atan2(dy, dx) + strength * torch.exp(-rho / radius_value)
    source_x = center + rho * torch.cos(theta)
    source_y = center + rho * torch.sin(theta)
    grid_x = 2.0 * source_x / (size - 1) - 1.0
    grid_y = 2.0 * source_y / (size - 1) - 1.0
    return torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)


def swirl(images: Tensor, strength: float = 3.0) -> Tensor:
    size = images.shape[-1]
    grid = _swirl_grid(size, float(strength), str(images.dtype).split(".")[-1]).to(images.device)
    return F.grid_sample(
        images,
        grid.expand(images.shape[0], -1, -1, -1),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).clamp(0.0, 1.0)


def apply_primitive(
    images: Tensor,
    primitive: str,
    sample_ids: Tensor,
    *,
    noise_seed: int = 2778,
) -> Tensor:
    _validate_images(images)
    if primitive == "Identity":
        return images
    if primitive == "Contrast":
        return contrast(images)
    if primitive == "GaussianBlur":
        return gaussian_blur(images)
    if primitive == "ImpulseNoise":
        return impulse_noise(images, sample_ids, seed=noise_seed)
    if primitive == "Invert":
        return invert(images)
    if primitive == "Rotate90":
        return rotate90(images)
    if primitive == "Swirl":
        return swirl(images)
    raise KeyError(f"unknown primitive {primitive}")


def apply_word(
    images: Tensor,
    word: Sequence[str],
    sample_ids: Tensor,
    *,
    noise_seed: int = 2778,
) -> Tensor:
    """Apply primitives chronologically: ``word[0]`` is applied first."""

    output = images
    for primitive in word:
        output = apply_primitive(output, primitive, sample_ids, noise_seed=noise_seed)
    return output
