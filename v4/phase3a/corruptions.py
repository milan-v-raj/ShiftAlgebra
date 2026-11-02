"""Deterministic clean-room ImageNet-C-style corruption operators.

These operators cover noise, blur, illumination, and digital artifacts. They
are inspired by common-corruption categories but are not claimed to reproduce
ImageNet-C implementations or severities bit-for-bit.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Sequence

import torch
import torch.nn.functional as F

from v4.phase1.corruptions import _deterministic_uniform

PRIMITIVES = (
    "Brightness",
    "GaussianNoise",
    "JPEGCompression",
    "MotionBlur",
    "Pixelate",
    "ShotNoise",
)

PARAMETERS = {
    "Brightness": {"offset": 0.12},
    "GaussianNoise": {"sigma": 0.06, "seed": 31_001},
    "JPEGCompression": {"quantization_scale": 0.55},
    "MotionBlur": {"kernel_size": 5},
    "Pixelate": {"downsample_size": 16},
    "ShotNoise": {"count_scale": 80.0, "seed": 31_101},
}


def _validate(images, sample_ids):
    if images.ndim != 4 or images.shape[1:] != (3, 32, 32):
        raise ValueError("expected RGB CIFAR tensors [N,3,32,32]")
    if not images.is_floating_point() or not torch.isfinite(images).all():
        raise ValueError("images must be finite floating-point tensors")
    if sample_ids.shape != (len(images),):
        raise ValueError("sample_ids must align with images")


def _normal_field(images, sample_ids, seed):
    first = _deterministic_uniform(images, sample_ids, seed).clamp_min(1e-7)
    second = _deterministic_uniform(images, sample_ids, seed + 1)
    return torch.sqrt(-2.0 * torch.log(first)) * torch.cos(2.0 * math.pi * second)


def brightness(images, offset=0.12):
    return (images + offset).clamp(0.0, 1.0)


def gaussian_noise(images, sample_ids, sigma=0.06, seed=31_001):
    return (images + sigma * _normal_field(images, sample_ids, seed)).clamp(0.0, 1.0)


def shot_noise(images, sample_ids, count_scale=80.0, seed=31_101):
    # Deterministic normal approximation to signal-dependent Poisson shot noise.
    standard_deviation = torch.sqrt(images.clamp_min(0.0) / count_scale)
    return (images + standard_deviation * _normal_field(images, sample_ids, seed)).clamp(0.0, 1.0)


@lru_cache(maxsize=8)
def _motion_kernel(size, dtype_name):
    dtype = getattr(torch, dtype_name)
    kernel = torch.zeros(size, size, dtype=dtype)
    kernel[torch.arange(size), torch.arange(size)] = 1.0 / size
    return kernel


def motion_blur(images, kernel_size=5):
    kernel = _motion_kernel(kernel_size, str(images.dtype).split(".")[-1]).to(images.device)
    weights = kernel.view(1, 1, kernel_size, kernel_size).repeat(images.shape[1], 1, 1, 1)
    radius = kernel_size // 2
    padded = F.pad(images, (radius, radius, radius, radius), mode="reflect")
    return F.conv2d(padded, weights, groups=images.shape[1]).clamp(0.0, 1.0)


def pixelate(images, downsample_size=16):
    reduced = F.interpolate(images, size=(downsample_size, downsample_size), mode="bilinear", align_corners=False)
    return F.interpolate(reduced, size=(32, 32), mode="nearest").clamp(0.0, 1.0)


@lru_cache(maxsize=8)
def _dct_matrix(size, dtype_name):
    dtype = getattr(torch, dtype_name)
    coordinates = torch.arange(size, dtype=dtype)
    frequencies = coordinates.view(-1, 1)
    matrix = torch.cos(math.pi * (2 * coordinates + 1) * frequencies / (2 * size))
    matrix[0] *= math.sqrt(1.0 / size)
    matrix[1:] *= math.sqrt(2.0 / size)
    return matrix


@lru_cache(maxsize=8)
def _jpeg_quantization(dtype_name):
    dtype = getattr(torch, dtype_name)
    return torch.tensor([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ], dtype=dtype) / 255.0


def jpeg_compression(images, quantization_scale=0.55):
    n, channels, height, width = images.shape
    block_size = 8
    blocks = images.reshape(n, channels, height // block_size, block_size, width // block_size, block_size)
    blocks = blocks.permute(0, 1, 2, 4, 3, 5).reshape(-1, block_size, block_size)
    matrix = _dct_matrix(block_size, str(images.dtype).split(".")[-1]).to(images.device)
    quantization = _jpeg_quantization(str(images.dtype).split(".")[-1]).to(images.device)
    centered = blocks - 0.5
    coefficients = matrix @ centered @ matrix.T
    restored = torch.round(coefficients / (quantization * quantization_scale)) * (
        quantization * quantization_scale
    )
    blocks = matrix.T @ restored @ matrix + 0.5
    output = blocks.reshape(n, channels, height // block_size, width // block_size, block_size, block_size)
    output = output.permute(0, 1, 2, 4, 3, 5).reshape(n, channels, height, width)
    return output.clamp(0.0, 1.0)


def apply_primitive(images, primitive, sample_ids):
    _validate(images, sample_ids)
    if primitive == "Identity":
        return images
    if primitive == "Brightness":
        return brightness(images, **PARAMETERS[primitive])
    if primitive == "GaussianNoise":
        return gaussian_noise(images, sample_ids, **PARAMETERS[primitive])
    if primitive == "JPEGCompression":
        return jpeg_compression(images, **PARAMETERS[primitive])
    if primitive == "MotionBlur":
        return motion_blur(images, **PARAMETERS[primitive])
    if primitive == "Pixelate":
        return pixelate(images, **PARAMETERS[primitive])
    if primitive == "ShotNoise":
        return shot_noise(images, sample_ids, **PARAMETERS[primitive])
    raise KeyError(primitive)


def apply_word(images, word: Sequence[str], sample_ids):
    output = images
    for primitive in word:
        output = apply_primitive(output, primitive, sample_ids)
    return output
