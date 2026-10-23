"""Architecture-independent, fixed-size primitive-shift fingerprints.

The fingerprint uses no target behavior label.  It summarizes paired clean and
shifted samples with deterministic random projections, logits, pixels, and
frequency statistics.  Its output dimension is independent of backbone feature
width, class count, and image channel count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


def _distribution_summary(values: Tensor) -> Tensor:
    values = values.reshape(-1)
    if values.numel() < 2 or not torch.isfinite(values).all():
        raise ValueError("summary values must contain at least two finite entries")
    quantiles = torch.quantile(
        values, torch.tensor([0.1, 0.5, 0.9], dtype=values.dtype, device=values.device)
    )
    return torch.cat((values.mean()[None], values.std(unbiased=False)[None], quantiles))


def _depth_indices(layer_count: int, depth_slots: int) -> list[int]:
    if layer_count < 1 or depth_slots < 1:
        raise ValueError("at least one layer and one depth slot are required")
    if depth_slots == 1:
        return [layer_count - 1]
    return [round(i * (layer_count - 1) / (depth_slots - 1)) for i in range(depth_slots)]


def _projection(width: int, output_dim: int, seed: int, *, dtype, device) -> Tensor:
    # Generate on CPU for identical values regardless of accelerator, then move.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    matrix = torch.randn(width, output_dim, generator=generator, dtype=torch.float64)
    matrix = matrix / matrix.norm(dim=0, keepdim=True).clamp_min(1e-12)
    return matrix.to(dtype=dtype, device=device)


def _high_frequency_ratio(images: Tensor) -> Tensor:
    if images.ndim != 4 or min(images.shape[-2:]) < 2:
        raise ValueError("pixels must have shape [sample, channel, height, width] with H,W >= 2")
    centered = images - images.mean(dim=(-2, -1), keepdim=True)
    spectrum = torch.fft.rfft2(centered, norm="ortho").abs().square()
    fy = torch.fft.fftfreq(images.shape[-2], device=images.device, dtype=images.dtype)[:, None]
    fx = torch.fft.rfftfreq(images.shape[-1], device=images.device, dtype=images.dtype)[None, :]
    radius = torch.sqrt(fx.square() + fy.square())
    mask = radius >= 0.25
    high = spectrum[..., mask].sum(dim=(-1, -2))
    total = spectrum.sum(dim=(-1, -2, -3)).clamp_min(1e-12)
    return high / total


@dataclass(frozen=True)
class FingerprintConfig:
    depth_slots: int = 4
    projection_dim: int = 16
    seed: int = 20261023

    @property
    def output_dim(self) -> int:
        # Per depth: shifted mean/std and displacement mean/std.
        representation = self.depth_slots * 4 * self.projection_dim
        # Four logit scalar distributions and five pixel scalar distributions;
        # each emits mean, std, q10, q50, q90.
        return representation + 4 * 5 + 5 * 5


class FixedFingerprint:
    """Stateless deterministic fingerprint extractor."""

    def __init__(self, config: FingerprintConfig | None = None) -> None:
        self.config = config or FingerprintConfig()
        if self.config.depth_slots < 1 or self.config.projection_dim < 1:
            raise ValueError("fingerprint dimensions must be positive")

    @property
    def output_dim(self) -> int:
        return self.config.output_dim

    def _representation(
        self, clean_layers: Sequence[Tensor], shifted_layers: Sequence[Tensor]
    ) -> Tensor:
        if len(clean_layers) != len(shifted_layers) or not clean_layers:
            raise ValueError("clean and shifted layer lists must have the same positive length")
        sample_count = clean_layers[0].shape[0]
        output = []
        for slot, layer_index in enumerate(_depth_indices(len(clean_layers), self.config.depth_slots)):
            clean, shifted = clean_layers[layer_index], shifted_layers[layer_index]
            if clean.ndim != 2 or clean.shape != shifted.shape or clean.shape[0] != sample_count:
                raise ValueError("each clean/shifted layer pair must match [sample, feature]")
            if not torch.isfinite(clean).all() or not torch.isfinite(shifted).all():
                raise ValueError("layer features must be finite")
            projection = _projection(
                clean.shape[1],
                self.config.projection_dim,
                self.config.seed + slot * 1009 + clean.shape[1],
                dtype=clean.dtype,
                device=clean.device,
            )
            shifted_projected = shifted @ projection
            displacement = (shifted - clean) @ projection
            output.extend(
                (
                    shifted_projected.mean(dim=0),
                    shifted_projected.std(dim=0, unbiased=False),
                    displacement.mean(dim=0),
                    displacement.std(dim=0, unbiased=False),
                )
            )
        return torch.cat(output)

    @staticmethod
    def _logits(clean_logits: Tensor, shifted_logits: Tensor) -> Tensor:
        if clean_logits.ndim != 2 or clean_logits.shape != shifted_logits.shape:
            raise ValueError("clean and shifted logits must match [sample, class]")
        if clean_logits.shape[1] < 2 or not torch.isfinite(clean_logits).all() \
                or not torch.isfinite(shifted_logits).all():
            raise ValueError("logits require at least two finite class scores")
        clean_prob = F.softmax(clean_logits, dim=-1)
        shifted_prob = F.softmax(shifted_logits, dim=-1)
        top2 = shifted_prob.topk(2, dim=-1).values
        entropy = -(shifted_prob * shifted_prob.clamp_min(1e-12).log()).sum(dim=-1)
        margin = top2[:, 0] - top2[:, 1]
        confidence = top2[:, 0]
        probability_displacement = (shifted_prob - clean_prob).square().sum(dim=-1).sqrt()
        return torch.cat(
            [_distribution_summary(values) for values in (
                entropy, margin, confidence, probability_displacement
            )]
        )

    @staticmethod
    def _pixels(clean_pixels: Tensor, shifted_pixels: Tensor) -> Tensor:
        if clean_pixels.shape != shifted_pixels.shape or clean_pixels.ndim != 4:
            raise ValueError("clean and shifted pixels must match [sample, channel, height, width]")
        if not torch.isfinite(clean_pixels).all() or not torch.isfinite(shifted_pixels).all():
            raise ValueError("pixels must be finite")
        dimensions = tuple(range(1, shifted_pixels.ndim))
        shifted_mean = shifted_pixels.mean(dim=dimensions)
        shifted_std = shifted_pixels.std(dim=dimensions, unbiased=False)
        absolute_displacement = (shifted_pixels - clean_pixels).abs().mean(dim=dimensions)
        shifted_high_frequency = _high_frequency_ratio(shifted_pixels)
        clean_high_frequency = _high_frequency_ratio(clean_pixels)
        high_frequency_change = shifted_high_frequency - clean_high_frequency
        return torch.cat(
            [_distribution_summary(values) for values in (
                shifted_mean,
                shifted_std,
                absolute_displacement,
                shifted_high_frequency,
                high_frequency_change,
            )]
        )

    def __call__(
        self,
        clean_layers: Sequence[Tensor],
        shifted_layers: Sequence[Tensor],
        clean_logits: Tensor,
        shifted_logits: Tensor,
        clean_pixels: Tensor,
        shifted_pixels: Tensor,
    ) -> Tensor:
        sample_counts = {
            clean_layers[0].shape[0] if clean_layers else -1,
            clean_logits.shape[0],
            shifted_logits.shape[0],
            clean_pixels.shape[0],
            shifted_pixels.shape[0],
        }
        if len(sample_counts) != 1 or next(iter(sample_counts)) < 2:
            raise ValueError("all inputs must share at least two paired samples")
        output = torch.cat(
            (
                self._representation(clean_layers, shifted_layers),
                self._logits(clean_logits, shifted_logits),
                self._pixels(clean_pixels, shifted_pixels),
            )
        )
        if output.numel() != self.output_dim or not torch.isfinite(output).all():
            raise AssertionError("fingerprint dimension/finite invariant failed")
        return output
