"""Preregistered attenuation homotopy for the Phase-1 Mason primitives.

For a primitive T and strength alpha, Phase 1B uses

    H[T, alpha](x) = (1 - alpha) * x + alpha * T(x).

This gives every primitive, including the originally fixed Invert and Rotate90,
a common continuous path from identity (alpha=0) to the exact Phase-1 operator
(alpha=1).  Compositions apply the homotopies chronologically; attenuation is
performed at every step rather than once at the end of a word.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch

from v4.phase1.corruptions import PRIMITIVES, apply_primitive

Tensor = torch.Tensor

# Dense where the mild calibration target is expected, coarser thereafter.
STRENGTH_GRID = tuple(round(0.025 * index, 3) for index in range(21)) + (
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
)
TARGET_NORMALIZED_DROP = 0.08
ACCEPTANCE_BAND = (0.04, 0.12)


def validate_strengths(strengths: Mapping[str, float]) -> dict[str, float]:
    """Return a normalized strength dictionary or reject an incomplete map."""

    if set(strengths) != set(PRIMITIVES):
        missing = sorted(set(PRIMITIVES) - set(strengths))
        extra = sorted(set(strengths) - set(PRIMITIVES))
        raise ValueError(f"strength map mismatch: missing={missing}, extra={extra}")
    normalized = {name: float(strengths[name]) for name in PRIMITIVES}
    if any(not 0.0 <= value <= 1.0 for value in normalized.values()):
        raise ValueError("all attenuation strengths must lie in [0, 1]")
    return normalized


def apply_attenuated_primitive(
    images: Tensor,
    primitive: str,
    sample_ids: Tensor,
    alpha: float,
    *,
    noise_seed: int = 2778,
) -> Tensor:
    """Apply one endpoint-preserving convex attenuation of a primitive."""

    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    if primitive == "Identity" or alpha == 0.0:
        return images
    shifted = apply_primitive(images, primitive, sample_ids, noise_seed=noise_seed)
    if alpha == 1.0:
        return shifted
    return torch.lerp(images, shifted, alpha).clamp(0.0, 1.0)


def apply_calibrated_word(
    images: Tensor,
    word: Sequence[str],
    sample_ids: Tensor,
    strengths: Mapping[str, float],
    *,
    noise_seed: int = 2778,
) -> Tensor:
    """Apply a word chronologically with one frozen strength per primitive."""

    normalized = validate_strengths(strengths)
    output = images
    for primitive in word:
        if primitive == "Identity":
            continue
        output = apply_attenuated_primitive(
            output,
            primitive,
            sample_ids,
            normalized[primitive],
            noise_seed=noise_seed,
        )
    return output
