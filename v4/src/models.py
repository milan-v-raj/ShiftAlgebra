"""V4 composition models built on the exact algebraic state.

The exact-core model decodes once after composing the complete word.  This
keeps the end-to-end parenthesization claim separate from the historical V3
Jordan/tree-averaged readout.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .algebra import CompositionOutput, compose_streaming


class MBCore(nn.Module):
    """Decode one exact associative state after the entire word is composed.

    The caller supplies primitive behavior, symmetric features ``z``, and
    directional ``pi``/``kappa`` coordinates.  Separate fingerprint encoders
    may be connected later without changing this audited core.
    """

    def __init__(
        self,
        behavior_dim: int,
        symmetric_dim: int,
        order_dim: int,
        hidden_dim: int = 64,
        initial_p: float = 2.0,
    ) -> None:
        super().__init__()
        if behavior_dim <= 0 or symmetric_dim < 0 or order_dim < 0:
            raise ValueError("invalid model dimensions")
        if initial_p <= 1:
            raise ValueError("initial_p must be greater than one")
        self.behavior_dim = behavior_dim
        self.symmetric_dim = symmetric_dim
        self.order_dim = order_dim
        raw_initial = math.log(math.expm1(initial_p - 1.0))
        self.raw_p = nn.Parameter(torch.full((behavior_dim,), raw_initial))
        feature_dim = behavior_dim + symmetric_dim + order_dim
        self.residual = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, behavior_dim),
        )
        # A small non-zero initialization avoids a zero-gradient saddle while
        # keeping the initial predictor close to the signed-power base.
        nn.init.normal_(self.residual[-1].weight, std=1e-3)
        nn.init.zeros_(self.residual[-1].bias)

    @property
    def p(self) -> torch.Tensor:
        return 1.0 + F.softplus(self.raw_p)

    def compose(
        self,
        behaviors: torch.Tensor,
        z: torch.Tensor,
        pi: torch.Tensor,
        kappa: torch.Tensor,
    ) -> CompositionOutput:
        return compose_streaming(behaviors, z, pi, kappa, self.p)

    def forward(
        self,
        behaviors: torch.Tensor,
        z: torch.Tensor,
        pi: torch.Tensor,
        kappa: torch.Tensor,
    ) -> torch.Tensor:
        state = self.compose(behaviors, z, pi, kappa)
        return state.base + self.residual(state.features())


class LpOnly(nn.Module):
    """Five-parameter (or generally d-parameter) signed-power baseline."""

    def __init__(self, behavior_dim: int, initial_p: float = 2.0) -> None:
        super().__init__()
        if behavior_dim <= 0 or initial_p <= 1:
            raise ValueError("invalid LpOnly configuration")
        raw_initial = math.log(math.expm1(initial_p - 1.0))
        self.raw_p = nn.Parameter(torch.full((behavior_dim,), raw_initial))

    @property
    def p(self) -> torch.Tensor:
        return 1.0 + F.softplus(self.raw_p)

    def forward(self, behaviors: torch.Tensor) -> torch.Tensor:
        from .algebra import signed_power_inverse, signed_power_transform

        return signed_power_inverse(
            signed_power_transform(behaviors, self.p).sum(dim=0), self.p
        )
