"""Order-channel alternatives and a cyclic representational counterexample."""

from __future__ import annotations

import torch
import torch.nn as nn

Tensor = torch.Tensor


def potential_order_effect(pi_i: Tensor, kappa_i: Tensor, pi_j: Tensor, kappa_j: Tensor) -> Tensor:
    """Historical transitive potential effect for chronological i then j."""

    if not (pi_i.shape == kappa_i.shape == pi_j.shape == kappa_j.shape):
        raise ValueError("potential tensors must have identical shapes")
    return 0.5 * pi_i * pi_j * (kappa_j - kappa_i)


class SkewBilinearOrder(nn.Module):
    """Low-dimensional antisymmetric order model that can represent cycles."""

    def __init__(self, input_dim: int, output_dim: int = 1) -> None:
        super().__init__()
        if input_dim < 2 or output_dim < 1:
            raise ValueError("invalid skew-bilinear dimensions")
        self.raw = nn.Parameter(torch.randn(output_dim, input_dim, input_dim) * 0.02)

    @property
    def skew(self) -> Tensor:
        return 0.5 * (self.raw - self.raw.transpose(-1, -2))

    def forward(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        if x_i.shape != x_j.shape or x_i.shape[-1] != self.raw.shape[-1]:
            raise ValueError("ordered inputs must have matching final dimensions")
        return torch.einsum("...d,kde,...e->...k", x_i, self.skew, x_j)


class AntisymmetrizedPairNetwork(nn.Module):
    """Flexible pair network with exact reversal antisymmetry by construction."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        if x_i.shape != x_j.shape:
            raise ValueError("ordered inputs must have matching shapes")
        ij = self.net(torch.cat((x_i, x_j), dim=-1))
        ji = self.net(torch.cat((x_j, x_i), dim=-1))
        return 0.5 * (ij - ji)


def rock_paper_scissors_target(*, dtype=torch.float64) -> Tensor:
    """Target matrix where 0->1, 1->2, and 2->0 all have effect +1."""

    target = torch.zeros(3, 3, dtype=dtype)
    target[0, 1] = target[1, 2] = target[2, 0] = 1.0
    return target - target.T


def potential_cycle_residual(target: Tensor) -> dict:
    """Fit scalar potentials by least squares and expose cyclic residual.

    Edges use the target chronology 0->1, 1->2, 2->0.  Any potential model's
    edge sum is zero, so a non-zero target circulation is unrepresentable.
    """

    if target.shape != (3, 3):
        raise ValueError("cycle diagnostic expects a 3x3 target")
    design = torch.tensor(
        [[-0.5, 0.5, 0.0], [0.0, -0.5, 0.5], [0.5, 0.0, -0.5]],
        dtype=target.dtype,
        device=target.device,
    )
    values = torch.stack((target[0, 1], target[1, 2], target[2, 0]))
    solution = torch.linalg.lstsq(design, values).solution
    prediction = design @ solution
    return {
        "target_circulation": float(values.sum()),
        "predicted_circulation": float(prediction.sum()),
        "minimum_mse": float((prediction - values).square().mean()),
        "potentials": solution,
        "predicted_edges": prediction,
    }
