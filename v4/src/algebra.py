"""Exact algebraic core for Shift Algebra V4.

This module is deliberately independent of the historical V1--V3 code.  It
implements the numerical transform and sufficient state described in
NEURIPS_DEVELOPMENT_PLAN.md, while preserving the old code for reproduction.

Chronology convention
---------------------
A word ``(X1, X2, ..., Xm)`` means X1 is applied first.  Literal operator
matrices therefore accumulate by left multiplication::

    T_word = T_Xm @ ... @ T_X2 @ T_X1

The closed-form and streaming implementations below use that same convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

Tensor = torch.Tensor
DEFAULT_EPS = 1e-6


def _validate_p(p: Tensor) -> None:
    if torch.any(~torch.isfinite(p)):
        raise ValueError("p must be finite")
    if torch.any(p < 1.0):
        raise ValueError("V4 saturation exponents must satisfy p >= 1")


def signed_power_transform(x: Tensor, p: Tensor, eps: float = DEFAULT_EPS) -> Tensor:
    """Numerically stable signed power transform with an exact paired inverse.

    ``phi_p(x) = sign(x) * ((abs(x) + eps)**p - eps**p)``.
    The subtraction makes zero map exactly to zero.
    """

    if eps <= 0:
        raise ValueError("eps must be positive")
    _validate_p(p)
    eps_t = torch.as_tensor(eps, dtype=x.dtype, device=x.device)
    return torch.sign(x) * ((x.abs() + eps_t).pow(p) - eps_t.pow(p))


def signed_power_inverse(u: Tensor, p: Tensor, eps: float = DEFAULT_EPS) -> Tensor:
    """Exact algebraic inverse of :func:`signed_power_transform`."""

    if eps <= 0:
        raise ValueError("eps must be positive")
    _validate_p(p)
    eps_t = torch.as_tensor(eps, dtype=u.dtype, device=u.device)
    return torch.sign(u) * ((u.abs() + eps_t.pow(p)).pow(1.0 / p) - eps_t)


@dataclass(frozen=True)
class CompositionOutput:
    """Extracted sufficient statistics for one chronological word."""

    base: Tensor
    symmetric: Tensor
    ordered: Tensor
    length: int

    def features(self) -> Tensor:
        """Concatenate base, symmetric, and ordered channels."""

        return torch.cat((self.base, self.symmetric, self.ordered), dim=-1)


@dataclass(frozen=True)
class AlgebraState:
    """Associative sufficient state for a contiguous chronological word.

    ``order_c`` stores the central coordinate of a Heisenberg block.  For two
    consecutive chunks A then B, the literal matrix product is ``T_B @ T_A``
    and its central coordinate is ``c_B + c_A + u_B * v_A``.
    """

    length: int
    sum_phi: Tensor
    sum_z: Tensor
    sum_z_sq: Tensor
    order_u: Tensor
    order_v: Tensor
    order_c: Tensor

    @classmethod
    def identity(
        cls,
        behavior_dim: int,
        symmetric_dim: int,
        order_dim: int,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> "AlgebraState":
        if min(behavior_dim, symmetric_dim, order_dim) < 0:
            raise ValueError("state dimensions must be non-negative")
        return cls(
            length=0,
            sum_phi=torch.zeros(behavior_dim, dtype=dtype, device=device),
            sum_z=torch.zeros(symmetric_dim, dtype=dtype, device=device),
            sum_z_sq=torch.zeros(symmetric_dim, dtype=dtype, device=device),
            order_u=torch.zeros(order_dim, dtype=dtype, device=device),
            order_v=torch.zeros(order_dim, dtype=dtype, device=device),
            order_c=torch.zeros(order_dim, dtype=dtype, device=device),
        )

    @classmethod
    def primitive(
        cls,
        behavior: Tensor,
        z: Tensor,
        pi: Tensor,
        kappa: Tensor,
        p: Tensor,
        *,
        eps: float = DEFAULT_EPS,
    ) -> "AlgebraState":
        _validate_primitive_shapes(behavior, z, pi, kappa, p)
        u = kappa * pi
        v = pi
        return cls(
            length=1,
            sum_phi=signed_power_transform(behavior, p, eps),
            sum_z=z,
            sum_z_sq=z.square(),
            order_u=u,
            order_v=v,
            order_c=0.5 * u * v,
        )

    def then(self, later: "AlgebraState") -> "AlgebraState":
        """Compose this chronological chunk followed by ``later``.

        This operation is exactly associative up to floating-point rounding.
        """

        self._validate_compatible(later)
        return AlgebraState(
            length=self.length + later.length,
            sum_phi=self.sum_phi + later.sum_phi,
            sum_z=self.sum_z + later.sum_z,
            sum_z_sq=self.sum_z_sq + later.sum_z_sq,
            order_u=self.order_u + later.order_u,
            order_v=self.order_v + later.order_v,
            order_c=later.order_c + self.order_c + later.order_u * self.order_v,
        )

    def extract(self, p: Tensor, *, eps: float = DEFAULT_EPS) -> CompositionOutput:
        _validate_p(p)
        if p.shape != self.sum_phi.shape:
            raise ValueError("p and behavior state must have identical shapes")
        base = signed_power_inverse(self.sum_phi, p, eps)
        if self.length < 2:
            symmetric = torch.zeros_like(self.sum_z)
            ordered = torch.zeros_like(self.order_c)
        else:
            pairs = self.length * (self.length - 1) / 2
            symmetric_sum = 0.5 * (self.sum_z.square() - self.sum_z_sq)
            ordered_sum = self.order_c - 0.5 * self.order_u * self.order_v
            symmetric = symmetric_sum / pairs
            ordered = ordered_sum / pairs
        return CompositionOutput(base, symmetric, ordered, self.length)

    def _validate_compatible(self, other: "AlgebraState") -> None:
        fields = ("sum_phi", "sum_z", "sum_z_sq", "order_u", "order_v", "order_c")
        for field in fields:
            a, b = getattr(self, field), getattr(other, field)
            if a.shape != b.shape:
                raise ValueError(f"incompatible {field} shapes: {a.shape} != {b.shape}")
            if a.dtype != b.dtype or a.device != b.device:
                raise ValueError(f"incompatible {field} dtype/device")


def _validate_primitive_shapes(
    behavior: Tensor, z: Tensor, pi: Tensor, kappa: Tensor, p: Tensor
) -> None:
    tensors = (behavior, z, pi, kappa, p)
    if any(t.ndim != 1 for t in tensors):
        raise ValueError("primitive tensors and p must all be one-dimensional")
    if behavior.shape != p.shape:
        raise ValueError("behavior and p must have identical shapes")
    if pi.shape != kappa.shape:
        raise ValueError("pi and kappa must have identical shapes")
    if any(torch.any(~torch.isfinite(t)) for t in tensors):
        raise ValueError("primitive tensors must be finite")
    _validate_p(p)


def _validate_sequence_inputs(
    behaviors: Tensor, z: Tensor, pi: Tensor, kappa: Tensor, p: Tensor
) -> None:
    if behaviors.ndim != 2 or z.ndim != 2 or pi.ndim != 2 or kappa.ndim != 2:
        raise ValueError("sequence inputs must be rank-2 tensors [length, dimension]")
    lengths = {behaviors.shape[0], z.shape[0], pi.shape[0], kappa.shape[0]}
    if len(lengths) != 1 or not lengths or behaviors.shape[0] < 1:
        raise ValueError("all sequence inputs must share a positive length")
    if behaviors.shape[1:] != p.shape:
        raise ValueError("p dimension must match behavior dimension")
    if pi.shape != kappa.shape:
        raise ValueError("pi and kappa must have identical shapes")
    if any(torch.any(~torch.isfinite(t)) for t in (behaviors, z, pi, kappa, p)):
        raise ValueError("sequence tensors must be finite")
    _validate_p(p)


def state_from_sequence(
    behaviors: Tensor,
    z: Tensor,
    pi: Tensor,
    kappa: Tensor,
    p: Tensor,
    *,
    eps: float = DEFAULT_EPS,
) -> AlgebraState:
    """Accumulate a chronological word into one associative state."""

    _validate_sequence_inputs(behaviors, z, pi, kappa, p)
    state = AlgebraState.identity(
        behaviors.shape[1], z.shape[1], pi.shape[1],
        dtype=behaviors.dtype, device=behaviors.device,
    )
    for i in range(behaviors.shape[0]):
        primitive = AlgebraState.primitive(behaviors[i], z[i], pi[i], kappa[i], p, eps=eps)
        state = state.then(primitive)
    return state


def compose_streaming(
    behaviors: Tensor,
    z: Tensor,
    pi: Tensor,
    kappa: Tensor,
    p: Tensor,
    *,
    eps: float = DEFAULT_EPS,
) -> CompositionOutput:
    return state_from_sequence(behaviors, z, pi, kappa, p, eps=eps).extract(p, eps=eps)


def compose_closed_form(
    behaviors: Tensor,
    z: Tensor,
    pi: Tensor,
    kappa: Tensor,
    p: Tensor,
    *,
    eps: float = DEFAULT_EPS,
) -> CompositionOutput:
    """Independent all-pairs formula used as an equivalence oracle."""

    _validate_sequence_inputs(behaviors, z, pi, kappa, p)
    m = behaviors.shape[0]
    base = signed_power_inverse(
        signed_power_transform(behaviors, p, eps).sum(dim=0), p, eps
    )
    if m < 2:
        return CompositionOutput(
            base, torch.zeros_like(z[0]), torch.zeros_like(pi[0]), m
        )

    symmetric_sum = torch.zeros_like(z[0])
    ordered_sum = torch.zeros_like(pi[0])
    for i in range(m):
        for j in range(i + 1, m):
            symmetric_sum = symmetric_sum + z[i] * z[j]
            ordered_sum = ordered_sum + 0.5 * pi[i] * pi[j] * (kappa[j] - kappa[i])
    pairs = m * (m - 1) / 2
    return CompositionOutput(base, symmetric_sum / pairs, ordered_sum / pairs, m)


def _heisenberg_matrix(u: Tensor, v: Tensor) -> Tensor:
    """Return one 3x3 block per coordinate with primitive central term 0.5uv."""

    d = u.numel()
    out = torch.eye(3, dtype=u.dtype, device=u.device).repeat(d, 1, 1)
    out[:, 0, 1] = u
    out[:, 1, 2] = v
    out[:, 0, 2] = 0.5 * u * v
    return out


def _translation_matrix(w: Tensor) -> Tensor:
    d = w.numel()
    out = torch.eye(3, dtype=w.dtype, device=w.device).repeat(d, 1, 1)
    out[:, 0, 2] = w
    return out


def compose_via_matrices(
    behaviors: Tensor,
    z: Tensor,
    pi: Tensor,
    kappa: Tensor,
    p: Tensor,
    *,
    eps: float = DEFAULT_EPS,
) -> CompositionOutput:
    """Literal block-matrix implementation for property/equivalence tests.

    This is intentionally less efficient than :func:`compose_streaming`.
    """

    _validate_sequence_inputs(behaviors, z, pi, kappa, p)
    m = behaviors.shape[0]
    phi = signed_power_transform(behaviors, p, eps)

    total_b = torch.eye(3, dtype=behaviors.dtype, device=behaviors.device).repeat(
        behaviors.shape[1], 1, 1
    )
    total_s = torch.eye(3, dtype=z.dtype, device=z.device).repeat(z.shape[1], 1, 1)
    total_o = torch.eye(3, dtype=pi.dtype, device=pi.device).repeat(pi.shape[1], 1, 1)

    for i in range(m):
        # X_i occurs later than the accumulated prefix, hence left multiply.
        total_b = torch.bmm(_translation_matrix(phi[i]), total_b)
        total_s = torch.bmm(_heisenberg_matrix(z[i], z[i]), total_s)
        total_o = torch.bmm(_heisenberg_matrix(kappa[i] * pi[i], pi[i]), total_o)

    base = signed_power_inverse(total_b[:, 0, 2], p, eps)
    if m < 2:
        symmetric = torch.zeros_like(z[0])
        ordered = torch.zeros_like(pi[0])
    else:
        pairs = m * (m - 1) / 2
        symmetric = (total_s[:, 0, 2] - 0.5 * z.square().sum(dim=0)) / pairs
        ordered = (
            total_o[:, 0, 2] - 0.5 * total_o[:, 0, 1] * total_o[:, 1, 2]
        ) / pairs
    return CompositionOutput(base, symmetric, ordered, m)


def concatenate_states(states: Sequence[AlgebraState]) -> AlgebraState:
    """Compose non-empty chronological chunks, useful for associativity tests."""

    if not states:
        raise ValueError("at least one state is required")
    result = states[0]
    for state in states[1:]:
        result = result.then(state)
    return result
