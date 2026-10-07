"""Shift-algebra modules: shift encoder, composition operators, behavior
decoder, and the training procedure shared by all methods.

Design (manuscript §IV): z_S = E(v_S) where v_S is a distribution-statistics
vector; a composition operator C predicts z_{A∘B} ≈ C(z_A, z_B); a behavior
decoder G predicts ∆B ≈ G(z_S). Shift Algebra's C decomposes into a symmetric
part S (commutative) and an antisymmetric part λR (order-sensitive, Eq. 30-33).

The training loop is vectorized over conditions (batched composition forwards,
precomputed index tensors) so thousands of full-batch steps are cheap on CPU.
"""
import numpy as np
import torch
import torch.nn as nn

from . import config as cfg


class ShiftEncoder(nn.Module):
    """v (ENCODER_IN_DIM) -> z (LATENT_DIM)."""

    def __init__(self, in_dim=cfg.ENCODER_IN_DIM, z_dim=cfg.LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, z_dim),
        )

    def forward(self, v):
        return self.net(v)


class BehaviorDecoder(nn.Module):
    """z (LATENT_DIM) -> ∆B̂ (BEHAVIOR_DIM)."""

    def __init__(self, z_dim=cfg.LATENT_DIM, out_dim=cfg.BEHAVIOR_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128), nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, z):
        return self.net(z)


class SymmetricComposition(nn.Module):
    """S(z_A, z_B) = S(z_B, z_A) by construction (symmetric inputs)."""

    def __init__(self, z_dim=cfg.LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * z_dim, 256), nn.ReLU(),
            nn.Linear(256, z_dim),
        )

    def forward(self, za, zb):
        return self.net(torch.cat([za + zb, za * zb], dim=-1))


class AntisymmetricComposition(nn.Module):
    """R(z_A, z_B) = -R(z_B, z_A), manuscript Eq. (33):
    R(z_A,z_B) = W (z_A ⊙ U z_B − z_B ⊙ U z_A)."""

    def __init__(self, z_dim=cfg.LATENT_DIM):
        super().__init__()
        self.U = nn.Parameter(torch.eye(z_dim) * 0.5)
        self.W = nn.Parameter(torch.eye(z_dim) * 0.5)

    def forward(self, za, zb):
        a = za * (zb @ self.U.t())
        b = zb * (za @ self.U.t())
        return (a - b) @ self.W.t()


class ShiftAlgebraComposition(nn.Module):
    """C(z_A,z_B) = S(z_A,z_B) + λ R(z_A,z_B), manuscript Eq. (30)."""

    def __init__(self, z_dim=cfg.LATENT_DIM, lam=cfg.LAM_ORDER):
        super().__init__()
        self.S = SymmetricComposition(z_dim)
        self.R = AntisymmetricComposition(z_dim)
        self.lam = lam

    def forward(self, za, zb):
        return self.S(za, zb) + self.lam * self.R(za, zb)


class AdditiveComposition(nn.Module):
    def forward(self, za, zb):
        return za + zb


class MLPComposition(nn.Module):
    def __init__(self, z_dim=cfg.LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * z_dim, 256), nn.ReLU(),
            nn.Linear(256, z_dim),
        )

    def forward(self, za, zb):
        return self.net(torch.cat([za, zb], dim=-1))


class AttentionComposition(nn.Module):
    """z_A cross-attends to the set {z_A, z_B} (single head)."""

    def __init__(self, z_dim=cfg.LATENT_DIM):
        super().__init__()
        self.q = nn.Linear(z_dim, z_dim)
        self.k = nn.Linear(z_dim, z_dim)
        self.v = nn.Linear(z_dim, z_dim)
        self.out = nn.Linear(z_dim, z_dim)
        self.scale = z_dim ** -0.5

    def forward(self, za, zb):
        kv = torch.stack([za, zb], dim=1)              # (N, 2, z)
        q = self.q(za).unsqueeze(1)                    # (N, 1, z)
        k = self.k(kv)                                 # (N, 2, z)
        v = self.v(kv)                                 # (N, 2, z)
        attn = torch.softmax(q @ k.transpose(1, 2) * self.scale, dim=-1)  # (N,1,2)
        return self.out((attn @ v).squeeze(1))         # (N, z)


def make_composition(method, z_dim=cfg.LATENT_DIM):
    if method in ("M5_shift_algebra", "M8_recon_only"):
        return ShiftAlgebraComposition(z_dim, lam=cfg.LAM_ORDER)
    if method == "M6_no_order":
        return ShiftAlgebraComposition(z_dim, lam=0.0)
    if method == "M1_additive":
        return AdditiveComposition()
    if method == "M2_mlp":
        return MLPComposition(z_dim)
    if method == "M3_attention":
        return AttentionComposition(z_dim)
    raise ValueError(f"no composition for method {method!r}")


def compose_recursive(comp, z_list):
    """Left fold: ((z_A ∘ z_B) ∘ z_C), manuscript Eq. (17)."""
    z = z_list[0]
    for zi in z_list[1:]:
        z = comp(z, zi)
    return z


def compose_mean(z_list):
    return torch.stack(z_list, dim=0).mean(dim=0)


def compose_sum(z_list):
    return torch.stack(z_list, dim=0).sum(dim=0)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))


def train_composition_model(method, conds, seed, verbose=False):
    """Train encoder + composition + decoder end-to-end on observed conditions.

    conds: list of dicts with keys:
      letters (tuple), sevs (tuple), kind ('prim'|'pair'),
      v (Tensor [ENCODER_IN_DIM]), y (Tensor [BEHAVIOR_DIM])
    For pairs, the two primitive conditions must also be present in conds
    (they are, by construction of the split).

    Returns dict of trained modules + in-sample predictions.
    """
    set_seed(seed)
    enc = ShiftEncoder()
    dec = BehaviorDecoder()

    V = torch.stack([c["v"] for c in conds])           # (N, in_dim)
    Y = torch.stack([c["y"] for c in conds])           # (N, 5)
    kinds = [c["kind"] for c in conds]

    # pair structure: (i_pair, i_x, i_y) with primitive indices
    pair_data = []
    for i, c in enumerate(conds):
        if c["kind"] == "pair":
            ix = next(j for j, p in enumerate(conds)
                      if p["kind"] == "prim" and p["letters"][0] == c["letters"][0]
                      and round(float(p["sevs"][0]), 6) == round(float(c["sevs"][0]), 6))
            iy = next(j for j, p in enumerate(conds)
                      if p["kind"] == "prim" and p["letters"][0] == c["letters"][1]
                      and round(float(p["sevs"][0]), 6) == round(float(c["sevs"][1]), 6))
            pair_data.append((i, ix, iy))

    use_comp = method in ("M5_shift_algebra", "M6_no_order", "M1_additive",
                          "M2_mlp", "M3_attention", "M8_recon_only")
    use_beh = method != "M8_recon_only"       # M8: reconstruction-only, then decoder
    use_repr = method in ("M5_shift_algebra", "M6_no_order", "M1_additive",
                          "M2_mlp", "M3_attention", "M8_recon_only")

    # ---- vectorized index tensors (precomputed once) ----
    prim_positions = [i for i, k in enumerate(kinds) if k == "prim"]
    ip_t = torch.tensor([ip for (ip, _, _) in pair_data], dtype=torch.long)
    ix_t = torch.tensor([ix for (_, ix, _) in pair_data], dtype=torch.long)
    iy_t = torch.tensor([iy for (_, _, iy) in pair_data], dtype=torch.long)
    prim_pos_t = torch.tensor(prim_positions, dtype=torch.long)
    # combined = [prim embs | pair embs]; order maps combined -> conds order
    perm = prim_positions + [ip for (ip, _, _) in pair_data]
    order = torch.tensor(np.argsort(perm), dtype=torch.long)

    comp = make_composition(method) if use_comp else None

    def condition_embeddings(Z):
        """(N, z) embeddings for every observed condition, in conds order."""
        if use_comp:
            pair_embs = comp(Z[ix_t], Z[iy_t])
        else:                                   # M0: mean-pooled primitives
            pair_embs = (Z[ix_t] + Z[iy_t]) / 2
        combined = torch.cat([Z[prim_pos_t], pair_embs], dim=0)
        return combined[order]

    if method == "M0_no_composition":
        params = list(enc.parameters()) + list(dec.parameters())
    else:
        params = list(enc.parameters()) + list(comp.parameters()) + list(dec.parameters())
    opt = torch.optim.Adam(params, lr=cfg.TRAIN_LR, weight_decay=1e-4)

    for step in range(cfg.TRAIN_STEPS):
        opt.zero_grad()
        Z_all = enc(V)
        loss = torch.tensor(0.0)
        if use_beh:
            embs = condition_embeddings(Z_all)
            loss = loss + cfg.LAMBDA_BEH * (dec(embs) - Y).pow(2).mean()
        if use_repr:
            zhat = comp(Z_all[ix_t], Z_all[iy_t])
            loss = loss + cfg.LAMBDA_REPR * (zhat - Z_all[ip_t]).pow(2).mean()
        loss.backward()
        opt.step()
        if verbose and (step + 1) % 1000 == 0:
            print(f"    step {step+1}/{cfg.TRAIN_STEPS} loss={loss.item():.6f}")

    # M8: encoder+composition were trained without behavior supervision;
    # freeze them and train the decoder on the observed embeddings.
    if method == "M8_recon_only":
        for p in enc.parameters():
            p.requires_grad_(False)
        for p in comp.parameters():
            p.requires_grad_(False)
        opt2 = torch.optim.Adam(dec.parameters(), lr=cfg.TRAIN_LR, weight_decay=1e-4)
        with torch.no_grad():
            embs = condition_embeddings(enc(V))
        for step in range(cfg.TRAIN_STEPS):
            opt2.zero_grad()
            loss = (dec(embs) - Y).pow(2).mean()
            loss.backward()
            opt2.step()

    with torch.no_grad():
        in_sample_pred = dec(condition_embeddings(enc(V)))

    out = dict(encoder=enc, decoder=dec, in_sample_pred=in_sample_pred,
               in_sample_y=Y, kinds=kinds)
    if use_comp:
        out["comp"] = comp
    return out


def predict_condition(trained, cond_letters, cond_sevs, prim_conds):
    """Zero-shot prediction for a held-out condition.

    prim_conds: dict (letter, sev) -> v (Tensor) for the primitive conditions
    (must cover every primitive in the condition at the matching severity).
    """
    enc = trained["encoder"]
    dec = trained["decoder"]
    method_comp = trained.get("comp", None)
    with torch.no_grad():
        zs = [enc(prim_conds[(l, round(float(s), 6))].unsqueeze(0))   # (1, z) each
              for l, s in zip(cond_letters, cond_sevs)]
        if method_comp is not None:
            z = compose_recursive(method_comp, zs)     # (1, z)
        elif len(zs) == 1:
            z = zs[0]                                  # (1, z)
        else:
            z = compose_mean(zs)                       # M0: mean-pooled primitives
        return dec(z).squeeze(0), z.squeeze(0)


def train_memorization_control(conds, seed):
    """MEMO control: a free embedding per observed condition + decoder.
    Held-out conditions get the mean of observed embeddings (must fail)."""
    set_seed(seed)
    z_dim = cfg.LATENT_DIM
    table = torch.nn.Parameter(torch.randn(len(conds), z_dim) * 0.1)
    dec = BehaviorDecoder()
    Y = torch.stack([c["y"] for c in conds])
    opt = torch.optim.Adam([table, *dec.parameters()], lr=cfg.TRAIN_LR, weight_decay=1e-4)
    for _ in range(cfg.TRAIN_STEPS):
        opt.zero_grad()
        loss = (dec(table) - Y).pow(2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        mean_emb = table.mean(dim=0, keepdim=True)
        in_sample_pred = dec(table)
        heldout_emb = mean_emb.expand(1, z_dim)
        heldout_pred = dec(heldout_emb).squeeze(0)
    return dict(table=table, decoder=dec, mean_emb=mean_emb,
                in_sample_pred=in_sample_pred, in_sample_y=Y,
                heldout_pred=heldout_pred, heldout_emb=heldout_emb.squeeze(0))


def train_permutation_control(conds, seed):
    """PERM control: Shift Algebra trained with composition targets permuted.
    Destroys compositional structure (must collapse on held-out)."""
    rng = np.random.RandomState(seed)
    pair_positions = [i for i, c in enumerate(conds) if c["kind"] == "pair"]
    perm = pair_positions.copy()
    rng.shuffle(perm)
    permuted = list(conds)
    for src, dst in zip(pair_positions, perm):
        permuted[dst] = dict(conds[dst])
        permuted[dst]["v"] = conds[src]["v"]
        permuted[dst]["y"] = conds[src]["y"]
    return train_composition_model("M5_shift_algebra", permuted, seed)
