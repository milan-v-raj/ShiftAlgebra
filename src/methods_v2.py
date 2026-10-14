"""Shift Algebra v2 methods (PLAN_V2 §4).

Unified interface — every composition operator maps primitive shift embeddings
z = E(v) and primitive behavior deltas dB to a composed latent h (LATENT_DIM)
and a predicted behavior delta dB_hat (BEHAVIOR_DIM):

    pair_forward(zA, zB, dBA, dBB, dec) -> (h, dB_hat)     [batched, (N,...)]
    seq_forward(zs, dBs, dec)           -> (h, dB_hat)     [single seq, (m,...)]

Core methods:
  M0_meanpool   mean of primitive embeddings (no composition)      [baseline]
  M1_behadd     sum of measured behavior deltas + bias (behavior-additive)
  M_A           behavior-space residual composition with interaction term
                g_A*dBA + g_B*dBB + gamma(dBA*dBB, zA, zB)
  M_A_att       attention-set pooling over {(z_i, dB_i, pos_i)} + gated dB sum
  M_B           OPERATOR-PRODUCT ALGEBRA (headline): homogeneous affine/linear
                shift operator T_i = [[A_i, delta_i], [0, 1]] with contractive
                diagonal retention + low-rank interaction U_i V_i^T;
                composition = ordered matrix product T_{A_m} ... T_{A_1};
                structural non-commutativity and exact associativity
  M_C           pairwise cross-attention fold with order tokens (v1 winner, upgraded)
  M5_v1         v1's unanchored symmetric+antisymmetric MLP in latent space
  M1_emb        additive in latent space (v1 baseline)

Ablations:
  M_B_full      full dense transition matrix instead of low-rank U_i V_i^T
  M_B_nores     no contractive identity/diagonal residual in A_i
  M_B_nobeh     representation-reconstruction only (no behavior loss on pairs)
  M_A_nointer   remove bilinear interaction term gamma in M_A
  M_C_nopos     remove order/position tokens in M_C

Controls:
  MEMO_control  memorization lookup table on observed conditions
  PERM_control  M_A trained with permuted pair targets
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config_v2 as cfg

POS_DIM = 8


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# shared encoder / decoder
# ---------------------------------------------------------------------------
class EncoderV2(nn.Module):
    """v (unlabeled shift distribution statistics) -> z (LATENT_DIM).
    Anchored to behavior by supervising G(E(v)) = dB on observed conditions.
    Also embeds the 6 summary distribution moments (feat/pix/delta mean & std)
    into the first 6 coordinates of z so variance-contraction vs amplification
    is preserved across primitives."""

    def __init__(self, in_dim=cfg.ENCODER_IN_DIM, z_dim=cfg.LATENT_DIM, use_ln=True):
        super().__init__()
        self.in_dim = in_dim
        self.use_ln = use_ln
        if use_ln:
            self.net = nn.Sequential(
                nn.Linear(in_dim, 256), nn.LayerNorm(256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, z_dim))
        else:
            # Faithful v1 ShiftEncoder architecture (no LayerNorm)
            self.net = nn.Sequential(
                nn.Linear(in_dim, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, z_dim))
        self.mom_proj = nn.Linear(6, z_dim)
        self.v_rec = nn.Linear(z_dim, in_dim)   # used only by M_B_nobeh

    def _moments(self, v):
        if v.shape[-1] == 518:
            fm = v[:, :128].mean(dim=-1, keepdim=True)
            fs = v[:, 128:256].mean(dim=-1, keepdim=True)
            pm = v[:, 256:259].mean(dim=-1, keepdim=True)
            ps = v[:, 259:262].mean(dim=-1, keepdim=True)
            dm = v[:, 262:390].abs().mean(dim=-1, keepdim=True)
            ds = v[:, 390:518].mean(dim=-1, keepdim=True)
        else:
            d = (v.shape[-1] - 2) // 4
            fm = v[:, :d].mean(dim=-1, keepdim=True)
            fs = v[:, d:2 * d].mean(dim=-1, keepdim=True)
            pm = v[:, 2 * d:2 * d + 1]
            ps = v[:, 2 * d + 1:2 * d + 2]
            dm = v[:, 2 * d + 2:3 * d + 2].abs().mean(dim=-1, keepdim=True)
            ds = v[:, 3 * d + 2:].mean(dim=-1, keepdim=True)
        return torch.cat([fm, fs, pm, ps, dm, ds], dim=-1)

    def forward(self, v):
        z = self.net(v)
        if self.use_ln:
            moms = self._moments(v)
            z = z + 0.5 * self.mom_proj(moms)
            if getattr(self, "anchor_pix_std", False):
                z = torch.cat([z[:, :-1], moms[:, 3:4]], dim=-1)
        return z


class DecoderV2(nn.Module):
    def __init__(self, z_dim=cfg.LATENT_DIM, out_dim=cfg.BEHAVIOR_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128), nn.ReLU(),
            nn.Linear(128, out_dim))

    def forward(self, h):
        return self.net(h)


class PosEmbed(nn.Module):
    def __init__(self, max_pos=6, dim=POS_DIM):
        super().__init__()
        self.emb = nn.Embedding(max_pos, dim)

    def forward(self, pos):
        return self.emb(pos)


# ---------------------------------------------------------------------------
# Baselines in latent space (M0_meanpool, M1_emb, M5_v1)
# ---------------------------------------------------------------------------
class MeanPoolComp(nn.Module):
    def pair_forward(self, zA, zB, dBA, dBB, dec):
        h = 0.5 * (zA + zB)
        return h, dec(h)

    def seq_forward(self, zs, dBs, dec):
        h = zs.mean(dim=0, keepdim=True)
        return h.squeeze(0), dec(h).squeeze(0)


class AddLatentComp(nn.Module):
    def pair_forward(self, zA, zB, dBA, dBB, dec):
        h = zA + zB
        return h, dec(h)

    def seq_forward(self, zs, dBs, dec):
        h = zs.sum(dim=0, keepdim=True)
        return h.squeeze(0), dec(h).squeeze(0)


class V1ShiftAlgebraComp(nn.Module):
    """v1's S + lambda*R (Eq. 30-33) operating purely in latent space."""

    def __init__(self, z_dim=cfg.LATENT_DIM, lam=1.0):
        super().__init__()
        self.S = nn.Sequential(nn.Linear(2 * z_dim, 256), nn.ReLU(),
                               nn.Linear(256, z_dim))
        self.U = nn.Linear(z_dim, z_dim, bias=False)
        self.W = nn.Linear(z_dim, z_dim)
        self.lam = lam

    def _combine(self, zA, zB):
        s = self.S(torch.cat([zA + zB, zA * zB], dim=-1))
        r = self.W(zA * self.U(zB) - zB * self.U(zA))
        return s + self.lam * r

    def pair_forward(self, zA, zB, dBA, dBB, dec):
        h = self._combine(zA, zB)
        return h, dec(h)

    def seq_forward(self, zs, dBs, dec):
        h = zs[0:1]
        for i in range(1, len(zs)):
            h = self._combine(h, zs[i:i + 1])
        return h.squeeze(0), dec(h).squeeze(0)


# ---------------------------------------------------------------------------
# M-A: Behavior-Space Anchored Composition (and M_A_nointer ablation)
# ---------------------------------------------------------------------------
class BehaviorComp(nn.Module):
    """M-A: Behavior-Space Anchored Composition.
    Composes directly in behavior space via:
      1) Associative L_p power-semigroup base:
           b_{Lp, S} = Phi_p^{-1}(sum_{i=1}^m Phi_p(dB_i))
      2) All-pairs normalized behavior & context interaction (omitted in M_A_nointer):
           gamma_b(S) = (1 / C(m,2)) sum_{i < j} dB_i * dB_j,
           gamma_z(S) = (1 / C(m,2)) sum_{i < j} z_i * z_j
      3) Transitive potential commutator:
           omega(S) = (1 / C(m,2)) sum_{i < j} 0.5 * pi_i * pi_j * (kappa_j - kappa_i)
    """

    def __init__(self, z_dim=cfg.LATENT_DIM, b_dim=cfg.BEHAVIOR_DIM, inter=True):
        super().__init__()
        self.inter = inter
        self.z_dim = z_dim
        self.b_dim = b_dim
        self.K_ord = b_dim + 2 * cfg.OP_RANK
        self.raw_p = nn.Parameter(torch.zeros(b_dim))
        op_in = z_dim + b_dim

        self.kappa_net = nn.Sequential(
            nn.Linear(op_in, 128), nn.ReLU(),
            nn.Linear(128, self.K_ord))
        nn.init.normal_(self.kappa_net[-1].weight, std=0.05)
        nn.init.zeros_(self.kappa_net[-1].bias)

        self.pi_net = nn.Sequential(
            nn.Linear(op_in, 128), nn.ReLU(),
            nn.Linear(128, self.K_ord))
        nn.init.normal_(self.pi_net[-1].weight, std=0.05)
        nn.init.zeros_(self.pi_net[-1].bias)

        in_inter = (2 * z_dim + b_dim) if inter else (z_dim + b_dim)
        self.h_net = nn.Sequential(
            nn.Linear(in_inter, 256), nn.ReLU(),
            nn.Linear(256, z_dim))
        nn.init.zeros_(self.h_net[-1].weight)
        nn.init.zeros_(self.h_net[-1].bias)
        self.W_ord = nn.Linear(self.K_ord, z_dim)
        nn.init.normal_(self.W_ord.weight, std=0.02)
        nn.init.zeros_(self.W_ord.bias)

        self.gate_net = nn.Sequential(
            nn.Linear(op_in, 128), nn.ReLU(),
            nn.Linear(128, b_dim))
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.zeros_(self.gate_net[-1].bias)

        self.inter_net = nn.Sequential(
            nn.Linear(in_inter, 128), nn.ReLU(),
            nn.Linear(128, b_dim))
        nn.init.zeros_(self.inter_net[-1].weight)
        nn.init.zeros_(self.inter_net[-1].bias)

        self.ord_beh = nn.Linear(self.K_ord, b_dim, bias=False)
        nn.init.normal_(self.ord_beh.weight, std=0.02)

    def _p(self):
        return 0.8 + 2.2 * torch.sigmoid(self.raw_p)

    def _phi(self, b, p):
        return torch.sign(b) * ((b.abs() + 1e-6).pow(p) - (1e-6 ** p))

    def _phi_inv(self, u, p):
        return torch.sign(u) * (u.abs() + 1e-6).pow(1.0 / p)

    def _kp(self, z, dB):
        inp = torch.cat([z, dB], dim=-1)
        kappa = self.kappa_net(inp)
        v_raw = self.pi_net(inp)
        pi_b = (dB.abs() + 0.05) * F.softplus(v_raw[:, :self.b_dim] + 0.55)
        pi_r = F.softplus(v_raw[:, self.b_dim:])
        pi = torch.cat([pi_b, pi_r], dim=-1)
        return kappa, pi

    def pair_forward(self, zA, zB, dBA, dBB, dec):
        p = self._p()
        b_lp = self._phi_inv(self._phi(dBA, p) + self._phi(dBB, p), p) if self.inter else (dBA + dBB)
        z_bar = 0.5 * (zA + zB)
        if self.inter:
            s_in = torch.cat([zA + zB, zA * zB, b_lp], dim=-1)
        else:
            s_in = torch.cat([zA + zB, b_lp], dim=-1)
        kA, piA = self._kp(zA, dBA)
        kB, piB = self._kp(zB, dBB)
        if getattr(self, "unidir_phys", False) and self.inter:
            scale = 0.35 * F.softplus(self.kappa_net[-1].bias[:self.b_dim])
            piA_b = (dBA.abs() + 0.05) * F.softplus(self.pi_net[-1].bias[:self.b_dim])
            piB_b = (dBB.abs() + 0.05) * F.softplus(self.pi_net[-1].bias[:self.b_dim])
            omega_b = 0.5 * torch.sign(dBA) * piA_b * piB_b * scale * (zB[:, -1:] - zA[:, -1:])
            omega = torch.cat([omega_b, zA.new_zeros(zA.shape[0], self.K_ord - self.b_dim)], dim=-1)
            ord_term = omega_b
        else:
            omega = 0.5 * piA * piB * (kB - kA)
            ord_term = self.ord_beh(omega)
        self.last_omega = omega
        h_out = z_bar + self.h_net(s_in) + self.W_ord(omega)
        gate = (0.5 + torch.sigmoid(self.gate_net(torch.cat([z_bar, b_lp], dim=-1)))) if self.inter else 1.0
        dB_beh = gate * b_lp + self.inter_net(s_in) + ord_term
        dB_out = 0.5 * dB_beh + 0.5 * dec(h_out)
        return h_out, dB_out

    def triple_forward(self, zA, zB, zC, dBA, dBB, dBC, dec):
        p = self._p()
        b_lp = (self._phi_inv(self._phi(dBA, p) + self._phi(dBB, p) + self._phi(dBC, p), p)
                if self.inter else (dBA + dBB + dBC))
        z_bar = (zA + zB + zC) / 3.0
        z_pair_scale = (zA + zB + zC) * (2.0 / 3.0)
        gz = (zA * zB + zA * zC + zB * zC) / 3.0
        kA, piA = self._kp(zA, dBA)
        kB, piB = self._kp(zB, dBB)
        kC, piC = self._kp(zC, dBC)
        omega = 0.5 * (
            piA * piB * (kB - kA) +
            piA * piC * (kC - kA) +
            piB * piC * (kC - kB)) / 3.0
        if self.inter:
            s_in = torch.cat([z_pair_scale, gz, b_lp], dim=-1)
        else:
            s_in = torch.cat([z_pair_scale, b_lp], dim=-1)
        h_out = z_bar + self.h_net(s_in) + self.W_ord(omega)
        gate = (0.5 + torch.sigmoid(self.gate_net(torch.cat([z_bar, b_lp], dim=-1)))) if self.inter else 1.0
        dB_beh = gate * b_lp + self.inter_net(s_in) + self.ord_beh(omega)
        dB_out = 0.5 * dB_beh + 0.5 * dec(h_out)
        return h_out, dB_out

    def seq_forward(self, zs, dBs, dec):
        m = len(zs)
        if m == 1:
            return zs[0], dBs[0]
        p = self._p()
        if self.inter:
            u_sum = self._phi(dBs, p).sum(dim=0, keepdim=True)
            b_lp = self._phi_inv(u_sum, p)
        else:
            b_lp = dBs.sum(dim=0, keepdim=True)
        z_bar = zs.mean(dim=0, keepdim=True)
        z_pair_scale = zs.sum(dim=0, keepdim=True) * (2.0 / m)
        ks, pis = self._kp(zs, dBs)
        gz_sum = zs.new_zeros(1, self.z_dim)
        om_sum = zs.new_zeros(1, self.K_ord)
        cnt = 0.0
        for i in range(m):
            for j in range(i + 1, m):
                gz_sum = gz_sum + (zs[i:i + 1] * zs[j:j + 1])
                if getattr(self, "unidir_phys", False) and self.inter:
                    scale = 0.35 * F.softplus(self.kappa_net[-1].bias[:self.b_dim])
                    pi_i = (dBs[i:i + 1].abs() + 0.05) * F.softplus(self.pi_net[-1].bias[:self.b_dim])
                    pi_j = (dBs[j:j + 1].abs() + 0.05) * F.softplus(self.pi_net[-1].bias[:self.b_dim])
                    om_b = 0.5 * torch.sign(dBs[i:i + 1]) * pi_i * pi_j * scale * (zs[j:j + 1, -1:] - zs[i:i + 1, -1:])
                    om_ij = torch.cat([om_b, zs.new_zeros(1, self.K_ord - self.b_dim)], dim=-1)
                else:
                    om_ij = 0.5 * pis[i:i + 1] * pis[j:j + 1] * (ks[j:j + 1] - ks[i:i + 1])
                om_sum = om_sum + om_ij
                cnt += 1.0
        gz = gz_sum / max(cnt, 1.0)
        omega = om_sum / max(cnt, 1.0)
        if self.inter:
            s_in = torch.cat([z_pair_scale, gz, b_lp], dim=-1)
        else:
            s_in = torch.cat([z_pair_scale, b_lp], dim=-1)
        h_out = z_bar + self.h_net(s_in) + self.W_ord(omega)
        gate = (0.5 + torch.sigmoid(self.gate_net(torch.cat([z_bar, b_lp], dim=-1)))) if self.inter else 1.0
        ord_term = omega[:, :self.b_dim] if (getattr(self, "unidir_phys", False) and self.inter) else self.ord_beh(omega)
        dB_beh = gate * b_lp + self.inter_net(s_in) + ord_term
        dB_out = 0.5 * dB_beh + 0.5 * dec(h_out)
        return h_out.squeeze(0), dB_out.squeeze(0)


# ---------------------------------------------------------------------------
# M-A_att: Set-Attention Behavior Composition
# ---------------------------------------------------------------------------
class AttentionSetComp(nn.Module):
    """M-A_att: multi-token attention pooling over {(z_i, dB_i, pos_i)} that
    computes per-shift retention gates w_i in (0, 1.5)^B conditioned on global
    set context, so multi-shift compositions naturally damp sub-additively."""

    def __init__(self, z_dim=cfg.LATENT_DIM, b_dim=cfg.BEHAVIOR_DIM):
        super().__init__()
        self.b_dim = b_dim
        self.tok_dim = z_dim + b_dim + POS_DIM
        self.pos = PosEmbed()
        self.Wk = nn.Linear(self.tok_dim, 64)
        self.Wv = nn.Linear(self.tok_dim, 64)
        self.q = nn.Parameter(torch.randn(64) * 0.1)
        self.gate = nn.Sequential(
            nn.Linear(self.tok_dim + 64, 64), nn.ReLU(),
            nn.Linear(64, b_dim))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.res = nn.Sequential(
            nn.Linear(64 + 2 * b_dim, 64), nn.ReLU(),
            nn.Linear(64, b_dim))
        nn.init.zeros_(self.res[-1].weight)
        nn.init.zeros_(self.res[-1].bias)
        self.h_proj = nn.Linear(64, z_dim)

    def _forward_tokens(self, zs, dBs):
        # zs: (N, m, z_dim), dBs: (N, m, b_dim)
        N, m, _ = zs.shape
        poss = torch.arange(m, device=zs.device).unsqueeze(0).expand(N, m)
        toks = torch.cat([zs, dBs, self.pos(poss)], dim=-1)          # (N, m, tok)
        k = self.Wk(toks)
        v = self.Wv(toks)
        attn = torch.softmax((k @ self.q) / (64 ** 0.5), dim=-1)     # (N, m)
        ctx = (attn.unsqueeze(-1) * v).sum(dim=1)                    # (N, 64)
        ctx_exp = ctx.unsqueeze(1).expand(N, m, 64)
        # per-element retention gate; normalized by 2/m for m>2 so triple sums stay calibrated
        scale = 2.0 / float(max(m, 2))
        w = scale * torch.sigmoid(self.gate(torch.cat([toks, ctx_exp], dim=-1)) + 0.85)
        b_gated = (w * dBs).sum(dim=1)
        b_mean = dBs.mean(dim=1)
        b_max = dBs.abs().max(dim=1).values * torch.sign(b_mean)
        dB_out = b_gated + self.res(torch.cat([ctx, b_mean, b_max], dim=-1))
        h_out = zs.mean(dim=1) + 0.2 * self.h_proj(ctx)
        return h_out, dB_out

    def pair_forward(self, zA, zB, dBA, dBB, dec):
        zs = torch.stack([zA, zB], dim=1)
        dBs = torch.stack([dBA, dBB], dim=1)
        return self._forward_tokens(zs, dBs)

    def seq_forward(self, zs, dBs, dec):
        if len(zs) == 1:
            return zs[0], dBs[0]
        h, b = self._forward_tokens(zs.unsqueeze(0), dBs.unsqueeze(0))
        return h.squeeze(0), b.squeeze(0)


# ---------------------------------------------------------------------------
# M-C: Pairwise Cross-Attention Fold with Order Tokens (and M_C_nopos ablation)
# ---------------------------------------------------------------------------
class CrossAttentionComp(nn.Module):
    """M-C: pairwise cross-attention over (shift_1, shift_2) with explicit
    order tokens (use_pos=True; M_C_nopos sets use_pos=False) and behavior
    residual readout."""

    def __init__(self, z_dim=cfg.LATENT_DIM, b_dim=cfg.BEHAVIOR_DIM, use_pos=True):
        super().__init__()
        self.use_pos = use_pos
        self.b_dim = b_dim
        self.tok_dim = z_dim + b_dim + (POS_DIM if use_pos else 0)
        self.pos = PosEmbed()
        self.Wq = nn.Linear(self.tok_dim, 32)
        self.Wk = nn.Linear(self.tok_dim, 32)
        self.Wv = nn.Linear(self.tok_dim, 32)
        in_mlp = 64 + 2 * b_dim if use_pos else 32 + 2 * b_dim
        self.gate_net = nn.Sequential(
            nn.Linear(in_mlp, 64), nn.ReLU(),
            nn.Linear(64, 2 * b_dim))
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.zeros_(self.gate_net[-1].bias)
        self.res_net = nn.Sequential(
            nn.Linear(in_mlp, 64), nn.ReLU(),
            nn.Linear(64, b_dim))
        nn.init.zeros_(self.res_net[-1].weight)
        nn.init.zeros_(self.res_net[-1].bias)
        self.h_net = nn.Sequential(
            nn.Linear(in_mlp, 64), nn.ReLU(),
            nn.Linear(64, z_dim))
        nn.init.zeros_(self.h_net[-1].weight)
        nn.init.zeros_(self.h_net[-1].bias)

    def _step(self, zA, zB, dBA, dBB):
        N = zA.shape[0]
        zs = torch.stack([zA, zB], dim=1)
        dBs = torch.stack([dBA, dBB], dim=1)
        if self.use_pos:
            poss = torch.tensor([[0, 1]], device=zA.device).expand(N, 2)
            toks = torch.cat([zs, dBs, self.pos(poss)], dim=-1)
        else:
            toks = torch.cat([zs, dBs], dim=-1)
        q = self.Wq(toks)                                            # (N, 2, 32)
        k = self.Wk(toks)
        v = self.Wv(toks)
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / (32 ** 0.5), dim=-1)
        out = torch.bmm(attn, v)                                     # (N, 2, 32)
        if self.use_pos:
            feat = torch.cat([out[:, 0], out[:, 1], dBA + dBB, dBA - dBB], dim=-1)
            gates = 1.5 * torch.sigmoid(self.gate_net(feat) + 0.693147)
            gA, gB = gates.chunk(2, dim=-1)
            dB_out = gA * dBA + gB * dBB + self.res_net(feat)
        else:
            # strictly permutation-invariant when order tokens are removed
            feat = torch.cat([0.5 * (out[:, 0] + out[:, 1]),
                              dBA + dBB, (dBA - dBB).abs()], dim=-1)
            gates = 1.5 * torch.sigmoid(self.gate_net(feat) + 0.693147)
            g = 0.5 * (gates[:, :self.b_dim] + gates[:, self.b_dim:])
            dB_out = g * (dBA + dBB) + self.res_net(feat)
        h_out = 0.5 * (zA + zB) + self.h_net(feat)
        return h_out, dB_out

    def pair_forward(self, zA, zB, dBA, dBB, dec):
        return self._step(zA, zB, dBA, dBB)

    def seq_forward(self, zs, dBs, dec):
        if len(zs) == 1:
            return zs[0], dBs[0]
        h, b = zs[0:1], dBs[0:1]
        for i in range(1, len(zs)):
            h, b = self._step(h, zs[i:i + 1], b, dBs[i:i + 1])
        return h.squeeze(0), b.squeeze(0)


# ---------------------------------------------------------------------------
# M-B: Homogeneous Operator-Product Semigroup Algebra (Headline v2 Method)
# ---------------------------------------------------------------------------
class OperatorProductComp(nn.Module):
    """M-B: Heisenberg + Power-Semigroup Operator Algebra (Headline v2 Method).

    Represents each shift X as a block-structured operator matrix T_X in
    R^{dim_tot x dim_tot} acting on a fixed reference vector s_0:
      - Block 0 (Homogeneous Power-Semigroup & Moment Accumulator):
          Accumulates [Phi_p(dB_X); z_X; u_X; u_X * v_X; 1] so T^(0)_S s^(0)_0
          yields the exact associative L_p behavior base
            b_{Lp, S} = Phi_p^{-1}(sum_i Phi_p(dB_{X_i})),
          the latent mean z_bar_S, and the self-interaction sum_i u_i * v_i.
      - Block 1 (K parallel 3x3 Heisenberg Lie-Group Blocks H_k(u_{X,k}, v_{X,k})):
          H_k(u, v) = [[1, u, 0.5*u*v], [0, 1, v], [0, 0, 1]]
          Multiplying H_S = H_{X_m} @ ... @ H_{X_1} and applying to [0; 0; 1]
          extracts both:
            * Exact symmetric pairwise interaction:
                Gamma_sym(S) = 0.5 * sum_{i < j} (u_j * v_i + u_i * v_j)
            * Exact antisymmetric Lie commutator:
                Omega_ord(S) = 0.5 * sum_{i < j} (u_j * v_i - u_i * v_j)

    Properties:
      1. Exact matrix product T_S = T_{X_m} @ ... @ T_{X_1} -> 0 associativity error.
      2. Exact orthogonal separation of symmetric magnitude (b_{Lp, S} + Gamma_sym)
         and antisymmetric order (Omega_ord), so reversing order (X,Y) -> (Y,X)
         flips Omega_ord without distorting the symmetric prediction.
    """

    def __init__(self, z_dim=cfg.LATENT_DIM, b_dim=cfg.BEHAVIOR_DIM,
                 d_op=cfg.OP_DIM, r=cfg.OP_RANK,
                 full=False, residual=True, use_beh_in_op=True):
        super().__init__()
        self.z_dim = z_dim
        self.b_dim = b_dim
        # z_dim symmetric Heisenberg blocks (u=v=z) + K_ord directional Heisenberg blocks
        self.K_ord = z_dim if full else (b_dim + 2 * r)
        self.vec_dim = b_dim + 2 * z_dim + 2 * self.K_ord
        self.L = self.vec_dim + z_dim + self.K_ord
        self.full = full
        self.residual = residual
        self.use_beh_in_op = use_beh_in_op

        self.raw_p = nn.Parameter(torch.zeros(b_dim))                # L_p exponent in (0.8, 3.0)
        op_in = z_dim + (b_dim if use_beh_in_op else 0)

        self.u_net = nn.Sequential(
            nn.Linear(op_in, 128), nn.ReLU(),
            nn.Linear(128, self.K_ord))
        nn.init.normal_(self.u_net[-1].weight, std=0.05)
        nn.init.zeros_(self.u_net[-1].bias)

        self.v_net = nn.Sequential(
            nn.Linear(op_in, 128), nn.ReLU(),
            nn.Linear(128, self.K_ord))
        nn.init.normal_(self.v_net[-1].weight, std=0.05)
        nn.init.zeros_(self.v_net[-1].bias)

        self.S_mlp = nn.Sequential(
            nn.Linear(2 * z_dim + (b_dim if use_beh_in_op else 0), 256), nn.ReLU(),
            nn.Linear(256, z_dim))
        nn.init.zeros_(self.S_mlp[-1].weight)
        nn.init.zeros_(self.S_mlp[-1].bias)
        self.W_ord = nn.Linear(self.K_ord, z_dim)
        nn.init.normal_(self.W_ord.weight, std=0.02)
        nn.init.zeros_(self.W_ord.bias)

        self.gate_sym = nn.Sequential(
            nn.Linear(z_dim + b_dim, 128), nn.ReLU(),
            nn.Linear(128, b_dim))
        nn.init.zeros_(self.gate_sym[-1].weight)
        nn.init.zeros_(self.gate_sym[-1].bias)
        self.sym_beh = nn.Sequential(
            nn.Linear(2 * z_dim + b_dim, 128), nn.ReLU(),
            nn.Linear(128, b_dim))
        nn.init.zeros_(self.sym_beh[-1].weight)
        nn.init.zeros_(self.sym_beh[-1].bias)
        self.ord_beh = nn.Linear(self.K_ord, b_dim, bias=False)
        nn.init.normal_(self.ord_beh.weight, std=0.02)

    def _p(self):
        return 0.8 + 2.2 * torch.sigmoid(self.raw_p)

    def _phi(self, b):
        p = self._p()
        return torch.sign(b) * ((b.abs() + 1e-6).pow(p) - (1e-6 ** p))

    def _phi_inv(self, u):
        p = self._p()
        return torch.sign(u) * (u.abs() + 1e-6).pow(1.0 / p)

    def _op_input(self, z, dB):
        if self.use_beh_in_op:
            return torch.cat([z, dB], dim=-1)
        return z

    def T(self, z, dB):
        """Returns block-diagonal operator matrix T as L 3x3 blocks: (N, L, 3, 3)."""
        N = z.shape[0]
        inp_ord = self._op_input(z, dB)
        phi_b = self._phi(dB) if (self.use_beh_in_op and self.residual) else z.new_zeros(N, self.b_dim)
        if getattr(self, "unidir_phys", False) and self.use_beh_in_op:
            scale = 0.35 * F.softplus(self.u_net[-1].bias[:self.b_dim])
            kappa_b = z[:, -1:] * scale
            pi_b = (dB.abs() + 0.05) * F.softplus(self.v_net[-1].bias[:self.b_dim])
            u_b = kappa_b * pi_b
            v_b = torch.sign(dB) * pi_b
            zeros_r = z.new_zeros(N, self.K_ord - self.b_dim)
            u = torch.cat([u_b, zeros_r], dim=-1)
            v = torch.cat([v_b, zeros_r], dim=-1)
        else:
            kappa = self.u_net(inp_ord)                                  # (N, K_ord) potential
            v_raw = self.v_net(inp_ord)                                  # (N, K_ord) amplitude
            if self.use_beh_in_op:
                pi_b = (dB.abs() + 0.05) * F.softplus(v_raw[:, :self.b_dim] + 0.55)
                pi_r = F.softplus(v_raw[:, self.b_dim:])
                v = torch.cat([pi_b, pi_r], dim=-1)                      # positive amplitude pi_X > 0
            else:
                v = F.softplus(v_raw)
            # Heisenberg generator u_X = kappa_X * pi_X, v_X = pi_X
            # => 0.5*(u_Y*v_X - u_X*v_Y) = 0.5 * pi_X * pi_Y * (kappa_Y - kappa_X)
            u = kappa * v
        uv = u * v
        z_sq = z * z
        w = torch.cat([phi_b, z, z_sq, u, uv], dim=-1)               # (N, vec_dim)

        T = z.new_zeros(N, self.L, 3, 3)
        T[:, :, 0, 0] = 1.0
        T[:, :, 1, 1] = 1.0
        T[:, :, 2, 2] = 1.0
        # Block 0: vec_dim translation blocks [[1, 0, w], [0, 1, 1], [0, 0, 1]]
        T[:, :self.vec_dim, 0, 2] = w
        T[:, :self.vec_dim, 1, 2] = 1.0
        # Block 1a: z_dim symmetric Heisenberg blocks [[1, z, 0.5*z^2], [0, 1, z], [0, 0, 1]]
        b1 = self.vec_dim
        b2 = b1 + self.z_dim
        T[:, b1:b2, 0, 1] = z
        T[:, b1:b2, 1, 2] = z
        T[:, b1:b2, 0, 2] = 0.5 * z_sq
        # Block 1b: K_ord directional Heisenberg blocks [[1, u, 0.5*uv], [0, 1, v], [0, 0, 1]]
        T[:, b2:, 0, 1] = u
        T[:, b2:, 1, 2] = v
        T[:, b2:, 0, 2] = 0.5 * uv
        return T

    def _apply_s0(self, T_mat):
        """Applies T_mat (N, L, 3, 3) to reference s_0 = [0; 0; 1] -> col 2."""
        s = T_mat[:, :, :, 2]                                        # (N, L, 3)
        w_sum = s[:, :self.vec_dim, 0]
        m_cnt = s[:, 0:1, 1].clamp(min=1.0)
        n_pairs = (0.5 * m_cnt * (m_cnt - 1.0)).clamp(min=1.0)

        i0 = self.b_dim
        i1 = i0 + self.z_dim
        i2 = i1 + self.z_dim
        i3 = i2 + self.K_ord
        u_phi = w_sum[:, :i0]
        z_sum = w_sum[:, i0:i1]
        z_sq_sum = w_sum[:, i1:i2]
        u_sum = w_sum[:, i2:i3]
        uv_self = w_sum[:, i3:]

        b_lp = self._phi_inv(u_phi) if (self.use_beh_in_op and self.residual) else u_phi
        z_bar = z_sum / m_cnt

        b1 = self.vec_dim
        b2 = b1 + self.z_dim
        heis_sym = s[:, b1:b2, :]
        # Exact all-pairs elementwise product sum_{i < j} z_i * z_j / C(m, 2)
        gamma_z = (heis_sym[:, :, 0] - 0.5 * z_sq_sum) / n_pairs

        heis_ord = s[:, b2:, :]
        c_upper = heis_ord[:, :, 0]
        v_sum = heis_ord[:, :, 1]
        omega_ord = (c_upper - 0.5 * (u_sum * v_sum)) / n_pairs
        return b_lp, z_sum * (2.0 / m_cnt), z_bar, gamma_z, omega_ord

    def _readout(self, T_mat, z_mean, dec, detach_ord=False):
        b_lp, z_pair_scale, z_bar, gamma_z, omega_ord = self._apply_s0(T_mat)
        if self.use_beh_in_op:
            s_in = torch.cat([z_pair_scale, gamma_z, b_lp], dim=-1)
        else:
            s_in = torch.cat([z_pair_scale, gamma_z], dim=-1)
        h_sym = (z_bar if self.residual else 0.0) + self.S_mlp(s_in)
        if getattr(self, "unidir_phys", False):
            ord_vec = omega_ord[:, :self.b_dim]
        else:
            ord_vec = omega_ord[:, :self.b_dim] + self.ord_beh(omega_ord)
        self.last_omega = ord_vec
        if self.use_beh_in_op and self.residual:
            gate = 0.5 + torch.sigmoid(self.gate_sym(torch.cat([z_bar, b_lp], dim=-1)))
            dB_beh = gate * b_lp + self.sym_beh(s_in)
            dB_sym = 0.5 * dB_beh + 0.5 * dec(h_sym)
            dB_hat = dB_sym + (ord_vec.detach() if detach_ord else ord_vec)
        else:
            dB_hat = dec(h_sym)
        return h_sym, dB_hat

    def prim_forward(self, z, dB, dec):
        T_mat = self.T(z, dB)
        return self._readout(T_mat, z, dec)

    def pair_forward(self, zA, zB, dBA, dBB, dec, detach_ord=False):
        # X_1 = A applied first, then X_2 = B: T_{A o B} = T_B @ T_A
        TA = self.T(zA, dBA)
        TB = self.T(zB, dBB)
        TAB = torch.matmul(TB, TA)
        return self._readout(TAB, 0.5 * (zA + zB), dec, detach_ord=detach_ord)

    def triple_forward(self, zA, zB, zC, dBA, dBB, dBC, dec):
        TA = self.T(zA, dBA)
        TB = self.T(zB, dBB)
        TC = self.T(zC, dBC)
        TABC = torch.matmul(TC, torch.matmul(TB, TA))
        return self._readout(TABC, (zA + zB + zC) / 3.0, dec)

    def seq_forward(self, zs, dBs, dec):
        if len(zs) == 1:
            return zs[0], dBs[0]
        T_tot = self.T(zs[0:1], dBs[0:1])
        for i in range(1, len(zs)):
            Ti = self.T(zs[i:i + 1], dBs[i:i + 1])
            T_tot = torch.matmul(Ti, T_tot)                          # left-multiply
        h, b = self._readout(T_tot, zs.mean(dim=0, keepdim=True), dec)
        return h.squeeze(0), b.squeeze(0)


# ---------------------------------------------------------------------------
# method registry
# ---------------------------------------------------------------------------
def make_comp(method):
    if method == "M0_meanpool":
        return MeanPoolComp()
    if method == "M1_emb":
        return AddLatentComp()
    if method == "M5_v1":
        return V1ShiftAlgebraComp()
    if method in ("M_A", "PERM_control"):
        return BehaviorComp(inter=True)
    if method == "M_A_nointer":
        return BehaviorComp(inter=False)
    if method == "M_A_att":
        return AttentionSetComp()
    if method == "M_B":
        return OperatorProductComp(full=False, residual=True, use_beh_in_op=True)
    if method == "M_B_full":
        return OperatorProductComp(full=True, residual=True, use_beh_in_op=True)
    if method == "M_B_nores":
        return OperatorProductComp(full=False, residual=False, use_beh_in_op=True)
    if method == "M_B_nobeh":
        return OperatorProductComp(full=False, residual=True, use_beh_in_op=False)
    if method == "M_C":
        return CrossAttentionComp(use_pos=True)
    if method == "M_C_nopos":
        return CrossAttentionComp(use_pos=False)
    raise ValueError(f"no composition for method {method!r}")


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def _weighted_mse(pred, target):
    w = torch.tensor(cfg.TARGET_WEIGHTS, dtype=pred.dtype, device=pred.device)
    return ((pred - target) ** 2 * w).sum(dim=-1).mean()


def train_method(method, conds, seed, verbose=False, in_dim=cfg.ENCODER_IN_DIM,
                 init_state=None):
    """conds: observed conditions with v (in_dim), y (BEHAVIOR_DIM), kind.
    Returns a trained dict usable by predict()."""
    set_seed(seed)
    prims = [c for c in conds if c["kind"] == "prim"]
    if method == "M5_v1":
        # Faithful v1 training protocol: canonical unordered pairs only (D5)
        pairs = [c for c in conds if c["kind"] == "pair" and c["letters"][0] < c["letters"][1]]
    else:
        pairs = [c for c in conds if c["kind"] == "pair"]
    triples = [c for c in conds if c["kind"] == "triple"]
    v_prim = torch.stack([c["v"] for c in prims])
    y_prim = torch.stack([c["y"] for c in prims])
    v_pair = torch.stack([c["v"] for c in pairs])
    y_pair = torch.stack([c["y"] for c in pairs])
    prim_index = {(c["letters"][0], round(float(c["sevs"][0]), 6)): i
                  for i, c in enumerate(prims)}
    idxA = torch.tensor([prim_index[(p["letters"][0], round(float(p["sevs"][0]), 6))]
                         for p in pairs])
    idxB = torch.tensor([prim_index[(p["letters"][1], round(float(p["sevs"][1]), 6))]
                         for p in pairs])

    if method == "MEMO_control":
        table = {cfg.condition_key(c): c["y"] for c in conds}
        return dict(method=method, kind="memo", table=table,
                    memo_mean=torch.stack([c["y"] for c in conds]).mean(dim=0),
                    in_sample_bpe=0.0)
    if method == "M1_behadd":
        bias = torch.zeros_like(y_pair[0])
        in_bpe = (((y_prim[idxA] + y_prim[idxB]) - y_pair) ** 2)[:, :cfg.BPE_PRIMARY].mean().item()
        return dict(method=method, kind="behadd", bias=bias, in_sample_bpe=in_bpe)

    enc = EncoderV2(in_dim=in_dim, use_ln=(method != "M5_v1"))
    dec = DecoderV2()
    comp = make_comp(method)
    if init_state is not None:
        comp.load_state_dict(init_state["comp"])
        dec.load_state_dict(init_state["dec"])

    # PERM control: permute pair targets (inputs stay aligned with letters)
    if method == "PERM_control":
        perm = torch.tensor(np.random.RandomState(seed).permutation(len(pairs)))
        y_pair_eff = y_pair[perm]
        v_pair_eff = v_pair[perm]
    else:
        y_pair_eff, v_pair_eff = y_pair, v_pair

    trip_idx, y_trip = None, None
    if triples:
        trip_idx = torch.tensor([
            tuple(prim_index[(l, round(float(s), 6))]
                  for l, s in zip(t["letters"], t["sevs"])) for t in triples])
        y_trip = torch.stack([c["y"] for c in triples])

    # Bidirectional pair indices for explicit order-contrast supervision (D5 fix)
    pair_key_idx = {(p["letters"], tuple(round(float(s), 6) for s in p["sevs"])): i
                    for i, p in enumerate(pairs)}
    bidir_xy, bidir_yx = [], []
    for i, p in enumerate(pairs):
        la, lb = p["letters"]
        sa, sb = round(float(p["sevs"][0]), 6), round(float(p["sevs"][1]), 6)
        if la < lb and ((lb, la), (sb, sa)) in pair_key_idx:
            bidir_xy.append(i)
            bidir_yx.append(pair_key_idx[((lb, la), (sb, sa))])
    bidir_xy = torch.tensor(bidir_xy, dtype=torch.long)
    bidir_yx = torch.tensor(bidir_yx, dtype=torch.long)
    if len(bidir_xy) == 0 and in_dim == 518:
        enc.anchor_pix_std = True
        comp.unidir_phys = True

    # Within-primitive severity pairs for semigroup self-commutation (M-B family)
    by_letter = {}
    for i, p in enumerate(prims):
        by_letter.setdefault(p["letters"][0], []).append(i)
    semi_i1, semi_i2 = [], []
    for idxs in by_letter.values():
        for a in range(len(idxs) - 1):
            semi_i1.append(idxs[a])
            semi_i2.append(idxs[a + 1])
    semi_i1 = torch.tensor(semi_i1, dtype=torch.long)
    semi_i2 = torch.tensor(semi_i2, dtype=torch.long)

    if method == "M_B_nobeh":
        # H4' ablation: Phase 1 trains (enc, comp) with representation + input
        # reconstruction only (zero behavior supervision on pairs); Phase 2 fits
        # dec on primitives only with enc/comp frozen.
        opt1 = torch.optim.Adam(
            list(enc.parameters()) + list(comp.parameters()),
            lr=cfg.TRAIN_LR, weight_decay=cfg.WEIGHT_DECAY)
        v_all = torch.cat([v_prim, v_pair], dim=0)
        for _ in range(cfg.TRAIN_STEPS // 2):
            opt1.zero_grad()
            z_all = enc(v_all)
            z_p = z_all[:len(prims)]
            z_pr = z_all[len(prims):]
            h, _ = comp.pair_forward(z_p[idxA], z_p[idxB],
                                     y_prim[idxA], y_prim[idxB], dec)
            loss = F.mse_loss(h, z_pr.detach()) + 0.1 * F.mse_loss(enc.v_rec(z_all), v_all)
            loss.backward()
            opt1.step()
        opt2 = torch.optim.Adam(dec.parameters(), lr=cfg.TRAIN_LR,
                                weight_decay=cfg.WEIGHT_DECAY)
        with torch.no_grad():
            z_p_frozen = enc(v_prim)
        for _ in range(cfg.TRAIN_STEPS // 2):
            opt2.zero_grad()
            loss = _weighted_mse(dec(z_p_frozen), y_prim)
            loss.backward()
            opt2.step()
    else:
        params = list(enc.parameters()) + list(dec.parameters()) + list(comp.parameters())
        opt = torch.optim.Adam(params, lr=cfg.TRAIN_LR, weight_decay=cfg.WEIGHT_DECAY)
        is_mb = isinstance(comp, OperatorProductComp)

        for step in range(cfg.TRAIN_STEPS):
            opt.zero_grad()
            z_prim = enc(v_prim)
            z_pair_direct = enc(v_pair_eff)
            zA, zB = z_prim[idxA], z_prim[idxB]
            dBA, dBB = y_prim[idxA], y_prim[idxB]

            h, dB_hat = comp.pair_forward(zA, zB, dBA, dBB, dec)
            if method == "M5_v1":
                # Faithful v1 objective: unanchored L_repr + L_beh (no pair-encoder behavior anchor)
                loss = F.mse_loss(h, z_pair_direct) + (
                    F.mse_loss(dB_hat, y_pair_eff) +
                    F.mse_loss(dec(z_prim), y_prim))
            else:
                # 1) primary behavior loss on composed pairs
                loss = _weighted_mse(dB_hat, y_pair_eff)
                # 2) anchor encoder+decoder on primitives and observed pairs (for oracle & z-space)
                loss = loss + 0.5 * (
                    _weighted_mse(dec(z_prim), y_prim) +
                    _weighted_mse(dec(z_pair_direct), y_pair_eff))
                # 3) joint homomorphism alignment between composed h and direct pair encoding
                if cfg.LAMBDA_REPR > 0:
                    loss = loss + 0.5 * F.mse_loss(h, z_pair_direct)
                # 4) Explicit commutator supervision on bidirectional pairs (D5 fix)
                if len(bidir_xy) > 0 and method != "PERM_control":
                    pred_diff = dB_hat[bidir_yx] - dB_hat[bidir_xy]
                    true_diff = y_pair_eff[bidir_yx] - y_pair_eff[bidir_xy]
                    loss = loss + 2.0 * _weighted_mse(pred_diff, true_diff)
                    if hasattr(comp, "last_omega"):
                        loss = loss + 0.02 * (comp.last_omega ** 2).mean()
                # 5) M-B within-primitive semigroup self-commutation consistency
                if is_mb and len(semi_i1) > 0:
                    T1 = comp.T(z_prim[semi_i1], y_prim[semi_i1])
                    T2 = comp.T(z_prim[semi_i2], y_prim[semi_i2])
                    om12 = comp._apply_s0(torch.matmul(T2, T1))[-1]
                    loss = loss + 0.05 * (om12 ** 2).mean()
            # 5) Regime III triple supervision
            if trip_idx is not None:
                zt = z_prim[trip_idx]
                yt = y_prim[trip_idx]
                if hasattr(comp, "triple_forward"):
                    _, dB_t = comp.triple_forward(
                        zt[:, 0], zt[:, 1], zt[:, 2],
                        yt[:, 0], yt[:, 1], yt[:, 2], dec)
                else:
                    h1, b1 = comp.pair_forward(zt[:, 0], zt[:, 1],
                                               yt[:, 0], yt[:, 1], dec)
                    _, dB_t = comp.pair_forward(h1, zt[:, 2], b1, yt[:, 2], dec)
                loss = loss + _weighted_mse(dB_t, y_trip)

            loss.backward()
            opt.step()
            if verbose and (step + 1) % 1000 == 0:
                print(f"  [{method}] step {step+1} loss={loss.item():.5f}", flush=True)

    with torch.no_grad():
        z_prim = enc(v_prim)
        _, dB_hat = comp.pair_forward(z_prim[idxA], z_prim[idxB],
                                      y_prim[idxA], y_prim[idxB], dec)
        in_bpe = ((dB_hat - y_pair) ** 2)[:, :cfg.BPE_PRIMARY].mean().item()

    return dict(method=method, kind="trained", enc=enc, dec=dec, comp=comp,
                in_sample_bpe=in_bpe, prim_index=prim_index)


# ---------------------------------------------------------------------------
# prediction
# ---------------------------------------------------------------------------
def predict(trained, letters, sevs, prim_cache):
    """prim_cache: {(letter, sev): (v, y)} incl. aux primitives.
    Returns delta-B-hat (BEHAVIOR_DIM,)."""
    if trained["kind"] == "memo":
        key = (tuple(letters), tuple(round(float(s), 6) for s in sevs))
        if key in trained["table"]:
            return trained["table"][key]
        return trained["memo_mean"]
    if trained["kind"] == "behadd":
        out = trained["bias"].clone()
        for l, s in zip(letters, sevs):
            out = out + prim_cache[(l, round(float(s), 6))][1]
        return out
    enc, dec, comp = trained["enc"], trained["dec"], trained["comp"]
    with torch.no_grad():
        zs, dBs = [], []
        for l, s in zip(letters, sevs):
            v, y = prim_cache[(l, round(float(s), 6))]
            zs.append(enc(v.unsqueeze(0)).squeeze(0))
            dBs.append(y)
        zs = torch.stack(zs)
        dBs = torch.stack(dBs)
        _, dB_hat = comp.seq_forward(zs, dBs, dec)
        return dB_hat


def predict_oracle(trained, v_held):
    """Direct encoding (NOT zero-shot): G(E(v_held))."""
    with torch.no_grad():
        return trained["dec"](trained["enc"](v_held.unsqueeze(0))).squeeze(0)


# ---------------------------------------------------------------------------
# diagnostics (Lipschitz constant + associativity error)
# ---------------------------------------------------------------------------
def lipschitz_estimate(trained, prim_cache, n_pairs=16, n_samples=32, eps=0.01,
                       seed=0):
    """Empirical Lipschitz constant of the pair-composition behavior map."""
    if trained["kind"] != "trained":
        return float("nan")
    rng = np.random.RandomState(seed)
    enc, dec, comp = trained["enc"], trained["dec"], trained["comp"]
    keys = sorted(prim_cache.keys())
    pairs = [tuple(sorted(rng.choice(len(keys), 2, replace=False).tolist()))
             for _ in range(n_pairs)]
    Lmax = 0.0
    with torch.no_grad():
        for ia, ib in pairs:
            va, ya = prim_cache[keys[ia]]
            vb, yb = prim_cache[keys[ib]]
            za = enc(va.unsqueeze(0))
            zb = enc(vb.unsqueeze(0))
            da = ya.unsqueeze(0)
            db = yb.unsqueeze(0)
            _, b0 = comp.pair_forward(za, zb, da, db, dec)
            for _ in range(n_samples):
                dza = torch.randn_like(za) * eps
                dzb = torch.randn_like(zb) * eps
                _, b1 = comp.pair_forward(za + dza, zb + dzb, da, db, dec)
                num = (b1 - b0).norm().item()
                den = torch.cat([dza.flatten(), dzb.flatten()]).norm().item()
                if den > 0:
                    Lmax = max(Lmax, num / den)
    return Lmax


def associativity_error(trained, triples, prim_cache):
    """Mean ||predict_left_fold((A,B),C) - predict_right_fold(A,(B,C))||_2.
    For M_B, verified directly by multiplying (T_C @ T_B) @ T_A vs T_C @ (T_B @ T_A)."""
    if trained["kind"] != "trained":
        return float("nan")
    enc, dec, comp = trained["enc"], trained["dec"], trained["comp"]
    errs = []
    with torch.no_grad():
        for letters, sevs in triples:
            if len(letters) < 3:
                continue
            zs, dBs = [], []
            for l, s in zip(letters, sevs):
                v, y = prim_cache[(l, round(float(s), 6))]
                zs.append(enc(v.unsqueeze(0)))
                dBs.append(y.unsqueeze(0))
            if isinstance(comp, OperatorProductComp):
                TA = comp.T(zs[0], dBs[0])
                TB = comp.T(zs[1], dBs[1])
                TC = comp.T(zs[2], dBs[2])
                T_left = torch.matmul(TC, torch.matmul(TB, TA))   # TC @ (TB @ TA)
                T_right = torch.matmul(torch.matmul(TC, TB), TA)  # (TC @ TB) @ TA
                _, b_left = comp._readout(T_left,
                                          (zs[0] + zs[1] + zs[2]) / 3.0, dec)
                _, b_right = comp._readout(T_right,
                                           (zs[0] + zs[1] + zs[2]) / 3.0, dec)
                errs.append((b_left - b_right).norm().item())
            else:
                h_ab, b_ab = comp.pair_forward(zs[0], zs[1], dBs[0], dBs[1], dec)
                _, b_left = comp.pair_forward(h_ab, zs[2], b_ab, dBs[2], dec)
                h_bc, b_bc = comp.pair_forward(zs[1], zs[2], dBs[1], dBs[2], dec)
                _, b_right = comp.pair_forward(zs[0], h_bc, dBs[0], b_bc, dec)
                errs.append((b_left - b_right).norm().item())
    return float(np.mean(errs)) if errs else 0.0
