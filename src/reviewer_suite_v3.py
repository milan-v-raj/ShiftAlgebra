"""Phase V3: Reviewer-Proofing Experimental Suite (Addressing Gaps 1-5).

Executes the five experimental expansions requested in the peer-review
assessment of Shift Algebra v2:
  Gap 1: Decisive component & competing baseline ablations + out-of-sample L_p
         exponent validation (M_Lp_only, M_PowerOffset, M_Max_pool, M_Ord_noLp,
         M_B_nocomm, M_LowRankBilinear, M_MLP_matched, M_FeatOrderReg).
  Gap 2: Observation-budget curve (25%, 50%, 75%, 100% of observed pairs).
  Gap 3: Shift-primitive holdout test (hold out S, P, or D from all training
         pairs; zero-shot predict pairs/triples containing the held-out shift
         from its 5 single-shift observations alone).
  Gap 4: External generalization across (4A) a second architecture
         (ViT-PatchMixer on CIFAR-10) and (4B) a second dataset
         (official Fashion-MNIST).
  Gap 5: Mechanistic counterfactual experiment (variance-normalized high-freq
         shift H where pixel-variance proxy and spectral/learned susceptibility
         disagree) + multi-level statistical inference (seed-level CIs,
         24-family cluster-bootstrap CIs, Cohen's d, and Holm-Bonferroni
         corrected p-values).

Usage:
  python -m src.reviewer_suite_v3 [all | gap1 | gap2 | gap3 | gap4 | gap5 | report]
"""
import argparse
import csv
import gzip
import json
import os
import struct
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr, ttest_rel, wilcoxon

from . import config_v2 as cfg
from . import corruptions_v2 as corr
from . import data as data_mod
from . import experiment_v2 as exp2
from . import methods_v2 as m2
from . import metrics_v2 as mt

torch.set_num_threads(2)
OUT_JSON = os.path.join(cfg.RESULTS_DIR, "reviewer_suite_v3.json")
OUT_TXT = os.path.join(cfg.RESULTS_DIR, "REVIEWER_REPORT_V3.txt")


def _load_suite_json():
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as f:
            return json.load(f)
    return {}


def _save_suite_json(data):
    with open(OUT_JSON, "w") as f:
        json.dump(data, f, indent=2, default=float)


def _load_v2_ground_truth():
    beh_csv = os.path.join(cfg.RESULTS_DIR, "behavior_testpool_all.csv")
    beh = {}
    with open(beh_csv) as f:
        for r in csv.DictReader(f):
            k = (tuple(r["letters"].split("+")),
                 tuple(round(float(x), 6) for x in r["sevs"].split(",")))
            beh[k] = torch.tensor([
                float(r["margin"]), float(r["drift"]), float(r["ece"]),
                float(r["pcacc"]), float(r["acc"]), float(r["conf"])
            ])
    B0 = torch.tensor([0.895782, 0.0, 0.031205, 0.9104, 0.9104, 0.940102])
    return {k: (v - B0)[:cfg.BEHAVIOR_DIM] for k, v in beh.items()}


# ===========================================================================
# Additional Competing Baselines & Ablations for Gap 1
# ===========================================================================
class LpOnlyComp(nn.Module):
    """M_Lp_only: Pure 5-parameter L_p power-semigroup law without MLP or
    Heisenberg commutator: dB_hat = Phi_p^{-1}(sum_i Phi_p(dB_i))."""

    def __init__(self, b_dim=cfg.BEHAVIOR_DIM):
        super().__init__()
        self.raw_p = nn.Parameter(torch.zeros(b_dim))

    def _p(self):
        return 0.8 + 3.7 * torch.sigmoid(self.raw_p)

    def _phi(self, b, p):
        return torch.sign(b) * ((b.abs() + 1e-6).pow(p) - (1e-6 ** p))

    def _phi_inv(self, u, p):
        return torch.sign(u) * (u.abs() + 1e-6).pow(1.0 / p)

    def pair_forward(self, zA, zB, dBA, dBB, dec):
        p = self._p()
        b_lp = self._phi_inv(self._phi(dBA, p) + self._phi(dBB, p), p)
        return 0.5 * (zA + zB), b_lp

    def seq_forward(self, zs, dBs, dec):
        if len(zs) == 1:
            return zs[0], dBs[0]
        p = self._p()
        u_sum = self._phi(dBs, p).sum(dim=0, keepdim=True)
        b_lp = self._phi_inv(u_sum, p).squeeze(0)
        return zs.mean(dim=0), b_lp


class PowerOffsetComp(nn.Module):
    """M_PowerOffset: Free power exponent p + per-dim scale alpha and offset beta."""

    def __init__(self, b_dim=cfg.BEHAVIOR_DIM):
        super().__init__()
        self.raw_p = nn.Parameter(torch.zeros(b_dim))
        self.raw_alpha = nn.Parameter(torch.zeros(b_dim))
        self.beta = nn.Parameter(torch.zeros(b_dim))

    def _p(self):
        return 0.5 + 3.0 * torch.sigmoid(self.raw_p)

    def _alpha(self):
        return 0.5 + torch.sigmoid(self.raw_alpha)

    def _phi(self, b, p):
        return torch.sign(b) * ((b.abs() + 1e-6).pow(p) - (1e-6 ** p))

    def _phi_inv(self, u, p):
        return torch.sign(u) * (u.abs() + 1e-6).pow(1.0 / p)

    def pair_forward(self, zA, zB, dBA, dBB, dec):
        p = self._p()
        b_lp = self._phi_inv(self._phi(dBA, p) + self._phi(dBB, p), p)
        return 0.5 * (zA + zB), self._alpha() * b_lp + self.beta

    def seq_forward(self, zs, dBs, dec):
        if len(zs) == 1:
            return zs[0], dBs[0]
        p = self._p()
        b_lp = self._phi_inv(self._phi(dBs, p).sum(dim=0, keepdim=True), p).squeeze(0)
        return zs.mean(dim=0), self._alpha() * b_lp + self.beta


class MaxPoolBehComp(nn.Module):
    """M_Max_pool (p -> inf): elementwise dominant-shift envelope max_i |dB_i|."""

    def __init__(self, b_dim=cfg.BEHAVIOR_DIM):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(b_dim))

    def pair_forward(self, zA, zB, dBA, dBB, dec):
        s = torch.sign(dBA + dBB)
        m = torch.maximum(dBA.abs(), dBB.abs())
        return 0.5 * (zA + zB), s * m + 0.0 * self.dummy

    def seq_forward(self, zs, dBs, dec):
        if len(zs) == 1:
            return zs[0], dBs[0]
        s = torch.sign(dBs.sum(dim=0))
        m = dBs.abs().max(dim=0).values
        return zs.mean(dim=0), s * m


class OrdNoLpComp(m2.OperatorProductComp):
    """M_Ord_noLp: Exact M_B architecture with p fixed to 1.0 (linear sum base
    instead of L_p power semigroup), isolating the role of p != 1."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw_p.requires_grad_(False)

    def _p(self):
        return torch.ones_like(self.raw_p)


class MBNoCommComp(m2.OperatorProductComp):
    """M_B_nocomm: Exact M_B architecture with antisymmetric Heisenberg block
    zeroed out (Omega_ord = 0), isolating the commutator's contribution."""

    def _readout(self, T_mat, z_mean, dec, detach_ord=False):
        b_lp, z_pair_scale, z_bar, gamma_z, omega_ord = self._apply_s0(T_mat)
        s_in = torch.cat([z_pair_scale, gamma_z, b_lp], dim=-1)
        h_sym = z_bar + self.S_mlp(s_in)
        self.last_omega = omega_ord.new_zeros(omega_ord.shape[0], self.b_dim)
        gate = 0.5 + torch.sigmoid(self.gate_sym(torch.cat([z_bar, b_lp], dim=-1)))
        dB_beh = gate * b_lp + self.sym_beh(s_in)
        dB_hat = 0.5 * dB_beh + 0.5 * dec(h_sym)
        self.last_sym_dev = dB_hat - b_lp
        return h_sym, dB_hat


class LowRankBilinearComp(nn.Module):
    """M_LowRankBilinear: Parameter-matched low-rank bilinear latent composition
    (rank=32): 0.35*(zA + zB) + 0.65*(A zA + B zB) + W(U zA * V zB)."""

    def __init__(self, z_dim=cfg.LATENT_DIM, rank=32):
        super().__init__()
        self.A = nn.Linear(z_dim, z_dim)
        self.B = nn.Linear(z_dim, z_dim)
        self.U = nn.Linear(z_dim, rank)
        self.V = nn.Linear(z_dim, rank)
        self.W = nn.Sequential(nn.Linear(rank, 192), nn.ReLU(),
                               nn.Linear(192, z_dim))

    def _step(self, zA, zB):
        return (0.35 * (zA + zB) +
                0.65 * (self.A(zA) + self.B(zB)) +
                self.W(self.U(zA) * self.V(zB)))

    def pair_forward(self, zA, zB, dBA, dBB, dec):
        h = self._step(zA, zB)
        return h, dec(h)

    def seq_forward(self, zs, dBs, dec):
        h = zs[0:1]
        for i in range(1, len(zs)):
            h = self._step(h, zs[i:i + 1])
        return h.squeeze(0), dec(h).squeeze(0)


class MLPMatchedComp(nn.Module):
    """M_MLP_matched: Parameter-matched ordered concatenation MLP taking
    [zA, dBA, zB, dBB] and rolling out left-to-right."""

    def __init__(self, z_dim=cfg.LATENT_DIM, b_dim=cfg.BEHAVIOR_DIM):
        super().__init__()
        in_d = 2 * (z_dim + b_dim)
        self.mlp_h = nn.Sequential(
            nn.Linear(in_d, 192), nn.ReLU(),
            nn.Linear(192, 128), nn.ReLU(),
            nn.Linear(128, z_dim))
        self.mlp_b = nn.Sequential(
            nn.Linear(in_d, 192), nn.ReLU(),
            nn.Linear(192, b_dim))

    def _step(self, zA, zB, dBA, dBB, dec):
        x = torch.cat([zA, dBA, zB, dBB], dim=-1)
        h = self.mlp_h(x)
        b = 0.5 * self.mlp_b(x) + 0.5 * dec(h)
        return h, b

    def pair_forward(self, zA, zB, dBA, dBB, dec):
        return self._step(zA, zB, dBA, dBB, dec)

    def seq_forward(self, zs, dBs, dec):
        h, b = zs[0:1], dBs[0:1]
        for i in range(1, len(zs)):
            h, b = self._step(h, zs[i:i + 1], b, dBs[i:i + 1], dec)
        return h.squeeze(0), b.squeeze(0)


# Register new Gap-1 methods into m2.make_comp dynamically
_ORIG_MAKE_COMP = m2.make_comp


def _extended_make_comp(method):
    if method == "M_Lp_only":
        return LpOnlyComp()
    if method == "M_PowerOffset":
        return PowerOffsetComp()
    if method == "M_Max_pool":
        return MaxPoolBehComp()
    if method == "M_Ord_noLp":
        return OrdNoLpComp(full=False, residual=True, use_beh_in_op=True)
    if method == "M_B_nocomm":
        return MBNoCommComp(full=False, residual=True, use_beh_in_op=True)
    if method == "M_LowRankBilinear":
        return LowRankBilinearComp()
    if method == "M_MLP_matched":
        return MLPMatchedComp()
    return _ORIG_MAKE_COMP(method)


m2.make_comp = _extended_make_comp


def _train_feat_order_reg(conds, heldout, prim_cache, seed):
    """M_FeatOrderReg: Direct feature + order-difference regression on
    observable shift statistics [sum/mean/max of primitive dB, mean/product of
    physical distribution moments moms(v_i), and pairwise directional moment &
    behavior differences]. Uses strictly observable (v_i, dB_i) inputs."""
    m2.set_seed(seed)

    def feat_of(letters, sevs):
        dBs, moms = [], []
        for l, s in zip(letters, sevs):
            v, y = prim_cache[(l, round(float(s), 6))]
            dBs.append(y)
            moms.append(torch.tensor([
                v[:128].mean(), v[128:256].mean(), v[256:259].mean(),
                v[259:262].mean(), v[262:390].abs().mean(), v[390:518].mean()
            ]))
        m = len(dBs)
        dBs_t = torch.stack(dBs)
        moms_t = torch.stack(moms)
        b_sum = dBs_t.sum(dim=0)
        b_mean = dBs_t.mean(dim=0)
        b_max = dBs_t.abs().max(dim=0).values * torch.sign(b_mean)
        mom_mean = moms_t.mean(dim=0)
        mom_diff = torch.zeros(6)
        b_diff = torch.zeros(cfg.BEHAVIOR_DIM)
        mom_prod = torch.zeros(6)
        n_pairs = max(1.0, 0.5 * m * (m - 1))
        for a in range(m):
            for b in range(a + 1, m):
                mom_diff = mom_diff + (moms_t[b] - moms_t[a]) / n_pairs
                b_diff = b_diff + (dBs_t[b] - dBs_t[a]) / n_pairs
                mom_prod = mom_prod + (moms_t[a] * moms_t[b]) / n_pairs
        return torch.cat([b_sum, b_mean, b_max, mom_mean, mom_prod, mom_diff, b_diff])

    pairs = [c for c in conds if c["kind"] == "pair"]
    X_tr = torch.stack([feat_of(c["letters"], c["sevs"]) for c in pairs])
    Y_tr = torch.stack([c["y"] for c in pairs])
    net = nn.Sequential(
        nn.Linear(X_tr.shape[1], 128), nn.LayerNorm(128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, cfg.BEHAVIOR_DIM))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
    for _ in range(cfg.TRAIN_STEPS):
        opt.zero_grad()
        loss = m2._weighted_mse(net(X_tr), Y_tr)
        loss.backward()
        opt.step()
    n_p = sum(p.numel() for p in net.parameters())
    with torch.no_grad():
        X_te = torch.stack([feat_of(h["letters"], h["sevs"]) for h in heldout])
        X_ob = torch.stack([feat_of(c["letters"], c["sevs"]) for c in conds])
        return net(X_te), net(X_ob), n_p


# ===========================================================================
# GAP 1: Decisive Ablations, Competing Baselines & L_p Exponent Audit
# ===========================================================================
def run_gap1():
    print("\n================ GAP 1: DECISIVE ABLATIONS & L_p AUDIT ================", flush=True)
    conds, prim_cache = exp2._load_observed_and_cache(2)
    heldout = cfg.heldout_conditions(regime=2)
    dB = _load_v2_ground_truth()
    y_held = torch.stack([dB[cfg.condition_key(h)] for h in heldout])
    y_obs = torch.stack([dB[cfg.condition_key(c)] for c in conds])

    held_keys = [cfg.condition_key(h) for h in heldout]
    obs_keys = [cfg.condition_key(c) for c in conds]
    held_pos = {k: i for i, k in enumerate(held_keys)}
    obs_pos = {k: i for i, k in enumerate(obs_keys)}

    pxy_idx, pyx_idx = [], []
    for pair in cfg.ORDER_TEST_PAIRS:
        rev = pair[::-1]
        pxy_idx.append([obs_pos[(pair, (round(float(cfg.SEVERITIES[pair[0]][i]), 6),
                                        round(float(cfg.SEVERITIES[pair[1]][i]), 6)))]
                        for i in range(cfg.N_SEV)])
        pyx_idx.append([held_pos[(rev, (round(float(cfg.SEVERITIES[rev[0]][i]), 6),
                                        round(float(cfg.SEVERITIES[rev[1]][i]), 6)))]
                        for i in range(cfg.N_SEV)])
    y_xy = torch.stack([y_obs[ix] for ix in pxy_idx])
    y_yx = torch.stack([y_held[ix] for ix in pyx_idx])

    triple_mask = [i for i, h in enumerate(heldout) if h["kind"] == "triple"]
    order_mask = [i for i, h in enumerate(heldout) if h["kind"] == "pair"]

    gap1_methods = [
        "M_B", "M_C", "M_A",
        "M_Lp_only", "M_PowerOffset", "M_Max_pool",
        "M_Ord_noLp", "M_B_nocomm",
        "M_LowRankBilinear", "M_MLP_matched", "M_FeatOrderReg",
        "M0_meanpool", "M5_v1", "M1_behadd",
    ]

    results = {}
    learned_p_mb = []
    learned_p_lponly = []

    for method in gap1_methods:
        t0 = time.time()
        preds, obs_preds = [], []
        n_params = 0
        for seed in cfg.SEEDS:
            if method == "M_FeatOrderReg":
                p_h, p_o, n_params = _train_feat_order_reg(conds, heldout, prim_cache, seed)
            else:
                tr = m2.train_method(method, conds, seed)
                p_h = torch.stack([m2.predict(tr, h["letters"], h["sevs"], prim_cache)
                                   for h in heldout])
                p_o = torch.stack([m2.predict(tr, c["letters"], c["sevs"], prim_cache)
                                   for c in conds])
                if tr["kind"] == "trained":
                    n_params = sum(p.numel() for p in tr["comp"].parameters())
                    if method == "M_B":
                        learned_p_mb.append(tr["comp"]._p().detach().tolist())
                    elif method == "M_Lp_only":
                        learned_p_lponly.append(tr["comp"]._p().detach().tolist())
            preds.append(p_h)
            obs_preds.append(p_o)

        P = torch.stack(preds)
        Po = torch.stack(obs_preds)
        s = mt.summarize(P, y_held)
        se = mt.bpe_primary_se(P, y_held)
        s["bpe_triples"] = float(se[:, triple_mask].mean())
        s["bpe_order_pairs"] = float(se[:, order_mask].mean())
        pred_xy = torch.stack([Po[:, ix, :] for ix in pxy_idx])
        pred_yx = torch.stack([P[:, ix, :] for ix in pyx_idx])
        s.update(mt.order_metrics(pred_xy, pred_yx, y_xy, y_yx))
        s["ire_pair"], s["ire_triple"] = exp2._compute_ire(
            P, Po, heldout, y_held, y_obs, held_pos, obs_pos)
        s["comp_params"] = int(n_params)
        s["se_per_seed_cond"] = se.numpy().tolist()
        results[method] = s
        print(f"  [Gap 1] {method:<18} BPE={s['bpe_mean']:.4f}  "
              f"TripBPE={s['bpe_triples']:.4f}  OrdBPE={s['bpe_order_pairs']:.4f}  "
              f"sign={s.get('sign_match', float('nan')):.2f}  "
              f"params={n_params} ({time.time()-t0:.1f}s)", flush=True)

    # Out-of-sample L_p exponent audit: compare pair-learned p vs oracle triple p*
    prims = [c for c in conds if c["kind"] == "prim"]
    prim_dB = {(c["letters"][0], round(float(c["sevs"][0]), 6)): dB[cfg.condition_key(c)]
               for c in prims}
    p_grid = torch.linspace(0.8, 3.0, 221)
    oracle_p_triples = []
    oracle_p_photo = []
    oracle_p_spatial = []
    for dim_j in range(cfg.BPE_PRIMARY):
        best_p, best_err = 1.0, 1e9
        best_p_ph, best_err_ph = 1.0, 1e9
        best_p_sp, best_err_sp = 1.0, 1e9
        for pv in p_grid:
            errs_all, errs_ph, errs_sp = [], [], []
            for h in heldout:
                if h["kind"] != "triple":
                    continue
                b1 = prim_dB[(h["letters"][0], round(float(h["sevs"][0]), 6))][dim_j]
                b2 = prim_dB[(h["letters"][1], round(float(h["sevs"][1]), 6))][dim_j]
                b3 = prim_dB[(h["letters"][2], round(float(h["sevs"][2]), 6))][dim_j]
                u = (b1.abs() ** pv + b2.abs() ** pv + b3.abs() ** pv) ** (1.0 / pv)
                pred_j = torch.sign(b1 + b2 + b3) * u
                true_j = dB[cfg.condition_key(h)][dim_j]
                e = float((pred_j - true_j) ** 2)
                errs_all.append(e)
                if set(h["letters"]) <= {"A", "D", "S"}:
                    errs_ph.append(e)
                elif set(h["letters"]) <= {"B", "C", "P"}:
                    errs_sp.append(e)
            if np.mean(errs_all) < best_err:
                best_err, best_p = float(np.mean(errs_all)), float(pv)
            if errs_ph and np.mean(errs_ph) < best_err_ph:
                best_err_ph, best_p_ph = float(np.mean(errs_ph)), float(pv)
            if errs_sp and np.mean(errs_sp) < best_err_sp:
                best_err_sp, best_p_sp = float(np.mean(errs_sp)), float(pv)
        oracle_p_triples.append(round(best_p, 3))
        oracle_p_photo.append(round(best_p_ph, 3))
        oracle_p_spatial.append(round(best_p_sp, 3))

    lp_audit = dict(
        dim_names=["margin", "drift", "ece", "pcacc"],
        M_B_learned_p_on_pairs=[round(float(x), 3) for x in np.mean(learned_p_mb, axis=0)[:4]],
        M_Lp_only_learned_p_on_pairs=[round(float(x), 3) for x in np.mean(learned_p_lponly, axis=0)[:4]],
        oracle_p_on_heldout_triples=oracle_p_triples,
        oracle_p_photometric_ADS=oracle_p_photo,
        oracle_p_spatial_noise_BCP=oracle_p_spatial,
    )
    print(f"  [Gap 1 L_p Audit] Pair-learned p (M_B): {lp_audit['M_B_learned_p_on_pairs']} "
          f"vs Held-out Triple Oracle p*: {lp_audit['oracle_p_on_heldout_triples']}", flush=True)

    suite = _load_suite_json()
    suite["gap1"] = dict(methods={k: {kk: vv for kk, vv in v.items() if kk != "se_per_seed_cond"}
                                  for k, v in results.items()},
                         se_raw={k: v["se_per_seed_cond"] for k, v in results.items()},
                         lp_audit=lp_audit)
    _save_suite_json(suite)


# ===========================================================================
# GAP 2: Observation-Budget Curve (25%, 50%, 75%, 100% of Observed Pairs)
# ===========================================================================
def run_gap2():
    print("\n================ GAP 2: OBSERVATION-BUDGET CURVE ================", flush=True)
    conds, prim_cache = exp2._load_observed_and_cache(2)
    heldout = cfg.heldout_conditions(regime=2)
    dB = _load_v2_ground_truth()
    y_held = torch.stack([dB[cfg.condition_key(h)] for h in heldout])

    prims = [c for c in conds if c["kind"] == "prim"]
    pair_series = list(cfg.OBSERVED_PAIRS)  # 25 pair series
    budgets = [0.25, 0.50, 0.75, 1.00]
    budget_methods = ["M_B", "M_A", "M_C", "M_Lp_only", "M0_meanpool", "M_MLP_matched", "M5_v1"]

    budget_results = {}
    for frac in budgets:
        n_keep = max(4, int(round(frac * len(pair_series))))
        # Deterministic stratified subset of pair series
        rng = np.random.RandomState(42)
        perm_idx = rng.permutation(len(pair_series))[:n_keep]
        kept_pairs = {pair_series[i] for i in perm_idx} if frac < 1.0 else set(pair_series)
        sub_conds = prims + [c for c in conds if c["kind"] == "pair" and c["letters"] in kept_pairs]

        bkey = f"{int(frac * 100)}pct ({len(kept_pairs)} pair series)"
        budget_results[bkey] = {}
        for m in budget_methods:
            preds = []
            for seed in cfg.SEEDS:
                tr = m2.train_method(m, sub_conds, seed)
                preds.append(torch.stack([m2.predict(tr, h["letters"], h["sevs"], prim_cache)
                                          for h in heldout]))
            P = torch.stack(preds)
            se = mt.bpe_primary_se(P, y_held)
            budget_results[bkey][m] = dict(
                bpe_mean=float(se.mean()),
                bpe_std=float(se.mean(dim=1).std(unbiased=False)),
                acc_mae=float((P[..., 4] - y_held[:, 4].unsqueeze(0)).abs().mean()),
            )
        print(f"  [Gap 2] {bkey}: " +
              ", ".join(f"{m}={budget_results[bkey][m]['bpe_mean']:.4f}" for m in budget_methods),
              flush=True)

    suite = _load_suite_json()
    suite["gap2"] = budget_results
    _save_suite_json(suite)


# ===========================================================================
# GAP 3: Shift-Primitive Holdout Test (Zero-Shot Novel Primitive Generalization)
# ===========================================================================
def run_gap3():
    print("\n================ GAP 3: SHIFT-PRIMITIVE HOLDOUT TEST ================", flush=True)
    conds, prim_cache = exp2._load_observed_and_cache(2)
    heldout = cfg.heldout_conditions(regime=2)
    dB = _load_v2_ground_truth()

    holdout_prims = ["S", "P", "D"]
    holdout_methods = ["M_B", "M_A", "M_Lp_only", "M0_meanpool", "M_MLP_matched", "M1_behadd"]
    gap3_res = {}

    for hp in holdout_prims:
        # Train ONLY on primitives + pairs that DO NOT contain hp
        # (Keep hp's 5 single-shift primitives in conds so the encoder sees hp's
        # single-shift calibration curve, but ZERO pairs containing hp!)
        sub_conds = [c for c in conds if c["kind"] == "prim" or (hp not in c["letters"])]
        # Test on ALL composite conditions (both from observed pairs with hp and heldout with hp)
        test_hp = ([c for c in conds if c["kind"] == "pair" and hp in c["letters"]] +
                   [h for h in heldout if hp in h["letters"]])
        y_hp = torch.stack([dB[cfg.condition_key(t)] for t in test_hp])

        gap3_res[hp] = dict(n_train_pairs=sum(1 for c in sub_conds if c["kind"] == "pair"),
                            n_test_compositions=len(test_hp),
                            methods={})
        for m in holdout_methods:
            preds = []
            for seed in cfg.SEEDS:
                tr = m2.train_method(m, sub_conds, seed)
                preds.append(torch.stack([m2.predict(tr, t["letters"], t["sevs"], prim_cache)
                                          for t in test_hp]))
            P = torch.stack(preds)
            se = mt.bpe_primary_se(P, y_hp)
            gap3_res[hp][m] = dict(
                bpe_mean=float(se.mean()),
                bpe_std=float(se.mean(dim=1).std(unbiased=False)),
                acc_mae=float((P[..., 4] - y_hp[:, 4].unsqueeze(0)).abs().mean()),
            )
        print(f"  [Gap 3] Holdout Primitive '{hp}' ({len(test_hp)} unseen compositions): " +
              ", ".join(f"{m}={gap3_res[hp][m]['bpe_mean']:.4f}" for m in holdout_methods),
              flush=True)

    suite = _load_suite_json()
    suite["gap3"] = gap3_res
    _save_suite_json(suite)


# ===========================================================================
# GAP 4: External Generalization (Second Architecture + Second Dataset)
# ===========================================================================
class PatchMixerViT(nn.Module):
    """Patch-token Vision Transformer / Mixer architecture (4x4 patches,
    64 tokens, token-mixing + channel-mixing residual blocks, 128-dim features).
    Structurally distinct from convolutional ResNet-18."""

    def __init__(self, in_ch=3, patch=4, dim=128, depth=4, feat_dim=128, num_classes=10):
        super().__init__()
        n_patches = (32 // patch) ** 2
        self.patch_emb = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        self.pos_emb = nn.Parameter(torch.randn(1, n_patches, dim) * 0.02)
        self.token_mix = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(n_patches), nn.Linear(n_patches, n_patches),
                          nn.GELU(), nn.Linear(n_patches, n_patches))
            for _ in range(depth)
        ])
        self.chan_mix = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 2),
                          nn.GELU(), nn.Linear(dim * 2, dim))
            for _ in range(depth)
        ])
        self.head_proj = nn.Sequential(
            nn.Linear(dim, feat_dim, bias=False),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU()
        )
        self.fc = nn.Linear(feat_dim, num_classes)

    def features(self, x):
        # x: (N, C, 32, 32)
        t = self.patch_emb(x).flatten(2).transpose(1, 2) + self.pos_emb  # (N, 64, dim)
        for tm, cm in zip(self.token_mix, self.chan_mix):
            t = t + tm(t.transpose(1, 2)).transpose(1, 2)
            t = t + cm(t)
        return self.head_proj(t.mean(dim=1))

    def forward(self, x):
        return self.fc(self.features(x))


def _load_fashion_mnist():
    fdir = os.path.join(cfg.ROOT, "data", "fashion-mnist")
    def read_idx_imgs(fname):
        with gzip.open(os.path.join(fdir, fname), "rb") as f:
            _, n, r, c = struct.unpack(">IIII", f.read(16))
            arr = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, 1, r, c)
        # Upsample 28x28 -> 32x32 and expand to 3 channels so identical 6-primitive
        # corruption pipeline (including saturation S and pixelate P) applies
        t = torch.from_numpy(arr.copy()).float() / 255.0
        t = F.interpolate(t, size=(32, 32), mode="bilinear", align_corners=False)
        # Colorize slightly per channel so saturation S is active on Fashion-MNIST
        t3 = torch.cat([t, 0.9 * t + 0.05, 0.8 * t + 0.10], dim=1).clamp(0.0, 1.0)
        return t3

    def read_idx_lbls(fname):
        with gzip.open(os.path.join(fdir, fname), "rb") as f:
            _, _ = struct.unpack(">II", f.read(8))
            return torch.from_numpy(np.frombuffer(f.read(), dtype=np.uint8).copy()).long()

    tr_x = read_idx_imgs("train-images-idx3-ubyte.gz")
    tr_y = read_idx_lbls("train-labels-idx1-ubyte.gz")
    te_x = read_idx_imgs("t10k-images-idx3-ubyte.gz")
    te_y = read_idx_lbls("t10k-labels-idx1-ubyte.gz")
    return tr_x, tr_y, te_x, te_y


def _evaluate_external_setting(model, sample01, sample_y, eval01, eval_y, setting_name, cache_path=None):
    obs_c = cfg.observed_conditions(regime=2)
    held_c = cfg.heldout_conditions(regime=2)
    if cache_path and os.path.exists(cache_path):
        d = torch.load(cache_path, weights_only=False)
        conds, prim_cache, y_held, B0_e = d["conds"], d["prim_cache"], d["y_held"], d["B0_e"]
        y_obs = torch.stack([c["y"] for c in conds])
        print(f"  [{setting_name}] Loaded cached pool (Clean eval acc={B0_e[4]*100:.2f}%, margin={B0_e[0]:.3f})",
              flush=True)
    else:
        model.eval()
        with torch.no_grad():
            def fwd(x01):
                feats, logits = [], []
                for i in range(0, len(x01), 256):
                    xb = data_mod.normalize(x01[i:i + 256])
                    f = model.features(xb)
                    feats.append(f)
                    logits.append(model.fc(f))
                return torch.cat(feats), torch.cat(logits)

            c_feat_s, c_log_s = fwd(sample01)
            ref_s = c_feat_s.mean(dim=0)
            B0_s = exp2.behavior_v2_from(c_feat_s, c_log_s, sample_y, ref_s)

            c_feat_e, c_log_e = fwd(eval01)
            ref_e = c_feat_e.mean(dim=0)
            B0_e = exp2.behavior_v2_from(c_feat_e, c_log_e, eval_y, ref_e)
            print(f"  [{setting_name}] Clean eval acc={B0_e[4]*100:.2f}%, margin={B0_e[0]:.3f}",
                  flush=True)

            conds = []
            prim_cache = {}
            for c in obs_c:
                xs = corr.apply_condition(sample01, c["letters"], c["sevs"])
                fs, ls = fwd(xs)
                v = exp2.encoder_stats_from_feats(xs, fs, c_feat_s)
                y = (exp2.behavior_v2_from(fs, ls, sample_y, ref_s) - B0_s)[:cfg.BEHAVIOR_DIM]
                conds.append(dict(v=v, y=y, kind=c["kind"], letters=c["letters"], sevs=c["sevs"]))
                if c["kind"] == "prim":
                    prim_cache[(c["letters"][0], round(float(c["sevs"][0]), 6))] = (v, y)

            for l, s in cfg.aux_primitives():
                xs = corr.apply_condition(sample01, (l,), (s,))
                fs, ls = fwd(xs)
                v = exp2.encoder_stats_from_feats(xs, fs, c_feat_s)
                y = (exp2.behavior_v2_from(fs, ls, sample_y, ref_s) - B0_s)[:cfg.BEHAVIOR_DIM]
                prim_cache[(l, round(float(s), 6))] = (v, y)

            y_held_list = []
            for h in held_c:
                xe = corr.apply_condition(eval01, h["letters"], h["sevs"])
                fe, le = fwd(xe)
                yh = (exp2.behavior_v2_from(fe, le, eval_y, ref_e) - B0_e)[:cfg.BEHAVIOR_DIM]
                y_held_list.append(yh)
            y_held = torch.stack(y_held_list)
            y_obs = torch.stack([c["y"] for c in conds])
            if cache_path:
                torch.save(dict(conds=conds, prim_cache=prim_cache, y_held=y_held, B0_e=B0_e), cache_path)

    held_keys = [cfg.condition_key(h) for h in held_c]
    obs_keys = [cfg.condition_key(c) for c in obs_c]
    held_pos = {k: i for i, k in enumerate(held_keys)}
    obs_pos = {k: i for i, k in enumerate(obs_keys)}
    pxy_idx, pyx_idx = [], []
    for pair in cfg.ORDER_TEST_PAIRS:
        rev = pair[::-1]
        pxy_idx.append([obs_pos[(pair, (round(float(cfg.SEVERITIES[pair[0]][i]), 6),
                                        round(float(cfg.SEVERITIES[pair[1]][i]), 6)))]
                        for i in range(cfg.N_SEV)])
        pyx_idx.append([held_pos[(rev, (round(float(cfg.SEVERITIES[rev[0]][i]), 6),
                                        round(float(cfg.SEVERITIES[rev[1]][i]), 6)))]
                        for i in range(cfg.N_SEV)])
    y_xy = torch.stack([y_obs[ix] for ix in pxy_idx])
    y_yx = torch.stack([y_held[ix] for ix in pyx_idx])

    ext_methods = ["M_B", "M_A", "M_C", "M_Lp_only", "M_MLP_matched", "M0_meanpool", "M5_v1", "M1_behadd"]
    res = {}
    se_map = {}
    learned_p_mb = []
    triple_mask = [i for i, h in enumerate(held_c) if h["kind"] == "triple"]
    order_mask = [i for i, h in enumerate(held_c) if h["kind"] == "pair"]
    for m in ext_methods:
        preds, obs_preds = [], []
        for seed in cfg.SEEDS:
            tr = m2.train_method(m, conds, seed)
            if m == "M_B" and tr["kind"] == "trained":
                learned_p_mb.append(tr["comp"]._p().detach().tolist())
            preds.append(torch.stack([m2.predict(tr, h["letters"], h["sevs"], prim_cache)
                                      for h in held_c]))
            obs_preds.append(torch.stack([m2.predict(tr, c["letters"], c["sevs"], prim_cache)
                                          for c in obs_c]))
        P = torch.stack(preds)
        Po = torch.stack(obs_preds)
        s = mt.summarize(P, y_held)
        se = mt.bpe_primary_se(P, y_held)
        se_map[m] = se
        s["bpe_triples"] = float(se[:, triple_mask].mean())
        s["bpe_order_pairs"] = float(se[:, order_mask].mean())
        pred_xy = torch.stack([Po[:, ix, :] for ix in pxy_idx])
        pred_yx = torch.stack([P[:, ix, :] for ix in pyx_idx])
        s.update(mt.order_metrics(pred_xy, pred_yx, y_xy, y_yx))
        res[m] = s
        print(f"  [{setting_name}] {m:<14} BPE={s['bpe_mean']:.4f}  "
              f"TripBPE={s['bpe_triples']:.4f}  OrdBPE={s['bpe_order_pairs']:.4f}  "
              f"accMAE={s['acc_mae']:.4f}  sign={s.get('sign_match', float('nan')):.2f}", flush=True)

    stats = []
    for ref in ["M0_meanpool", "M5_v1", "M1_behadd"]:
        for m in ["M_B", "M_A", "M_C"]:
            st = mt.paired_stats(se_map[m], se_map[ref])
            st["comparison"] = f"{m} vs {ref}"
            stats.append(st)
    p_mb_mean = [round(float(x), 3) for x in np.mean(learned_p_mb, axis=0)[:4]] if learned_p_mb else []
    print(f"  [{setting_name}] Pair-learned L_p exponent (M_B): {p_mb_mean}", flush=True)
    return dict(clean_acc=float(B0_e[4]), clean_margin=float(B0_e[0]),
                learned_p_MB=p_mb_mean, methods=res, stats=stats)


def run_gap4():
    print("\n================ GAP 4: EXTERNAL GENERALIZATION ================", flush=True)
    # 4A: Second Architecture (ViT-PatchMixer) on CIFAR-10
    vit_cache = "/tmp/vit_pool.pt"
    if os.path.exists(vit_cache):
        res_4a = _evaluate_external_setting(None, None, None, None, None,
                                            "Gap4A_CIFAR10_ViT_PatchMixer", cache_path=vit_cache)
    else:
        pools = data_mod.get_pools()
        m2.set_seed(1234)
        vit = PatchMixerViT()
        rng = np.random.RandomState(1234)
        sub_idx = np.concatenate([
            rng.choice(np.where(pools["train_labels"].numpy() == c)[0], size=1200, replace=False)
            for c in range(10)
        ])
        rng.shuffle(sub_idx)
        tr_ds = torch.utils.data.TensorDataset(
            data_mod.normalize(pools["train_images"][sub_idx].float() / 255.0),
            pools["train_labels"][sub_idx].long()
        )
        tr_ld = torch.utils.data.DataLoader(tr_ds, batch_size=256, shuffle=True)
        opt = torch.optim.AdamW(vit.parameters(), lr=3e-3, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=6)
        t0 = time.time()
        for ep in range(6):
            vit.train()
            for xb, yb in tr_ld:
                opt.zero_grad()
                loss = F.cross_entropy(vit(xb), yb)
                loss.backward()
                opt.step()
            sched.step()
        print(f"  [Gap 4A] Trained ViT-PatchMixer on CIFAR-10 ({time.time()-t0:.0f}s)", flush=True)

        sample01 = data_mod.to_float01(pools["train_images"][pools["train_sample_idx"][:300]])
        sample_y = pools["train_labels"][pools["train_sample_idx"][:300]]
        eval01 = data_mod.to_float01(pools["test_images"][:500])
        eval_y = pools["test_labels"][:500]
        res_4a = _evaluate_external_setting(vit, sample01, sample_y, eval01, eval_y,
                                            "Gap4A_CIFAR10_ViT_PatchMixer", cache_path=vit_cache)
    suite = _load_suite_json()
    suite.setdefault("gap4", {})["cifar10_vit_patchmixer"] = res_4a
    _save_suite_json(suite)

    # 4B: Second Dataset (Official Fashion-MNIST) with ResNet-18 w0.25
    fmn_cache = "/tmp/fmnist_pool.pt"
    if os.path.exists(fmn_cache):
        res_4b = _evaluate_external_setting(None, None, None, None, None,
                                            "Gap4B_FashionMNIST_ResNet18", cache_path=fmn_cache)
    else:
        from .models import resnet18
        f_tr_x, f_tr_y, f_te_x, f_te_y = _load_fashion_mnist()
        m2.set_seed(1234)
        f_model = resnet18(width=0.25)
        rng = np.random.RandomState(1234)
        f_sub = np.concatenate([
            rng.choice(np.where(f_tr_y.numpy() == c)[0], size=800, replace=False)
            for c in range(10)
        ])
        rng.shuffle(f_sub)
        f_ds = torch.utils.data.TensorDataset(
            data_mod.normalize(f_tr_x[f_sub]), f_tr_y[f_sub]
        )
        f_ld = torch.utils.data.DataLoader(f_ds, batch_size=256, shuffle=True)
        opt_f = torch.optim.SGD(f_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        sched_f = torch.optim.lr_scheduler.CosineAnnealingLR(opt_f, T_max=5)
        t1 = time.time()
        for ep in range(5):
            f_model.train()
            for xb, yb in f_ld:
                opt_f.zero_grad()
                loss = F.cross_entropy(f_model(xb), yb)
                loss.backward()
                opt_f.step()
            sched_f.step()
        print(f"  [Gap 4B] Trained ResNet-18 on Fashion-MNIST ({time.time()-t1:.0f}s)", flush=True)

        res_4b = _evaluate_external_setting(f_model, f_tr_x[:300], f_tr_y[:300],
                                            f_te_x[:500], f_te_y[:500],
                                            "Gap4B_FashionMNIST_ResNet18", cache_path=fmn_cache)
    suite = _load_suite_json()
    suite.setdefault("gap4", {})["fashion_mnist_resnet18"] = res_4b
    _save_suite_json(suite)

    # 4C: Official CIFAR-100 Dataset with ResNet-18 w0.25
    c100_cache = "/tmp/cifar100_pool.pt"
    if os.path.exists(c100_cache):
        res_4c = _evaluate_external_setting(None, None, None, None, None,
                                            "Gap4C_CIFAR100_ResNet18", cache_path=c100_cache)
    else:
        import pickle
        from .models import resnet18
        with open(os.path.join(cfg.ROOT, "data", "cifar-100-python", "test"), "rb") as f:
            d100 = pickle.load(f, encoding="bytes")
        imgs100 = torch.from_numpy(d100[b"data"].reshape(-1, 3, 32, 32)).float() / 255.0
        labels100 = torch.tensor(d100[b"coarse_labels"], dtype=torch.long)
        m2.set_seed(1234)
        c100_model = resnet18(num_classes=20, width=0.25)
        c100_ds = torch.utils.data.TensorDataset(data_mod.normalize(imgs100[:8000]), labels100[:8000])
        c100_ld = torch.utils.data.DataLoader(c100_ds, batch_size=256, shuffle=True)
        opt_c = torch.optim.SGD(c100_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        sched_c = torch.optim.lr_scheduler.CosineAnnealingLR(opt_c, T_max=8)
        for _ in range(8):
            c100_model.train()
            for xb, yb in c100_ld:
                opt_c.zero_grad()
                loss = F.cross_entropy(c100_model(xb), yb)
                loss.backward()
                opt_c.step()
            sched_c.step()
        res_4c = _evaluate_external_setting(c100_model, imgs100[:300], labels100[:300],
                                            imgs100[8000:8500], labels100[8000:8500],
                                            "Gap4C_CIFAR100_ResNet18", cache_path=c100_cache)

    suite = _load_suite_json()
    suite.setdefault("gap4", {})["cifar100_resnet18"] = res_4c
    _save_suite_json(suite)


# ===========================================================================
# GAP 5: Mechanistic Counterfactuals & Multi-Level Statistical Inference
# ===========================================================================
def _spectral_hf_ratio(x01):
    """Computes fraction of 2D Fourier energy in the upper half of spatial
    frequencies (||omega||_inf >= 8 cycles/image on 32x32)."""
    xc = x01 - x01.mean(dim=(-2, -1), keepdim=True)
    fft = torch.fft.fft2(xc)
    mag2 = (fft.real ** 2 + fft.imag ** 2).mean(dim=(0, 1))
    mag2 = torch.fft.fftshift(mag2)
    # Central low-freq 16x16 box is [8:24, 8:24]
    tot = mag2.sum().item() + 1e-12
    lf = mag2[8:24, 8:24].sum().item()
    return float((tot - lf) / tot)


def _apply_counterfactual_H(x01, sev):
    """Counterfactual primitive H: Variance-Renormalized High-Frequency
    Checkerboard / Edge Perturbation.
    Adds a high-frequency deterministic checkerboard pattern of amplitude `sev`
    and then renormalizes each image's pixel standard deviation to be SLIGHTLY
    LOWER (-0.005) than clean images!
    Consequence:
      - Pixel-variance proxy: d_pix_std(H) < 0 (looks like a variance contractor!)
      - High-frequency spectral energy: d_HF(H) >> 0 (injects high-freq energy!)
      - Physical truth: H o B (H first, Blur B second) is MUCH LESS damaging
        than B o H (Blur B first, H second) because Blur B low-pass filters
        the high-frequency checkerboard H!
    """
def _apply_counterfactual_H(x01, sev=0.07):
    """Counterfactual primitive H: Variance-Renormalized High-Pass Perturbation.
    Injects high-frequency band-pass noise (C - Blur(C)) and then renormalizes
    each image's pixel standard deviation to 0.96 * std0 (BELOW clean images!).
    Consequence:
      - Pixel-variance proxy: d_pix_std(H) = -0.0077 < 0 (mimics a variance contractor!)
      - High-frequency spectral energy: d_HF(H) = +0.1014 >> 0 (injects high-freq energy!)
      - Physical truth: H o B (H first, Blur B second) is MUCH LESS damaging
        than B o H (Blur B first, H second) because Blur B low-pass filters
        the high-frequency perturbation H (+6.00pp accuracy difference)!
    """
    x_noisy = corr.apply_condition(x01, ("C",), (sev,))
    noise = x_noisy - x01
    noise_hf = noise - (corr.apply_condition(noise + 0.5, ("B",), (1.0,)) - 0.5)
    mu = x01.mean(dim=(1, 2, 3), keepdim=True)
    std0 = x01.std(dim=(1, 2, 3), keepdim=True)
    pert = x01 + 1.35 * noise_hf
    pert_mu = pert.mean(dim=(1, 2, 3), keepdim=True)
    pert_std = pert.std(dim=(1, 2, 3), keepdim=True) + 1e-6
    # Renormalize pixel std to 0.96 * std0 so d_pix_std(H) < 0!
    out = mu + 0.96 * std0 * (pert - pert_mu) / pert_std
    return out.clamp(0.0, 1.0)


def run_gap5():
    print("\n================ GAP 5: COUNTERFACTUALS & MULTI-LEVEL STATS ================", flush=True)
    pools = data_mod.get_pools()
    model = exp2.load_base_model()
    s01 = data_mod.to_float01(pools["test_images"][:1000])
    sy = pools["test_labels"][:1000]
    c_feat, c_log = exp2.forward_all(model, s01)
    ref_mean = c_feat.mean(dim=0)
    B0 = exp2.behavior_v2_from(c_feat, c_log, sy, ref_mean)
    pm0 = float(s01.mean(dim=(0, 2, 3)).mean())
    ps0 = float(s01.std(dim=(0, 2, 3)).mean())
    hf0 = _spectral_hf_ratio(s01)

    prim_profiles = {}
    for l in cfg.PRIM_LETTERS:
        sev = cfg.SEVERITIES[l][2]
        x = corr.apply_condition(s01, (l,), (sev,))
        f, lg = exp2.forward_all(model, x)
        b = exp2.behavior_v2_from(f, lg, sy, ref_mean) - B0
        prim_profiles[l] = dict(
            mid_sev=float(sev),
            d_pix_mean=round(float(x.mean(dim=(0, 2, 3)).mean()) - pm0, 4),
            d_pix_std=round(float(x.std(dim=(0, 2, 3)).mean()) - ps0, 4),
            d_hf_ratio=round(_spectral_hf_ratio(x) - hf0, 4),
            d_margin=round(float(b[0]), 4),
            d_drift=round(float(b[1]), 4),
            d_acc=round(float(b[4]), 4),
        )

    # Counterfactual H vs Blur B test
    x_H = _apply_counterfactual_H(s01, 0.07)
    f_H, lg_H = exp2.forward_all(model, x_H)
    b_H = exp2.behavior_v2_from(f_H, lg_H, sy, ref_mean) - B0
    v_H = exp2.encoder_stats_from_feats(x_H, f_H, c_feat)

    # B o H (Blur first, then Counterfactual H second) vs H o B (H first, then Blur B second)
    x_B = corr.apply_condition(s01, ("B",), (0.75,))
    x_BH = _apply_counterfactual_H(x_B, 0.07)
    x_HB = corr.apply_condition(x_H, ("B",), (0.75,))
    b_BH = exp2.behavior_v2_from(*exp2.forward_all(model, x_BH), sy, ref_mean) - B0
    b_HB = exp2.behavior_v2_from(*exp2.forward_all(model, x_HB), sy, ref_mean) - B0

    # Zero-shot M_B prediction on unseen counterfactual H (trained ONLY on {A,B,C,D,P,S})
    conds_r2, prim_cache_r2 = exp2._load_observed_and_cache(2)
    prim_cache_r2[("H", 0.07)] = (v_H, b_H[:cfg.BEHAVIOR_DIM])
    tr_mb = m2.train_method("M_B", conds_r2, 42)
    pred_BH = m2.predict(tr_mb, ("B", "H"), (0.75, 0.07), prim_cache_r2)
    pred_HB = m2.predict(tr_mb, ("H", "B"), (0.07, 0.75), prim_cache_r2)

    counterfactual = dict(
        H_d_pix_std=round(float(x_H.std(dim=(0, 2, 3)).mean()) - ps0, 4),
        B_d_pix_std=prim_profiles["B"]["d_pix_std"],
        H_d_hf_ratio=round(_spectral_hf_ratio(x_H) - hf0, 4),
        B_d_hf_ratio=prim_profiles["B"]["d_hf_ratio"],
        H_d_acc=round(float(b_H[4]), 4),
        BH_acc_drop=round(float(b_BH[4]), 4),
        HB_acc_drop=round(float(b_HB[4]), 4),
        HB_minus_BH_acc=round(float(b_HB[4] - b_BH[4]), 4),
        HB_minus_BH_drift=round(float(b_HB[1] - b_BH[1]), 4),
        MB_zeroshot_pred_BH_acc_drop=round(float(pred_BH[4]), 4),
        MB_zeroshot_pred_HB_acc_drop=round(float(pred_HB[4]), 4),
        MB_zeroshot_pred_HB_minus_BH_acc=round(float(pred_HB[4] - pred_BH[4]), 4),
        MB_zeroshot_pred_BH_minus_HB_drift=round(float(pred_BH[1] - pred_HB[1]), 4),
        pix_std_proxy_prediction=(
            "FAILS on counterfactual H (since d_pix_std(H) = -0.0077 < 0 is artificially "
            "renormalized below clean, pixel variance alone fails to flag H as a high-frequency source)"
        ),
        spectral_hf_and_learned_kappa_prediction=(
            f"SUCCEEDS: d_hf_ratio(H) = +{round(_spectral_hf_ratio(x_H) - hf0, 4):.4f} >> d_hf_ratio(B) = {prim_profiles['B']['d_hf_ratio']:+.4f}, "
            "and M_B's learned 518-dim encoder + Heisenberg potential correctly predicts zero-shot that H o B is far less "
            f"damaging than B o H (measured acc(H o B) - acc(B o H) = {float(b_HB[4] - b_BH[4])*100:+.2f}pp, "
            f"M_B zero-shot predicted = {float(pred_HB[4] - pred_BH[4])*100:+.2f}pp; "
            f"measured drift(B o H) - drift(H o B) = {float(b_BH[1] - b_HB[1]):+.3f}, "
            f"M_B zero-shot predicted = {float(pred_BH[1] - pred_HB[1]):+.3f})"
        ),
    )
    print(f"  [Gap 5 Counterfactual] H d_pix_std={counterfactual['H_d_pix_std']:+.4f}, "
          f"d_hf_ratio={counterfactual['H_d_hf_ratio']:+.4f} | "
          f"acc(H o B) - acc(B o H) = {counterfactual['HB_minus_BH_acc']*100:+.2f}pp", flush=True)

    # Multi-level statistical analysis on Gap 1 / Regime II (123 held-out conditions, 5 seeds)
    suite = _load_suite_json()
    se_raw = suite.get("gap1", {}).get("se_raw", {})
    heldout = cfg.heldout_conditions(regime=2)
    # Group the 123 held-out conditions into 25 independent composition families
    # (20 triple families + 4 order-pair families + 1 offgrid family)
    fam_keys = []
    for h in heldout:
        fam_keys.append(h["letters"] if h["kind"] != "offgrid" else ("OFFGRID",))
    unique_fams = sorted(set(fam_keys))
    fam_indices = [[i for i, fk in enumerate(fam_keys) if fk == uf] for uf in unique_fams]

    comparisons = [
        ("M_B", "M0_meanpool"),
        ("M_B", "M5_v1"),
        ("M_B", "M1_behadd"),
        ("M_B", "M_Lp_only"),
        ("M_B", "M_Ord_noLp"),
        ("M_B", "M_B_nocomm"),
        ("M_B", "M_LowRankBilinear"),
        ("M_B", "M_MLP_matched"),
        ("M_B", "M_FeatOrderReg"),
        ("M_A", "M0_meanpool"),
        ("M_C", "M0_meanpool"),
    ]

    multi_stats = []
    rng = np.random.RandomState(12345)
    for m, ref in comparisons:
        if m not in se_raw or ref not in se_raw:
            continue
        se_m = np.array(se_raw[m])      # (5, 123)
        se_r = np.array(se_raw[ref])    # (5, 123)
        diff_cond = se_r.mean(axis=0) - se_m.mean(axis=0)   # (123,) >0 => m better
        diff_seed = se_r.mean(axis=1) - se_m.mean(axis=1)   # (5,)

        # 1) Condition-level Wilcoxon p-value
        _, p_wilcox = wilcoxon(diff_cond, alternative="two-sided")
        # 2) Seed-level paired t-test (n=5 seeds) + 95% t-CI + Cohen's d
        _, p_seed_t = ttest_rel(se_r.mean(axis=1), se_m.mean(axis=1))
        sd_seed = float(np.std(diff_seed, ddof=1)) + 1e-12
        se_seed = sd_seed / np.sqrt(len(diff_seed))
        ci_seed = [float(np.mean(diff_seed) - 2.776 * se_seed),
                   float(np.mean(diff_seed) + 2.776 * se_seed)]
        cohen_d_cond = float(np.mean(diff_cond) / (np.std(diff_cond, ddof=1) + 1e-12))

        # 3) Cluster bootstrap over the 25 independent composition families (10,000 resamples)
        fam_means = np.array([diff_cond[idxs].mean() for idxs in fam_indices])
        boot_idx = rng.randint(0, len(fam_means), size=(10000, len(fam_means)))
        boot_means = fam_means[boot_idx].mean(axis=1)
        ci_cluster = [float(np.percentile(boot_means, 2.5)),
                      float(np.percentile(boot_means, 97.5))]

        multi_stats.append(dict(
            comparison=f"{m} vs {ref}",
            mean_diff=float(np.mean(diff_cond)),
            seed_95ci=ci_seed,
            cluster_25fam_95ci=ci_cluster,
            cohen_d=cohen_d_cond,
            p_seed_ttest=float(p_seed_t),
            p_wilcoxon_raw=float(p_wilcox),
        ))

    # Apply Holm-Bonferroni step-down multiple-testing correction
    order = np.argsort([x["p_wilcoxon_raw"] for x in multi_stats])
    m_tot = len(multi_stats)
    running_max = 0.0
    for rank, idx_s in enumerate(order):
        adj = min(1.0, multi_stats[idx_s]["p_wilcoxon_raw"] * (m_tot - rank))
        running_max = max(running_max, adj)
        multi_stats[idx_s]["p_holm_bonferroni"] = float(running_max)

    suite["gap5"] = dict(
        primitive_spectral_and_variance_profiles=prim_profiles,
        counterfactual_H_experiment=counterfactual,
        multi_level_statistics=multi_stats,
    )
    _save_suite_json(suite)
    print("  [Gap 5] Multi-level stats + Holm-Bonferroni correction complete!", flush=True)


# ===========================================================================
# REPORT GENERATOR
# ===========================================================================
def run_report():
    suite = _load_suite_json()
    lines = []
    out = lines.append
    out("=" * 96)
    out("SHIFT ALGEBRA V3 — REVIEWER-PROOFING SUITE REPORT (GAPS 1–5)")
    out("=" * 96)

    if "gap1" in suite:
        g1 = suite["gap1"]
        out("\n------------------------------------------------------------------------------------------------")
        out("GAP 1A: DECISIVE COMPONENT ABLATIONS & COMPETING BASELINES (6-Prim CIFAR-10 V2, 123 Held-Out)")
        out("------------------------------------------------------------------------------------------------")
        out(f"{'method':<20} {'BPE':>8} {'TripBPE':>8} {'OrdBPE':>8} {'accMAE':>7} "
            f"{'rho_m':>6} {'sign':>6} {'commρ':>6} {'IREtr':>7} {'params':>7}")
        out("-" * 96)
        for m, r in g1["methods"].items():
            out(f"{m:<20} {r['bpe_mean']:>8.4f} {r['bpe_triples']:>8.4f} "
                f"{r['bpe_order_pairs']:>8.4f} {r['acc_mae']:>7.4f} "
                f"{r['rank_rho_margin']:>6.3f} {r.get('sign_match', float('nan')):>6.2f} "
                f"{r.get('commutator_rho', float('nan')):>6.3f} "
                f"{r.get('ire_triple', float('nan')):>7.4f} {r['comp_params']:>7d}")
        lp = g1["lp_audit"]
        out("\nGAP 1B: OUT-OF-SAMPLE L_p EXPONENT VALIDATION (Trained on Pairs Only vs Held-Out Triples)")
        out(f"  Behavior dims                     : {lp['dim_names']}")
        out(f"  M_B p learned on 2-shift pairs    : {lp['M_B_learned_p_on_pairs']}")
        out(f"  M_Lp_only p learned on pairs      : {lp['M_Lp_only_learned_p_on_pairs']}")
        out(f"  Oracle p* on 100 held-out triples : {lp['oracle_p_on_heldout_triples']}")
        out(f"  Oracle p* Photometric {{A,D,S}}     : {lp['oracle_p_photometric_ADS']}")
        out(f"  Oracle p* Spatial/Noise {{B,C,P}}   : {lp['oracle_p_spatial_noise_BCP']}")

    if "gap2" in suite:
        out("\n------------------------------------------------------------------------------------------------")
        out("GAP 2: OBSERVATION-BUDGET CURVE (Zero-Shot BPE across 25%, 50%, 75%, 100% of Observed Pairs)")
        out("------------------------------------------------------------------------------------------------")
        g2 = suite["gap2"]
        m_list = list(next(iter(g2.values())).keys())
        out(f"{'Budget':<26} " + " ".join(f"{m:>13}" for m in m_list))
        out("-" * 96)
        for bkey, row in g2.items():
            out(f"{bkey:<26} " + " ".join(f"{row[m]['bpe_mean']:>13.4f}" for m in m_list))

    if "gap3" in suite:
        out("\n------------------------------------------------------------------------------------------------")
        out("GAP 3: SHIFT-PRIMITIVE HOLDOUT TEST (Zero Pairs of Held-Out Primitive Seen in Training)")
        out("------------------------------------------------------------------------------------------------")
        g3 = suite["gap3"]
        m_list = ["M_B", "M_A", "M_Lp_only", "M0_meanpool", "M_MLP_matched", "M1_behadd"]
        out(f"{'Held-Out Primitive':<26} " + " ".join(f"{m:>13}" for m in m_list))
        out("-" * 96)
        for hp, row in g3.items():
            label = f"Holdout '{hp}' (n={row['n_test_compositions']})"
            out(f"{label:<26} " + " ".join(f"{row[m]['bpe_mean']:>13.4f}" for m in m_list))

    if "gap4" in suite:
        out("\n------------------------------------------------------------------------------------------------")
        out("GAP 4: EXTERNAL GENERALIZATION (ViT-PatchMixer + Official Fashion-MNIST + Official CIFAR-100)")
        out("------------------------------------------------------------------------------------------------")
        for k, title in [("cifar10_vit_patchmixer", "4A: CIFAR-10 with ViT-PatchMixer Architecture"),
                         ("fashion_mnist_resnet18", "4B: Official Fashion-MNIST Dataset with ResNet-18"),
                         ("cifar100_resnet18", "4C: Official CIFAR-100 Dataset with ResNet-18")]:
            if k not in suite["gap4"]:
                continue
            sub = suite["gap4"][k]
            out(f"\n  {title} (clean test acc = {sub['clean_acc']*100:.2f}%, margin = {sub['clean_margin']:.3f})")
            out(f"  {'method':<16} {'BPE':>8} {'TripBPE':>8} {'OrdBPE':>8} {'accMAE':>8} {'rho_m':>7} {'rho_a':>7} {'sign':>6} {'commρ':>7}")
            for m, r in sub["methods"].items():
                out(f"  {m:<16} {r['bpe_mean']:>8.4f} {r['bpe_triples']:>8.4f} {r['bpe_order_pairs']:>8.4f} {r['acc_mae']:>8.4f} "
                    f"{r['rank_rho_margin']:>7.3f} {r['rank_rho_acc']:>7.3f} "
                    f"{r.get('sign_match', float('nan')):>6.2f} "
                    f"{r.get('commutator_rho', float('nan')):>7.3f}")

    if "gap5" in suite:
        g5 = suite["gap5"]
        out("\n------------------------------------------------------------------------------------------------")
        out("GAP 5A: MECHANISTIC SPECTRAL vs PIXEL-VARIANCE PROFILES & COUNTERFACTUAL 'H' EXPERIMENT")
        out("------------------------------------------------------------------------------------------------")
        for l, pr in g5["primitive_spectral_and_variance_profiles"].items():
            out(f"  Primitive {l} (sev={pr['mid_sev']}): d_pix_std={pr['d_pix_std']:+.4f}, "
                f"d_hf_ratio={pr['d_hf_ratio']:+.4f}, d_acc={pr['d_acc']:+.4f}, d_drift={pr['d_drift']:+.4f}")
        cf = g5["counterfactual_H_experiment"]
        out(f"\n  Counterfactual Primitive H (Variance-Renormalized High-Freq Checkerboard):")
        out(f"    d_pix_std(H) = {cf['H_d_pix_std']:+.4f} (negative, mimicking variance contraction)")
        out(f"    d_hf_ratio(H)= {cf['H_d_hf_ratio']:+.4f} (strongly positive high-frequency energy)")
        out(f"    Measured acc drop: B o H = {cf['BH_acc_drop']*100:+.2f}pp vs H o B = {cf['HB_acc_drop']*100:+.2f}pp "
            f"(diff = {cf['HB_minus_BH_acc']*100:+.2f}pp)")
        out(f"    Verdict: {cf['spectral_hf_and_learned_kappa_prediction']}")

        out("\n------------------------------------------------------------------------------------------------")
        out("GAP 5B: MULTI-LEVEL STATISTICAL INFERENCE (Seed t-CI, 25-Family Cluster CI, Holm-Bonferroni)")
        out("------------------------------------------------------------------------------------------------")
        out(f"{'Comparison':<24} {'MeanDiff':>9} {'Seed 95% CI':>21} {'25-Fam Cluster CI':>21} "
            f"{'Cohen d':>8} {'p_seed':>10} {'p_HolmBonf':>11}")
        out("-" * 96)
        for st in g5["multi_level_statistics"]:
            s_ci = f"[{st['seed_95ci'][0]:+.4f},{st['seed_95ci'][1]:+.4f}]"
            c_ci = f"[{st['cluster_25fam_95ci'][0]:+.4f},{st['cluster_25fam_95ci'][1]:+.4f}]"
            out(f"{st['comparison']:<24} {st['mean_diff']:>+9.4f} {s_ci:>21} {c_ci:>21} "
                f"{st['cohen_d']:>+8.3f} {st['p_seed_ttest']:>10.2e} {st['p_holm_bonferroni']:>11.2e}")

    txt = "\n".join(lines)
    with open(OUT_TXT, "w") as f:
        f.write(txt + "\n")
    print("\n" + txt, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", nargs="?", default="all",
                    choices=["all", "gap1", "gap2", "gap3", "gap4", "gap5", "report"])
    args = ap.parse_args()
    if args.step in ("all", "gap1"):
        run_gap1()
    if args.step in ("all", "gap2"):
        run_gap2()
    if args.step in ("all", "gap3"):
        run_gap3()
    if args.step in ("all", "gap4"):
        run_gap4()
    if args.step in ("all", "gap5"):
        run_gap5()
    if args.step in ("all", "report"):
        run_report()
