"""V2 metrics (PLAN_V2 §7). All metrics are computed on the 5-dim target
[margin, drift, ece, pcacc, acc]; BPE uses the 4 primary (non-flooring) dims.

Inputs per method: heldout_pred (S,H,5) [S seeds], heldout_y (H,5) measured
ground-truth target deltas, in_sample_bpe (S,), oracle_bpe (S,).
"""
import numpy as np
import torch
from scipy.stats import spearmanr, wilcoxon

from . import config_v2 as cfg

PRIMARY = cfg.BPE_PRIMARY          # 4
ACC_IDX = 4                        # accuracy is target dim 4 (secondary)


def bpe_primary_se(pred, y):
    """pred (S,H,5), y (H,5) -> per-condition MSE (S,H) over the
    4 primary dims [margin, drift, ece, pcacc]."""
    return ((pred - y.unsqueeze(0)) ** 2)[..., :PRIMARY].mean(dim=-1)


def summarize(pred, y, in_sample_bpe=None, oracle_bpe=None):
    """Returns the per-method metric dict."""
    S, H, _ = pred.shape
    se = bpe_primary_se(pred, y)                       # (S,H)
    out = {}
    out["bpe_mean"] = float(se.mean())
    out["bpe_std"] = float(se.mean(dim=1).std(unbiased=False).item())
    out["bpe_per_seed"] = [float(x) for x in se.mean(dim=1)]
    acc_mae = (pred[..., ACC_IDX] - y[:, ACC_IDX].unsqueeze(0)).abs()   # (S,H)
    out["acc_mae"] = float(acc_mae.mean())
    # rank correlations (per seed, then averaged)
    rho_margin, rho_acc = [], []
    for s in range(S):
        r1 = spearmanr(pred[s, :, 0].numpy(), y[:, 0].numpy()).statistic
        r2 = spearmanr(pred[s, :, ACC_IDX].numpy(), y[:, ACC_IDX].numpy()).statistic
        rho_margin.append(r1 if r1 == r1 else 0.0)     # NaN -> 0
        rho_acc.append(r2 if r2 == r2 else 0.0)
    out["rank_rho_margin"] = float(np.mean(rho_margin))
    out["rank_rho_acc"] = float(np.mean(rho_acc))
    if in_sample_bpe is not None and oracle_bpe is not None:
        e_seen = float(np.mean(in_sample_bpe))
        e_unseen = out["bpe_mean"]
        e_oracle = float(np.mean(oracle_bpe))
        out["cgg"] = e_unseen - e_seen
        out["rce"] = (e_unseen - e_seen) / (e_oracle - e_seen + 1e-12)
    return out


def sign_match_rate(pred, y, ordered_pairs):
    """ordered_pairs: list of ((X,Y), (Y,X)) letter pairs with severity-aligned
    condition indices in y. Returns the fraction of (pair, severity) tests
    where sign(pred_margin(XY) - pred_margin(YX)) == sign(true)."""
    S, H, _ = pred.shape
    # build index maps: key -> position in y
    hits, total = 0, 0
    return hits, total   # filled by caller (needs key->index map); see below


def order_metrics(pred_xy, pred_yx, y_xy, y_yx):
    """Order test over G ordered-pair groups (G = 8 in V2).

    pred_xy, pred_yx: (G, S, k, 5) — model predictions for the OBSERVED order
    (X,Y) and the HELD-OUT reverse (Y,X), aligned by severity (k = 5).
    y_xy, y_yx: (G, k, 5) — measured ground-truth target deltas.

    Returns:
      sign_match   — fraction of (group, severity, seed) tests where
                     sign(pred_YX - pred_XY) == sign(true_YX - true_XY)
                     on the margin component (primary, graded);
      commutator_rho — Spearman between predicted and measured order-effect
                     magnitudes ||dB_YX - dB_XY|| over all groups/severities;
      behavior_order_err — mean | ||pred diff|| - ||true diff|| |.
    """
    G, S, k, _ = pred_xy.shape
    signs_hit, signs_tot = 0, 0
    pred_eff, true_eff, order_errs = [], [], []
    for g in range(G):
        pxy, pyx = pred_xy[g], pred_yx[g]              # (S,k,5)
        txy, tyx = y_xy[g], y_yx[g]                    # (k,5)
        # Sign match on the dominant graded behavior shift (drift, index 1)
        sgn_p = torch.sign(pyx[..., 1] - pxy[..., 1])          # (S,k)
        sgn_t = torch.sign(tyx[:, 1] - txy[:, 1]).unsqueeze(0)
        signs_hit += int((sgn_p == sgn_t).sum())
        signs_tot += sgn_p.numel()
        pe = (pyx[..., :PRIMARY] - pxy[..., :PRIMARY]).norm(dim=-1).mean(dim=0).numpy()
        te = (tyx[:, :PRIMARY] - txy[:, :PRIMARY]).norm(dim=-1).numpy()
        pred_eff.append(pe)
        true_eff.append(te)
        order_errs.append(pe - te)
    sign_rate = signs_hit / signs_tot if signs_tot else float("nan")
    pred_eff = np.concatenate(pred_eff) if pred_eff else np.array([0.0])
    true_eff = np.concatenate(true_eff) if true_eff else np.array([0.0])
    if len(pred_eff) >= 3 and np.std(pred_eff) > 1e-12 and np.std(true_eff) > 1e-12:
        comm_rho = float(spearmanr(pred_eff, true_eff).statistic)
    else:
        comm_rho = float("nan")
    order_err = float(np.mean(np.abs(np.concatenate(order_errs))))
    return dict(sign_match=sign_rate, commutator_rho=comm_rho,
                behavior_order_err=order_err)


def paired_stats(se_ref, se_other):
    """Paired Wilcoxon over H conditions on seed-averaged per-condition SE,
    with a 95% bootstrap CI of the mean difference (other - ref).
    diff = other_err - ref_err, so POSITIVE = other WORSE = ref BETTER."""
    diff = (se_other.mean(dim=0) - se_ref.mean(dim=0)).numpy()
    try:
        p = wilcoxon(diff, alternative="two-sided").pvalue
    except ValueError:
        p = float("nan")
    rng = np.random.RandomState(0)
    boots = []
    for _ in range(2000):
        idx = rng.randint(0, len(diff), len(diff))
        boots.append(diff[idx].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(mean_diff=float(diff.mean()), ci_lo=float(lo), ci_hi=float(hi),
                p_value=float(p))


def per_severity(pred, y, sev_index):
    """BPE split by severity index (list of H ints 0..4)."""
    se = bpe_primary_se(pred, y).mean(dim=0).numpy()      # (H,)
    out = {}
    for s in sorted(set(sev_index)):
        m = np.array([i for i, x in enumerate(sev_index) if x == s])
        out[int(s)] = float(se[m].mean()) if len(m) else float("nan")
    return out
