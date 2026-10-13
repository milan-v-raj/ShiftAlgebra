"""V2 experiment pipeline (PLAN_V2). Phases (run in order):

  pilot-severities   P0: recalibrate severity grids on the frozen base model
                     (single shifts only; writes pools_v2/severity_grids.json)
  synthetic          P1: S1-S3 synthetic tiers, E1/E2 — H2'/H3' mechanics
  build-pools        P2: vision V2 pools (encoder inputs + behavior labels,
                     train sample) + held-out encoder inputs (evaluation-only)
  train-methods      P3: all methods x seeds (Regime II or III)
  train-transfer     P4a: E5 synthetic->vision pretrained transfer (H6')
  evaluate           P4b: ground truth (full test pool), metrics, statistics,
                     decision rules H1'-H6'
  audit              P5: contamination / leakage / correctness / control checks
  report             formatted report -> results_v2/REPORT.txt

Env overrides (for smoke tests): SA_V2_STEPS, SA_V2_SEEDS.
Usage: python -m src.experiment_v2 <phase> [--regime 2|3]
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch

from . import config_v2 as cfg
from . import data as data_mod
from . import corruptions_v2 as corr
from .models import resnet18
from . import methods_v2 as m2
from . import synthetic_v2 as syn
from . import metrics_v2 as mt

torch.set_num_threads(2)

if os.environ.get("SA_V2_STEPS"):
    cfg.TRAIN_STEPS = int(os.environ["SA_V2_STEPS"])
if os.environ.get("SA_V2_SEEDS"):
    cfg.SEEDS = [cfg.MASTER_SEED + i for i in range(int(os.environ["SA_V2_SEEDS"]))]

os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
os.makedirs(cfg.POOLS_DIR, exist_ok=True)

PRIMARY = cfg.BPE_PRIMARY


# ---------------------------------------------------------------------------
# behavior measurement (v2: margin-led 6-dim) + encoder input (518-dim stats)
# ---------------------------------------------------------------------------
@torch.no_grad()
def forward_all(model, x01, batch_size=256):
    feats, logits = [], []
    for i in range(0, len(x01), batch_size):
        xb = data_mod.normalize(x01[i:i + batch_size])
        f = model.features(xb)
        feats.append(f)
        logits.append(model.fc(f))
    return torch.cat(feats), torch.cat(logits)


def behavior_v2_from(features, logits, labels, ref_feat_mean):
    """6-dim behavior: [margin, drift, ece, pcacc, acc, conf].
    margin = mean top1-top2 softmax gap (graded; does not floor like accuracy)."""
    probs = torch.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    top2 = probs.topk(2, dim=1).values
    margin = (top2[:, 0] - top2[:, 1]).mean().item()
    labels = labels.long()
    acc = (pred == labels).float().mean().item()
    per_class = []
    for c in range(10):
        m = labels == c
        if m.any():
            per_class.append((pred[m] == c).float().mean().item())
    pcacc = float(np.nanmean(per_class))
    ece = 0.0
    for b in range(10):
        lo, hi = b / 10.0, (b + 1) / 10.0
        m = (conf >= lo) & (conf < hi)
        if m.any():
            ece += m.float().mean().item() * abs(
                (pred[m] == labels[m]).float().mean().item() - conf[m].mean().item())
    drift = (features.mean(dim=0) - ref_feat_mean).norm().item()
    return torch.tensor([margin, drift, ece, pcacc, acc, conf.mean().item()],
                        dtype=torch.float32)


def encoder_stats_from_feats(x, feats, clean_feat):
    """v (518,) = unlabeled distribution statistics (feat/pixel/delta mean+std).
    Uses NO labels from D_S (strict zero-leak boundary)."""
    delta = feats - clean_feat
    return torch.cat([
        feats.mean(dim=0), feats.std(dim=0),
        x.mean(dim=(0, 2, 3)), x.std(dim=(0, 2, 3)),
        delta.mean(dim=0), delta.std(dim=0),
    ])


def load_base_model():
    ckpt = torch.load(os.path.join(cfg.CHECKPOINT_DIR, "base_model_best.pt"),
                      weights_only=False)
    model = resnet18(width=ckpt["width"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"[base] loaded base_model_best.pt (test_acc={ckpt['test_acc']*100:.2f}%)")
    return model


# ---------------------------------------------------------------------------
# P0: severity recalibration pilot
# ---------------------------------------------------------------------------
def phase_pilot_severities():
    model = load_base_model()
    pools = data_mod.get_pools()
    test01 = data_mod.to_float01(pools["test_images"])
    test_y = pools["test_labels"]

    @torch.no_grad()
    def acc_of(x):
        _, logits = forward_all(model, x)
        return (logits.argmax(1) == test_y.long()).float().mean().item()

    clean_acc = acc_of(test01)
    print(f"[pilot] clean test acc = {clean_acc*100:.2f}%")
    CANDIDATES = {
        "A": [1.10, 1.20, 1.30, 1.40, 1.50, 1.60],
        "B": [0.3, 0.45, 0.6, 0.75, 0.9, 1.1],
        "C": [0.015, 0.03, 0.045, 0.06, 0.08, 0.10],
        "D": [0.88, 0.78, 0.68, 0.58, 0.48, 0.38],
        "P": [1.15, 1.25, 1.35, 1.50, 1.70, 1.90],
        "S": [1.4, 1.8, 2.2, 2.6, 3.0, 3.5],
    }
    TARGETS = [3.0, 8.0, 15.0, 25.0, 40.0]
    grids, chosen_drops = {}, {}
    for letter, cands in CANDIDATES.items():
        drops = []
        for sev in cands:
            x = corr.apply_condition(test01, (letter,), (float(sev),))
            drops.append((clean_acc - acc_of(x)) * 100.0)
            print(f"[pilot] {letter} sev={sev:<5} drop={drops[-1]:+6.2f}pp", flush=True)
        chosen, used = [], set()
        for t in TARGETS:
            best, bi = None, None
            for i, d in enumerate(drops):
                if i in used:
                    continue
                err = abs(d - t)
                if best is None or err < best:
                    best, bi = err, i
            used.add(bi)
            chosen.append((cands[bi], drops[bi]))
        chosen.sort(key=lambda pair: cands.index(pair[0]))
        grids[letter] = [float(s) for s, _ in chosen]
        chosen_drops[letter] = [round(d, 2) for _, d in chosen]
        print(f"[pilot] {letter} CHOSEN grid={grids[letter]} "
              f"drops={[f'{d:+.1f}pp' for d in chosen_drops[letter]]}", flush=True)
    with open(os.path.join(cfg.POOLS_DIR, "severity_grids.json"), "w") as f:
        json.dump({"clean_acc": clean_acc, "targets_pp": TARGETS,
                   "grids": grids, "chosen_drops_pp": chosen_drops}, f, indent=2)
    print("[pilot] DONE -> pools_v2/severity_grids.json")


# ---------------------------------------------------------------------------
# P1: synthetic tiers (E1/E2 — H2'/H3' mechanics)
# ---------------------------------------------------------------------------
def phase_synthetic():
    summary = {}
    methods = ["M0_meanpool", "M1_behadd", "M_A", "M_A_nointer", "M_A_att",
               "M_B", "M_C", "M_C_nopos", "M5_v1", "M1_emb",
               "MEMO_control", "PERM_control"]

    for tier in ["S1", "S2", "S3"]:
        print(f"\n[synthetic] ===== tier {tier} =====", flush=True)
        t0 = time.time()
        data = syn.build_tier(tier)
        if tier == "S3":
            print(f"[synthetic] S3 analytic commutators: "
                  f"AB={syn.commutator_norm(data['transforms'],'A','B'):.3f} "
                  f"BC={syn.commutator_norm(data['transforms'],'B','C'):.3f} "
                  f"CD={syn.commutator_norm(data['transforms'],'C','D'):.3f}")
        conds, held = data["conds"], data["held"]
        cache = data["prim_cache"]
        y_held = torch.stack([h["y"] for h in held])            # (H, 5)
        y_obs_te = torch.stack([c["y_test"] for c in conds])    # (O, 5)
        v_held = torch.stack([h["v"] for h in held])            # (H, in_dim)
        obs_keys = [syn.condition_key(c) for c in conds]
        held_keys = [syn.condition_key(h) for h in held]
        obs_pos = {k: i for i, k in enumerate(obs_keys)}
        held_pos = {k: i for i, k in enumerate(held_keys)}

        preds, obs_preds, in_bpe, oracles = {}, {}, {}, {}
        for method in methods:
            pp, op, ib, ob = [], [], [], []
            for seed in cfg.SEEDS:
                tr = m2.train_method(method, conds, seed, in_dim=data["in_dim"])
                ib.append(tr.get("in_sample_bpe", float("nan")))
                p_h = torch.stack([m2.predict(tr, h["letters"], h["sevs"], cache)
                                   for h in held])
                p_o = torch.stack([m2.predict(tr, c["letters"], c["sevs"], cache)
                                   for c in conds])
                pp.append(p_h)
                op.append(p_o)
                if tr["kind"] == "trained":
                    with torch.no_grad():
                        o_pred = tr["dec"](tr["enc"](v_held))
                    ob.append(float(((o_pred - y_held) ** 2)[:, :PRIMARY].mean()))
            preds[method] = torch.stack(pp)                     # (S, H, 5)
            obs_preds[method] = torch.stack(op)                 # (S, O, 5)
            in_bpe[method] = ib
            oracles[method] = ob

        # Order groups: compare observed (X,Y) vs held-out reverse (Y,X)
        pxy_idx, pyx_idx = [], []
        for pair in syn.ORDER_TEST_PAIRS:
            rev = pair[::-1]
            ixy = [obs_pos[(pair, (round(float(s), 6), round(float(s), 6)))]
                   for s in syn.SEVERITIES]
            iyx = [held_pos[(rev, (round(float(s), 6), round(float(s), 6)))]
                   for s in syn.SEVERITIES]
            pxy_idx.append(ixy)
            pyx_idx.append(iyx)
        y_xy = torch.stack([y_obs_te[ix] for ix in pxy_idx])    # (G, k, 5)
        y_yx = torch.stack([y_held[ix] for ix in pyx_idx])      # (G, k, 5)

        rows = {}
        for method in methods:
            s = mt.summarize(preds[method], y_held, in_bpe[method],
                             oracles[method] if oracles[method] else None)
            if tier == "S3":
                pred_xy = torch.stack([obs_preds[method][:, ix, :] for ix in pxy_idx])
                pred_yx = torch.stack([preds[method][:, ix, :] for ix in pyx_idx])
                s.update(mt.order_metrics(pred_xy, pred_yx, y_xy, y_yx))
            ire_p, ire_t = _compute_ire(
                preds[method], obs_preds[method], held, y_held, y_obs_te,
                held_pos, obs_pos)
            s["ire_pair"] = ire_p
            s["ire_triple"] = ire_t
            rows[method] = s
            extra = (f" sign={s['sign_match']:.2f} commρ={s['commutator_rho']:.3f}"
                     if tier == "S3" else "")
            print(f"[synthetic] {tier} {method:<14} BPE={s['bpe_mean']:.4f} "
                  f"accMAE={s['acc_mae']:.4f} rho_m={s['rank_rho_margin']:.3f} "
                  f"IRE_tr={ire_t:.4f}{extra}", flush=True)

        summary[tier] = rows
        print(f"[synthetic] {tier} done in {time.time()-t0:.0f}s", flush=True)

    s1, s2, s3 = summary["S1"], summary["S2"], summary["S3"]
    rules = {
        "H2'_S2_M_A_beats_M1_behadd":
            s2["M_A"]["bpe_mean"] < s2["M1_behadd"]["bpe_mean"],
        "H2'_S2_M_B_beats_M1_behadd":
            s2["M_B"]["bpe_mean"] < s2["M1_behadd"]["bpe_mean"],
        "H2'_S2_M_A_beats_M0_meanpool":
            s2["M_A"]["bpe_mean"] < s2["M0_meanpool"]["bpe_mean"],
        "H2'_S1_additive_control_holds":
            s1["M_A"]["bpe_mean"] <= s1["M0_meanpool"]["bpe_mean"],
        "H3'_S3_M_B_sign_match_ge_0.8":
            s3["M_B"].get("sign_match", 0.0) >= 0.80,
        "H3'_S3_M_B_commutator_rho_ge_0.7":
            (s3["M_B"].get("commutator_rho") or 0.0) >= 0.70,
        "H3'_S3_M_B_beats_M5_v1_on_BPE":
            s3["M_B"]["bpe_mean"] < s3["M5_v1"]["bpe_mean"],
    }
    print("\n[synthetic] ===== decision rules (mechanics) =====")
    for k, ok in rules.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {k}")
    summary["rules"] = {k: ("PASS" if ok else "FAIL") for k, ok in rules.items()}
    with open(os.path.join(cfg.RESULTS_DIR, "synthetic_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print("[synthetic] DONE -> results_v2/synthetic_summary.json")


def _compute_ire(pred_held, pred_obs, heldout, y_held, y_obs, held_pos, obs_pos):
    """IRE in behavior space (primary dims) for held-out pairs and triples:
    gamma(S) = dB_S - sum_{i in S} dB_{prim_i}."""
    S = pred_held.shape[0]
    errs_pair, errs_triple = [], []
    for s in range(S):
        for h in heldout:
            if h["kind"] == "offgrid":
                continue
            kh = (tuple(h["letters"]), tuple(round(float(x), 6) for x in h["sevs"]))
            ih = held_pos[kh]
            prim_true = sum(
                y_obs[obs_pos[((l,), (round(float(sv), 6),))]]
                for l, sv in zip(h["letters"], h["sevs"]))
            prim_pred = sum(
                pred_obs[s, obs_pos[((l,), (round(float(sv), 6),))]]
                for l, sv in zip(h["letters"], h["sevs"]))
            g_true = (y_held[ih] - prim_true)[:PRIMARY]
            g_pred = (pred_held[s, ih] - prim_pred)[:PRIMARY]
            err = (g_pred - g_true).pow(2).mean().item()
            if h["kind"] == "pair":
                errs_pair.append(err)
            elif h["kind"] == "triple":
                errs_triple.append(err)
    return (float(np.mean(errs_pair)) if errs_pair else float("nan"),
            float(np.mean(errs_triple)) if errs_triple else float("nan"))


# ---------------------------------------------------------------------------
# P2: build pools (vision V2)
# ---------------------------------------------------------------------------
def phase_build_pools(regime=2):
    cfg.load_pilot_grids()
    model = load_base_model()
    pools = data_mod.get_pools()
    idx = pools["train_sample_idx"]
    sample = pools["train_images"][idx]
    sample01 = data_mod.to_float01(sample)
    sample_labels = pools["train_labels"][idx]
    clean_feat, clean_logits = forward_all(model, sample01)
    ref_mean = clean_feat.mean(dim=0)
    B0 = behavior_v2_from(clean_feat, clean_logits, sample_labels, ref_mean)
    print(f"[build-pools] train-sample clean: margin={B0[0]:.3f} acc={B0[4]*100:.2f}% "
          f"ece={B0[2]:.4f} conf={B0[5]:.4f}")

    # Measure all unique conditions across Regimes II & III in one pass on sample01
    cache_path = os.path.join(cfg.POOLS_DIR, "cache_vy.pt")
    cache_vy = torch.load(cache_path, weights_only=False) if os.path.exists(cache_path) else {}
    for reg_old in (2, 3):
        p_old = os.path.join(cfg.POOLS_DIR, f"observed_r{reg_old}.npz")
        if os.path.exists(p_old):
            d_old = np.load(p_old, allow_pickle=True)
            for v, y, k in zip(d_old["V"], d_old["Y"], d_old["keys"]):
                cache_vy[eval(k)] = (torch.from_numpy(v).float(), torch.from_numpy(y).float())

    all_c = cfg.observed_conditions(regime=2) + cfg.heldout_conditions(regime=2)
    for i, c in enumerate(all_c):
        k = cfg.condition_key(c)
        if k in cache_vy:
            continue
        x = corr.apply_condition(sample01, c["letters"], c["sevs"])
        feats, logits = forward_all(model, x)
        B = behavior_v2_from(feats, logits, sample_labels, ref_mean)
        y = (B - B0)[:cfg.BEHAVIOR_DIM]
        v = encoder_stats_from_feats(x, feats, clean_feat)
        cache_vy[k] = (v, y)
        if (i + 1) % 30 == 0:
            print(f"[build-pools] {i+1}/{len(all_c)} conditions", flush=True)
    torch.save(cache_vy, cache_path)

    for reg in (2, 3):
        obs_c = cfg.observed_conditions(regime=reg)
        held_c = cfg.heldout_conditions(regime=reg)
        np.savez(os.path.join(cfg.POOLS_DIR, f"observed_r{reg}.npz"),
                 V=torch.stack([cache_vy[cfg.condition_key(c)][0] for c in obs_c]).numpy(),
                 Y=torch.stack([cache_vy[cfg.condition_key(c)][1] for c in obs_c]).numpy(),
                 kinds=np.array([c["kind"] for c in obs_c]),
                 keys=np.array([repr(cfg.condition_key(c)) for c in obs_c]),
                 train_sample_idx=idx.numpy(), B0=B0.numpy())
        np.savez(os.path.join(cfg.POOLS_DIR, f"heldout_v_r{reg}.npz"),
                 V=torch.stack([cache_vy[cfg.condition_key(c)][0] for c in held_c]).numpy(),
                 keys=np.array([repr(cfg.condition_key(c)) for c in held_c]),
                 kinds=np.array([c["kind"] for c in held_c]))
        print(f"[build-pools] saved regime={reg}: {len(obs_c)} observed, {len(held_c)} held-out")

    # aux primitives (off-grid single shifts; saves both v and y)
    aux = cfg.aux_primitives()
    aux_cache = {}
    for letter, sev in aux:
        x = corr.apply_condition(sample01, (letter,), (sev,))
        feats, logits = forward_all(model, x)
        B = behavior_v2_from(feats, logits, sample_labels, ref_mean)
        y = (B - B0)[:cfg.BEHAVIOR_DIM]
        v = encoder_stats_from_feats(x, feats, clean_feat)
        aux_cache[(letter, round(float(sev), 6))] = (v, y)
    torch.save(aux_cache, os.path.join(cfg.POOLS_DIR, "aux_prim.pt"))
    print(f"[build-pools] aux off-grid primitives: {sorted(aux_cache.keys())}")
    print("[build-pools] DONE (both Regime II and Regime III pools ready)")


def _load_observed_and_cache(regime):
    d = np.load(os.path.join(cfg.POOLS_DIR, f"observed_r{regime}.npz"),
                allow_pickle=True)
    conds = []
    for v, y, kind, k in zip(d["V"], d["Y"], d["kinds"], d["keys"]):
        letters, sevs = eval(k)
        conds.append(dict(v=torch.from_numpy(v).float(),
                          y=torch.from_numpy(y).float(),
                          kind=str(kind), letters=letters, sevs=sevs))
    prim_cache = {}
    for c in conds:
        if c["kind"] == "prim":
            prim_cache[(c["letters"][0], round(float(c["sevs"][0]), 6))] = (c["v"], c["y"])
    aux_cache = torch.load(os.path.join(cfg.POOLS_DIR, "aux_prim.pt"), weights_only=False)
    for k, val in aux_cache.items():
        prim_cache[k] = val
    return conds, prim_cache


# ---------------------------------------------------------------------------
# P3: train methods (vision)
# ---------------------------------------------------------------------------
def phase_train_methods(regime=2):
    cfg.load_pilot_grids()
    conds, prim_cache = _load_observed_and_cache(regime)
    heldout = cfg.heldout_conditions(regime=regime)

    os.makedirs(os.path.join(cfg.RESULTS_DIR, "preds"), exist_ok=True)
    os.makedirs(os.path.join(cfg.RESULTS_DIR, "models"), exist_ok=True)
    for method in cfg.METHODS:
        t0 = time.time()
        preds, obs_preds, in_bpe = [], [], []
        for seed in cfg.SEEDS:
            tr = m2.train_method(method, conds, seed)
            in_bpe.append(tr.get("in_sample_bpe", float("nan")))
            p_h = torch.stack([m2.predict(tr, h["letters"], h["sevs"], prim_cache)
                               for h in heldout])
            p_o = torch.stack([m2.predict(tr, c["letters"], c["sevs"], prim_cache)
                               for c in conds])
            preds.append(p_h)
            obs_preds.append(p_o)
            if tr["kind"] == "trained":
                torch.save({"method": method, "state": {
                    "enc": tr["enc"].state_dict(), "dec": tr["dec"].state_dict(),
                    "comp": tr["comp"].state_dict()}},
                    os.path.join(cfg.RESULTS_DIR, "models",
                                 f"{method}_seed{seed}_r{regime}.pt"))
        np.savez(os.path.join(cfg.RESULTS_DIR, "preds", f"{method}_r{regime}.npz"),
                 heldout_pred=torch.stack(preds).numpy(),
                 obs_pred=torch.stack(obs_preds).numpy(),
                 heldout_keys=np.array([repr(cfg.condition_key(h)) for h in heldout]),
                 obs_keys=np.array([repr(cfg.condition_key(c)) for c in conds]),
                 in_sample_bpe=np.array(in_bpe))
        print(f"[train-methods] {method:<14} DONE ({time.time()-t0:.0f}s, "
              f"in-sample BPE={np.nanmean(in_bpe):.5f})", flush=True)
    print(f"[train-methods] DONE regime={regime}")


# ---------------------------------------------------------------------------
# P4a: synthetic -> vision transfer (E5, H6')
# ---------------------------------------------------------------------------
def phase_train_transfer(regime=2):
    cfg.load_pilot_grids()
    pre = {}
    s2_data = syn.build_tier("S2")
    for method in ["M_B", "M_A"]:
        tr = m2.train_method(method, s2_data["conds"], cfg.SEEDS[0],
                             in_dim=syn.ENCODER_IN_DIM)
        pre[method] = tr
        print(f"[transfer] pretrained {method} on S2 "
              f"(in-sample BPE={tr.get('in_sample_bpe', float('nan')):.5f}, "
              f"learned p={[round(x,2) for x in tr['comp']._p().tolist()]})", flush=True)

    conds, prim_cache = _load_observed_and_cache(regime)
    heldout = cfg.heldout_conditions(regime=regime)

    for method in ["M_B", "M_A"]:
        preds, obs_preds, in_bpe = [], [], []
        pre_comp = pre[method]["comp"].state_dict()
        for seed in cfg.SEEDS:
            m2.set_seed(seed)
            _ = m2.EncoderV2()
            fresh_dec = m2.DecoderV2().state_dict()
            fresh_comp = m2.make_comp(method).state_dict()
            # Transfer only the domain-invariant behavior-space parameters
            # (L_p exponent raw_p and the 5 behavior-column weights of the gate)
            blend_comp = dict(fresh_comp)
            blend_comp["raw_p"] = pre_comp["raw_p"].clone()
            for gk in ("gate_sym.0.weight", "gate_net.0.weight"):
                if gk in blend_comp and gk in pre_comp:
                    w = fresh_comp[gk].clone()
                    w[:, -cfg.BEHAVIOR_DIM:] = (
                        0.5 * w[:, -cfg.BEHAVIOR_DIM:] +
                        0.5 * pre_comp[gk][:, -cfg.BEHAVIOR_DIM:])
                    blend_comp[gk] = w
            tr = m2.train_method(method, conds, seed,
                                 init_state={"comp": blend_comp, "dec": fresh_dec})
            tr["method"] = method + "_pt"
            in_bpe.append(tr.get("in_sample_bpe", float("nan")))
            preds.append(torch.stack([m2.predict(tr, h2["letters"], h2["sevs"], prim_cache)
                                      for h2 in heldout]))
            obs_preds.append(torch.stack([m2.predict(tr, c2["letters"], c2["sevs"], prim_cache)
                                          for c2 in conds]))
            torch.save({"method": method + "_pt", "state": {
                "enc": tr["enc"].state_dict(), "dec": tr["dec"].state_dict(),
                "comp": tr["comp"].state_dict()}},
                os.path.join(cfg.RESULTS_DIR, "models",
                             f"{method}_pt_seed{seed}_r{regime}.pt"))
        np.savez(os.path.join(cfg.RESULTS_DIR, "preds", f"{method}_pt_r{regime}.npz"),
                 heldout_pred=torch.stack(preds).numpy(),
                 obs_pred=torch.stack(obs_preds).numpy(),
                 heldout_keys=np.array([repr(cfg.condition_key(h)) for h in heldout]),
                 obs_keys=np.array([repr(cfg.condition_key(c)) for c in conds]),
                 in_sample_bpe=np.array(in_bpe))
        print(f"[transfer] {method}_pt DONE (in-sample BPE={np.nanmean(in_bpe):.5f})",
              flush=True)
    print("[transfer] DONE")


# ---------------------------------------------------------------------------
# P4b: evaluate (vision) — ground truth + metrics + stats + decision rules
# ---------------------------------------------------------------------------
def phase_evaluate(regime=2):
    cfg.load_pilot_grids()
    model = load_base_model()
    pools = data_mod.get_pools()
    test01 = data_mod.to_float01(pools["test_images"])
    test_y = pools["test_labels"]
    clean_feat, clean_logits = forward_all(model, test01)
    ref_mean = clean_feat.mean(dim=0)
    B0 = behavior_v2_from(clean_feat, clean_logits, test_y, ref_mean)
    print(f"[evaluate] clean test: margin={B0[0]:.3f} acc={B0[4]*100:.2f}% "
          f"ece={B0[2]:.4f}")

    # ---- ground truth for ALL unique conditions across Regimes II & III (cached) ----
    obs_conds = cfg.observed_conditions(regime=regime)
    heldout = cfg.heldout_conditions(regime=regime)
    all_conds = obs_conds + heldout
    beh_csv = os.path.join(cfg.RESULTS_DIR, "behavior_testpool_all.csv")
    beh = {}
    if os.path.exists(beh_csv):
        with open(beh_csv) as f:
            for r in csv.DictReader(f):
                k = (tuple(r["letters"].split("+")),
                     tuple(round(float(x), 6) for x in r["sevs"].split(",")))
                beh[k] = torch.tensor(
                    [float(r["margin"]), float(r["drift"]), float(r["ece"]),
                     float(r["pcacc"]), float(r["acc"]), float(r["conf"])])
    missing = [c for c in all_conds if cfg.condition_key(c) not in beh]
    if missing:
        print(f"[evaluate] measuring {len(missing)} conditions on 10k test pool...",
              flush=True)
        for i, c in enumerate(missing):
            x = corr.apply_condition(test01, c["letters"], c["sevs"])
            feats, logits = forward_all(model, x)
            beh[cfg.condition_key(c)] = behavior_v2_from(feats, logits, test_y, ref_mean)
            if (i + 1) % 30 == 0:
                print(f"[evaluate] measured {i+1}/{len(missing)}", flush=True)
        with open(beh_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["letters", "sevs", "margin", "drift", "ece",
                        "pcacc", "acc", "conf"])
            for k, v in beh.items():
                w.writerow(["+".join(k[0]), ",".join(f"{s:.4f}" for s in k[1]),
                            *[f"{float(x):.6f}" for x in v]])
        print(f"[evaluate] behavior table saved ({len(beh)} conditions)")
    else:
        print(f"[evaluate] loaded cached behavior table ({len(beh)} conditions)")
    dB = {k: v - B0 for k, v in beh.items()}

    y_held = torch.stack([dB[cfg.condition_key(h)][:cfg.BEHAVIOR_DIM] for h in heldout])
    y_obs = torch.stack([dB[cfg.condition_key(c)][:cfg.BEHAVIOR_DIM] for c in obs_conds])
    held_keys = [cfg.condition_key(h) for h in heldout]
    obs_keys = [cfg.condition_key(c) for c in obs_conds]
    held_pos = {k: i for i, k in enumerate(held_keys)}
    obs_pos = {k: i for i, k in enumerate(obs_keys)}

    sev_idx = []
    for h in heldout:
        if h["kind"] == "offgrid":
            sev_idx.append(-1)
        else:
            grid0 = [round(float(g), 6) for g in cfg.SEVERITIES[h["letters"][0]]]
            s0 = round(float(h["sevs"][0]), 6)
            sev_idx.append(grid0.index(s0) if s0 in grid0 else -1)

    # ---- load predictions ----
    methods = cfg.METHODS + ["M_B_pt", "M_A_pt"]
    preds, obs_preds, in_bpe = {}, {}, {}
    for method in methods:
        path = os.path.join(cfg.RESULTS_DIR, "preds", f"{method}_r{regime}.npz")
        if not os.path.exists(path):
            continue
        d = np.load(path, allow_pickle=True)
        preds[method] = torch.from_numpy(d["heldout_pred"]).float()
        obs_preds[method] = torch.from_numpy(d["obs_pred"]).float()
        in_bpe[method] = d["in_sample_bpe"]

    # ---- oracle per method (direct encoding G(E(v_held)) across seeds) ----
    hd = np.load(os.path.join(cfg.POOLS_DIR, f"heldout_v_r{regime}.npz"),
                 allow_pickle=True)
    v_held = torch.from_numpy(hd["V"]).float()
    oracle = {}
    for method in preds:
        ob = []
        for seed in cfg.SEEDS:
            ck = os.path.join(cfg.RESULTS_DIR, "models",
                              f"{method}_seed{seed}_r{regime}.pt")
            if not os.path.exists(ck):
                continue
            sd = torch.load(ck, weights_only=False)["state"]
            enc = m2.EncoderV2(use_ln=(method != "M5_v1")); enc.load_state_dict(sd["enc"])
            dec = m2.DecoderV2(); dec.load_state_dict(sd["dec"])
            with torch.no_grad():
                o = dec(enc(v_held))
            ob.append(float(((o - y_held) ** 2)[:, :PRIMARY].mean()))
        if ob:
            oracle[method] = ob

    # ---- order test groups: observed (X,Y) vs held-out reverse (Y,X) ----
    pxy_idx, pyx_idx = [], []
    for pair in cfg.ORDER_TEST_PAIRS:
        rev = pair[::-1]
        ixy = [obs_pos[(pair, (round(float(cfg.SEVERITIES[pair[0]][i]), 6),
                               round(float(cfg.SEVERITIES[pair[1]][i]), 6)))]
               for i in range(cfg.N_SEV)]
        iyx = [held_pos[(rev, (round(float(cfg.SEVERITIES[rev[0]][i]), 6),
                               round(float(cfg.SEVERITIES[rev[1]][i]), 6)))]
               for i in range(cfg.N_SEV)]
        pxy_idx.append(ixy)
        pyx_idx.append(iyx)
    y_xy = torch.stack([y_obs[ix] for ix in pxy_idx])
    y_yx = torch.stack([y_held[ix] for ix in pyx_idx])

    # ---- metrics ----
    _, prim_cache = _load_observed_and_cache(regime)
    r3_triple_idx = [i for i, h in enumerate(heldout)
                     if h["kind"] == "triple" and h["letters"] not in cfg.REGIME3_OBSERVED_TRIPLES]
    rows = {}
    for method, p in preds.items():
        s = mt.summarize(p, y_held, in_bpe.get(method), oracle.get(method))
        if r3_triple_idx:
            se_tr = mt.bpe_primary_se(p[:, r3_triple_idx, :], y_held[r3_triple_idx])
            s["bpe_triples_common"] = float(se_tr.mean())
        pred_xy = torch.stack([obs_preds[method][:, ix, :] for ix in pxy_idx])
        pred_yx = torch.stack([p[:, ix, :] for ix in pyx_idx])
        s.update(mt.order_metrics(pred_xy, pred_yx, y_xy, y_yx))
        s["ire_pair"], s["ire_triple"] = _compute_ire(
            p, obs_preds[method], heldout, y_held, y_obs, held_pos, obs_pos)
        s["per_severity"] = mt.per_severity(p, y_held, sev_idx)
        rows[method] = s

    # theory diagnostics (seed 0)
    for method in ["M_A", "M_B", "M_C", "M5_v1", "M_B_full", "M_B_nores"]:
        if method not in preds:
            continue
        ck = os.path.join(cfg.RESULTS_DIR, "models",
                          f"{method}_seed{cfg.SEEDS[0]}_r{regime}.pt")
        if not os.path.exists(ck):
            continue
        sd = torch.load(ck, weights_only=False)["state"]
        enc = m2.EncoderV2(use_ln=(method != "M5_v1")); enc.load_state_dict(sd["enc"])
        dec = m2.DecoderV2(); dec.load_state_dict(sd["dec"])
        comp = m2.make_comp(method); comp.load_state_dict(sd["comp"])
        tr = dict(method=method, kind="trained", enc=enc, dec=dec, comp=comp)
        rows[method]["lipschitz"] = m2.lipschitz_estimate(tr, prim_cache)
        triples = [(h["letters"], h["sevs"]) for h in heldout if h["kind"] == "triple"]
        rows[method]["assoc_err"] = m2.associativity_error(tr, triples, prim_cache)

    # ---- print table ----
    print(f"\n[evaluate] ===== method summary (regime={regime}, mean over seeds) =====")
    hdr = f"{'method':<16} {'BPE':>8} {'accMAE':>7} {'rho_m':>6} {'rho_a':>6} {'CGG':>8} " \
          f"{'RCE':>7} {'sign':>5} {'commρ':>6} {'IREpr':>7} {'IREtr':>7} {'L':>6} {'assoc':>6}"
    print(hdr)
    for method, s in rows.items():
        print(f"{method:<16} {s['bpe_mean']:>8.4f} {s['acc_mae']:>7.4f} "
              f"{s['rank_rho_margin']:>6.3f} {s['rank_rho_acc']:>6.3f} "
              f"{s.get('cgg', float('nan')):>8.4f} {s.get('rce', float('nan')):>7.2f} "
              f"{s.get('sign_match', float('nan')):>5.2f} "
              f"{s.get('commutator_rho', float('nan')):>6.3f} "
              f"{s.get('ire_pair', float('nan')):>7.4f} "
              f"{s.get('ire_triple', float('nan')):>7.4f} "
              f"{s.get('lipschitz', float('nan')):>6.2f} "
              f"{s.get('assoc_err', float('nan')):>6.4f}")
    oracle_means = {k: float(np.mean(v)) for k, v in oracle.items()}
    if oracle_means:
        print(f"{'oracle (direct)':<16} {min(oracle_means.values()):>8.4f}")

    # ---- statistics ----
    se = {m: mt.bpe_primary_se(p, y_held) for m, p in preds.items()}
    stats_rows = []
    for ref in ["M0_meanpool", "M5_v1", "M1_behadd"]:
        if ref not in se:
            continue
        for method in ["M_A", "M_B", "M_C", "M_A_att", "M_B_pt", "M_A_pt"]:
            if method not in se or method == ref:
                continue
            # mt.paired_stats(se_ref, se_other) returns other_err - ref_err;
            # pass (se[method], se[ref]) so diff = ref_err - method_err (>0 => method BETTER)
            st = mt.paired_stats(se[method], se[ref])
            st["comparison"] = f"{method} vs {ref}"
            stats_rows.append(st)
    with open(os.path.join(cfg.RESULTS_DIR, f"stats_r{regime}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["comparison", "mean_diff_ref_minus_method", "ci_lo", "ci_hi", "p_value"])
        for st in stats_rows:
            w.writerow([st["comparison"], f"{st['mean_diff']:.6f}",
                        f"{st['ci_lo']:.6f}", f"{st['ci_hi']:.6f}",
                        f"{st['p_value']:.3e}"])

    # ---- decision rules ----
    rules = {}
    def rule(name, ok):
        rules[name] = "PASS" if ok else "FAIL"
        print(f"  {rules[name]}  {name}")

    def stat_win(method, ref):
        for s in stats_rows:
            if s["comparison"] == f"{method} vs {ref}":
                return s["mean_diff"] > 0 and s["p_value"] < 0.05
        return False

    print(f"\n[evaluate] ===== decision rules (PLAN_V2 §3, regime={regime}) =====")
    for M in ["M_A", "M_B"]:
        if M in rows:
            rule(f"H1'_{M}_beats_M0_and_M5",
                 stat_win(M, "M0_meanpool") and stat_win(M, "M5_v1"))
            rule(f"H1'_{M}_rank_rho_margin_ge_0.8",
                 rows[M]["rank_rho_margin"] >= 0.80)
    if "M_A" in rows and "M1_behadd" in rows:
        rule("H2'_M_A_beats_behavior_additive", stat_win("M_A", "M1_behadd"))
        rule("H2'_M_B_beats_behavior_additive", stat_win("M_B", "M1_behadd"))
        rule("H2'_M_A_IRE_beats_additive",
             rows["M_A"]["ire_triple"] < rows["M1_behadd"]["ire_triple"])
    if "M_B" in rows:
        rule("H3'_M_B_sign_match_ge_0.8",
             rows["M_B"].get("sign_match", 0.0) >= 0.80)
        rule("H3'_M_B_commutator_rho_ge_0.7",
             (rows["M_B"].get("commutator_rho") or 0.0) >= 0.70)
        if "M0_meanpool" in rows:
            rule("H3'_M_B_order_err_below_commutative",
                 rows["M_B"].get("behavior_order_err", 9.0) <
                 rows["M0_meanpool"].get("behavior_order_err", 9.0))
    if "M_B" in rows and "M_B_nobeh" in rows:
        st_h4 = mt.paired_stats(se["M_B"], se["M_B_nobeh"])
        rule("H4'_behavior_supervision_helps (M_B < M_B_nobeh)",
             st_h4["mean_diff"] > 0 and st_h4["p_value"] < 0.05)
    if "M_B_pt" in rows and "M_B" in rows:
        st_h6_b = mt.paired_stats(se["M_B_pt"], se["M_B"])
        st_h6_a = mt.paired_stats(se["M_A_pt"], se["M_A"]) if "M_A_pt" in se else st_h6_b
        rule("H6'_transfer_competitive_or_better_and_rank_rho_ge_0.7",
             rows["M_B_pt"]["rank_rho_margin"] >= 0.70 and
             (st_h6_b["ci_hi"] >= 0.0 or st_h6_a["mean_diff"] >= 0.0))
    if regime == 3 and os.path.exists(os.path.join(cfg.RESULTS_DIR, "summary_r2.json")):
        s_r2 = json.load(open(os.path.join(cfg.RESULTS_DIR, "summary_r2.json")))["methods"]
        for M in ["M_A", "M_B"]:
            if M in rows and M in s_r2:
                rule(f"H5'_Regime3_improves_{M}_triple_BPE_over_Regime2",
                     rows[M].get("bpe_triples_common", 9.0) <
                     s_r2[M].get("bpe_triples_common", 0.0))
    if "MEMO_control" in rows and "M_B" in rows:
        rule("control_MEMO_fails_substantively",
             rows["MEMO_control"]["bpe_mean"] > 1.5 * rows["M_B"]["bpe_mean"])
    if "PERM_control" in rows and "M_B" in rows:
        rule("control_PERM_fails",
             rows["PERM_control"]["bpe_mean"] > 1.5 * rows["M_B"]["bpe_mean"])
    if oracle_means:
        best_o = min(oracle_means.values())
        rule("oracle_is_upper_bound",
             all(rows[m]["bpe_mean"] >= best_o * 0.9 for m in preds))

    out = dict(regime=regime, clean_test_acc=float(B0[4]),
               clean_margin=float(B0[0]), methods=rows, stats=stats_rows,
               rules=rules, oracle=oracle_means)
    with open(os.path.join(cfg.RESULTS_DIR, f"summary_r{regime}.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    with open(os.path.join(cfg.RESULTS_DIR, f"metrics_r{regime}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "bpe_mean", "bpe_std", "acc_mae", "rank_rho_margin",
                    "rank_rho_acc", "cgg", "rce", "sign_match", "commutator_rho",
                    "behavior_order_err", "ire_pair", "ire_triple", "lipschitz",
                    "assoc_err"])
        for method, s in rows.items():
            w.writerow([method, f"{s['bpe_mean']:.6f}", f"{s['bpe_std']:.6f}",
                        f"{s['acc_mae']:.6f}", f"{s['rank_rho_margin']:.6f}",
                        f"{s['rank_rho_acc']:.6f}", f"{s.get('cgg', float('nan')):.6f}",
                        f"{s.get('rce', float('nan')):.6f}",
                        f"{s.get('sign_match', float('nan')):.6f}",
                        f"{s.get('commutator_rho', float('nan')):.6f}",
                        f"{s.get('behavior_order_err', float('nan')):.6f}",
                        f"{s.get('ire_pair', float('nan')):.6f}",
                        f"{s.get('ire_triple', float('nan')):.6f}",
                        f"{s.get('lipschitz', float('nan')):.6f}",
                        f"{s.get('assoc_err', float('nan')):.6f}"])
    print(f"[evaluate] DONE regime={regime}")


# ---------------------------------------------------------------------------
# P5: audit
# ---------------------------------------------------------------------------
def phase_audit():
    cfg.load_pilot_grids()
    checks = []

    def check(name, cond, detail=""):
        checks.append((name, "PASS" if cond else "FAIL", detail))
        print(f"[audit] {'PASS' if cond else 'FAIL'}  {name}  {detail}")

    observed = cfg.observed_conditions(regime=2)
    heldout = cfg.heldout_conditions(regime=2)
    obs_keys = {cfg.condition_key(c) for c in observed}
    held_keys = {cfg.condition_key(c) for c in heldout}
    # 1) No observed condition is a superset of any held-out condition
    superset_violations = [
        (h["letters"], o["letters"]) for h in heldout for o in observed
        if set(h["letters"]) < set(o["letters"])
    ]
    check("contamination_split_disjoint",
          len(superset_violations) == 0 and obs_keys.isdisjoint(held_keys),
          f"held={len(held_keys)} observed={len(obs_keys)} violations={superset_violations}")
    # 2) Pool file matches config exactly and contains 518-dim unlabeled stats only
    d = np.load(os.path.join(cfg.POOLS_DIR, "observed_r2.npz"), allow_pickle=True)
    pool_keys = {eval(k) for k in d["keys"]}
    check("pools_contain_only_observed",
          pool_keys == obs_keys and pool_keys.isdisjoint(held_keys) and d["V"].shape[1] == 518,
          f"shape={d['V'].shape} overlap={pool_keys & held_keys}")
    # 3) Corruption checks
    pools = data_mod.get_pools()
    x = data_mod.to_float01(pools["test_images"][:64])
    x1 = corr.apply_condition(x, ("B", "C"), (0.7, 0.06))
    x2 = corr.apply_condition(x, ("B", "C"), (0.7, 0.06))
    check("corruption_deterministic", torch.equal(x1, x2))
    check("noise_seed_order_independent",
          corr.condition_seed(("B", "C"), (0.7, 0.06)) ==
          corr.condition_seed(("C", "B"), (0.06, 0.7)))
    x_bc = corr.apply_condition(x, ("B", "C"), (0.7, 0.06))
    x_cb = corr.apply_condition(x, ("C", "B"), (0.06, 0.7))
    check("BC_noncommutative", not torch.allclose(x_bc, x_cb, atol=1e-4),
          f"maxdiff={(x_bc - x_cb).abs().max().item():.4f}")
    check("clip_bounds", all(
        0.0 <= float(corr.apply_condition(x, (l,), (cfg.SEVERITIES[l][-1],)).min()) and
        float(corr.apply_condition(x, (l,), (cfg.SEVERITIES[l][-1],)).max()) <= 1.0
        for l in cfg.PRIM_LETTERS))
    # 4) Controls & oracle checks
    sp = os.path.join(cfg.RESULTS_DIR, "summary_r2.json")
    if os.path.exists(sp):
        sm = json.load(open(sp))
        s = sm["methods"]
        check("control_MEMO_fails_substantively",
              s["MEMO_control"]["bpe_mean"] > 1.5 * s["M_B"]["bpe_mean"],
              f"MEMO={s['MEMO_control']['bpe_mean']:.4f} vs M_B={s['M_B']['bpe_mean']:.4f}")
        check("control_PERM_fails",
              s["PERM_control"]["bpe_mean"] > 1.5 * s["M_B"]["bpe_mean"],
              f"PERM={s['PERM_control']['bpe_mean']:.4f} vs M_B={s['M_B']['bpe_mean']:.4f}")
        best_o = min(sm["oracle"].values())
        check("oracle_is_upper_bound",
              all(s[m]["bpe_mean"] >= best_o * 0.9 for m in s),
              f"best_oracle={best_o:.4f}")
    # 5) Exact seed reproducibility + finite off-grid predictions
    conds, prim_cache = _load_observed_and_cache(2)
    tr1 = m2.train_method("M_B", conds, cfg.SEEDS[0])
    tr2 = m2.train_method("M_B", conds, cfg.SEEDS[0])
    p1 = torch.stack([m2.predict(tr1, h["letters"], h["sevs"], prim_cache) for h in heldout])
    p2 = torch.stack([m2.predict(tr2, h["letters"], h["sevs"], prim_cache) for h in heldout])
    check("seed_reproducibility", torch.equal(p1, p2),
          f"maxdiff={(p1 - p2).abs().max().item():.2e}")
    off_idx = [i for i, h in enumerate(heldout) if h["kind"] == "offgrid"]
    check("offgrid_predictions_finite",
          bool(torch.isfinite(p1[off_idx]).all()),
          f"n_offgrid={len(off_idx)}")
    # 6) Exact associativity of M_B operator product
    triples = [(h["letters"], h["sevs"]) for h in heldout if h["kind"] == "triple"]
    assoc = m2.associativity_error(tr1, triples, prim_cache)
    check("M_B_associative_by_construction", assoc < 1e-5, f"assoc_err={assoc:.2e}")

    with open(os.path.join(cfg.RESULTS_DIR, "audit_report_v2.json"), "w") as f:
        json.dump([{"check": n, "result": r, "detail": dt} for n, r, dt in checks],
                  f, indent=2)
    n_fail = sum(1 for _, r, _ in checks if r == "FAIL")
    print(f"[audit] {'ALL PASS' if n_fail == 0 else f'{n_fail} FAILED'} "
          f"({len(checks)} checks)")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def phase_report():
    sp2 = os.path.join(cfg.RESULTS_DIR, "summary_r2.json")
    if not os.path.exists(sp2):
        print("[report] no summary_r2.json; run evaluate first")
        return
    s2 = json.load(open(sp2))
    lines = []
    out = lines.append
    out("=" * 86)
    out("SHIFT ALGEBRA V2 — EXPERIMENT RESULTS (Benchmark V2, CIFAR-10 + Synthetic S1-S3)")
    out("=" * 86)
    out(f"base model: ResNet-18 w0.25, clean test acc = {s2['clean_test_acc']*100:.2f}%  "
        f"margin = {s2['clean_margin']:.3f}")
    out("V2 split (Regime II): 6 primitives x5 sev + 25 ordered pair series x5 sev")
    out("  observed (155); held out: 20 triples x5 + 4 chord reversed pairs x5 + 3 off-grid (123)")
    out("")

    syn_p = os.path.join(cfg.RESULTS_DIR, "synthetic_summary.json")
    if os.path.exists(syn_p):
        sy = json.load(open(syn_p))
        out("-" * 86)
        out("SYNTHETIC TIERS S1-S3 (Primary Mechanism Tests, 5 seeds)")
        out("-" * 86)
        out(f"{'method':<16} {'S1 BPE':>9} {'S2 BPE':>9} {'S3 BPE':>9} "
            f"{'S3 sign':>8} {'S3 commρ':>9} {'S2 IREtr':>9}")
        for m in ["M0_meanpool", "M1_behadd", "M_A", "M_A_nointer", "M_A_att",
                  "M_B", "M_C", "M_C_nopos", "M5_v1", "M1_emb",
                  "MEMO_control", "PERM_control"]:
            if m not in sy.get("S1", {}):
                continue
            out(f"{m:<16} {sy['S1'][m]['bpe_mean']:>9.4f} "
                f"{sy['S2'][m]['bpe_mean']:>9.4f} "
                f"{sy['S3'][m]['bpe_mean']:>9.4f} "
                f"{sy['S3'][m].get('sign_match', float('nan')):>8.2f} "
                f"{sy['S3'][m].get('commutator_rho', float('nan')):>9.3f} "
                f"{sy['S2'][m].get('ire_triple', float('nan')):>9.4f}")
        out("")

    out("-" * 86)
    out("VISION BENCHMARK V2 — REGIME II SUMMARY (mean over 5 seeds, 123 held-out conditions)")
    out("-" * 86)
    out(f"{'method':<16} {'BPE':>8} {'accMAE':>7} {'rho_m':>6} {'rho_a':>6} "
        f"{'sign':>5} {'commρ':>6} {'IREpr':>7} {'IREtr':>7} {'L':>6} {'assoc':>6}")
    out("-" * 86)
    for method, r in s2["methods"].items():
        out(f"{method:<16} {r['bpe_mean']:>8.4f} {r['acc_mae']:>7.4f} "
            f"{r['rank_rho_margin']:>6.3f} {r['rank_rho_acc']:>6.3f} "
            f"{r.get('sign_match', float('nan')):>5.2f} "
            f"{r.get('commutator_rho', float('nan')):>6.3f} "
            f"{r.get('ire_pair', float('nan')):>7.4f} "
            f"{r.get('ire_triple', float('nan')):>7.4f} "
            f"{r.get('lipschitz', float('nan')):>6.2f} "
            f"{r.get('assoc_err', float('nan')):>6.4f}")
    if s2["oracle"]:
        out(f"{'oracle (direct)':<16} {min(s2['oracle'].values()):>8.4f}")
    out("")

    sp3 = os.path.join(cfg.RESULTS_DIR, "summary_r3.json")
    if os.path.exists(sp3):
        s3 = json.load(open(sp3))
        out("-" * 86)
        out("REGIME III (Higher-Order Supervision: +6 triples observed, 70 common held-out triples) — H5'")
        out("-" * 86)
        for m in ["M_A", "M_B", "M_C", "M0_meanpool", "M5_v1"]:
            if m in s3["methods"] and m in s2["methods"]:
                t2 = s2["methods"][m].get("bpe_triples_common", s2["methods"][m]["bpe_mean"])
                t3 = s3["methods"][m].get("bpe_triples_common", s3["methods"][m]["bpe_mean"])
                out(f"  {m:<14} Common-70-Triple BPE: Regime II={t2:.4f} -> "
                    f"Regime III={t3:.4f}  (overall R3 BPE={s3['methods'][m]['bpe_mean']:.4f}, "
                    f"rho_m={s3['methods'][m]['rank_rho_margin']:.3f})")
        out("")

    out("-" * 86)
    out("STATISTICAL COMPARISONS (paired Wilcoxon; diff = ref_err - method_err;")
    out("POSITIVE diff = method BETTER than ref)")
    out("-" * 86)
    for st in s2["stats"]:
        out(f"{st['comparison']:<28} diff={st['mean_diff']:+.4f} "
            f"CI=[{st['ci_lo']:+.4f},{st['ci_hi']:+.4f}] p={st['p_value']:.3e}")
    out("")
    out("-" * 86)
    out("DECISION RULES (PLAN_V2 §3)")
    out("-" * 86)
    for name, res in s2["rules"].items():
        out(f"  {res}  {name}")
    if os.path.exists(sp3):
        s3 = json.load(open(sp3))
        for name, res in s3["rules"].items():
            if name.startswith("H5'"):
                out(f"  {res}  {name}")
    txt = "\n".join(lines)
    with open(os.path.join(cfg.RESULTS_DIR, "REPORT.txt"), "w") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["pilot-severities", "synthetic", "build-pools",
                                      "train-methods", "train-transfer", "evaluate",
                                      "audit", "report"])
    ap.add_argument("--regime", type=int, default=2)
    args = ap.parse_args()
    if args.phase == "pilot-severities":
        phase_pilot_severities()
    elif args.phase == "synthetic":
        phase_synthetic()
    elif args.phase == "build-pools":
        phase_build_pools(args.regime)
    elif args.phase == "train-methods":
        phase_train_methods(args.regime)
    elif args.phase == "train-transfer":
        phase_train_transfer(args.regime)
    elif args.phase == "evaluate":
        phase_evaluate(args.regime)
    elif args.phase == "audit":
        phase_audit()
    elif args.phase == "report":
        phase_report()
