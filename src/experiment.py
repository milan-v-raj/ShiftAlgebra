"""Experiment pipeline. Phases (run in order):

  prepare-data   verify CIFAR-10 md5s, build + cache tensors
  train-base     train the frozen base model (long; run in background)
  build-pools    encoder inputs + behavior labels for the 45 observed
                 conditions (train sample)  [needs base model]
  train-methods  train all methods x seeds; zero-shot held-out predictions
  evaluate       ground-truth behavior on the full test pool (all 78
                 conditions), oracle stats, metrics, statistics, decision rules
  audit          contamination / leakage / correctness / control checks

Usage: python -m src.experiment <phase>
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch

from . import config as cfg
from . import data as data_mod
from . import corruptions as corr
from .models import resnet18, count_params
from . import shift_algebra as sa

torch.set_num_threads(2)

os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(cfg.POOLS_DIR, exist_ok=True)


# ----------------------------------------------------------------------------
# behavior measurement
# ----------------------------------------------------------------------------
@torch.no_grad()
def forward_all(model, x01, batch_size=256):
    """Returns (features (N,F), logits (N,10)) for float [0,1] images."""
    feats, logits = [], []
    for i in range(0, len(x01), batch_size):
        xb = data_mod.normalize(x01[i:i + batch_size])
        f = model.features(xb)
        feats.append(f)
        logits.append(model.fc(f))
    return torch.cat(feats), torch.cat(logits)


def behavior_from(features, logits, labels, ref_feat_mean):
    """5-dim behavior vector: [acc, per-class acc, ECE, feature drift, mean conf]."""
    probs = torch.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    labels = labels.long()
    acc = (pred == labels).float().mean().item()
    per_class = []
    for c in range(10):
        m = labels == c
        if m.any():
            per_class.append((pred[m] == c).float().mean().item())
    pcacc = float(np.nanmean(per_class))
    ece = 0.0
    for b in range(10):                                   # 10 equal-width bins
        lo, hi = b / 10.0, (b + 1) / 10.0
        m = (conf >= lo) & (conf < hi)
        if m.any():
            ece += m.float().mean().item() * abs(
                (pred[m] == labels[m]).float().mean().item() - conf[m].mean().item())
    drift = (features.mean(dim=0) - ref_feat_mean).norm().item()
    return torch.tensor([acc, pcacc, ece, drift, conf.mean().item()],
                        dtype=torch.float32)


# ----------------------------------------------------------------------------
# phase: prepare-data
# ----------------------------------------------------------------------------
def phase_prepare_data():
    tr_x, tr_y, te_x, te_y = data_mod.load_batches(verify=True)
    print(f"[prepare-data] train {tuple(tr_x.shape)} test {tuple(te_x.shape)}")
    print(f"[prepare-data] train label counts: {torch.bincount(tr_y).tolist()}")
    print(f"[prepare-data] test  label counts: {torch.bincount(te_y).tolist()}")
    print("[prepare-data] OK")


# ----------------------------------------------------------------------------
# phase: train-base
# ----------------------------------------------------------------------------
class CifarTrain(torch.utils.data.Dataset):
    def __init__(self, images, labels, seed):
        self.images = images.numpy()
        self.labels = labels.numpy()
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = self.images[i]                                   # (3,32,32) uint8
        p = np.pad(img, ((0, 0), (4, 4), (4, 4)), mode="constant")
        oy, ox = self.rng.randint(0, 9, size=2)
        img = p[:, oy:oy + 32, ox:ox + 32]
        if self.rng.rand() < 0.5:
            img = img[:, :, ::-1]
        x = torch.from_numpy(np.ascontiguousarray(img)).float() / 255.0
        return data_mod.normalize(x.unsqueeze(0)).squeeze(0), int(self.labels[i])


class CifarEval(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        x = self.images[i].float() / 255.0
        return data_mod.normalize(x.unsqueeze(0)).squeeze(0), int(self.labels[i])


def evaluate_base(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb).argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += len(yb)
    return correct / total


def phase_train_base():
    pools = data_mod.get_pools()
    sa.set_seed(cfg.BASE_MODEL["seed"])
    model = resnet18(width=cfg.BASE_MODEL["width"])
    print(f"[train-base] ResNet-18 width={cfg.BASE_MODEL['width']} "
          f"params={count_params(model)/1e6:.2f}M (Amendment 1)", flush=True)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.BASE_MODEL["lr"],
                          momentum=cfg.BASE_MODEL["momentum"],
                          weight_decay=cfg.BASE_MODEL["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.BASE_MODEL["epochs"])
    train_ds = CifarTrain(pools["train_images"], pools["train_labels"],
                          cfg.BASE_MODEL["seed"])
    train_ld = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.BASE_MODEL["batch_size"], shuffle=True,
        num_workers=0, drop_last=False)
    test_ld = torch.utils.data.DataLoader(
        CifarEval(pools["test_images"], pools["test_labels"]), batch_size=256,
        shuffle=False, num_workers=0)

    log = {"epochs": [], "test_acc": [], "epoch_seconds": []}
    best_acc, best_state = -1.0, None
    for epoch in range(cfg.BASE_MODEL["epochs"]):
        t0 = time.time()
        model.train()
        running = 0.0
        for xb, yb in train_ld:
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item()
        sched.step()
        dt = time.time() - t0
        msg = (f"[train-base] epoch {epoch+1}/{cfg.BASE_MODEL['epochs']} "
               f"loss={running/len(train_ld):.4f} time={dt:.0f}s")
        if (epoch + 1) % 5 == 0 or epoch == cfg.BASE_MODEL["epochs"] - 1:
            acc = evaluate_base(model, test_ld)
            msg += f" test_acc={acc*100:.2f}%"
            log["epochs"].append(epoch + 1)
            log["test_acc"].append(acc)
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        log["epoch_seconds"].append(dt)
        print(msg, flush=True)

    final_acc = evaluate_base(model, test_ld)
    torch.save({"state_dict": model.state_dict(), "test_acc": final_acc,
                "width": cfg.BASE_MODEL["width"], "seed": cfg.BASE_MODEL["seed"],
                "kind": "final"}, os.path.join(cfg.CHECKPOINT_DIR, "base_model.pt"))
    torch.save({"state_dict": best_state, "test_acc": best_acc,
                "width": cfg.BASE_MODEL["width"], "seed": cfg.BASE_MODEL["seed"],
                "kind": "best"}, os.path.join(cfg.CHECKPOINT_DIR, "base_model_best.pt"))
    log["final_test_acc"] = final_acc
    log["best_test_acc"] = best_acc
    with open(os.path.join(cfg.CHECKPOINT_DIR, "base_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print(f"[train-base] DONE final test acc={final_acc*100:.2f}% "
          f"best={best_acc*100:.2f}%")


def load_base_model(best=True):
    name = "base_model_best.pt" if best else "base_model.pt"
    ckpt = torch.load(os.path.join(cfg.CHECKPOINT_DIR, name), weights_only=False)
    model = resnet18(width=ckpt["width"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"[base] loaded {name} (test_acc={ckpt['test_acc']*100:.2f}%)")
    return model


# ----------------------------------------------------------------------------
# phase: build-pools
# ----------------------------------------------------------------------------
def encoder_input_from(model, clean_sample01, clean_feat, cond):
    """v (518,) for a condition: feature stats + pixel stats + feature deltas.
    Computed from the TRAIN sample only (leakage boundary)."""
    x = corr.apply_condition(clean_sample01, cond["letters"], cond["sevs"])
    feats, _ = forward_all(model, x)
    delta = feats - clean_feat
    pix_mean = x.mean(dim=(0, 2, 3))
    pix_std = x.std(dim=(0, 2, 3))
    v = torch.cat([
        feats.mean(dim=0), feats.std(dim=0),
        pix_mean, pix_std,
        delta.mean(dim=0), delta.std(dim=0),
    ])
    return v, feats


def phase_build_pools():
    model = load_base_model(best=True)
    pools = data_mod.get_pools()
    idx = pools["train_sample_idx"]
    sample = pools["train_images"][idx]                       # (2000,3,32,32) uint8
    sample01 = data_mod.to_float01(sample)
    sample_labels = pools["train_labels"][idx]
    clean_feat, clean_logits = forward_all(model, sample01)
    ref_mean = clean_feat.mean(dim=0)
    B0 = behavior_from(clean_feat, clean_logits, sample_labels, ref_mean)
    print(f"[build-pools] train-sample clean behavior: acc={B0[0]*100:.2f}% "
          f"ece={B0[2]:.4f} conf={B0[4]:.4f}")

    conds = cfg.observed_conditions()
    Vs, Ys, kinds, keys = [], [], [], []
    for c in conds:
        v, feats = encoder_input_from(model, sample01, clean_feat, c)
        x = corr.apply_condition(sample01, c["letters"], c["sevs"])
        _, logits = forward_all(model, x)
        B = behavior_from(feats, logits, sample_labels, ref_mean)
        Vs.append(v)
        Ys.append(B - B0)
        kinds.append(c["kind"])
        keys.append(cfg.condition_key(c))
        if len(Vs) % 10 == 0:
            print(f"[build-pools] {len(Vs)}/{len(conds)} conditions "
                  f"(last acc drop={float(B[0]-B0[0])*100:+.2f}pp)", flush=True)

    np.savez(os.path.join(cfg.POOLS_DIR, "observed.npz"),
             V=torch.stack(Vs).numpy(), Y=torch.stack(Ys).numpy(),
             kinds=np.array(kinds),
             keys=np.array([repr(k) for k in keys]),
             train_sample_idx=idx.numpy(),
             B0=B0.numpy())

    # auxiliary primitives at off-grid severities (evaluation-only inputs;
    # single shifts are not held-out compositions)
    aux = cfg.aux_primitives()
    aux_v = {}
    for letter, sev in aux:
        x = corr.apply_condition(sample01, (letter,), (sev,))
        feats, _ = forward_all(model, x)
        delta = feats - clean_feat
        v = torch.cat([feats.mean(dim=0), feats.std(dim=0),
                       x.mean(dim=(0, 2, 3)), x.std(dim=(0, 2, 3)),
                       delta.mean(dim=0), delta.std(dim=0)])
        aux_v[(letter, round(float(sev), 6))] = v
    torch.save(aux_v, os.path.join(cfg.POOLS_DIR, "aux_prim.pt"))
    print(f"[build-pools] aux off-grid primitives: {sorted(aux_v.keys())}")

    with open(os.path.join(cfg.POOLS_DIR, "meta.json"), "w") as f:
        json.dump({"n_observed": len(conds), "train_sample": cfg.TRAIN_SAMPLE,
                   "B0": B0.tolist()}, f, indent=2)
    print(f"[build-pools] DONE: {len(conds)} observed conditions -> "
          f"{cfg.POOLS_DIR}/observed.npz")


# ----------------------------------------------------------------------------
# phase: train-methods
# ----------------------------------------------------------------------------
def phase_train_methods():
    d = np.load(os.path.join(cfg.POOLS_DIR, "observed.npz"), allow_pickle=True)
    V, Y, kinds = torch.from_numpy(d["V"]), torch.from_numpy(d["Y"]), d["kinds"]
    keys = [eval(k) for k in d["keys"]]
    conds = []
    for i, k in enumerate(keys):
        conds.append(dict(letters=k[0], sevs=k[1], kind=str(kinds[i]),
                          v=V[i], y=Y[i]))
    prim_v = {}
    for c in conds:
        if c["kind"] == "prim":
            prim_v[(c["letters"][0], round(float(c["sevs"][0]), 6))] = c["v"]
    # evaluation-only auxiliary primitives at off-grid severities
    aux_v = torch.load(os.path.join(cfg.POOLS_DIR, "aux_prim.pt"), weights_only=False)
    prim_v.update(aux_v)
    prim_keys = [cfg.condition_key(c) for c in conds if c["kind"] == "prim"]
    pair_keys = [cfg.condition_key(c) for c in conds if c["kind"] == "pair"]
    prim_mask = torch.tensor([k == "prim" for k in kinds])
    pair_mask = torch.tensor([k == "pair" for k in kinds])

    heldout = cfg.heldout_conditions()
    os.makedirs(os.path.join(cfg.RESULTS_DIR, "preds"), exist_ok=True)
    os.makedirs(os.path.join(cfg.RESULTS_DIR, "models"), exist_ok=True)

    for method in cfg.METHODS:
        t0 = time.time()
        held_preds, held_zs, prim_preds, pair_preds, in_bpes = [], [], [], [], []
        for seed in cfg.SEEDS:
            if method == "MEMO_control":
                tr = sa.train_memorization_control(conds, seed)
                pred = tr["heldout_pred"].unsqueeze(0).expand(len(heldout), -1)
                z = tr["heldout_emb"].unsqueeze(0).expand(len(heldout), -1)
            elif method == "PERM_control":
                tr = sa.train_permutation_control(conds, seed)
                preds, zs = [], []
                for h in heldout:
                    p, z = sa.predict_condition(tr, h["letters"], h["sevs"], prim_v)
                    preds.append(p)
                    zs.append(z)
                pred, z = torch.stack(preds), torch.stack(zs)
            else:
                tr = sa.train_composition_model(method, conds, seed)
                preds, zs = [], []
                for h in heldout:
                    p, z = sa.predict_condition(tr, h["letters"], h["sevs"], prim_v)
                    preds.append(p)
                    zs.append(z)
                pred, z = torch.stack(preds), torch.stack(zs)
                if method == "M5_shift_algebra":
                    torch.save({"encoder": tr["encoder"].state_dict(),
                                "decoder": tr["decoder"].state_dict(),
                                "comp": tr["comp"].state_dict()},
                               os.path.join(cfg.RESULTS_DIR, "models",
                                            f"M5_seed{seed}.pt"))
            in_pred = tr["in_sample_pred"]
            in_bpe = (in_pred - Y).pow(2).mean().item()
            in_bpes.append(in_bpe)
            held_preds.append(pred)
            held_zs.append(z)
            prim_preds.append(in_pred[prim_mask])
            pair_preds.append(in_pred[pair_mask])
            print(f"[train-methods] {method} seed={seed} in-sample BPE={in_bpe:.6f}",
                  flush=True)
        np.savez(os.path.join(cfg.RESULTS_DIR, "preds", f"{method}.npz"),
                 heldout_pred=torch.stack(held_preds).numpy(),
                 heldout_z=torch.stack(held_zs).numpy(),
                 prim_pred=torch.stack(prim_preds).numpy(),
                 pair_pred=torch.stack(pair_preds).numpy(),
                 in_sample_bpe=np.array(in_bpes),
                 heldout_keys=np.array([repr(cfg.condition_key(h)) for h in heldout]),
                 prim_keys=np.array([repr(k) for k in prim_keys]),
                 pair_keys=np.array([repr(k) for k in pair_keys]))
        print(f"[train-methods] {method} DONE ({time.time()-t0:.0f}s)", flush=True)


# ----------------------------------------------------------------------------
# phase: evaluate
# ----------------------------------------------------------------------------
def phase_evaluate():
    from scipy import stats as sps

    model = load_base_model(best=True)
    pools = data_mod.get_pools()
    test01 = data_mod.to_float01(pools["test_images"])
    test_labels = pools["test_labels"]
    clean_feat, clean_logits = forward_all(model, test01)
    ref_mean = clean_feat.mean(dim=0)
    B0_test = behavior_from(clean_feat, clean_logits, test_labels, ref_mean)
    print(f"[evaluate] clean test behavior: acc={B0_test[0]*100:.2f}% "
          f"ece={B0_test[2]:.4f} conf={B0_test[4]:.4f}")

    # ---- ground-truth behavior for ALL conditions on the full test pool ----
    # (cached: the table is deterministic, so reruns of evaluate reuse it)
    all_conds = cfg.observed_conditions() + cfg.heldout_conditions()
    beh_csv = os.path.join(cfg.RESULTS_DIR, "behavior_testpool.csv")
    beh = None
    if os.path.exists(beh_csv):
        with open(beh_csv) as f:
            _rows = list(csv.DictReader(f))
        if len(_rows) == len(all_conds):
            beh = {}
            for c, r in zip(all_conds, _rows):
                beh[cfg.condition_key(c)] = torch.tensor(
                    [float(r["acc"]), float(r["per_class_acc"]), float(r["ece"]),
                     float(r["feature_drift"]), float(r["mean_conf"])])
            print(f"[evaluate] loaded cached behavior table ({len(beh)} conditions)")
    if beh is None:
        beh = {}
        for i, c in enumerate(all_conds):
            x = corr.apply_condition(test01, c["letters"], c["sevs"])
            feats, logits = forward_all(model, x)
            beh[cfg.condition_key(c)] = behavior_from(feats, logits, test_labels, ref_mean)
            if (i + 1) % 10 == 0:
                print(f"[evaluate] measured {i+1}/{len(all_conds)} conditions", flush=True)
        with open(beh_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["letters", "sevs", "kind", "acc", "per_class_acc", "ece",
                        "feature_drift", "mean_conf", "d_acc", "d_per_class_acc",
                        "d_ece", "d_drift", "d_conf"])
            for c in all_conds:
                k = cfg.condition_key(c)
                w.writerow(["+".join(k[0]), ",".join(f"{s:.3f}" for s in k[1]), c["kind"],
                            *[f"{float(x):.6f}" for x in beh[k]],
                            *[f"{float(x):.6f}" for x in beh[k] - B0_test]])
        print(f"[evaluate] behavior table saved ({len(all_conds)} conditions)")
    dB = {k: v - B0_test for k, v in beh.items()}

    # ---- oracle encoder inputs for held-out conditions (train sample) ----
    idx = torch.from_numpy(np.load(os.path.join(cfg.POOLS_DIR, "observed.npz"),
                                    allow_pickle=True)["train_sample_idx"])
    sample01 = data_mod.to_float01(pools["train_images"][idx])
    s_feat, _ = forward_all(model, sample01)
    heldout = cfg.heldout_conditions()
    v_held = {}
    for h in heldout:
        v, _ = encoder_input_from(model, sample01, s_feat, h)
        v_held[cfg.condition_key(h)] = v
    print(f"[evaluate] oracle encoder inputs computed for {len(v_held)} held-out conditions")

    # ---- load method predictions ----
    held_keys = [cfg.condition_key(h) for h in heldout]
    dB_held = torch.stack([dB[k] for k in held_keys])              # (33,5)
    per_method = {}
    for method in cfg.METHODS:
        d = np.load(os.path.join(cfg.RESULTS_DIR, "preds", f"{method}.npz"),
                    allow_pickle=True)
        lookup = {}
        for arr, keyname in [("prim_pred", "prim_keys"), ("pair_pred", "pair_keys"),
                             ("heldout_pred", "heldout_keys")]:
            t = torch.from_numpy(d[arr])                           # (S,n,5)
            ks = [eval(k) for k in d[keyname]]
            for i, k in enumerate(ks):
                lookup[k] = t[:, i]
        per_method[method] = dict(
            hp=torch.from_numpy(d["heldout_pred"]),                # (S,33,5)
            hz=torch.from_numpy(d["heldout_z"]),                   # (S,33,64)
            lookup=lookup,
            in_bpe=torch.from_numpy(d["in_sample_bpe"]).float())   # (S,)

    S = len(cfg.SEEDS)
    # on-grid pair conditions only (off-grid duplicates are kind == "offgrid")
    bc_idx = [i for i, h in enumerate(heldout)
              if h["letters"] == ("B", "C") and h["kind"] == "pair"]
    cb_idx = [i for i, h in enumerate(heldout)
              if h["letters"] == ("C", "B") and h["kind"] == "pair"]

    # ---- oracle (direct encoding, NOT zero-shot) + true embedding diffs ----
    oracle_bpe = np.zeros(S)
    osi_true = np.zeros((S, 5))
    for si, seed in enumerate(cfg.SEEDS):
        sd = torch.load(os.path.join(cfg.RESULTS_DIR, "models", f"M5_seed{seed}.pt"),
                        weights_only=False)
        enc = sa.ShiftEncoder(); enc.load_state_dict(sd["encoder"]); enc.eval()
        dec = sa.BehaviorDecoder(); dec.load_state_dict(sd["decoder"]); dec.eval()
        with torch.no_grad():
            Z = torch.stack([enc(v_held[k].unsqueeze(0)).squeeze(0) for k in held_keys])
            opred = dec(Z)
        oracle_bpe[si] = (opred - dB_held).pow(2).sum(dim=1).mean().item()
        osi_true[si] = (Z[bc_idx] - Z[cb_idx]).norm(dim=1).numpy()

    # ---- per-method metrics ----
    summary_rows, se_by_method = [], {}
    for method, pm in per_method.items():
        hp, hz = pm["hp"], pm["hz"]
        se = (hp - dB_held.unsqueeze(0)).pow(2).sum(dim=2)         # (S,33)
        se_by_method[method] = se.numpy()
        bpe = se.mean(dim=1)                                       # (S,)
        acc_pred = B0_test[0] + hp[..., 0]                         # (S,33)
        acc_true = B0_test[0] + dB_held[:, 0]                      # (33,)
        acc_mae = (acc_pred - acc_true.unsqueeze(0)).abs().mean(dim=1)
        rho = sps.spearmanr(hp.mean(dim=0)[:, 0].numpy(),
                            dB_held[:, 0].numpy()).statistic
        if np.isnan(rho):
            rho = None
        cgg = bpe - pm["in_bpe"]                                   # (S,)
        rce = (bpe - pm["in_bpe"]) / (torch.from_numpy(oracle_bpe).float()
                                      - pm["in_bpe"] + 1e-8)       # (S,)
        # order sensitivity
        pred_diff = (hz[:, bc_idx] - hz[:, cb_idx]).norm(dim=2)    # (S,5)
        osi_abs = float(np.abs(pred_diff.numpy() - osi_true).mean())
        bdiff_pred = (hp[:, bc_idx] - hp[:, cb_idx]).norm(dim=2)   # (S,5)
        bc_true = torch.stack([dB[held_keys[i]] for i in bc_idx])  # (5,5)
        cb_true = torch.stack([dB[held_keys[i]] for i in cb_idx])
        bdiff_true = (bc_true - cb_true).norm(dim=1).numpy()       # (5,)
        beh_order_err = float(np.abs(bdiff_pred.numpy() - bdiff_true).mean())
        comm = sps.spearmanr(pred_diff.mean(dim=0).numpy(),
                             osi_true.mean(axis=0)).statistic
        if np.isnan(comm):
            comm = None
        # interaction recovery in behavior space (identifiable ground truth)
        # (off-grid conditions are excluded: their constituent primitives/pairs
        # at off-grid severities have no saved per-condition predictions)
        pair_errs, trip_errs = [], []
        for h in heldout:
            if h["kind"] == "offgrid":
                continue
            key = cfg.condition_key(h)
            letters, sevs = h["letters"], h["sevs"]
            pk = [cfg.condition_key(dict(letters=(l,), sevs=(s,)))
                  for l, s in zip(letters, sevs)]
            gamma_pred = pm["lookup"][key] - sum(pm["lookup"][q] for q in pk)
            gamma_true = dB[key] - sum(dB[q] for q in pk)
            if len(letters) == 3:
                for a in range(3):
                    for b in range(a + 1, 3):
                        pq = cfg.condition_key(dict(
                            letters=(letters[a], letters[b]),
                            sevs=(sevs[a], sevs[b])))
                        gamma_pred = gamma_pred - (pm["lookup"][pq]
                                                   - pm["lookup"][pk[a]]
                                                   - pm["lookup"][pk[b]])
                        gamma_true = gamma_true - (dB[pq] - dB[pk[a]] - dB[pk[b]])
                trip_errs.append((gamma_pred - gamma_true).pow(2).sum(dim=1)
                                 .mean().item())
            else:
                pair_errs.append((gamma_pred - gamma_true).pow(2).sum(dim=1)
                                 .mean().item())
        summary_rows.append(dict(
            method=method,
            bpe_mean=float(bpe.mean()), bpe_std=float(bpe.std()),
            acc_mae_mean=float(acc_mae.mean()),
            rank_rho=rho,
            cgg_mean=float(cgg.mean()), rce_mean=float(rce.mean()),
            osi_abs=osi_abs, beh_order_err=beh_order_err, commutator_rho=comm,
            ire_pair=float(np.mean(pair_errs)),
            ire_triple=float(np.mean(trip_errs)),
            oracle_bpe_mean=float(oracle_bpe.mean()),
        ))
        print(f"[evaluate] {method}: BPE={bpe.mean():.6f}±{bpe.std():.6f} "
              f"accMAE={acc_mae.mean():.4f} rankρ={rho} CGG={cgg.mean():.6f} "
              f"OSI={osi_abs:.4f} IRE_pair={np.mean(pair_errs):.6f}", flush=True)

    # ---- statistics: paired Wilcoxon across 33 conditions (seed-averaged SE) ----
    stats_rows = []
    base_se = se_by_method["M5_shift_algebra"].mean(axis=0)
    for other in ["M0_no_composition", "M1_additive", "M6_no_order",
                  "M8_recon_only", "M2_mlp", "M3_attention",
                  "MEMO_control", "PERM_control"]:
        o_se = se_by_method[other].mean(axis=0)
        diff = base_se - o_se                       # >0: M5 error larger => M5 WORSE
        try:
            p = float(sps.wilcoxon(base_se, o_se).pvalue)
        except ValueError:
            p = float("nan")
        rng = np.random.RandomState(0)
        boots = [diff[rng.randint(0, len(diff), len(diff))].mean()
                 for _ in range(10000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        stats_rows.append(dict(comparison=f"M5 vs {other}",
                               mean_diff=float(diff.mean()),
                               ci_lo=float(lo), ci_hi=float(hi), p_value=p))

    with open(os.path.join(cfg.RESULTS_DIR, "metrics_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    with open(os.path.join(cfg.RESULTS_DIR, "stats_vs_m5.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
        w.writeheader()
        w.writerows(stats_rows)

    # ---- decision rules (plan §7) ----
    def mval(method, key):
        return next(r[key] for r in summary_rows if r["method"] == method)

    m5_bpe = mval("M5_shift_algebra", "bpe_mean")
    rules = {
        "H1_M5_CGG_beats_M0": bool(mval("M5_shift_algebra", "cgg_mean")
                                   < mval("M0_no_composition", "cgg_mean")),
        "H2_M5_BPE_beats_M1_additive": bool(m5_bpe < mval("M1_additive", "bpe_mean")),
        "H2_M5_IRE_pair_beats_M1": bool(mval("M5_shift_algebra", "ire_pair")
                                        < mval("M1_additive", "ire_pair")),
        "H3_M5_OSI_beats_M6_no_order": bool(mval("M5_shift_algebra", "osi_abs")
                                            < mval("M6_no_order", "osi_abs")),
        "H3_commutator_rho_ge_0.7": bool((mval("M5_shift_algebra", "commutator_rho")
                                          or 0) >= 0.7),
        "H4_M5_BPE_beats_M8_recon_only": bool(m5_bpe < mval("M8_recon_only", "bpe_mean")),
        "H4_rank_rho_ge_0.8": bool((mval("M5_shift_algebra", "rank_rho")
                                    or 0) >= 0.8),
        "control_MEMO_fails": bool(mval("MEMO_control", "bpe_mean") > 3 * m5_bpe),
        "control_PERM_fails": bool(mval("PERM_control", "bpe_mean") > 3 * m5_bpe),
        "oracle_is_upper_bound": bool(mval("M5_shift_algebra", "oracle_bpe_mean")
                                      <= m5_bpe + 1e-9),
    }

    summary = dict(
        clean_test_acc=float(B0_test[0]),
        clean_test_ece=float(B0_test[2]),
        clean_test_conf=float(B0_test[4]),
        methods=summary_rows,
        stats=stats_rows,
        decision_rules=rules,
        oracle_bpe_per_seed=oracle_bpe.tolist(),
    )
    with open(os.path.join(cfg.RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print("[evaluate] ===== decision rules =====")
    for k, v in rules.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print("[evaluate] DONE")


# ----------------------------------------------------------------------------
# phase: audit
# ----------------------------------------------------------------------------
def phase_audit():
    report = {}
    ok_all = True

    def check(name, cond, detail=""):
        nonlocal ok_all
        report[name] = {"pass": bool(cond), "detail": str(detail)}
        ok_all &= bool(cond)
        print(f"[audit] {'PASS' if cond else 'FAIL'}  {name}  {detail}")

    # 1. contamination: no observed condition's letter-set is a superset of a
    #    held-out condition's letter-set; no exact key overlap
    obs, held = cfg.observed_conditions(), cfg.heldout_conditions()
    obs_sets = [cfg.condition_set(c) for c in obs]
    obs_keys = {cfg.condition_key(c) for c in obs}
    contam = []
    for h in held:
        hs = cfg.condition_set(h)
        if cfg.condition_key(h) in obs_keys:
            contam.append((h, "exact key overlap"))
        for o, os_ in zip(obs, obs_sets):
            if hs <= os_ and hs != os_:
                contam.append((h, o))
    check("contamination_split_disjoint", len(contam) == 0, f"violations={contam}")

    # 2. provenance: pools contain only observed conditions
    d = np.load(os.path.join(cfg.POOLS_DIR, "observed.npz"), allow_pickle=True)
    pool_keys = {eval(k) for k in d["keys"]}
    held_keys = {cfg.condition_key(h) for h in held}
    check("pools_contain_only_observed", pool_keys.isdisjoint(held_keys),
          f"overlap={pool_keys & held_keys}")
    check("pools_match_config", pool_keys == obs_keys,
          f"missing={obs_keys - pool_keys} extra={pool_keys - obs_keys}")

    # 3. determinism of corruptions
    x = torch.rand(8, 3, 32, 32)
    a = corr.apply_condition(x, ("B", "C"), (1.5, 0.15))
    b = corr.apply_condition(x, ("B", "C"), (1.5, 0.15))
    check("corruption_deterministic", torch.equal(a, b))

    # 4. noise field shared across orders (pure order effect for BC vs CB)
    check("noise_seed_order_independent",
          corr.condition_seed(("B", "C"), (1.5, 0.15))
          == corr.condition_seed(("C", "B"), (0.15, 1.5)))

    # 5. corruption semantics: BC != CB (non-commutative); A/D commute exactly
    #    when no clipping occurs (brightness x0.9 darkens, contrast x0.5 about
    #    the mean stays in [0,1]; at x1.5 brightness clipping breaks exact
    #    commutativity, which is expected behavior, not a bug)
    bc = corr.apply_condition(x, ("B", "C"), (2.0, 0.2))
    cb = corr.apply_condition(x, ("C", "B"), (0.2, 2.0))
    check("BC_noncommutative", (bc - cb).abs().max().item() > 1e-3,
          f"maxdiff={(bc-cb).abs().max().item():.4f}")
    ad = corr.apply_condition(x, ("A", "D"), (0.9, 0.5))
    da = corr.apply_condition(x, ("D", "A"), (0.5, 0.9))
    check("AD_commutative_no_clip", torch.allclose(ad, da, atol=1e-5),
          f"maxdiff={(ad-da).abs().max().item():.2e}")
    check("clip_bounds", bool(a.min() >= 0 and a.max() <= 1))

    # 6. controls + oracle from summary
    s = json.load(open(os.path.join(cfg.RESULTS_DIR, "summary.json")))
    rules = s["decision_rules"]
    check("control_MEMO_fails", rules["control_MEMO_fails"])
    check("control_PERM_fails", rules["control_PERM_fails"])
    check("oracle_is_upper_bound", rules["oracle_is_upper_bound"])

    # 7. seed reproducibility: retrain M5 seed 0, compare predictions
    V, Y, kinds = torch.from_numpy(d["V"]), torch.from_numpy(d["Y"]), d["kinds"]
    conds = [dict(letters=k[0], sevs=k[1], kind=str(kinds[i]), v=V[i], y=Y[i])
             for i, k in enumerate([eval(kk) for kk in d["keys"]])]
    prim_v = {(c["letters"][0], round(float(c["sevs"][0]), 6)): c["v"]
              for c in conds if c["kind"] == "prim"}
    aux_v = torch.load(os.path.join(cfg.POOLS_DIR, "aux_prim.pt"), weights_only=False)
    prim_v.update(aux_v)
    tr = sa.train_composition_model("M5_shift_algebra", conds, cfg.SEEDS[0])
    preds = torch.stack([sa.predict_condition(tr, h["letters"], h["sevs"], prim_v)[0]
                         for h in held])
    saved = np.load(os.path.join(cfg.RESULTS_DIR, "preds", "M5_shift_algebra.npz"),
                    allow_pickle=True)["heldout_pred"][0]
    check("seed_reproducibility",
          torch.allclose(preds, torch.from_numpy(saved), atol=1e-5),
          f"maxdiff={(preds - torch.from_numpy(saved)).abs().max().item():.2e}")

    # 8. off-grid predictions are finite (aux primitives wired correctly)
    d5 = np.load(os.path.join(cfg.RESULTS_DIR, "preds", "M5_shift_algebra.npz"),
                 allow_pickle=True)
    og = [i for i, h in enumerate(held) if h["kind"] == "offgrid"]
    check("offgrid_predictions_finite",
          len(og) > 0 and bool(np.isfinite(d5["heldout_pred"][:, og, :]).all()),
          f"n_offgrid={len(og)}")

    report["all_pass"] = ok_all
    with open(os.path.join(cfg.RESULTS_DIR, "audit_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[audit] {'ALL CHECKS PASS' if ok_all else 'SOME CHECKS FAILED'}")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["prepare-data", "train-base", "build-pools",
                                      "train-methods", "evaluate", "audit"])
    args = ap.parse_args()
    {"prepare-data": phase_prepare_data,
     "train-base": phase_train_base,
     "build-pools": phase_build_pools,
     "train-methods": phase_train_methods,
     "evaluate": phase_evaluate,
     "audit": phase_audit}[args.phase]()
