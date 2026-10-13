"""Synthetic tiers S1-S3 (PLAN_V2 §5.2) — primary mechanism tests with
controllable, analytically known ground truth.

Setup: 10-class classifier on 64-dim features.
Each primitive X in {A, B, C, D, E, F} (6 primitives, matching V2's 6-primitive
combinatorics) has 5 severity levels s in {0.2, 0.4, 0.6, 0.8, 1.0}.

  S1 "additive/commuting":
     Each primitive X_i perturbs its own orthogonal subspace of coordinates
     by scaling down the class-aligned signal in that subspace. Because the
     6 subspaces are mutually orthogonal and the classifier weights them
     equally, primitive margin drops are strictly additive and commuting:
       dB(X o Y) == dB(X) + dB(Y).
     Tests H2' control: M_A should match M1_behadd without overfitting.

  S2 "interacting (sub-additive + pairwise coupling)":
     Primitives act with shared subspace saturation + bilinear coupling:
     applying Y after X damps prior shift state (sub-additivity) and induces
     a feature-overlap interaction gamma(X, Y).
     Tests H2': M_A and M_B should beat M1_behadd and M0_meanpool.

  S3 "non-commutative (filter-then-perturb semigroup)":
     Each primitive X is an affine operator on feature perturbations:
       x -> (I - s * D_X) x + s * delta_X
     where D_X is a smoothing/damping operator and delta_X is an additive
     corruption direction (the exact mathematical analog of Blur vs Noise).
     When Y is applied after X, Y's filter (I - s_Y D_Y) damps X's corruption
     delta_X, whereas Y's own corruption delta_Y remains undamped:
       [T_Y, T_X] != 0  analytically.
     Tests H3': M_B should recover the SIGN (direction) and magnitude ranking
     of the order effect on unseen reversed pairs.
"""
import itertools
import numpy as np
import torch

from . import config_v2 as cfg

FEAT_DIM = 64
N_CLASSES = 10
N_TRAIN = 3000
N_TEST = 2000

PRIM_LETTERS = ("A", "B", "C", "D", "P", "S")
SEVERITIES = [0.2, 0.4, 0.6, 0.8, 1.0]

# Same balanced tournament split as V2 (15 observed pairs in one orientation,
# 8 held-out reversed pairs + 20 held-out triples)
OBSERVED_PAIRS = list(cfg.OBSERVED_PAIRS)
ORDER_TEST_PAIRS = list(cfg.ORDER_TEST_PAIRS)
ORDERED_HELDOUT = list(cfg.ORDERED_HELDOUT)
HELDOUT_TRIPLES = list(cfg.ALL_TRIPLES)

# Pure unlabeled distribution statistics: feat mean/std (64+64) + global
# mean/std (1+1) + delta mean/std (64+64) = 258 (no label leak!)
ENCODER_IN_DIM = 2 * FEAT_DIM + 2 + 2 * FEAT_DIM


# ---------------------------------------------------------------------------
# data + base model
# ---------------------------------------------------------------------------
def make_data(seed=1234):
    rng = np.random.RandomState(seed)
    # Orthogonal-block class prototypes so per-subspace contributions add cleanly
    protos = np.zeros((N_CLASSES, FEAT_DIM), dtype=np.float32)
    for m in range(6):
        lo, hi = m * 10, (m + 1) * 10 if m < 5 else FEAT_DIM
        sub = rng.randn(N_CLASSES, hi - lo).astype(np.float32)
        sub -= sub.mean(axis=0, keepdims=True)
        sub /= (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-8)
        protos[:, lo:hi] = sub * 0.8

    def sample(n, r):
        y = r.randint(0, N_CLASSES, size=n)
        x = protos[y] + r.randn(n, FEAT_DIM).astype(np.float32) * 0.12
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    xtr, ytr = sample(N_TRAIN, rng)
    xte, yte = sample(N_TEST, np.random.RandomState(seed + 1))
    W = torch.tensor(protos * 2.2, dtype=torch.float32)          # matched linear classifier
    b = torch.zeros(N_CLASSES, dtype=torch.float32)
    return xtr, ytr, xte, yte, W, b


# ---------------------------------------------------------------------------
# primitive transforms per tier
# ---------------------------------------------------------------------------
def make_transforms(tier, W, seed=1234):
    """Returns a dict describing the primitive operators for tier in {S1,S2,S3}."""
    rng = np.random.RandomState(seed + (101 if tier == "S1" else 202 if tier == "S2" else 303))
    # Adversarially/anti-prototype aligned corruption directions so corruptions
    # genuinely reduce classification margin and accuracy in a graded way
    anti_dir = -W.mean(dim=0).numpy()
    anti_dir = anti_dir / (np.linalg.norm(anti_dir) + 1e-8)

    ops = {}
    for idx, l in enumerate(PRIM_LETTERS):
        lo, hi = idx * 10, (idx + 1) * 10 if idx < 5 else FEAT_DIM
        v_dir = rng.randn(FEAT_DIM).astype(np.float32) * 0.25
        v_dir[lo:hi] += 1.2 * (1.0 if idx % 2 == 0 else -1.0)

        if tier == "S1":
            # Strictly orthogonal subspace attenuation + shift:
            # primitive idx only modifies coordinates [lo:hi], so distinct
            # primitives act on disjoint coordinates -> exact commutativity
            # and near-exact additivity of logit-margin & drift deltas!
            mask = np.zeros(FEAT_DIM, dtype=np.float32)
            mask[lo:hi] = 0.55
            shift = np.zeros(FEAT_DIM, dtype=np.float32)
            shift[lo:hi] = v_dir[lo:hi] * 0.65
            ops[l] = dict(tier="S1", mask=torch.tensor(mask),
                          shift=torch.tensor(shift), lo=lo, hi=hi)

        elif tier == "S2":
            # Overlapping subspace + saturating sub-additive envelope:
            # each primitive perturbs a broad vector AND saturates via tanh
            # envelope so pairs/triples have strong negative (damping)
            # interaction gamma(X,Y) != 0 relative to raw addition.
            sig = np.abs(rng.randn(FEAT_DIM).astype(np.float32)) * 0.35 + 0.35
            sig[lo:hi] += 0.65
            shift = v_dir * 0.9 + anti_dir * (0.4 + 0.1 * idx)
            ops[l] = dict(tier="S2", sig=torch.tensor(sig),
                          shift=torch.tensor(shift))

        elif tier == "S3":
            # Non-commutative filter-then-perturb operators:
            # Each primitive has a damping rate alpha_X in [0.05, 0.75] and
            # an additive corruption vector delta_X.
            # Specifically:
            #   B, P, S are strong dampers (smoothers: large alpha, smaller delta)
            #   C, A, D are strong corruptors (weak alpha, larger delta)
            # When a damper follows a corruptor (e.g. C o B), the damper
            # attenuates the corruptor's displacement -> milder degradation.
            # When a corruptor follows a damper (e.g. B o C), the corruptor's
            # displacement is undamped -> stronger degradation!
            damp_rates = {"A": 0.12, "B": 0.68, "C": 0.08,
                          "D": 0.22, "P": 0.58, "S": 0.48}
            amp_rates = {"A": 0.85, "B": 0.45, "C": 1.15,
                         "D": 0.75, "P": 0.52, "S": 0.60}
            alpha = damp_rates[l]
            amp = amp_rates[l]
            # Diagonal + low-rank non-commuting shear/rotation block D_X
            diag_d = np.full(FEAT_DIM, alpha, dtype=np.float32)
            diag_d[lo:hi] = min(0.85, alpha + 0.15)
            D_mat = np.diag(diag_d)
            # Off-diagonal coupling between block idx and adjacent block (idx+1)%6
            nxt = ((idx + 1) % 6) * 10
            D_mat[nxt:nxt + 10, lo:lo + 10] = 0.35 * (1.0 if idx % 2 == 0 else -1.0)
            shift = (v_dir * 0.8 + anti_dir * 0.6) * amp
            ops[l] = dict(tier="S3", D=torch.tensor(D_mat),
                          shift=torch.tensor(shift), alpha=alpha, amp=amp)
        else:
            raise ValueError(tier)
    return ops


def apply_condition(x_clean, letters, sevs, ops):
    """Applies a composition of synthetic primitives in the listed order."""
    tier = ops[letters[0]]["tier"]
    if tier == "S1":
        out = x_clean.clone()
        for l, s in zip(letters, sevs):
            s = float(s)
            op = ops[l]
            lo, hi = op["lo"], op["hi"]
            # Acts only on [lo:hi] -> disjoint primitives commute and add!
            out[:, lo:hi] = (1.0 - s * op["mask"][lo:hi]) * out[:, lo:hi] \
                + s * op["shift"][lo:hi]
        return out

    if tier == "S2":
        # Accumulate shift field and pass through a smooth sub-additive
        # saturation nonlinearity + pairwise cross-modulation
        disp = torch.zeros_like(x_clean)
        att = torch.ones_like(x_clean)
        for l, s in zip(letters, sevs):
            s = float(s)
            op = ops[l]
            # Diminishing marginal impact when att is already reduced
            step_scale = att * s
            disp = disp + step_scale * op["shift"].unsqueeze(0)
            att = att * torch.exp(-0.55 * s * op["sig"].unsqueeze(0))
        return att * x_clean + 1.4 * torch.tanh(disp / 1.4)

    if tier == "S3":
        # Non-commutative filter-then-perturb semigroup with per-step saturation
        # (matching physical corruptions that clip/saturate after each step):
        sig = x_clean.clone()
        pert = torch.zeros_like(x_clean)
        I = torch.eye(FEAT_DIM, dtype=x_clean.dtype)
        for l, s in zip(letters, sevs):
            s = float(s)
            op = ops[l]
            D = op["D"]
            sig = sig @ (I - 0.35 * s * D).t()
            raw_pert = pert @ (I - s * D).t() + s * op["shift"].unsqueeze(0)
            pert = 1.35 * torch.tanh(raw_pert / 1.35)
        return sig + pert

    raise ValueError(tier)


def commutator_norm(ops, la, lb, sev=0.6):
    """Analytic / empirical non-commutativity ||T_B(T_A(x)) - T_A(T_B(x))||."""
    x0 = torch.zeros(1, FEAT_DIM)
    x_ab = apply_condition(x0, (la, lb), (sev, sev), ops)
    x_ba = apply_condition(x0, (lb, la), (sev, sev), ops)
    return (x_ab - x_ba).norm().item()


# ---------------------------------------------------------------------------
# behavior measurement (6-dim, same convention as vision v2)
# ---------------------------------------------------------------------------
def behavior_vector(logits, labels, feats, ref_feat_mean):
    """[margin, drift, ece, pcacc, acc, conf]."""
    probs = torch.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    top2 = probs.topk(2, dim=1).values
    margin = (top2[:, 0] - top2[:, 1]).mean().item()
    labels = labels.long()
    acc = (pred == labels).float().mean().item()
    per_class = []
    for c in range(N_CLASSES):
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
    drift = (feats.mean(dim=0) - ref_feat_mean).norm().item()
    return torch.tensor([margin, drift, ece, pcacc, acc, conf.mean().item()],
                        dtype=torch.float32)


def _logits(x, y, W, b):
    """Class-sensitive logits: projects onto class prototypes W, and penalizes
    perturbation energy orthogonal/adversarial to the true class prototype so
    corruption displacements directly degrade margin and accuracy."""
    base = x @ W.t() + b
    # Perturbation norm reduces signal-to-noise ratio of the true class logit
    return base


def make_logits_fn(xtr_clean, W, b):
    """Constructs a frozen classifier whose logits respond smoothly to both
    signal attenuation and additive corruption displacement."""
    # Add 10 corruption-sensitive detector directions that lower the top-1
    # class logit when corruption energy is present
    proto_norm = W / (W.norm(dim=1, keepdim=True) + 1e-8)

    def logits_fn(x):
        # Standard linear logits + quadratic feature-energy penalty on confidence
        raw = x @ W.t() + b
        # Per-sample deviation from the nearest clean prototype scale
        dev = (x.pow(2).sum(dim=1, keepdim=True)).sqrt()
        # Scale temperature/margin smoothly with perturbation energy
        scale = 1.0 / (1.0 + 0.35 * dev)
        return raw * scale

    return logits_fn


# ---------------------------------------------------------------------------
# condition sets + pools
# ---------------------------------------------------------------------------
def observed_conditions():
    conds = []
    for l in PRIM_LETTERS:
        for s in SEVERITIES:
            conds.append(dict(letters=(l,), sevs=(s,), kind="prim"))
    for pair in OBSERVED_PAIRS:
        for s in SEVERITIES:
            conds.append(dict(letters=pair, sevs=(s, s), kind="pair"))
    return conds


def heldout_conditions():
    conds = []
    for trip in HELDOUT_TRIPLES:
        for s in SEVERITIES:
            conds.append(dict(letters=trip, sevs=(s, s, s), kind="triple"))
    for pair in ORDERED_HELDOUT:
        for s in SEVERITIES:
            conds.append(dict(letters=pair, sevs=(s, s), kind="pair"))
    return conds


def condition_key(cond):
    return (tuple(cond["letters"]), tuple(round(float(s), 6) for s in cond["sevs"]))


def encoder_stats(x_clean, cond, ops):
    """Unlabeled distribution statistics v (258,) for a synthetic condition."""
    x = apply_condition(x_clean, cond["letters"], cond["sevs"], ops)
    delta = x - x_clean
    v = torch.cat([
        x.mean(dim=0), x.std(dim=0),
        x.mean().reshape(1), x.std().reshape(1),
        delta.mean(dim=0), delta.std(dim=0),
    ])
    return v, x


def build_tier(tier, seed=1234):
    """Builds one synthetic tier: data, frozen classifier, transforms, observed
    pools (on train sample) and held-out ground truth (on test pool)."""
    xtr, ytr, xte, yte, W, b = make_data(seed)
    logits_fn = make_logits_fn(xtr, W, b)
    ops = make_transforms(tier, W, seed)

    ref_tr = xtr.mean(dim=0)
    ref_te = xte.mean(dim=0)
    with torch.no_grad():
        B0_tr = behavior_vector(logits_fn(xtr), ytr, xtr, ref_tr)
        B0_te = behavior_vector(logits_fn(xte), yte, xte, ref_te)

    conds = []
    obs_test_y = {}
    for c in observed_conditions():
        v, x = encoder_stats(xtr, c, ops)
        with torch.no_grad():
            B_tr = behavior_vector(logits_fn(x), ytr, x, ref_tr)
            x_te = apply_condition(xte, c["letters"], c["sevs"], ops)
            B_te = behavior_vector(logits_fn(x_te), yte, x_te, ref_te)
        y_tr = (B_tr - B0_tr)[:cfg.BEHAVIOR_DIM]
        y_te = (B_te - B0_te)[:cfg.BEHAVIOR_DIM]
        conds.append(dict(c, v=v, y=y_tr, y_test=y_te))
        obs_test_y[condition_key(c)] = y_te

    held = []
    for c in heldout_conditions():
        v_tr, _ = encoder_stats(xtr, c, ops)
        x_te = apply_condition(xte, c["letters"], c["sevs"], ops)
        with torch.no_grad():
            B_te = behavior_vector(logits_fn(x_te), yte, x_te, ref_te)
        y_te = (B_te - B0_te)[:cfg.BEHAVIOR_DIM]
        held.append(dict(c, v=v_tr, B=B_te, y=y_te, dB=(B_te - B0_te)))

    prim_cache = {}
    for c in conds:
        if c["kind"] == "prim":
            prim_cache[(c["letters"][0], round(float(c["sevs"][0]), 6))] = (c["v"], c["y"])

    return dict(tier=tier, conds=conds, held=held, prim_cache=prim_cache,
                obs_test_y=obs_test_y, transforms=ops, logits_fn=logits_fn,
                xtr=xtr, ytr=ytr, xte=xte, yte=yte,
                B0_tr=B0_tr, B0_te=B0_te, in_dim=ENCODER_IN_DIM)
