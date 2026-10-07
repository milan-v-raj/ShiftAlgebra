"""Central configuration — all experiment constants (pre-registered).

Amendment 1 (compute-constrained sandbox: 2 CPU cores, no GPU):
  The plan specifies a ResNet-18 base model (~93.2% clean CIFAR-10).
  Full-width ResNet-18 training is infeasible on 2 CPU cores, so the base
  model is a ResNet-18 at width 0.25 ([16,32,64,128] channels, [2,2,2,2]
  blocks, ~0.7M params), trained on the full CIFAR-10 train pool. The
  shift-algebra protocol is unchanged; only the base model capacity differs.
  Target: >= 88% clean test accuracy (full-width ResNet-18 reaches ~93%).

Amendment 2 (data source):
  CIFAR-10 was obtained from a GitHub mirror of the official extracted
  batches; all 7 file md5 checksums verified against the official
  torchvision reference values (see src/data.py).
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
BATCHES_DIR = os.path.join(DATA_DIR, "cifar-10-batches-py")
RESULTS_DIR = os.path.join(ROOT, "results")
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")
POOLS_DIR = os.path.join(ROOT, "pools")

MASTER_SEED = 1234
N_SEEDS = 5
SEEDS = [MASTER_SEED + i for i in range(N_SEEDS)]

# ---------------- Base model ----------------
BASE_MODEL = dict(
    arch="resnet18",
    width=0.25,          # Amendment 1
    epochs=40,
    batch_size=128,
    lr=0.1,
    momentum=0.9,
    weight_decay=5e-4,
    seed=MASTER_SEED,
)

# ---------------- Data pools ----------------
N_TRAIN_POOL = 50000
N_TEST_POOL = 10000
TRAIN_SAMPLE = 2000      # train-pool images per condition: encoder inputs + behavior labels
TEST_EVAL_FULL = True    # held-out ground truth measured on the full 10k test pool

# ---------------- Primitive shift library (Appendix A.2) ----------------
PRIM_LETTERS = ("A", "B", "C", "D")
SEVERITIES = {
    "A": [1.20, 1.35, 1.50, 1.65, 1.80],   # brightness scale (x -> s*x)
    "B": [0.5, 1.0, 1.5, 2.0, 2.5],        # gaussian blur sigma (pixels)
    "C": [0.05, 0.10, 0.15, 0.20, 0.25],   # gaussian noise sigma ([0,1] units)
    "D": [0.80, 0.65, 0.50, 0.35, 0.20],   # contrast factor about per-image mean
}
N_SEV = 5

# ---------------- ShiftCompose split (Appendix A.3, Regime II revised) ----------------
OBSERVED_PAIRS = [("A", "B"), ("A", "C"), ("A", "D"), ("B", "D"), ("C", "D")]
HELDOUT_PAIRS = [("B", "C"), ("C", "B")]
HELDOUT_TRIPLES = [("A", "B", "C"), ("B", "C", "D"), ("A", "B", "D"), ("A", "C", "D")]
# Off-grid severities (extrapolation eval; not in any training grid).
# Severity tuples are aligned with the letter order: ("C","B") applies
# noise at 0.175 FIRST, then blur at 1.75.
OFFGRID = {
    ("B", "C"): (1.75, 0.175),
    ("C", "B"): (0.175, 1.75),
    ("A", "B", "C"): (1.65, 1.75, 0.175),
}

# ---------------- Shift-algebra model ----------------
LATENT_DIM = 64
ENCODER_IN_DIM = 518     # feat mean(128)+std(128)+pixel mean(3)+std(3)+delta mean(128)+std(128)
BEHAVIOR_DIM = 5         # acc, mean per-class acc, ECE, feature drift, mean confidence
LAM_ORDER = 1.0          # lambda for the antisymmetric (order) pathway
TRAIN_STEPS = 4000
TRAIN_LR = 1e-3
LAMBDA_REPR = 1.0
LAMBDA_BEH = 1.0

# ---------------- Methods ----------------
METHODS = ["M5_shift_algebra", "M6_no_order", "M8_recon_only",
           "M0_no_composition", "M1_additive", "M2_mlp", "M3_attention",
           "MEMO_control", "PERM_control"]


def observed_conditions():
    """All conditions the model may observe (Regime II): 4 primitives x 5 sev
    + 5 pairs x 5 diagonal severity combos = 45 conditions."""
    conds = []
    for l in PRIM_LETTERS:
        for i in range(N_SEV):
            conds.append(dict(letters=(l,), sevs=(SEVERITIES[l][i],), kind="prim"))
    for pair in OBSERVED_PAIRS:
        for i in range(N_SEV):
            conds.append(dict(letters=pair,
                              sevs=(SEVERITIES[pair[0]][i], SEVERITIES[pair[1]][i]),
                              kind="pair"))
    return conds


def heldout_conditions():
    """Strictly zero-shot conditions: 2 pairs x 5 sev + 4 triples x 5 sev
    + 3 off-grid = 33 conditions."""
    conds = []
    for pair in HELDOUT_PAIRS:
        for i in range(N_SEV):
            conds.append(dict(letters=pair,
                              sevs=(SEVERITIES[pair[0]][i], SEVERITIES[pair[1]][i]),
                              kind="pair"))
    for trip in HELDOUT_TRIPLES:
        for i in range(N_SEV):
            conds.append(dict(letters=trip,
                              sevs=tuple(SEVERITIES[l][i] for l in trip),
                              kind="triple"))
    for letters, sevs in OFFGRID.items():
        conds.append(dict(letters=letters, sevs=tuple(sevs), kind="offgrid"))
    return conds


def condition_set(cond):
    return frozenset(cond["letters"])


def condition_key(cond):
    return (tuple(cond["letters"]), tuple(round(float(s), 6) for s in cond["sevs"]))


def aux_primitives():
    """Primitive (letter, severity) pairs needed to *evaluate* off-grid
    held-out conditions. These are single shifts at off-grid severities:
    observing a primitive at a new severity is NOT a held-out violation
    (only compositions are held out)."""
    needed = set()
    for letters, sevs in OFFGRID.items():
        for l, s in zip(letters, sevs):
            s = float(s)
            if all(abs(s - g) > 1e-9 for g in SEVERITIES[l]):
                needed.add((l, s))
    return sorted(needed)
