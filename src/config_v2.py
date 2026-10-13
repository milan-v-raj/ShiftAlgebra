"""V2 configuration (PLAN_V2). Pre-registered before any v2 training run.

Key changes vs v1 (src/config.py), each fixing a diagnosed v1 failure:
- 6 primitives (adds P=pixelate, S=saturation)  [fixes D6: more composition data]
- severity grids recalibrated by the P0 pilot so single shifts cause
  ~3/8/15/25/40pp accuracy drops (below saturation)  [fixes D4]
- behavior vector: margin-led, accuracy demoted  [fixes D4]
    measured (6): [margin, drift, ece, pcacc, acc, conf]
    target   (5): [margin, drift, ece, pcacc, acc]   (conf reported only)
    BPE is computed on the 4 primary (non-flooring) dims; the accuracy
    component is trained at reduced weight and reported as secondary.
- all 15 unordered pairs observed in a balanced tournament orientation on K_6
  (every primitive appears 2-3 times first and 2-3 times second, with at most
  one orientation observed per pair); held out: 20 triples + 8 reversed
  non-commutative pairs + 3 off-grid  [fixes D5/D6: order + higher-order tests]
- Regime III additionally observes 6 triples  [tests H5']
- shift encoder E(v) takes the 518-dim unlabeled shift statistics and is
  anchored to behavior by supervising G(E(v)) = dB on all observed shifts,
  while primitive dB vectors enter the composition operators directly  [fixes D2]
"""
import itertools
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results_v2")
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")     # shared with v1 (frozen base)
POOLS_DIR = os.path.join(ROOT, "pools_v2")

MASTER_SEED = 1234
N_SEEDS = 5
SEEDS = [MASTER_SEED + i for i in range(N_SEEDS)]

N_TEST_POOL = 10000
TRAIN_SAMPLE = 2000

# ---------------- Primitive shift library (6 primitives) ----------------
PRIM_LETTERS = ("A", "B", "C", "D", "P", "S")
# Recalibrated by the P0 pilot (pools_v2/severity_grids.json); values below
# are defaults until the pilot runs (targets: ~3/8/15/25/40pp drops).
SEVERITIES = {
    "A": [1.10, 1.25, 1.40, 1.55, 1.70],   # brightness scale
    "B": [0.3, 0.6, 0.9, 1.2, 1.5],        # gaussian blur sigma (px)
    "C": [0.02, 0.05, 0.08, 0.11, 0.14],   # gaussian noise sigma
    "D": [0.85, 0.70, 0.55, 0.40, 0.25],   # contrast factor
    "P": [2, 3, 4, 6, 8],                  # pixelate block size (px)
    "S": [1.5, 2.0, 2.5, 3.0, 3.5],        # saturation factor
}
N_SEV = 5

# ---------------- Behavior vector (margin-led) ----------------
# measured: [margin, drift, ece, pcacc, acc, conf]; target = first 5.
BEHAVIOR_MEASURED = 6
BEHAVIOR_DIM = 5
TARGET_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 0.3]     # acc downweighted (floors)
BPE_PRIMARY = 4                                # BPE over [margin, drift, ece, pcacc]

# ---------------- Encoder / model dims ----------------
STATS_DIM = 518          # unlabeled shift distribution stats (feat/pixel/delta mean+std)
ENCODER_IN_DIM = STATS_DIM
LATENT_DIM = 64
OP_DIM = 24              # auxiliary state dimension in Operator-Product Algebra (M-B)
OP_RANK = 8              # low-rank operator interaction rank (M-B)
TRAIN_STEPS = 500
TRAIN_LR = 1e-3
WEIGHT_DECAY = 1e-4
LAMBDA_REPR = 0.2        # secondary: anchor composed latent to E(v_pair)

# ---------------- V2 split ----------------
# All 15 unordered pairs observed, PLUS the reverse orientations of the 9
# training-only pairs (BIDIR_TRAIN_PAIRS) so the order pathway has leak-free
# training supervision on both non-commuting pairs (AC, BP, CP) and commuting
# pairs (AP, AS, BD, BS, DP, PS) [fixes D5]. The 6 ORDER_TEST_PAIRS (including
# the flagship BC, plus CD, CS, AB, AD, DS) are observed in ONE orientation
# only; their 6 reverses (ORDERED_HELDOUT, 30 conditions) are strictly held out
# for zero-shot order evaluation.
# ---------------- V2 split ----------------
# All 15 unordered pairs observed, PLUS the reverse orientations of the 5
# spanning-tree training pairs (BP, CP, AC, AD, DS) and 4 commuting pairs
# (AS, BD, BS, DP, PS) in BIDIR_TRAIN_PAIRS so the order potential kappa_X is
# identified on a spanning tree of the 6 primitives [fixes D5].
# The 4 chord edges ORDER_TEST_PAIRS (balanced: 2 negative BC->CB, CS->SC and
# 2 positive BA->AB, PA->AP) are observed in ONE orientation only; their 4
# reverses (ORDERED_HELDOUT, 20 conditions) are strictly held out to test
# zero-shot transitive order generalization across the spanning tree.
ORDER_TEST_PAIRS = [
    ("B", "C"), ("B", "A"), ("C", "S"), ("P", "C"),
]
ORDERED_HELDOUT = [pair[::-1] for pair in ORDER_TEST_PAIRS]

BIDIR_TRAIN_PAIRS = [
    ("B", "P"), ("A", "C"), ("A", "D"), ("A", "P"), ("D", "S"),
    ("A", "S"), ("B", "D"), ("B", "S"), ("D", "P"), ("P", "S"),
]
OBSERVED_PAIRS = (
    list(ORDER_TEST_PAIRS)
    + [("C", "D")]
    + list(BIDIR_TRAIN_PAIRS)
    + [pair[::-1] for pair in BIDIR_TRAIN_PAIRS]
)

ALL_TRIPLES = [tuple(t) for t in itertools.combinations(PRIM_LETTERS, 3)]   # 20
# Regime III: additionally observe 6 triples; 14 remain held out.
REGIME3_OBSERVED_TRIPLES = [
    ("A", "B", "C"), ("A", "B", "D"), ("A", "C", "D"),
    ("B", "C", "D"), ("A", "B", "S"), ("C", "D", "P"),
]

# Off-grid extrapolation conditions (updated dynamically relative to SEVERITIES
# after load_pilot_grids so they always sit between/beyond grid points).
OFFGRID = {
    ("B", "C"): (0.8, 0.07),
    ("C", "B"): (0.07, 0.8),
    ("A", "B", "C"): (1.35, 0.8, 0.07),
}

# ---------------- Methods ----------------
CORE_METHODS = ["M0_meanpool", "M1_behadd", "M_A", "M_A_att", "M_B", "M_C",
                "M5_v1", "M1_emb"]
ABLATION_METHODS = ["M_B_full", "M_B_nores", "M_B_nobeh", "M_A_nointer",
                    "M_C_nopos"]
CONTROL_METHODS = ["MEMO_control", "PERM_control"]
METHODS = CORE_METHODS + ABLATION_METHODS + CONTROL_METHODS


def observed_conditions(regime=2):
    """Regime II: 6 prims x 5 sev + 15 pairs x 5 sev = 105 conditions.
    Regime III: additionally 6 triples x 5 sev = 135 conditions."""
    conds = []
    for l in PRIM_LETTERS:
        for i in range(N_SEV):
            conds.append(dict(letters=(l,), sevs=(SEVERITIES[l][i],), kind="prim"))
    for pair in OBSERVED_PAIRS:
        for i in range(N_SEV):
            conds.append(dict(letters=pair,
                              sevs=(SEVERITIES[pair[0]][i], SEVERITIES[pair[1]][i]),
                              kind="pair"))
    if regime >= 3:
        for trip in REGIME3_OBSERVED_TRIPLES:
            for i in range(N_SEV):
                conds.append(dict(letters=trip,
                                  sevs=tuple(SEVERITIES[l][i] for l in trip),
                                  kind="triple"))
    return conds


def heldout_conditions(regime=2):
    """Regime II: 20 triples x 5 + 8 reversed pairs x 5 + 3 off-grid = 143.
    Regime III: 14 triples x 5 + 40 + 3 = 113."""
    conds = []
    triples = ALL_TRIPLES if regime == 2 else [
        t for t in ALL_TRIPLES if t not in REGIME3_OBSERVED_TRIPLES]
    for trip in triples:
        for i in range(N_SEV):
            conds.append(dict(letters=trip,
                              sevs=tuple(SEVERITIES[l][i] for l in trip),
                              kind="triple"))
    for pair in ORDERED_HELDOUT:
        for i in range(N_SEV):
            conds.append(dict(letters=pair,
                              sevs=(SEVERITIES[pair[0]][i],
                                    SEVERITIES[pair[1]][i]),
                              kind="pair"))
    for letters, sevs in OFFGRID.items():
        conds.append(dict(letters=letters, sevs=tuple(sevs), kind="offgrid"))
    return conds


def condition_key(cond):
    return (tuple(cond["letters"]), tuple(round(float(s), 6) for s in cond["sevs"]))


def aux_primitives():
    """Primitive (letter, severity) pairs needed to EVALUATE off-grid held-out
    conditions (single shifts at off-grid severities; not held-out compositions)."""
    needed = set()
    for letters, sevs in OFFGRID.items():
        for l, s in zip(letters, sevs):
            s = float(s)
            if all(abs(s - g) > 1e-9 for g in SEVERITIES[l]):
                needed.add((l, round(s, 6)))
    return sorted(needed)


def load_pilot_grids():
    """If the P0 pilot has run, override SEVERITIES with the calibrated grids
    and place OFFGRID severities at midpoints between grid levels 2 and 3."""
    global SEVERITIES, OFFGRID
    path = os.path.join(POOLS_DIR, "severity_grids.json")
    if os.path.exists(path):
        with open(path) as f:
            g = json.load(f)["grids"]
        SEVERITIES = {k: [float(x) for x in v] for k, v in g.items()}
        sa = round(0.5 * (SEVERITIES["A"][2] + SEVERITIES["A"][3]) + 0.013, 4)
        sb = round(0.5 * (SEVERITIES["B"][2] + SEVERITIES["B"][3]) + 0.013, 4)
        sc = round(0.5 * (SEVERITIES["C"][2] + SEVERITIES["C"][3]) + 0.003, 4)
        OFFGRID = {
            ("B", "C"): (sb, sc),
            ("C", "B"): (sc, sb),
            ("A", "B", "C"): (sa, sb, sc),
        }
    return SEVERITIES
