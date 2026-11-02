# Shift Algebra V4 Development Workspace

This folder executes [`NEURIPS_DEVELOPMENT_PLAN.md`](../NEURIPS_DEVELOPMENT_PLAN.md) without changing the historical V1--V3 implementation or results.

## Phase status

| Phase | Status | Current evidence |
|---|---|---|
| **0. Claim and code audit** | **IN PROGRESS** | Exact core, architecture-independent fingerprint, cyclic-order control, property tests, standard metrics, source/family baseline, schema, claim matrix, limitations, and Phase-1 draft created |
| **1. Independent composition bridge** | **COMPLETE — METHOD GATE FAILED** | One-shot CIFAR-10 and EMNIST confirmation found severe saturation; max-envelope baseline wins |
| **1B. Severity-calibrated extension** | **COMPLETE — METHOD GATE FAILED** | Max envelope loses after calibration, but pair residual beats potential algebra by 9.22%; long-word dynamic-range guard also fails |
| **2. CIFAR model bank** | **COMPLETE — CPU PILOT PASSED** | Composition-aware `g1` cuts MAE 47.4% vs `g0` and improves rank/regret; later full-vector CIFAR-100 evidence is recorded in 2F/2G |
| **2B. Adaptive hazard mixture** | **COMPLETE — PRIMARY GATE FAILED** | Fresh ACHM MAE improves 1.719pp to 1.593pp, but clustered CI includes zero and controlled rank declines; prior `g1` remains best at 1.241pp |
| **2C. Active pair calibration** | **COMPLETE — PRIMARY GATE PASSED** | Five selected pair groups reach 3.176pp vs 3.636pp expected random MAE; clustered improvement CI [0.119, 0.889]pp |
| **2D. Architecture-family holdout** | **COMPLETE — PRIMARY GATE FAILED** | `g1` improves 2.978pp to 1.800pp and rank/regret, but six-lineage CI [-0.042, 2.293]pp includes zero |
| **2E. Architecture-breadth extension** | **COMPLETE — PRIMARY GATE FAILED** | Four new families: `g1` improves 4.443pp to 2.034pp, but family-hierarchical CI [-0.072, 5.287]pp includes zero |
| **2F. CIFAR-100 full vector** | **COMPLETE — BOTH GATES PASSED** | Six-coordinate macro normalized MAE 0.554→0.254; all metrics improve; magnitude five-pair 0.271 vs 0.338 expected random |
| **2G. No-refit architecture transfer** | **COMPLETE — BOTH GATES PASSED** | Frozen Phase-2F method transfers to six unseen families: 0.912→0.326; all metrics/families improve; frozen magnitude 0.329 vs 0.484 random |
| **2H. Order-awareness ablation** | **COMPLETE — PRIMARY GATE FAILED** | Symmetric composition 0.352 vs order-aware 0.349; 4/6 metrics and reversal contrast improve, but controlled rank declines 0.732→0.729 |
| **3A. Realistic-corruption composition** | **COMPLETE — PRIMARY GATE FAILED** | Six common-corruption categories: all six MAEs and rank improve, aggregate CI is positive, but accuracy-selection regret worsens 0.1083→0.1367pp; gate 6/7 |
| 3B. Larger/official corruption benchmark | NOT STARTED | Phase 3A is not official ImageNet-C or ImageNet |
| 4. Natural/hybrid shifts | NOT STARTED | — |
| 5. Method freeze/preregistration | NOT STARTED | — |
| 6. One-shot final evaluation | NOT STARTED | — |
| 7. Paper and artifact | NOT STARTED | — |

Completed Phase-1, Phase-1B, Phase-2 pilot, Phase-2B, Phase-2C, Phase-2D, Phase-2E, Phase-2F, Phase-2G, Phase-2H, and Phase-3A results are documented in their frozen reports; no result should be inferred for unstarted Phase 3B or Phases 4--7.

## Layout

```text
v4/
├── src/                       # Exact V4 algebra and prospective models
├── tests/                     # Invariant, schema, metric, and baseline tests
├── benchmark/                 # Versioned schema and external protocol manifests
│   └── manifests/
├── baselines/                 # R0/R1/R2/R3 baselines
├── analysis/                  # Comparable metrics and later inference
├── literature/                # Prior-work claim matrix
├── protocol/                  # Preregistrations and freeze contracts
├── KNOWN_LIMITATIONS.md       # Claim boundaries and current failure modes
└── verify_phase0.py           # One-command Phase-0 verifier
```

## Verify Phase 0

From the repository root:

```bash
python3 v4/verify_phase0.py
```

The verifier:

1. runs all V4 unit/property tests;
2. validates the Mason manifest and exact word counts;
3. checks the literature matrix schema;
4. checks that all required Phase-0 artifacts exist;
5. prints a machine-readable summary to stdout.

Completed composition bridges can be verified separately:

```bash
python3 v4/phase1/verify_phase1.py
python3 v4/phase1b/verify_phase1b.py
python3 v4/phase2/verify_phase2.py
python3 v4/phase2b/verify_phase2b.py
python3 v4/phase2c/verify_phase2c.py
python3 v4/phase2d/verify_phase2d.py
python3 v4/phase2e/verify_phase2e.py
python3 v4/phase2f/verify_phase2f.py
python3 v4/phase2g/verify_phase2g.py
python3 v4/phase2h/verify_phase2h.py
python3 -m v4.phase3a.verify
python3 -m v4.reproduce_cpu --verify --summary
```

## Scientific rules

1. V1--V3 files are immutable except for documentation corrections.
2. R0--R4 information access must be declared for every method.
3. Matrix, streaming, and closed-form implementations must remain numerically equivalent.
4. The matrix form is not compared as if it were different from its exact closed form.
5. Natural factors are commutative unless a physical order actually exists.
6. Final conditions remain sealed after method freeze.
7. Failed preregistered gates are reported, not rescued under the same protocol ID.

## Immediate remaining Phase-0 tasks

- complete the multi-index literature search outside GitHub and refresh the claim matrix;
- freeze Phase-1 model/checkpoint provenance and hyperparameter budgets;
- complete license review, especially FaceScrub;
- run a five-model CIFAR compute/variance pilot on development-only conditions.
