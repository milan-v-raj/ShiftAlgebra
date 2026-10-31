# Shift Algebra V4 Development Workspace

This folder executes [`NEURIPS_DEVELOPMENT_PLAN.md`](../NEURIPS_DEVELOPMENT_PLAN.md) without changing the historical V1--V3 implementation or results.

## Phase status

| Phase | Status | Current evidence |
|---|---|---|
| **0. Claim and code audit** | **IN PROGRESS** | Exact core, architecture-independent fingerprint, cyclic-order control, property tests, standard metrics, source/family baseline, schema, claim matrix, limitations, and Phase-1 draft created |
| **1. Independent composition bridge** | **COMPLETE — METHOD GATE FAILED** | One-shot CIFAR-10 and EMNIST confirmation found severe saturation; max-envelope baseline wins |
| **1B. Severity-calibrated extension** | **COMPLETE — METHOD GATE FAILED** | Max envelope loses after calibration, but pair residual beats potential algebra by 9.22%; long-word dynamic-range guard also fails |
| **2. CIFAR model bank** | **IN PROGRESS — CPU PILOT PASSED** | Composition-aware `g1` cuts MAE 47.4% vs `g0` and improves rank/regret; final 75+ model/CIFAR-100 bank remains |
| **2B. Adaptive hazard mixture** | **COMPLETE — PRIMARY GATE FAILED** | Fresh ACHM MAE improves 1.719pp to 1.593pp, but clustered CI includes zero and controlled rank declines; prior `g1` remains best at 1.241pp |
| 3. ImageNet/realistic corruptions | NOT STARTED | — |
| 4. Natural/hybrid shifts | NOT STARTED | — |
| 5. Method freeze/preregistration | NOT STARTED | — |
| 6. One-shot final evaluation | NOT STARTED | — |
| 7. Paper and artifact | NOT STARTED | — |

Completed Phase-1, Phase-1B, Phase-2 pilot, and Phase-2B results are documented in their frozen reports; no result should be inferred for unstarted Phases 3--7.

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
