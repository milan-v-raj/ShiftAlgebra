# V4 CPU Benchmark Artifact

This package consolidates the prospectively frozen Phase-2 through Phase-3B CPU banks. Phases 2--2E are CIFAR-10 accuracy studies; Phases 2F--3A are six-coordinate CIFAR-100 confirmation benchmarks; Phase 3B is the fresh-bank CIFAR-10 six-coordinate replication.

## Scope

- 189 separately seeded training lineages and 378 epoch snapshots overall;
- 8,748 CIFAR-10 accuracy prediction rows across five frozen banks;
- 8,640 sealed CIFAR-100 full-vector prediction rows, each with six outputs;
- 2,160 sealed CIFAR-10 Phase-3B full-vector prediction rows;
- twelve CIFAR-100 families plus fresh six-family CIFAR-10 development/confirmation banks of lightweight CPU models;
- canonical CIFAR-10 and CIFAR-100 provenance verified by pinned hashes.

## Fast verification and summary

```bash
python reproduce.py --v4-cpu
# or
python -m v4.reproduce_cpu --verify --summary
```

Verification uses committed CSV/JSON artifacts and does not require datasets or checkpoints. Install `requirements.txt` first.

## Recompute analyses

```bash
python -m v4.reproduce_cpu --reanalyze --summary
```

This deterministically regenerates Phase-2B, Phase-2C, Phase-2D, Phase-2E, Phase-2F, Phase-2G, Phase-2H, Phase-3A, Phase-3B, and cross-phase analysis JSON from committed forecasts. It does not retrain models or reopen a confirmation.

## Core artifacts

- `phase2/PHASE2_PILOT_FINAL_REPORT.md` — positive incremental model-bank pilot;
- `phase2b/PHASE2B_FINAL_REPORT.md` — failed adaptive-mixture gate;
- `phase2c/PHASE2C_FINAL_REPORT.md` — positive active pair-calibration gate;
- `phase2d/PHASE2D_FINAL_REPORT.md` — failed formal architecture-holdout gate with encouraging point transfer;
- `phase2d/SELECTOR_COMPARISON.md` — equal-budget alternative selector comparison;
- `phase2d/POST_OUTCOME_ROBUSTNESS_AUDIT.md` — hash, leak, heterogeneity, and sensitivity review;
- `phase2e/PHASE2E_FINAL_REPORT.md` — prospective four-family breadth extension;
- `phase2f/PHASE2F_FINAL_REPORT.md` — positive CIFAR-100 six-coordinate confirmation;
- `phase2g/PHASE2G_FINAL_REPORT.md` — positive strict no-refit transfer to six unseen architectures;
- `phase2h/PHASE2H_FINAL_REPORT.md` — failed prospective order-awareness gate with small positive MAE/contrast effects;
- `phase3a/PHASE3A_FINAL_REPORT.md` — failed 6/7 realistic-corruption gate with favorable error/rank transfer but slightly worse selection regret;
- `phase3b/PHASE3B_FINAL_REPORT.md` — prospective CIFAR-10 replication of the favorable error/rank pattern, with the same strict regret failure and 6/7 gate;
- `SELECTOR_TRANSFER_SYNTHESIS.md` — retrospective equal-budget transfer table;
- `results/CPU_CROSS_PHASE_SYNTHESIS.{json,md}` — no-refit retrospective bank synthesis;
- `results/ARCHITECTURE_SYNTHESIS.{json,md}` — retrospective six-family sensitivity analysis;
- each phase's `INFORMATION_AUDIT.md`, manifest, frozen method, and verifier.

## Frozen gate record

| Stage | Gate |
|---|---|
| Phase 2 incremental composition value | PASS |
| Phase 2B ACHM superiority | FAIL |
| Phase 2C active pair selection | PASS |
| Phase 2D architecture holdout | FAIL |
| Phase 2E architecture breadth | FAIL |
| Phase 2F CIFAR-100 full vector | PASS |
| Phase 2F magnitude five-pair selector | PASS |
| Phase 2G strict no-refit architecture transfer | PASS |
| Phase 2G frozen magnitude selector transfer | PASS |
| Phase 2H incremental order awareness | FAIL |
| Phase 3A realistic-corruption composition | FAIL (6/7) |
| Phase 3B prospective CIFAR-10 replication | FAIL (6/7) |

The artifact intentionally preserves this mixed record. The Phases 2--2E cross-phase interval is a retrospective sensitivity analysis: those five banks share CIFAR-10 and related shift semantics. Phase 2F is a separate CIFAR-100 development/confirmation experiment; Phase 2G transfers its method without refitting to six unseen architecture families. Both remain within CIFAR-100. Phase 2H retains a failed all-required order gate despite small positive error and reversal-contrast effects. Phase 3A moves to six clean-room common-corruption categories and improves all six MAEs, controlled rank, the aggregate interval, and all family signs, but fails because top-1 accuracy-selection regret worsens slightly. Phase 3B repeats that favorable error/rank/family pattern with fresh CIFAR-10 banks, partitions, seeds, and words, but fails the same strict regret criterion. The two phases agree on 6/6 metric directions and 7/7 gate directions; both complete gates remain failed. Neither is an official CIFAR-C or ImageNet-C evaluation, and historical Phase 1 used some Phase-3B images. Verification success means artifacts and arithmetic are correct; it does not turn failed scientific gates into passes.
