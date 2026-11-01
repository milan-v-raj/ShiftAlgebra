# V4 CPU Benchmark Artifact

This package consolidates the independently frozen Phase-2, Phase-2B, Phase-2C, Phase-2D, and Phase-2E confirmation banks.

## Scope

- 63 independent training lineages;
- 126 epoch snapshots;
- 342 bank-specific composed environments;
- 8,748 target prediction rows;
- CIFAR-10 and lightweight CPU models only.

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

This deterministically regenerates Phase-2B, Phase-2C, Phase-2D, Phase-2E, and cross-phase analysis JSON from committed forecasts. It does not retrain models or reopen a confirmation.

## Core artifacts

- `phase2/PHASE2_PILOT_FINAL_REPORT.md` — positive incremental model-bank pilot;
- `phase2b/PHASE2B_FINAL_REPORT.md` — failed adaptive-mixture gate;
- `phase2c/PHASE2C_FINAL_REPORT.md` — positive active pair-calibration gate;
- `phase2d/PHASE2D_FINAL_REPORT.md` — failed formal architecture-holdout gate with encouraging point transfer;
- `phase2d/SELECTOR_COMPARISON.md` — equal-budget alternative selector comparison;
- `phase2d/POST_OUTCOME_ROBUSTNESS_AUDIT.md` — hash, leak, heterogeneity, and sensitivity review;
- `phase2e/PHASE2E_FINAL_REPORT.md` — prospective four-family breadth extension;
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

The artifact intentionally preserves this mixed record. The cross-phase interval is a retrospective sensitivity analysis: all banks share CIFAR-10 and related shift semantics, so it is not independent-dataset confirmation. Verification success means artifacts and arithmetic are correct; it does not turn failed scientific gates into passes.
