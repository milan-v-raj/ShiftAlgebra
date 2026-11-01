# V4 CPU Benchmark Artifact

This package consolidates the independently frozen Phase-2 through Phase-2F CPU banks. Phases 2--2E are CIFAR-10 accuracy studies; Phase 2F is a six-coordinate CIFAR-100 development/confirmation benchmark.

## Scope

- 99 independent training lineages and 198 epoch snapshots overall;
- 8,748 CIFAR-10 accuracy prediction rows across five frozen banks;
- 2,160 sealed CIFAR-100 full-vector prediction rows, each with six outputs;
- six architecture families and lightweight CPU models;
- canonical CIFAR-100 test provenance verified pixel-for-pixel.

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

This deterministically regenerates Phase-2B, Phase-2C, Phase-2D, Phase-2E, Phase-2F, and cross-phase analysis JSON from committed forecasts. It does not retrain models or reopen a confirmation.

## Core artifacts

- `phase2/PHASE2_PILOT_FINAL_REPORT.md` — positive incremental model-bank pilot;
- `phase2b/PHASE2B_FINAL_REPORT.md` — failed adaptive-mixture gate;
- `phase2c/PHASE2C_FINAL_REPORT.md` — positive active pair-calibration gate;
- `phase2d/PHASE2D_FINAL_REPORT.md` — failed formal architecture-holdout gate with encouraging point transfer;
- `phase2d/SELECTOR_COMPARISON.md` — equal-budget alternative selector comparison;
- `phase2d/POST_OUTCOME_ROBUSTNESS_AUDIT.md` — hash, leak, heterogeneity, and sensitivity review;
- `phase2e/PHASE2E_FINAL_REPORT.md` — prospective four-family breadth extension;
- `phase2f/PHASE2F_FINAL_REPORT.md` — positive CIFAR-100 six-coordinate confirmation;
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

The artifact intentionally preserves this mixed record. The Phases 2--2E cross-phase interval is a retrospective sensitivity analysis: those five banks share CIFAR-10 and related shift semantics. Phase 2F is a separate CIFAR-100 development/confirmation experiment, but its coefficients were refit on development outcomes, so it is not strict no-refit dataset transfer. Verification success means artifacts and arithmetic are correct; it does not turn failed scientific gates into passes.
