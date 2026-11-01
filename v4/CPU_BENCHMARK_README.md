# V4 CPU Benchmark Artifact

This package consolidates the independently frozen Phase-2 through Phase-2H CPU banks. Phases 2--2E are CIFAR-10 accuracy studies; Phases 2F--2H are six-coordinate CIFAR-100 confirmation benchmarks.

## Scope

- 135 independent training lineages and 270 epoch snapshots overall;
- 8,748 CIFAR-10 accuracy prediction rows across five frozen banks;
- 6,480 sealed CIFAR-100 full-vector prediction rows, each with six outputs;
- twelve CIFAR-100 architecture families and lightweight CPU models;
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

This deterministically regenerates Phase-2B, Phase-2C, Phase-2D, Phase-2E, Phase-2F, Phase-2G, Phase-2H, and cross-phase analysis JSON from committed forecasts. It does not retrain models or reopen a confirmation.

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

The artifact intentionally preserves this mixed record. The Phases 2--2E cross-phase interval is a retrospective sensitivity analysis: those five banks share CIFAR-10 and related shift semantics. Phase 2F is a separate CIFAR-100 development/confirmation experiment; Phase 2G transfers its method without refitting to six unseen architecture families. Both remain within CIFAR-100, so neither is independent-dataset transfer. Phase 2H retains a failed all-required order gate despite small positive error and reversal-contrast effects. Verification success means artifacts and arithmetic are correct; it does not turn failed scientific gates into passes.
