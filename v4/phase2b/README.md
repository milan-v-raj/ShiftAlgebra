# Phase 2B — Adaptive Composition Hazard Mixture (CPU Pilot)

**Protocol:** `V4-P2B-ACHM-CPU-001`  
**Status:** fresh confirmation complete; **primary ACHM gate failed** despite a small MAE improvement

Phase 2B tests whether target-free forecasting benefits from selecting among four interpretable hazard laws rather than using one law everywhere:

1. dominant primitive;
2. independent retention;
3. symmetric pair compounding;
4. ordered pair compounding.

A small softmax gate uses only pre-target calibration information. It forms a convex hazard mixture and therefore returns a bounded accuracy forecast. Five-fold lineage-group validation selects regularization using only the original Phase-2 `method_development` rows. The original Phase-2 confirmation file is forbidden.

The fresh test contains 30 snapshots from 15 new CPU-trained lineages and 64 completely new, reversal-paired words. The behavior partition is also disjoint from both Phase-2 partitions. Secondary variants use 0, 5, 10, or 15 bidirectional pair groups.

## Frozen development summary

| Pair groups | Directed pair measurements | Development MAE | Lineage-CV MAE |
|---:|---:|---:|---:|
| 0 | 0 | 4.687pp | 4.686pp |
| 5 | 10 | 2.667pp | 2.667pp |
| 10 | 20 | 1.307pp | 1.482pp |
| 15 | 30 | 1.074pp | 1.421pp |

These are development results, not evidence for a new claim.

## Fresh outcome

ACHM improved MAE over the frozen ordered hazard from **1.719pp to 1.593pp** (7.3%), but the clustered 95% improvement interval `[-0.025, 0.295]pp` included zero and controlled model-rank Spearman declined from `0.822` to `0.811`. The complete gate therefore **failed**. The previously frozen metadata-plus-composition `g1` remained best at **1.241pp**.

The pair-budget curve was `5.357pp` (0 groups), `3.431pp` (5), `1.726pp` (10), and `1.593pp` (15). Order and expert-disagreement uncertainty diagnostics were negative. See `PHASE2B_FINAL_REPORT.md` and `INFORMATION_AUDIT.md`.

## Files

- `PHASE2B_MANIFEST.json` — frozen models, words, image partition, and source hash.
- `FROZEN_ACHM.json` — all budget-specific gates, feature scales, CV scores, and pair selections.
- `method.py` — hazard experts, active selection, gate fitting, and application.
- `train_bank.py` / `evaluate_bank.py` — fresh CPU bank pipeline.
- `forecast.py` / `analyze.py` — frozen prediction and one-shot analysis.
- `tests/test_phase2b.py` — information-boundary, manifest, and convex-mixture checks.
- `PHASE2B_FINAL_REPORT.md` — complete positive and negative fresh results.
- `INFORMATION_AUDIT.md` — chronology, access, clustering, and numerical audit.
- `../protocol/phase2b_achm_preregistration.md` — protocol and failure policy.

This remains a CPU pilot. It does not satisfy the roadmap's 75+ model, CIFAR-100, architecture-holdout, or official ODP baseline requirements.
