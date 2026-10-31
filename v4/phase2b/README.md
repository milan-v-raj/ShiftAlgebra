# Phase 2B — Adaptive Composition Hazard Mixture (CPU Pilot)

**Protocol:** `V4-P2B-ACHM-CPU-001`  
**Status:** ACHM and fresh manifest frozen; fresh confirmation not yet opened

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

These are development results, not evidence for a new claim. The fresh outcome is one-shot and must be retained whether the gate passes or fails.

## Files

- `PHASE2B_MANIFEST.json` — frozen models, words, image partition, and source hash.
- `FROZEN_ACHM.json` — all budget-specific gates, feature scales, CV scores, and pair selections.
- `method.py` — hazard experts, active selection, gate fitting, and application.
- `train_bank.py` / `evaluate_bank.py` — fresh CPU bank pipeline.
- `forecast.py` / `analyze.py` — frozen prediction and one-shot analysis.
- `tests/test_phase2b.py` — information-boundary, manifest, and convex-mixture checks.
- `../protocol/phase2b_achm_preregistration.md` — protocol and failure policy.

This remains a CPU pilot. It does not satisfy the roadmap's 75+ model, CIFAR-100, architecture-holdout, or official ODP baseline requirements.
