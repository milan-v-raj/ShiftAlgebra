# Phase 2 — CIFAR Model-Bank Development Pilot

**Protocol:** `V4-P2-CIFAR-BANK-PILOT-001`
**Status:** one-shot CPU pilot complete; incremental-value gate passed; full Phase 2 remains in progress

This CPU pilot used 30 development and 30 confirmation snapshots from 15 separately trained lineages per split, spanning five architecture families and three training recipes.

## Main result

On 90 sealed length-3--6 words:

- source/model/word baseline `g0`: **3.304pp accuracy MAE**;
- `g1` with frozen composition forecasts: **1.737pp**;
- relative reduction: **47.4%**;
- lineage/word-group bootstrap improvement: **1.529pp**, 95% CI **[0.981, 2.191]pp**;
- model-rank Spearman: **0.242 -> 0.665**;
- top-1 model-selection regret: **2.158pp -> 0.489pp**.

All preregistered incremental pilot criteria passed.

The order diagnostic did not pass: the symmetric hazard model slightly beat the ordered version. The positive result is therefore evidence for composition-aware extensive retention forecasting, not yet for a general non-commutative order law.

## Scope boundary

This is not the final Phase-2 benchmark. The roadmap still requires 75+ final models, at least six architecture and four recipe families, CIFAR-100, modern/robust checkpoints, and architecture/recipe holdouts.

## Files

- `PHASE2_MANIFEST.json` — frozen 60-snapshot bank and word roles;
- `FROZEN_METHOD.json` — q=0.6 and frozen `g0`/`g1` regressions;
- `PHASE2_DEVELOPMENT_REPORT.md` — development-only record before confirmation;
- `PHASE2_PILOT_FINAL_REPORT.md` — complete result;
- `INFORMATION_AUDIT.md` — leakage/correctness review;
- `results/` — compact training, behavior, forecast, and analysis artifacts.

## Verify

```bash
python v4/phase2/verify_phase2.py
```

See [`../protocol/phase2_pilot_preregistration.md`](../protocol/phase2_pilot_preregistration.md) for the frozen design and gates.
