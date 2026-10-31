# Phase 2C Active Pair Calibration — Outcome-Blind CPU Protocol

**Protocol:** `V4-P2C-ACTIVE-PAIR-CPU-001`
**Status:** protocol, designs, and analysis frozen before fresh behavior outcomes
**Compute:** CPU only, two PyTorch threads

## Distinct question

Phase 2B's ACHM method gate failed and remains failed. Phase 2C does not alter or rescue that claim. It tests the separate calibration-efficiency question:

> Under a budget of five bidirectional corruption pairs, does a development-selected interaction design outperform the expected performance of uniformly random five-pair designs?

## Fixed method and development boundary

The active five-group design is exactly the Phase-2B development-frozen design; it is not changed after Phase-2B outcomes. Twenty random five-group designs are sampled without replacement using seed 22403. Every design receives the same ACHM gate family and fixed L2 coefficient `0.001`, selected for the five-group gate during Phase-2B development.

All gates are fitted only from `calibration` and `method_development` roles in the original Phase-2 development CSV. The original Phase-2 confirmation and all Phase-2B behavior outcomes are forbidden during fitting.

Five unordered groups cost ten directed pair measurements. Unmeasured pair residuals are set to zero. The full fifteen-group ACHM and prior frozen `g1` are reference methods, not same-budget comparators.

## Fresh confirmation

- 15 new lineages and 30 epoch snapshots;
- seeds 52100 onward, disjoint from Phases 2 and 2B;
- five lightweight families and three recipes;
- 1,000 balanced test images at per-class offsets 300--399;
- 64 new reversal-paired words, 16 at each length 3--6;
- zero word overlap with either the Phase-2 or Phase-2B target sets;
- 37 clean/single/pair calibration conditions.

## Primary estimand and gate

The random strategy is represented by the expected absolute error across twenty independently frozen uniform-random designs. This estimates expected performance before drawing a random calibration set; it is not an operational ensemble prediction.

The active design passes only if all conditions hold:

1. active MAE is below mean random-design MAE;
2. a 10,000-draw paired bootstrap interval for expected-random error minus active error, clustered by reversal group and lineage, excludes zero;
3. active controlled model-rank Spearman is at least the median random design;
4. active top-1 selection regret is no worse than the median random design.

The analysis must also disclose the full random range, active percentile, full-budget ACHM, prior `g1`, and per-length MAE.

## Failure and interpretation policy

No design, seed, word, model, metric, or gate may be changed after fresh outcomes are opened. Passing would support only the tested development-selected five-pair heuristic on weak CIFAR CPU models. It would not establish global optimality. Failure would show that the heuristic does not beat random calibration reliably and must be retained without rescue.
