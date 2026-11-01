# Phase 2E Prospective Architecture-Breadth Extension

**Protocol:** `V4-P2E-ARCH-BREADTH-CPU-001`  
**Status:** methods, families, seeds, words, images, metrics, and gates frozen before outcomes

## Purpose

Phase 2D remains failed and is not rerun or replaced. Phase 2E independently asks whether unchanged `g1` improves over unchanged `g0` on four additional CPU architecture families: TinyInception, TinySqueezeNet, TinyMixer, and TinyAttention.

## Fresh bank

- four families absent from Phase-2 fitting and Phase 2D;
- three recipes and independent lineages per family, twelve lineages total;
- epoch-one and epoch-two snapshots, 24 models;
- CIFAR-10 per-class offsets 500--599, disjoint from all earlier banks;
- 60 fresh reversal-paired words: twenty each at lengths 4, 5, and 6;
- zero word overlap with Phases 2, 2B, 2C, and 2D.

Length 3 is omitted because all 120 distinct-primitive length-3 permutations have already been consumed by prior frozen phases. No repeated-primitive words are introduced because the frozen pair-residual methods do not define self-pair calibration.

## Methods and information boundary

`g0`, `g1`, full ACHM, active, magnitude, variance, coverage, D-optimal, and twenty random designs are inherited byte-for-byte from prior frozen artifacts. No Phase-2D or Phase-2E outcome is used for fitting. Alternative selectors remain descriptive.

## Primary gate

All five conditions are required:

1. `g1` MAE is lower than `g0`;
2. a 10,000-draw hierarchical bootstrap resampling architecture family, lineage within family, and reversal group has a positive 95% interval for `g0` minus `g1` absolute error;
3. at least three of four family mean improvements are positive;
4. `g1` controlled model-rank Spearman exceeds `g0`;
5. `g1` top-1 selection regret is no worse than `g0`.

The result is retained without architecture, seed, word, or gate replacement. Any combined Phase-2D/2E analysis is retrospective and cannot change either gate.
