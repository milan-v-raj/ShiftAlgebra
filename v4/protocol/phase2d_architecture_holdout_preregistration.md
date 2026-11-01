# Phase 2D Architecture-Holdout CPU Protocol

**Protocol:** `V4-P2D-ARCH-HOLDOUT-CPU-001`
**Status:** all methods, models, words, metrics, and gates frozen before outcomes

## Primary question

Does the previously frozen composition-aware `g1` retain incremental value over `g0` on architecture families absent from every fitting bank?

## Fresh bank

- TinyVGG and TinyDenseNet, neither present in Phase-2 development;
- three existing recipes per family;
- six independent lineages and twelve epoch snapshots;
- seeds 62100 onward;
- 1,000 balanced CIFAR-10 images at per-class offsets 400--499;
- 64 new reversal-paired words, with zero overlap with Phases 2, 2B, and 2C;
- the standard 37 calibration conditions.

## Frozen methods

No method is fitted on Phase-2D:

- prior `g0` and `g1` regressions;
- full fifteen-group ACHM;
- active five-group selector;
- twenty random five-group controls;
- magnitude-only, variance-only, coverage-only, and graph D-optimal five-group selectors, all fit only on original Phase-2 development roles.

Unknown architecture categories map to all-zero known-family indicators, as already implemented; no new family coefficient is estimated.

## Primary gate

All four conditions are required:

1. `g1` MAE lower than `g0`;
2. 10,000-draw lineage/reversal clustered CI for `g0` error minus `g1` error excludes zero;
3. `g1` controlled model-rank Spearman higher than `g0`;
4. `g1` top-1 model-selection regret no worse than `g0`.

## Secondary gates

The previously positive active selector is replicated against expected random five-pair selection using the Phase-2C four-part gate. Alternative selectors are reported descriptively without selecting a winner after outcomes.

No outcome-guided rerun, architecture replacement, word replacement, or hyperparameter change is permitted.
