# Phase 2 — CIFAR Model-Bank Development Pilot

**Protocol:** `V4-P2-CIFAR-BANK-PILOT-001`
**Status:** Stage-B method frozen; confirmation models and outcomes unopened

This CPU pilot starts the roadmap's model-bank phase with 30 development and 30 confirmation snapshots from 15 separately trained lineages per split, spanning five architecture families and three training recipes.

It tests whether a new extensive ordered-hazard forecast improves model ranking and selection beyond source accuracy, architecture, recipe, and word metadata. It also includes target-image ATC/DoC references under the explicitly advantaged R3 regime.

This is not the final Phase-2 benchmark: the roadmap still requires 75+ final models, six or more architecture families, four recipe families, and CIFAR-100.

## Stages

1. Commit code, model/word manifest, methods, metrics, and gates.
2. Train/evaluate development lineages only.
3. Freeze pair-scaling exponent and `g0`/`g1` regressions.
4. Train/evaluate confirmation lineages once.
5. Run ranking, regret, bootstrap, and correctness audits.

See [`../protocol/phase2_pilot_preregistration.md`](../protocol/phase2_pilot_preregistration.md).
