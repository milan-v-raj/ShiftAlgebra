# Phase 2E Information and Correctness Audit

**Protocol:** `V4-P2E-ARCH-BREADTH-CPU-001`  
**Outcome:** no leak detected; failed primary gate retained

## Chronology

1. Commit `8115a11` froze architectures, seeds, image offsets, words, inherited method hashes, hierarchy, metrics, and the five-part gate.
2. Commit `2a7350b` recorded all twelve training lineages and 24 checkpoint hashes before target evaluation.
3. Evaluation, forecasting, and analysis occurred once afterward.
4. No architecture, word, seed, method, or criterion was replaced.

## Information boundaries

- `g0` and `g1` are the original Phase-2 regressions.
- ACHM and all five-pair selectors are inherited from the pre-Phase-2D artifact.
- Frozen dependency hashes are checked before forecasting.
- No Phase-2D or Phase-2E target outcome enters fitting.
- Unknown families map to the already-defined all-zero known-family indicators.
- Recipes are known categories; no holdout-family coefficient is learned.

## Separation

- Four families are disjoint from Phase-2 development and Phase 2D.
- Seeds and lineages are new.
- CIFAR-10 offsets 500--599 are disjoint from offsets 0--499 used previously.
- All sixty words are disjoint from Phases 2, 2B, 2C, and 2D.
- Length 3 was prospectively omitted because no unused distinct-primitive permutation remained.

## Accounting

- 4 families, 12 lineages, 24 snapshots;
- 97 conditions per snapshot: 37 calibration and 60 targets;
- 2,328 behavior rows and 1,440 forecast rows;
- 30 reversal groups;
- 10,000-draw hierarchy: family, lineage within family, and reversal group;
- behavior CSV hash recorded in `evaluation.json`;
- `verify_phase2e.py`: 4/4 checks pass.

## Decision handling

Four gate conditions passed. The family-hierarchical interval included zero, so the all-required gate failed. The strong pooled point estimate, positive length breakdown, and favorable three-family majority were not used to override it. Phase 2D also remains failed.

## Limitations

Four families remain a small family-level sample, each with only three independent lineages. The models are low-accuracy CPU proxies on CIFAR-10. TinyMixer contributes a large share of the average benefit, while TinyAttention is negative. This is architecture breadth within one dataset and one transformation system, not dataset or natural-shift breadth.
