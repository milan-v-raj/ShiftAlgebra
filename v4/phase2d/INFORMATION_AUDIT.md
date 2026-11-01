# Phase 2D Information and Correctness Audit

**Protocol:** `V4-P2D-ARCH-HOLDOUT-CPU-001`
**Outcome:** no target leak detected; failed primary and active-replication gates retained

## Chronology

- `e71b360` froze the scientific conclusion, unseen families, words, image partition, all methods, selectors, metrics, and gates.
- `9cbdc6a` recorded six trained lineages before behavior evaluation.
- Behavior evaluation and analysis occurred afterward.
- The only post-outcome code correction converted NumPy boolean values to Python booleans for JSON serialization; it changed no prediction, metric, gate, or method.

## Architecture separation

TinyVGG and TinyDenseNet are implemented in `v4/phase2d/models.py` and do not occur among the five Phase-2 development families. Frozen family one-hot features therefore take their already-defined unknown/all-zero representation. No holdout-family coefficient is learned.

## Target separation

- seeds, model IDs, and lineages are new;
- image offsets 400--499 do not overlap prior behavior partitions;
- all 64 words are disjoint from Phase 2, Phase 2B, and Phase 2C words;
- reverse pairs share clusters;
- no architecture-holdout outcome enters fitting.

## Method access

`g0`/`g1`, full ACHM, active/random pair gates, and all alternative selector gates use original Phase-2 development roles only. Alternative pair sets and gates were frozen before outcomes. Full methods use thirty directed pair measurements; budgeted methods use ten.

## Accounting and verification

- 12 snapshots from six independent lineages;
- 101 evaluated conditions per snapshot, giving 1,212 behavior rows;
- 64 targets per snapshot, giving 768 forecast rows;
- 32 reversal groups;
- bootstrap clustering by lineage and reversal group;
- `verify_phase2d.py`: 4/4 checks pass, including independent MAE recomputation and bounded forecasts;
- Phase-2D unit tests: 4/4 pass.

## Gate handling

The primary gate failed solely because the clustered interval included zero. The active-selector replication also failed. Neither result was rescued using favorable family breakdowns, alternative selectors, or point estimates.

## Limitations

Six lineages and two holdout families provide low inferential power. The architectures are lightweight CPU models on CIFAR-10, not modern high-accuracy public models. Recipe names transfer, but datasets and corruption mechanisms do not. Controlled rank uses twelve correlated snapshots with many controls and is descriptive. These results do not satisfy the roadmap's final architecture-breadth gate.

A subsequent sensitivity review (`POST_OUTCOME_ROBUSTNESS_AUDIT.md`) verified all twelve checkpoint hashes, found one negative lineage and both positive family means, and obtained an exact lineage sign-flip p=0.0625 one-sided. It found no leak and did not revise the failed gate.
