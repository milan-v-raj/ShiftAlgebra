# Phase 2C Final Report — Active Pair Calibration

**Protocol:** `V4-P2C-ACTIVE-PAIR-CPU-001`
**Decision:** **ACTIVE-PAIR PRIMARY GATE PASSED**
**Scope:** CPU-only calibration-efficiency pilot; not the final 75+ model benchmark

## Executive result

With a budget of five bidirectional pair groups (ten directed measurements), the development-selected design achieved **3.176pp accuracy MAE** on fresh models and words. Twenty frozen uniform-random five-group designs had mean MAE **3.636pp**, median **3.818pp**, and range **2.592--4.459pp**. The active design reduced error by **0.459pp**, or **12.6% relative to expected random-design MAE**.

A 10,000-draw bootstrap clustered by fresh reversal group and training lineage gave a 95% interval of **[0.119, 0.889]pp** for expected-random error minus active error. Controlled model-rank Spearman was **0.474** versus a random-design median of **0.417**, and top-1 regret was **3.870pp** versus random median **3.984pp**. All four preregistered criteria passed.

This establishes that the tested development-selected heuristic is better than expected random five-pair calibration in this CPU setting. It does **not** establish globally optimal pair selection: three of twenty random designs had lower MAE, with the best reaching 2.592pp.

## Fresh confirmation design

- 15 new training lineages and 30 epoch snapshots;
- seeds 52100--52502, disjoint from Phases 2 and 2B;
- five lightweight architecture families and three recipes;
- 1,000 balanced CIFAR-10 test images at per-class offsets 300--399;
- image partition disjoint from Phase-2 and Phase-2B behavior partitions;
- 64 new reversal-paired words, 16 at each length 3--6;
- zero word overlap with both earlier target sets;
- 37 clean/single/pair calibration conditions.

Fresh clean accuracy ranged from **28.7% to 51.9%**, with mean **41.7%**. The evidence remains limited to weak two-epoch CPU classifiers.

## Chronology and information boundary

1. `fa6f5ed` froze the protocol, model/word manifest, active design, twenty random controls, gates, analysis, and failure policy before fresh outcomes.
2. `b201678` recorded all fresh training lineages before behavior evaluation.
3. Behavior evaluation was opened only afterward.

All gates were fitted from the original Phase-2 `calibration` and `method_development` roles. Phase-2 confirmation and every Phase-2B behavior outcome were forbidden. The active five-group design was exactly the design frozen before Phase-2B; it was not changed after seeing its results.

## Pair designs

The active design measured:

1. Contrast--Invert;
2. Rotate90--Swirl;
3. GaussianBlur--ImpulseNoise;
4. Invert--Swirl;
5. Invert--Rotate90.

Both directions of each group were measured. The selector balanced development interaction magnitude, development word frequency, and primitive coverage. Random designs sampled five of the fifteen unordered groups uniformly without replacement. Every design used the same ACHM gate family and fixed L2 coefficient.

## Primary results

| Method/strategy | Pair groups | Directed measurements | MAE ↓ | Controlled rank ↑ | Top-1 regret ↓ |
|---|---:|---:|---:|---:|---:|
| Prior frozen `g1` | 15 | 30 | **1.410pp** | **0.695** | 0.656pp |
| Full-budget ACHM | 15 | 30 | 1.603pp | 0.661 | **0.363pp** |
| **Active design** | **5** | **10** | **3.176pp** | **0.474** | **3.870pp** |
| Random strategy, median | 5 | 10 | 3.818pp | 0.417 | 3.984pp |
| Random strategy, mean MAE | 5 | 10 | 3.636pp | -- | -- |
| Best random design | 5 | 10 | 2.592pp | -- | -- |
| Worst random design | 5 | 10 | 4.459pp | -- | -- |

The full-budget references are not same-cost competitors: they use all thirty directed pair measurements. They show the accuracy sacrificed when calibration cost is reduced by two thirds.

## Preregistered gate

| Criterion | Result | Decision |
|---|---:|---|
| Active MAE below mean random MAE | 3.176pp < 3.636pp | PASS |
| Clustered 95% CI excludes zero | [0.119, 0.889]pp | PASS |
| Controlled rank at least random median | 0.474 > 0.417 | PASS |
| Top-1 regret no worse than random median | 3.870pp < 3.984pp | PASS |
| Complete gate | all four required | **PASS** |

The active design beat **17 of 20 random designs by MAE**, placing it at the 85th percentile under the declared convention. The three better random draws are important: development selection improves expected efficiency but does not identify the best subset reliably.

## Performance by word length

| Length | Active five-group MAE | Full ACHM MAE | Prior `g1` MAE |
|---:|---:|---:|---:|
| 3 | 2.346pp | 1.226pp | 1.278pp |
| 4 | 3.561pp | 1.696pp | **1.316pp** |
| 5 | 3.689pp | 1.768pp | **1.465pp** |
| 6 | 3.110pp | 1.723pp | **1.581pp** |

The five-group design is substantially less accurate than full pair calibration at every length. Its contribution is calibration efficiency relative to random selection, not full-budget forecasting superiority.

## Scientific interpretation

Phase 2C gives fresh positive evidence for **budgeted pair selection**, the practical secondary direction proposed after Phase 2. Under an equal ten-measurement budget, selecting pairs using prior interaction evidence and primitive coverage is better on average than random calibration.

The safe claim is narrow:

> On new weak CIFAR-10 CPU models and unseen length-3--6 words, a development-selected five-pair design reduced target-free forecasting MAE relative to the expected performance of twenty frozen random five-pair designs.

The result does not repair the failed Phase-2B adaptive-mixture gate, prove optimal experimental design, transfer to held-out architectures, or complete full Phase 2. A stronger future study would compare graph-D-optimal, uncertainty-based, magnitude-only, coverage-only, and random selectors across CIFAR-100 and architecture-family holdouts. Those revisions must use another fresh confirmation.
