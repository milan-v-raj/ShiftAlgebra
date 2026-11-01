# Phase 2F Final Report — CIFAR-100 Full-Vector Confirmation

**Protocol:** `V4-P2F-CIFAR100-FULLVECTOR-CPU-001`
**Primary decision:** **FULL-VECTOR GATE PASSED (5/5)**
**Budgeted decision:** **MAGNITUDE FIVE-PAIR GATE PASSED (4/4)**

## Executive result

On a sealed CIFAR-100 confirmation bank, the composition-aware full-pair model reduced macro normalized six-coordinate MAE from **0.554 (`g0`) to 0.254**, a **54.2% relative reduction**. It improved every reported coordinate, increased macro controlled model-rank Spearman from **0.302 to 0.811**, and reduced accuracy top-1 selection regret from **1.560pp to 0.382pp**. The family/lineage/reversal hierarchical interval for normalized error improvement was **[0.152, 0.538]**, excluding zero. All five preregistered criteria passed.

The prospectively selected magnitude five-pair design also passed: macro normalized MAE was **0.271**, compared with **0.338** for expected random five-pair calibration, a 19.8% reduction. Its clustered improvement interval was **[0.024, 0.133]**.

## What was predicted

Every forecast contains the complete six-coordinate output required by the historical project:

1. probability margin;
2. penultimate-feature centroid drift;
3. ten-bin ECE;
4. balanced accuracy;
5. overall accuracy;
6. mean confidence.

Balanced and overall accuracy are numerically redundant because the test partition contains exactly twenty images from each class. Both were retained because the protocol and user requirement froze all six outputs. A post-outcome five-unique-coordinate sensitivity analysis reaches the same conclusions.

## Prospective design

- canonical CIFAR-100 fine labels, 100 classes;
- all 10,000 transported test images pixel-matched to the canonical MD5-verified test pickle;
- six architecture families and three recipes;
- 18 development and 18 confirmation lineages;
- 36 snapshots in each split;
- disjoint 2,000-image development and confirmation partitions;
- 60 fresh development and 60 fresh confirmation words at lengths 4--6;
- all development behavior and methods committed before confirmation outcomes;
- development-only per-coordinate exponent selection, scaling, ridge fitting, and pair selection.

The setup was frozen in `0c6d846` and corrected for a transport-commit metadata path in `4e65720`, before outcomes. Both banks were recorded in `3e2e1ca`. Development behavior and all methods were frozen in `af5f5d2` before confirmation evaluation.

## Full-vector results

Raw MAEs are in each coordinate's natural units. Accuracy values are also shown in percentage points.

| Coordinate | Metadata-only `g0` | Full composition | Relative reduction |
|---|---:|---:|---:|
| Margin | 0.01039 | **0.00505** | 51.4% |
| Feature drift | 0.72046 | **0.20796** | 71.1% |
| ECE | 0.02798 | **0.01539** | 45.0% |
| Balanced accuracy | 0.00839 (0.839pp) | **0.00493 (0.493pp)** | 41.2% |
| Accuracy | 0.00839 (0.839pp) | **0.00493 (0.493pp)** | 41.2% |
| Mean confidence | 0.02265 | **0.00724** | 68.0% |

| Aggregate endpoint | `g0` | Full composition |
|---|---:|---:|
| Macro normalized MAE ↓ | 0.554 | **0.254** |
| Macro controlled model-rank Spearman ↑ | 0.302 | **0.811** |
| Accuracy top-1 regret ↓ | 1.560pp | **0.382pp** |

## Primary gate

| Criterion | Result | Decision |
|---|---:|---|
| Macro normalized MAE lower | 0.254 < 0.554 | PASS |
| Hierarchical CI excludes zero | [0.152, 0.538] | PASS |
| At least four coordinates improve | 6/6 | PASS |
| Macro controlled rank improves | 0.811 > 0.302 | PASS |
| Accuracy regret no worse | 0.382pp < 1.560pp | PASS |
| Complete gate | all required | **PASS** |

Every architecture family had positive mean normalized improvement. This result is materially stronger than the earlier CIFAR-10 architecture tests, whose family-level intervals included zero.

## Five-pair magnitude confirmation

Development selected:

- `Contrast|Invert`;
- `Invert|Swirl`;
- `GaussianBlur|Invert`;
- `Invert|Rotate90`;
- `Contrast|Swirl`.

| Endpoint | Magnitude five-pair | Random strategy |
|---|---:|---:|
| Macro normalized MAE ↓ | **0.271** | 0.338 mean / 0.333 median |
| Expected-random improvement | 0.067 | — |
| Hierarchical 95% interval | **[0.024, 0.133]** | — |
| Macro controlled rank ↑ | **0.802** | gate comparator: random median |
| Accuracy regret ↓ | **0.665pp** | gate comparator: random median |

All four selector criteria passed. This is the first fresh confirmation of multi-metric magnitude-based pair selection; it does not retroactively change the Phase-2D active-selector failure.

## Redundancy sensitivity

Removing balanced accuracy from the aggregate gives:

- `g0`: 0.586 normalized MAE;
- full composition: 0.258;
- magnitude five-pair: 0.274 versus 0.350 expected random;
- full controlled rank: 0.801 versus 0.301 for `g0`;
- magnitude controlled rank: 0.795 versus 0.677 random median;
- all five remaining coordinates improve;
- full improvement interval: **[0.155, 0.617]**;
- magnitude-versus-random interval: **[0.023, 0.160]**.

Accuracy regret is unchanged by removal of a different coordinate. All five full-vector sensitivity criteria and all four magnitude sensitivity criteria pass. Thus duplicated accuracy weighting is not responsible for either decision.

## Interpretation

Phase 2F supplies the strongest evidence so far that primitive and pair measurements contain useful information about unseen compositions beyond clean/model/word metadata. It also restores the complete behavior vector rather than reporting accuracy alone.

The result does **not** establish a universal Shift Algebra. The metric-specific ridge coefficients and exponents were developed on CIFAR-100 development outcomes, so this is a sealed within-dataset confirmation of the forecasting methodology—not strict coefficient transfer from CIFAR-10. Transformations remain synthetic, models remain small and low-accuracy, and feature-drift scale is architecture dependent. A future claim of broad transfer still requires a second dataset using these now-frozen full-vector methods without refitting.
