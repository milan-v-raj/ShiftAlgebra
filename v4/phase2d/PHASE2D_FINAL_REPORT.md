# Phase 2D Final Report — Architecture-Family Holdout

**Protocol:** `V4-P2D-ARCH-HOLDOUT-CPU-001`
**Decision:** **PRIMARY ARCHITECTURE-HOLDOUT GATE FAILED (3/4 criteria passed)**

## Executive result

On two architecture families absent from all fitting data, frozen `g1` reduced MAE from **2.978pp (`g0`) to 1.800pp**, a 39.6% relative reduction. Controlled model-rank Spearman improved from **0.270 to 0.621**, and top-1 regret improved from **2.806pp to 2.147pp**. However, the lineage/reversal-clustered 95% interval for the 1.178pp MAE improvement was **[-0.042, 2.293]pp**. Because the lower endpoint did not exclude zero, the complete preregistered gate failed.

The correct interpretation is encouraging point transfer with insufficient independent-family evidence, not confirmed architecture generalization.

## Fresh test

- unseen TinyVGG and TinyDenseNet families;
- three recipes each, six independent lineages, twelve snapshots;
- 1,000 disjoint CIFAR-10 images per condition at offsets 400--499;
- 64 new reversal-paired words absent from Phases 2, 2B, and 2C;
- all methods frozen in commit `e71b360` before outcomes;
- training recorded in `9cbdc6a` before behavior evaluation.

## Primary results

| Method | MAE ↓ | Model-rank Spearman ↑ | Controlled Spearman ↑ | Top-1 regret ↓ |
|---|---:|---:|---:|---:|
| **Prior frozen `g1`** | **1.800pp** | 0.632 | 0.621 | 2.147pp |
| Full fifteen-group ACHM | 2.312pp | **0.711** | **0.645** | **0.711pp** |
| Prior frozen `g0` | 2.978pp | -0.036 | 0.270 | 2.806pp |

`g1` was best for point accuracy; full ACHM was better for model ranking and selection. No method dominated all endpoints.

## Preregistered primary gate

| Criterion | Result | Decision |
|---|---:|---|
| `g1` MAE lower than `g0` | 1.800pp < 2.978pp | PASS |
| Clustered CI excludes zero | [-0.042, 2.293]pp | **FAIL** |
| Controlled rank higher | 0.621 > 0.270 | PASS |
| Top-1 regret no worse | 2.147pp < 2.806pp | PASS |
| Complete gate | all four required | **FAIL** |

Only six independent lineages were available, making the clustered interval wide. Increasing snapshots would not fix that inferential limitation because snapshots share lineages.

## Family breakdown

| Method | TinyDenseNet MAE | TinyVGG MAE |
|---|---:|---:|
| `g1` | 2.120pp | **1.479pp** |
| `g0` | 2.573pp | 3.383pp |
| Full ACHM | **2.034pp** | 2.589pp |

The incremental result was not uniform: `g1` was especially strong on TinyVGG, while full ACHM slightly beat it on TinyDenseNet.

## Five-pair selectors

| Selector | MAE ↓ | Controlled Spearman ↑ | Top-1 regret ↓ |
|---|---:|---:|---:|
| Magnitude-only | **4.517pp** | **0.510** | 0.828pp |
| Variance-only | **4.517pp** | **0.510** | 0.828pp |
| Previously active selector | 6.071pp | 0.479 | 0.916pp |
| Coverage-only | 6.519pp | 0.429 | **0.741pp** |
| Graph D-optimal | 8.004pp | 0.385 | 0.828pp |
| Random strategy mean | 6.452pp | -- | -- |

Magnitude and variance selected the same five-group set in a different order, so their predictions are identical. These alternatives were frozen before outcomes but are secondary; no winner was chosen after the test.

## Active-selector replication

The earlier active selector improved point MAE over expected random selection (**6.071pp versus 6.452pp**) and had better controlled rank than the random median. Nevertheless:

- improvement CI: **[-0.217, 1.102]pp**;
- top-1 regret was worse than the random median;
- it beat 70% rather than 85% of random subsets.

The complete active replication gate therefore failed. Phase 2C remains a positive result for its own bank, but selection did not transfer reliably enough to unseen architectures.

## Conclusion

The strongest CPU conclusion is now more precise:

- full composition forecasts show encouraging transfer to unseen lightweight families;
- the formal architecture-holdout confirmation gate does not pass because independent evidence is too small;
- `g1` transfers better for point MAE, while full ACHM can make better model-selection decisions;
- five-pair selection is architecture sensitive, and simple magnitude selection was more robust here than the previously active heuristic.

No method should be retuned on this opened holdout. A new architecture claim requires more independent families, not more seeds or snapshots of these two families.
