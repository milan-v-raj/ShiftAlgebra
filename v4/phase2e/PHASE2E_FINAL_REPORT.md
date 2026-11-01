# Phase 2E Final Report — Prospective Architecture Breadth

**Protocol:** `V4-P2E-ARCH-BREADTH-CPU-001`  
**Decision:** **PRIMARY GATE FAILED (4/5 criteria passed)**

## Executive result

On four additional architecture families absent from method fitting and Phase 2D, unchanged `g1` reduced MAE from **4.443pp (`g0`) to 2.034pp**, a 54.2% relative reduction. Controlled model-rank Spearman improved from **0.159 to 0.755**, and top-1 regret fell from **7.348pp to 0.000pp**. However, the preregistered family/lineage/reversal hierarchical interval for the 2.409pp improvement was **[-0.072, 5.287]pp**. Because the interval included zero, the complete gate failed.

Phase 2E therefore adds strong but heterogeneous point evidence. It does not convert the failed Phase-2D architecture claim into a pass.

## Prospective test

- TinyInception, TinySqueezeNet, TinyMixer, and TinyAttention;
- three recipes per family, twelve independent lineages, 24 snapshots;
- clean accuracy range 20.3%--48.6%;
- disjoint CIFAR-10 offsets 500--599;
- 60 fresh reversal-paired words, twenty each at lengths 4, 5, and 6;
- exact inherited methods and gates frozen in commit `8115a11`;
- training provenance committed as `2a7350b` before evaluation.

## Primary methods

| Method | MAE ↓ | Model-rank Spearman ↑ | Controlled Spearman ↑ | Top-1 regret ↓ |
|---|---:|---:|---:|---:|
| Frozen `g1` | 2.034pp | **0.854** | **0.755** | **0.000pp** |
| Full ACHM | **1.933pp** | 0.813 | 0.741 | **0.000pp** |
| Frozen `g0` | 4.443pp | 0.240 | 0.159 | 7.348pp |

Full ACHM slightly wins point MAE, while `g1` wins raw and controlled ranking. Both select an oracle top model on these environments.

## Preregistered gate

| Criterion | Outcome | Decision |
|---|---:|---|
| `g1` MAE lower | 2.034pp < 4.443pp | PASS |
| Family/lineage/reversal CI excludes zero | [-0.072, 5.287]pp | **FAIL** |
| At least three positive family means | 3/4 | PASS |
| Controlled rank improves | 0.755 > 0.159 | PASS |
| Regret no worse | 0.000pp < 7.348pp | PASS |
| Complete gate | all five required | **FAIL** |

## Architecture heterogeneity

| Family | `g0` MAE | `g1` MAE | Improvement |
|---|---:|---:|---:|
| TinyMixer | 8.826pp | 2.157pp | **+6.669pp** |
| TinySqueezeNet | 4.228pp | 1.944pp | **+2.285pp** |
| TinyInception | 2.958pp | 1.632pp | **+1.327pp** |
| TinyAttention | **1.760pp** | 2.402pp | **-0.642pp** |

The large TinyMixer effect drives much of the average. The negative TinyAttention result explains why family-level uncertainty remains material and must not be hidden by pooling snapshots.

## Equal-budget selectors

| Selector | MAE ↓ | Controlled Spearman ↑ | Top-1 regret |
|---|---:|---:|---:|
| Magnitude / variance | **3.055pp** | 0.722 | 0.000pp |
| Previously active | 3.664pp | **0.728** | 0.000pp |
| Random strategy mean | 4.699pp | -- | -- |
| Coverage | 4.861pp | 0.651 | 0.000pp |
| D-optimal | 5.080pp | 0.628 | 0.000pp |

These comparisons are descriptive because no selector replication gate was preregistered for Phase 2E. Magnitude and variance again select the same groups.

## Length breakdown

| Length | `g0` | `g1` | Full ACHM |
|---:|---:|---:|---:|
| 4 | 4.410pp | **1.810pp** | 1.601pp |
| 5 | 4.361pp | **2.072pp** | 1.994pp |
| 6 | 4.558pp | **2.219pp** | 2.204pp |

The `g1` advantage remains positive at every tested length, although absolute error increases modestly.

## Conclusion

Across these four families, composition measurements provide large practical point improvement over metadata-only forecasting. The evidence is not uniform: TinyAttention favors `g0`, and the family-hierarchical interval narrowly includes zero. The honest decision is therefore another failed formal architecture gate with substantially strengthened descriptive breadth.
