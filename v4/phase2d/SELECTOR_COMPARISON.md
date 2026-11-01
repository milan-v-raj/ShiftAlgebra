# Five-Pair Selector Comparison

All selectors use five unordered groups, ten directed measurements, the same ACHM gate family, fixed L2 regularization, and original Phase-2 development roles only. Every design was frozen before Phase-2D architecture outcomes.

| Selector | Selection signal | Development MAE | Architecture-holdout MAE | Controlled rank | Top-1 regret |
|---|---|---:|---:|---:|---:|
| Magnitude | Mean absolute symmetric + asymmetric residual | **2.151pp** | **4.517pp** | **0.510** | 0.828pp |
| Variance | Across-model interaction variability | **2.151pp** | **4.517pp** | **0.510** | 0.828pp |
| Active | Magnitude × word frequency + coverage bonus | 2.667pp | 6.071pp | 0.479 | 0.916pp |
| Coverage | New-node and degree-balance objective | 2.812pp | 6.519pp | 0.429 | **0.741pp** |
| Graph D-optimal | Log-determinant of primitive-incidence information | 3.287pp | 8.004pp | 0.385 | 0.828pp |
| Uniform random | Twenty frozen subsets | -- | 6.452pp mean | -- | -- |

Magnitude and variance selected the same set, so their forecasts are identical. The architecture holdout favors direct interaction magnitude for MAE/rank and coverage for selection regret. The originally active heuristic does not dominate these simpler alternatives.

This is a prospective secondary comparison, not a new primary gate. It covers only two weak architecture families. A future selector claim requires another fresh dataset or architecture bank and should compare additional objectives without selecting among these results post hoc.
