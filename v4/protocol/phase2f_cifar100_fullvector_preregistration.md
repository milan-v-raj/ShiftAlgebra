# Phase 2F CIFAR-100 Full-Vector CPU Protocol

**Protocol:** `V4-P2F-CIFAR100-FULLVECTOR-CPU-001`  
**Stage:** bank, metrics, development process, and confirmation gate frozen before any Phase-2F behavior outcome

## Question

Do composition measurements improve target-free prediction of the complete historical behavior vector on a new dataset, beyond clean/model/word metadata?

## Complete output vector

Every method must output all six coordinates for every model and word:

1. mean top-1/top-2 probability margin;
2. L2 drift of the mean penultimate feature from the clean-image centroid;
3. ten-bin expected calibration error;
4. macro per-class (balanced) accuracy;
5. overall accuracy;
6. mean maximum softmax confidence.

No result may be summarized using accuracy alone. Because CIFAR-100 test classes are balanced, balanced and overall accuracy are expected to be numerically redundant; both remain present for compatibility and this redundancy must be disclosed.

## Data and banks

- canonical CIFAR-100 fine labels (100 classes, chance 0.01);
- lossless class-folder transport pinned and audited against all 10,000 canonical test images;
- six architecture families inherited from Phases 2D/2E;
- three recipes per family;
- 18 development and 18 confirmation lineages, each with epoch-1 and epoch-2 snapshots;
- disjoint balanced test partitions: offsets 0--19 for development and 20--39 for confirmation;
- 60 development and 60 confirmation reversal-paired words, disjoint from each other and every prior phase;
- lengths 4--6 only because distinct-primitive length-3 space is exhausted.

## Development then freeze

Development outcomes may be used only to:

- select each metric's pair-scaling exponent from the frozen grid 0.0--2.0 by 0.1;
- fit metric-specific ridge `g0` and `g1` models;
- freeze development normalization scales;
- select five pair groups by standardized multi-metric residual magnitude;
- fit equal-budget magnitude and twenty deterministic random controls.

`g0` uses clean metric values, model metadata, length, and primitive counts. `g1` adds ordered, symmetric, independent/additive, and envelope composition forecasts. Confirmation outcomes are forbidden until the method artifact is committed.

## Primary full-vector gate

All conditions are required on confirmation:

1. `g1` macro normalized MAE is below `g0`;
2. a 10,000-draw family/lineage/reversal hierarchical interval for normalized error improvement excludes zero;
3. `g1` raw MAE is lower on at least four of six coordinates;
4. macro controlled model-rank Spearman across all six coordinates is higher;
5. accuracy top-1 selection regret is no worse.

Normalization uses only frozen development target standard deviations. Raw per-coordinate MAE is always reported.

## Magnitude-selector confirmation gate

The five-pair magnitude design must simultaneously beat expected random five-pair macro normalized MAE, have a clustered improvement interval excluding zero, attain macro controlled rank at least the random median, and have accuracy regret no worse than the random median.

## Information boundary

Composed confirmation images and labels are opened once after method freeze. Positive and negative outcomes are retained without replacing metrics, architectures, words, pair designs, or gates.
