# Phase 3B Preregistration — Independent-Dataset Realistic-Corruption Replication

**Protocol:** `V4-P3B-CIFAR10-REALISTIC-REPLICATION-001`  
**Frozen date:** 2026-10-07  
**Outcome status at freeze:** no Phase-3B development or confirmation behavior opened

## Question

Does the favorable error/rank result—and potentially the unfavorable selection-regret result—from Phase 3A replicate when the dataset, class count, model seeds, image partitions, and target words change from CIFAR-100 to CIFAR-10?

This is an independent-dataset replication of the Phase-3A method class, not a coefficient-level no-refit transfer. Phase-3B coefficients and scales may be fitted only on Phase-3B development behavior. The primary seven-part gate is unchanged.

## Fixed corruption system

Phase 3B reuses the Phase-3A corruption source byte-for-byte:

1. brightness offset;
2. Gaussian noise;
3. JPEG-style block-DCT quantization;
4. diagonal motion blur;
5. pixelation;
6. signal-dependent shot-noise approximation.

These are deterministic clean-room ImageNet-C-style categories. They are not official CIFAR-C or ImageNet-C implementations and their parameters are not official severity levels.

## Development and confirmation boundary

Development:

- 18 fresh CIFAR-10 training lineages and 36 epoch snapshots;
- six lightweight architecture families and three recipes;
- seeds 73100--73602;
- CIFAR-10 test offsets 600--799, 200 images/class;
- 60 development targets in 30 reversal groups.

Confirmation:

- 18 separately seeded lineages and 36 snapshots;
- seeds 74100--74602;
- test offsets 800--999, 200 images/class;
- 60 disjoint targets in 30 reversal groups;
- behavior remains sealed until the development-fitted method and confirmation checkpoint hashes are separately committed.

All 120 Phase-3B target words are disjoint from all 120 Phase-3A target words. Calibration in each split contains identity, six singles, and all 30 directed distinct-primitive pairs.

The Phase-3B image partitions are disjoint from one another and from the Phase-2 through Phase-2E model-bank partitions. They are not claimed globally untouched: historical Phase-1 broad confirmation used much of the CIFAR-10 test set.

## Frozen method

The primary remains the fitted order-blind symmetric composition method. It averages both directions of every calibrated pair before target feature construction. Comparators remain:

- metadata/clean-reference ridge `g0`;
- independent-retention formula;
- envelope formula;
- unfitted symmetric-pair formula.

The sole dataset-required structural change is the chance floor: 0.10 for CIFAR-10 rather than 0.01 for CIFAR-100. No Phase-3A confirmation result may select a Phase-3B exponent, coefficient, target scale, feature, word, model, or corruption parameter.

Every method forecasts margin, feature drift, ECE, balanced accuracy, overall accuracy, and mean confidence.

## Primary all-required gate

All seven criteria must pass:

1. fitted symmetric macro normalized MAE is below `g0`;
2. the architecture-family/lineage/reversal hierarchical 95% improvement interval excludes zero;
3. at least four of six raw coordinate MAEs improve;
4. fitted symmetric macro error is below both independent and envelope formulas;
5. macro controlled model-rank Spearman improves over `g0`;
6. accuracy top-1 selection regret is no worse than `g0`;
7. at least five of six architecture-family mean improvements are positive.

Ten thousand deterministic bootstrap draws resample family, lineage within family, and reversal group as a crossed axis. Epoch snapshots and reversal orientations remain clustered.

## Replication reporting

After the primary gate is evaluated, report descriptively:

- agreement with Phase 3A on each of the seven gate directions;
- agreement on all six coordinate improvement directions;
- reversal behavior and all family signs;
- sensitivities by epoch, word length, family, recipe, and removal of redundant balanced accuracy.

Replication agreement cannot modify the primary Phase-3B decision.

## Outcome handling

The result is retained whether favorable or unfavorable. No order-aware rescue, selection-specific tuning, gate tolerance, or corruption-severity adjustment is permitted after confirmation opens.

Passing would support replication of symmetric composition forecasting across CIFAR-100 and CIFAR-10 under one clean-room corruption implementation. It would not establish official corruption-benchmark, natural-shift, modern-checkpoint, independent-implementation, or universal operator-algebra generalization.
