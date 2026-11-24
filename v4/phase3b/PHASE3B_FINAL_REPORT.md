# Phase 3B Final Report — Prospective CIFAR-10 Replication

**Protocol:** `V4-P3B-CIFAR10-REALISTIC-REPLICATION-001`

**Decision:** **CROSS-DATASET REPLICATION GATE FAILED (6/7 criteria passed)**

**Interpretation:** the favorable six-metric error, rank, uncertainty, and family-sign pattern replicated on CIFAR-10, but the unchanged all-required gate failed because accuracy-selection regret was slightly worse than the frozen baseline.

## Scope

Phase 3B prospectively replicated Phase 3A on CIFAR-10 while retaining its six deterministic clean-room corruption operators, order-blind symmetric primary method, four comparators, six-coordinate behavior vector, and seven conjunctive gates. The sole dataset-required method change was the accuracy chance floor, from 0.01 for CIFAR-100 to 0.10 for CIFAR-10. All exponents, scales, and coefficients were fitted afresh using Phase-3B development behavior only.

The development and confirmation banks each contained 18 fresh training lineages and 36 epoch snapshots spanning six lightweight architecture families and three recipes. Each split evaluated 97 conditions on 2,000 images: identity, six singles, all 30 directed distinct-primitive pairs, and 60 length-four through length-six target words arranged as 30 exact reversal pairs.

These are **clean-room ImageNet-C-style categories**, not official CIFAR-C or ImageNet-C implementations or severity levels. CIFAR-10 is a different dataset from Phase 3A's CIFAR-100, but the Phase-3B images are not claimed globally untouched because historical Phase 1 used a broad CIFAR-10 confirmation partition.

## Prospective chronology

1. Protocol, corruption code, fresh seeds, image partitions, model rosters, target words, metrics, methods, comparators, and gates were frozen and pushed at `603e2b5` before any Phase-3B behavior was opened.
2. Development training and behavior were completed once. The fitted method and its source behavior were committed and pushed at `18282c5` before confirmation training began.
3. All 36 fresh confirmation checkpoints were trained and hashed. Their ledger was committed and pushed with `behavior_opened: false` at `c7d5b57`.
4. Confirmation behavior was then opened once, forecast, analyzed, stress-tested, and independently verified. No frozen exponent, coefficient, scale, corruption, partition, target, comparator, or gate was changed.

## Primary errors

All MAEs are in native coordinate units. Macro error divides each coordinate by its development-only target standard deviation and averages the six coordinates.

| Coordinate | Metadata/clean `g0` MAE | Fitted symmetric MAE | Relative reduction | Hierarchical normalized improvement 95% CI |
|---|---:|---:|---:|---:|
| Margin | 0.058991 | **0.041519** | 29.6% | [0.09312, 0.24132] |
| Feature drift | 0.581925 | **0.428119** | 26.4% | [0.06519, 0.23353] |
| ECE | 0.064706 | **0.047189** | 27.1% | [0.08873, 0.21987] |
| Balanced accuracy | 0.027605 | **0.018341** | 33.6% | [0.08987, 0.21371] |
| Accuracy | 0.027605 | **0.018341** | 33.6% | [0.08987, 0.21371] |
| Mean confidence | 0.046269 | **0.032125** | 30.6% | [0.08939, 0.21276] |

The fitted symmetric method improved all six point MAEs. Macro normalized MAE fell from **0.517437 to 0.362973**, a **29.9% relative reduction**. The family/lineage/reversal hierarchical improvement was **0.154464**, 95% CI **[0.108865, 0.195970]**. Every coordinate-specific interval also excluded zero.

Balanced and overall accuracy are numerically redundant on the class-balanced image partitions, but both were retained exactly as preregistered.

## Comparator results

`g0` uses clean-reference behavior plus static model and word metadata, but no single- or pair-corruption behavior. The inherited short label “metadata/clean” should not be read as behavior-free.

| Method | Macro normalized MAE ↓ | Macro controlled model-rank Spearman ↑ | Accuracy top-1 regret ↓ |
|---|---:|---:|---:|
| Metadata/clean ridge `g0` | 0.517437 | 0.184997 | **0.7125 pp** |
| **Fitted symmetric composition** | **0.362973** | **0.666781** | 0.7275 pp |
| Independent-retention formula | 1.014509 | 0.500997 | 0.8050 pp |
| Envelope formula | 0.482604 | 0.450510 | **0.2325 pp** |
| Unfitted symmetric-pair formula | 0.411646 | 0.660083 | 0.7667 pp |

The fitted method beat all four comparators on macro normalized MAE and sharply improved controlled rank over `g0`. Selection regret tells a different story: the envelope formula selected better models despite having worse full-vector error, and the fitted method was 0.0150 percentage points worse than `g0`.

## Unchanged preregistered gate

| Criterion | Result |
|---|---|
| Macro normalized MAE below `g0` | PASS |
| Hierarchical 95% improvement interval excludes zero | PASS |
| At least four of six coordinate MAEs improve | PASS (6/6) |
| Better macro error than independent and envelope formulas | PASS |
| Macro controlled rank improves over `g0` | PASS |
| Accuracy top-1 selection regret is no worse than `g0` | **FAIL** (0.7275 vs 0.7125 pp) |
| At least five of six family improvements are positive | PASS (6/6) |

Because the gate was conjunctive, the final decision is **FAILED**. The unfavorable selection result is retained without tuning. No tolerance was preregistered, so the strict numerical worsening fails even though the absolute difference is only 0.0150 percentage points.

## Replication, family, reversal, and robustness results

Phase 3B agreed with Phase 3A on the direction of all six coordinate improvements and all seven gate booleans. Both phases improved error, rank, uncertainty, and every family mean, and both failed only the strict accuracy-regret criterion. This agreement is descriptive and cannot change either phase's primary decision.

All six Phase-3B architecture-family mean normalized improvements were positive:

- ConvNeXt holdout: 0.221746;
- dilated holdout: 0.170341;
- depthwise holdout: 0.147392;
- shuffle holdout: 0.146692;
- residual holdout: 0.137892;
- gated holdout: 0.102723.

The symmetric model was exactly invariant under reversal. Real outcomes were not: mean absolute reversal gaps were 0.059542 margin, 0.619297 feature drift, 0.066449 ECE, 0.024701 balanced accuracy, 0.024701 accuracy, and 0.046468 mean confidence. This is an explicit limitation, not a reason to reopen the completed Phase-2H order-awareness program.

The labeled no-retuning sensitivity that removes redundant balanced accuracy retained an improvement of **0.155320**, hierarchical 95% CI **[0.107411, 0.197387]**. Improvements were positive for both epochs, all word lengths, all six families, and all three recipes. These checks do not replace the failed primary gate.

## Correctness and leakage audit

The post-outcome verifier passed **40/40** fail-closed checks. It independently recalculated all per-coordinate and macro MAEs, reconstructed all seven gate decisions, checked Phase-3A replication accounting, confirmed exact order blindness, validated checkpoint/behavior/forecast/method/development hashes, matched every forecast target to measured behavior, and confirmed disjoint Phase-3B model, image, and target-word partitions. It also confirmed that all Phase-3B targets are disjoint from Phase 3A and that the Phase-3A and Phase-3B corruption source files are byte-identical.

The detailed review is in [`INFORMATION_AUDIT.md`](INFORMATION_AUDIT.md). No confirmation-outcome path into fitting was found. As in Phase 3A, calibration and target outcomes are written by one evaluator before the frozen forecaster runs; isolation therefore rests on pushed chronology, immutable method hashes, fail-closed source checks, and audited dataflow rather than a separate cryptographic prediction service.

## Conclusion

Phase 3B supplies a favorable prospective cross-dataset replication of the **error and ranking pattern**: all six errors improve, every hierarchical interval is positive, controlled ranking rises substantially, every family sign is favorable, and the fitted symmetric method has the best macro error among all five methods. It nevertheless does **not** pass the complete replication gate because accuracy-selection regret worsens slightly.

The justified headline is narrow: **symmetric composition-aware forecasting replicated across CIFAR-100 and CIFAR-10 for six-metric error and ranking, but robust model-selection improvement did not replicate and the complete gates failed in both datasets.** This does not establish official CIFAR-C/ImageNet-C performance, globally independent data, natural-shift transfer, modern-model transfer, or a broadly reliable order-aware benefit.
