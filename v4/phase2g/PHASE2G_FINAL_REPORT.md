# Phase 2G Final Report — Strict No-Refit Architecture Transfer

**Protocol:** `V4-P2G-CIFAR100-NOREFIT-ARCH-001`

**Primary decision:** **STRICT NO-REFIT FULL-VECTOR GATE PASSED (6/6)**

**Budgeted decision:** **FROZEN MAGNITUDE FIVE-PAIR GATE PASSED (4/4)**

## Executive result

The complete Phase-2F method transferred byte-for-byte to six unseen CIFAR-100 architecture families. No exponent, coefficient, input scale, target scale, feature definition, or pair design was refit. Frozen full composition reduced macro normalized six-coordinate MAE from **0.912 (`g0`) to 0.326**, a **64.3% relative reduction**. It improved all six coordinates, raised macro controlled model-rank Spearman from **0.054 to 0.779**, and reduced accuracy top-1 selection regret from **2.872pp to 0.307pp**.

The architecture-family/lineage/reversal hierarchical interval for normalized improvement was **[0.381, 0.789]**. Every one of the six held-out families had positive mean improvement. All six preregistered criteria passed.

The already-frozen Phase-2F magnitude five-pair design also passed all four criteria. Its macro normalized MAE was **0.329**, versus **0.484** for the expected prediction of twenty frozen random five-pair designs. Its hierarchical improvement interval was **[0.089, 0.234]**.

## Strict transfer boundary

The transferred file was:

- `v4/phase2f/FROZEN_FULL_VECTOR_METHODS.json`;
- SHA-256 `243761ddd3749dcc0149e83cca7e40241d43a3800320d8f777bac624b1f5827e`;
- tied to Phase-2F development behavior SHA-256 `8c3289ac47c01267751a2f3d8104d1940f04e20b8fa0cec4dea0c2baef7b90a7`.

Phase 2G added no family indicator to the frozen model. Residual, depthwise-separable, ConvNeXt-style, grouped/shuffle, dilated-pyramid, and gated-convolution families therefore received zero in all six Phase-2F family columns. Existing recipe indicators were retained. Target outcomes were never available to feature construction or fitting.

## Complete six-coordinate result

Raw MAEs are in each coordinate's natural units.

| Coordinate | Frozen `g0` | Frozen full composition | Relative reduction |
|---|---:|---:|---:|
| Margin | 0.01293 | **0.00541** | 58.2% |
| Feature drift | 1.54717 | **0.31671** | 79.5% |
| ECE | 0.04188 | **0.02167** | 48.3% |
| Balanced accuracy | 0.01727 | **0.00599** | 65.3% |
| Accuracy | 0.01727 | **0.00599** | 65.3% |
| Mean confidence | 0.02625 | **0.00988** | 62.4% |

| Aggregate endpoint | Frozen `g0` | Frozen full composition |
|---|---:|---:|
| Macro normalized MAE ↓ | 0.912 | **0.326** |
| Macro controlled model-rank Spearman ↑ | 0.054 | **0.779** |
| Accuracy top-1 regret ↓ | 2.872pp | **0.307pp** |

## Primary gate

| Criterion | Result | Decision |
|---|---:|---|
| Macro normalized MAE lower | 0.326 < 0.912 | PASS |
| Hierarchical CI excludes zero | [0.381, 0.789] | PASS |
| At least four coordinates improve | 6/6 | PASS |
| Macro controlled rank improves | 0.779 > 0.054 | PASS |
| Accuracy regret no worse | 0.307pp < 2.872pp | PASS |
| At least five families positive | 6/6 | PASS |
| Complete strict-transfer gate | all required | **PASS** |

Family mean normalized improvements were positive for ConvNeXt (0.213), depthwise (0.780), dilated (0.377), gated (0.622), residual (0.866), and shuffle (0.660).

## Frozen magnitude five-pair result

The exact Phase-2F magnitude groups transferred unchanged:

- `Contrast|Invert`;
- `Invert|Swirl`;
- `GaussianBlur|Invert`;
- `Invert|Rotate90`;
- `Contrast|Swirl`.

| Endpoint | Frozen magnitude five-pair | Frozen random strategy |
|---|---:|---:|
| Macro normalized MAE ↓ | **0.329** | 0.484 mean / 0.474 median |
| Expected-random improvement | 0.154 | — |
| Hierarchical 95% interval | **[0.089, 0.234]** | — |
| Macro controlled rank ↑ | **0.764** | 0.694 median |
| Accuracy regret ↓ | **0.590pp** | 0.994pp median |

All four magnitude criteria passed.

## Redundancy sensitivity

Balanced and overall accuracy are effectively identical on the balanced partition. Excluding balanced accuracy preserves every criterion:

- macro normalized MAE: 0.933 (`g0`) to 0.335 (full);
- full controlled rank: 0.030 to 0.799;
- all five remaining coordinates and all six families improve;
- full improvement interval: **[0.385, 0.809]**;
- magnitude MAE: 0.337 versus 0.504 expected random;
- magnitude improvement interval: **[0.094, 0.254]**.

Thus duplicated accuracy weighting does not cause either passed gate.

## Interpretation

Phase 2G closes the largest qualification on Phase 2F: its positive result is not dependent on refitting coefficients for the confirmation architectures. The same six-output method and the same magnitude pair selector transfer strongly to six architecture families that were absent from all Phase-2F development and confirmation.

The scope remains limited. This is within-dataset transfer on CIFAR-100, using the same synthetic transformation system, three known recipes, tiny two-epoch CPU models, and pair calibration for each model. It is not independent-dataset transfer, zero-calibration prediction, a natural-shift result, an ImageNet result, evidence for modern public checkpoints, or a universal Shift Algebra.
