# Phase 2G Preregistration — Strict No-Refit Full-Vector Architecture Transfer

**Protocol:** `V4-P2G-CIFAR100-NOREFIT-ARCH-001`

## Question

Do the Phase-2F six-coordinate coefficients, exponents, normalization scales, and pair-selection designs transfer without refitting to six architecture families absent from every Phase-2F fit and confirmation?

This is a CIFAR-100 architecture-transfer test, not an independent-dataset test. It directly addresses coefficient and architecture transfer while retaining the CPU-only scope.

## Frozen information boundary

The complete Phase-2F method file transfers byte-for-byte:

- path: `v4/phase2f/FROZEN_FULL_VECTOR_METHODS.json`;
- SHA-256: `243761ddd3749dcc0149e83cca7e40241d43a3800320d8f777bac624b1f5827e`;
- frozen development behavior SHA-256: `8c3289ac47c01267751a2f3d8104d1940f04e20b8fa0cec4dea0c2baef7b90a7`.

No Phase-2G outcome may change metric-specific `q`, coefficients, feature normalization, target normalization, full-pair design, magnitude five-pair design, or twenty random controls. The six new family names are not added to the frozen regressions: all old-family indicator columns are zero. Existing recipe indicators remain available.

Permitted per-model inference inputs are clean, six single-primitive, and the pair conditions required by each frozen design, plus model/word metadata. Confirmation target values are forbidden from fitting and feature construction.

## Bank and target freeze

- Dataset: CIFAR-100, with Phase-2F provenance unchanged.
- Architectures: residual, depthwise-separable, ConvNeXt-style, grouped/shuffle, dilated-pyramid, and gated-convolution families.
- Recipes: ERM+flip, label smoothing, and noise augmentation.
- Independent units: 18 lineages, three per family; epochs 1 and 2 remain clustered within lineage.
- Confirmation images: 20 images/class at offsets 40--59, disjoint from Phase-2F offsets 0--39.
- Targets: 60 fresh words, 30 reversal groups, lengths 4--6, disjoint from all prior frozen word banks.
- Calibration: identity, six singles, and thirty directed distinct pairs.
- Outputs: margin, feature drift, ECE, balanced accuracy, accuracy, and mean confidence.

Training checkpoints and hashes will be recorded before behavior evaluation. Confirmation behavior is evaluated once after that record is committed.

## Primary full-vector gate

All six criteria are required:

1. frozen full-pair macro normalized MAE is below frozen metadata-only `g0`;
2. the architecture-family/lineage/reversal hierarchical 95% improvement interval excludes zero;
3. at least four of six raw coordinate MAEs improve;
4. macro controlled model-rank Spearman improves;
5. accuracy top-1 selection regret is no worse;
6. at least five of six held-out architecture-family mean improvements are positive.

## Frozen magnitude five-pair gate

All four criteria are required against the expected prediction of the twenty already-frozen random five-pair designs:

1. macro normalized MAE is below the random-design mean;
2. the hierarchical 95% improvement interval excludes zero;
3. macro controlled rank is at least the random-design median;
4. accuracy regret is no worse than the random-design median.

## Inference

The primary interval resamples architecture family, lineage within sampled family, and reversal group as a crossed target axis. Both epoch snapshots remain within a lineage. Ten thousand deterministic bootstrap draws are used. Raw per-coordinate errors, aggregate normalized errors, ranks, regret, and all family signs are retained regardless of outcome.

## Interpretation boundary

Passing supports strict no-refit architecture transfer on CIFAR-100 for these lightweight CPU models and synthetic compositions. Failing is retained and rejects that claim under this protocol. Neither outcome establishes independent-dataset, natural-shift, ImageNet, modern-checkpoint, or universal-algebra validity.
