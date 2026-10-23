# Phase 1 Preregistration: Mason Exact-Protocol Bridge

**Status:** DEVELOPMENT LABELS ACCESSED; CONFIRMATORY IMAGE SUBSETS NOT YET EVALUATED
**Protocol ID:** `V4-P1-MASON-001`
**Source manifest:** `v4/benchmark/manifests/mason_exact.json`
**Upstream source commit:** `fc62a92ccd11f64eec26c7a019e9e0957fd9bd33`
**Freeze requirement:** replace all `TBD` fields and commit before running benchmark inference

## 1. Question

Can a predictor calibrated on primitive and ordered-pair behavior forecast an independently designed set of length-3--6 corruption compositions without processing images from those test compositions at prediction time?

## 2. Independently fixed protocol

Use Mason et al.'s six primitives and upstream seed `2778`:

- Contrast;
- GaussianBlur;
- ImpulseNoise;
- Invert;
- Rotate90;
- Swirl.

The word set contains, per dataset:

- identity: 1;
- singles: 6;
- all distinct ordered pairs: 30;
- triples: 40;
- length four: 30;
- length five: 30;
- length six: 30;
- total: 167.

Apply this fixed word list separately to EMNIST, CIFAR-10, and FaceScrub, subject to license review. Do not resample words.

## 3. Information regimes

### Primary R2 condition

Available to the forecast model:

- clean/source validation labels;
- all six primitive behavior vectors and fingerprints;
- all 30 ordered-pair calibration labels;
- model metadata.

Unavailable at prediction time:

- images from length-3--6 target words;
- labels from length-3--6 target words.

### References

- R0 source/model baseline;
- R1 primitive-only baselines;
- R3 target-image methods only if the exact target images are supplied, clearly reported as information advantaged;
- R4 direct measurements as oracle only.

## 4. Models

The first reproduction should include:

1. upstream ERM classifier;
2. upstream contrastive classifier;
3. upstream modular classifier;
4. one independently trained standard classifier per dataset.

Exact checkpoint URLs, hashes, preprocessing, and training commands: **TBD before freeze**.

If official weights are unavailable, retrain from the pinned upstream implementation with three declared seeds and disclose that this is a protocol reproduction, not a checkpoint reproduction.

## 5. Primary methods

1. `M_B_core-small` with `p >= 1`;
2. `M_Lp_only`;
3. source/family ridge `g0`;
4. additive behavior;
5. mean/dominant-shift baseline;
6. pairwise ANOVA/ridge;
7. parameter-matched GRU;
8. parameter-matched ordered Transformer;
9. low-rank bilinear recurrent model.

Hyperparameter grids and selection metric: **TBD before freeze**.

## 6. Outcomes

### Primary

Macro accuracy MAE in percentage points over length-3--6 words, first averaged within `(dataset, model, length)`, then across those clusters.

### Secondary

- RMSE and worst-decile MAE;
- accuracy MAE separately at lengths 3, 4, 5, 6;
- loss MAE;
- within-model rank of words;
- reverse-word order sign and MAE where both words occur;
- calibration error if predictive probabilities are retained;
- compute and calibration cost.

### Algebra diagnostics

- exact inverse error;
- matrix/streaming/closed-form equivalence;
- symmetric reversal residual;
- ordered reversal residual;
- identity error;
- end-to-end associator.

## 7. Statistical unit and inference

- primary cluster: `(dataset, model, unordered primitive set)`;
- severities/word orders nested inside a cluster;
- paired hierarchical bootstrap over dataset -> model -> word family;
- 10,000 bootstrap draws in the frozen analysis;
- Holm correction across the declared primary baseline family;
- report effect sizes and 95% confidence intervals.

Seeds measure optimization stability and are not treated as independent dataset replicates.

## 8. Gate

Phase 1 supports continued full-model development only if:

1. `M_B_core-small` improves macro length-3--6 accuracy MAE over the best non-algebraic R2 baseline;
2. the family-clustered 95% CI excludes zero;
3. it improves over `M_Lp_only` on the preregistered order-sensitive subset;
4. no dataset has more than 5% relative material harm versus the best R2 baseline;
5. all invariant/property tests pass.

If (1) passes but (3) fails, continue with the saturation law as the primary model and demote the order component. If (1) fails, do not tune on the same held-out words; diagnose on a separately declared development subset.

## 9. Exclusions

A condition may be excluded only for:

- corrupt output that violates the pinned implementation contract;
- unavailable/invalid source image;
- class-mapping failure;
- non-finite model output.

Every exclusion must be emitted before target metrics are computed and recorded in a machine-readable file. Performance-based exclusions are forbidden.

## 10. Freeze checklist

- [ ] License review complete.
- [ ] Dataset versions and hashes recorded.
- [ ] Checkpoints and preprocessing frozen.
- [ ] Manifest hash copied below.
- [ ] Method hyperparameter budgets frozen.
- [ ] Statistical script frozen.
- [ ] Exclusion script frozen.
- [ ] Compute budget approved.
- [x] Development length-3--6 labels were confined to the declared development image subset.
- [ ] Non-overlapping confirmatory images remain unevaluated until the freeze commit.

## 11. Amendment P1-A1 after development, before confirmation

Official upstream checkpoints were not published, and the internal FaceScrub subset is unavailable. The executable bridge therefore uses independently trained frozen SmallCNN classifiers on CIFAR-10 and EMNIST. This is an exact word-list bridge, not a checkpoint or pixel-bitwise reproduction.

Frozen independent classifiers:

- CIFAR-10: SmallCNN width 24, seed 1301, 8 epochs, AdamW, batch 256, learning rate `3e-3`, weight decay `1e-4`;
- EMNIST Balanced: SmallCNN width 16, seed 1301, 5 epochs, AdamW, batch 512, learning rate `3e-3`, weight decay `1e-4`.

Development subsets:

- CIFAR-10: first 100 stable test examples/class;
- EMNIST Balanced: first 40 stable test examples/class.

Reserved one-shot confirmation:

- CIFAR-10: remaining 900 examples/class, offset 100;
- EMNIST Balanced: remaining 360 examples/class, offset 40.

Development exposed severe accuracy saturation. Before confirmation, the following rules are frozen for every method:

1. fit signed-power exponents on 30 calibration pairs only;
2. freeze exponents before fitting interaction parameters;
3. constrain predicted absolute accuracy to `[0,1]` and report the clipping fraction;
4. fit no parameter on length-3--6 confirmatory outcomes;
5. retain the max-absolute primitive envelope as a mandatory baseline.

The primary result remains macro accuracy MAE over 130 length-3--6 words. The expected outcome is not changed: failure to beat the max envelope is a failed Phase-1 method gate and must be reported.

**Generated manifest SHA-256:** `d2be1ca5c899659482204630d1f2b755587b71ee994372a7b716e5c5ad591264`
**Frozen executable Git commit:** `b761987bc691db191bd2617ad05382f67a68ba18`
**Container digest:** `TBD`
