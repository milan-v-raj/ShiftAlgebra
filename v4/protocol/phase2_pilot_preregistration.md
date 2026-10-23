# Phase 2 Preregistration — CIFAR Model-Bank CPU Pilot

**Protocol:** `V4-P2-CIFAR-BANK-PILOT-001`
**Status:** STAGE A — CODE, BANK, SPLITS, METHODS, AND GATES FROZEN; NO PHASE-2 OUTCOMES OPENED
**Date:** 2026-10-23

## 1. Scope

This protocol starts Phase 2 with the largest scientifically useful bank feasible in the current 2-CPU environment. It is a development pilot, not the roadmap's final 75-model CIFAR-10/CIFAR-100 benchmark.

The pilot asks whether a composition forecast adds cross-model prediction and model-selection value beyond clean accuracy, architecture family, training recipe, and word metadata.

It also tests a prospectively specified correction motivated by Phase 1B: pair interactions may need to accumulate extensively with word length rather than be averaged away.

## 2. Bank design

Dataset: hash-validated CIFAR-10 using the Phase-1 train/validation split.

Five lightweight architecture families:

- shallow convolutional network;
- residual network;
- depthwise-separable network;
- patch-convolution network;
- pointwise-heavy convolutional network.

Three training recipes:

- ERM with horizontal flips;
- ERM with horizontal flips and label smoothing 0.1;
- ERM with horizontal flips and Gaussian-noise augmentation.

Each lineage trains for two epochs with AdamW, learning rate `3e-3`, weight decay `1e-4`, and batch size 512. Epoch-one and epoch-two snapshots are both deployment candidates but are not treated as independent training replicates.

Model split:

- development: 15 lineages / 30 snapshots, seeds in the 22100 series;
- confirmation: 15 separately trained lineages / 30 snapshots, seeds in the 32100 series.

Inference and uncertainty cluster epoch snapshots by lineage.

This reaches the roadmap's 30-checkpoint development-bank target, but not its final 75+ checkpoint, six-family, four-recipe target. Phase 2 remains incomplete at final-benchmark scale regardless of pilot outcome.

## 3. Image partitions

- Development-bank behavior: 50 images/class at class-local test offset 0, 500 total.
- Confirmation-bank behavior: 100 images/class at offset 100, 1,000 total.

The image partitions are disjoint. Base images overlap images used by earlier phases under other models, so this is not a new dataset sample. No Phase-2 bank outcome exists before this protocol.

## 4. Conditions and word seal

The fixed 167-word Mason list is reused with the Phase-1B CIFAR attenuation strengths. This holds the intervention constant while changing the unit of generalization to models.

Roles:

- 37 identity/single/pair conditions: R2 calibration;
- 40 length-3--6 words, exactly 10 per length: method development;
- 90 remaining length-3--6 words: confirmation.

Exact reverse pairs are assigned together. The deterministic split uses seed 2202 and is stored in `v4/phase2/PHASE2_MANIFEST.json`.

Phase-1/1B outcomes motivated the model form, and those earlier outcomes included these word identities on different classifiers. Therefore the pilot confirmation is a joint held-out-model/held-out-Phase-2-outcome test, not a claim that word identities were never seen anywhere in the project.

## 5. Information regimes

For every bank model, R1/R2 methods may use its clean, six single, and 30 ordered-pair measurements. They may not use that model's length-3--6 confirmation accuracy.

Declared methods:

- R1 primitive mean;
- R1 maximum-damage primitive;
- R1 product of normalized accuracy retentions;
- R2 accuracy-space pair residual with learned global scaling;
- R2 symmetric hazard pair model;
- R2 ordered hazard pair model;
- `g0`: source accuracy + parameter count + epoch + architecture + recipe + word length/primitive counts;
- `g1`: `g0` plus frozen R1/R2 composition forecasts;
- R3 ATC and DoC references, which use unlabeled images from the actual target condition.

R3 methods are information advantaged and cannot serve as the primary R2 comparator.

## 6. Extensive hazard algebra

For model clean accuracy `a0`, chance `c=0.1`, and shifted accuracy `a`, define clipped retention and hazard

```text
r(a) = clip((a-c)/(a0-c), 0.02, 2.0)
h(a) = -log r(a).
```

For ordered pair `i -> j`, define

```text
R_ij = h_ij - h_i - h_j.
```

For word length at least three with `N` constituent ordered pairs:

```text
h_hat = sum_i h_i + N^(q-1) sum_(i before j) R_ij.
a_hat = clip(c + (a0-c) exp(-h_hat), 0, 1).
```

- `q=0` reproduces an average pair correction;
- `q=1` sums pair corrections extensively;
- intermediate/extrapolative values define a smooth scaling law.

A symmetric ablation replaces `R_ij` by `(R_ij+R_ji)/2`. This removes directional order information while preserving magnitude and capacity.

## 7. Method-development freeze

After development-bank evaluation only:

1. select `q` from `{0.0, 0.1, ..., 2.0}` by lowest ordered-hazard MAE on the 40 method-development words across development models;
2. break ties toward smaller `q`;
3. fit fixed-alpha (`alpha=1`) ridge versions of `g0` and `g1` on the same declared rows;
4. commit `FROZEN_METHOD.json`, complete q-search results, ridge coefficients, development behaviors, and development forecasts;
5. make no further method, feature, split, metric, or gate change.

Only after this Stage-B commit may confirmation lineages be trained/evaluated. Confirmation outcomes may not revise `q` or either ridge.

## 8. Primary endpoints

On 30 confirmation snapshots and 90 sealed words:

1. accuracy MAE in percentage points;
2. ODP-style mean Spearman ranking models within each environment;
3. source/family/recipe-controlled Spearman;
4. top-1 and top-3 model-selection regret;
5. within-model environment rank as a separately labeled secondary axis.

The primary incremental comparison is `g1` versus `g0`.

## 9. Statistical unit

Use 10,000 paired bootstrap draws, seed 22202. Resample:

- reverse-word groups;
- training lineages.

Epoch snapshots remain nested within lineage. The estimand is `g0 absolute error - g1 absolute error`; positive values favor the compositional score.

The model count is 30 but the independent training-lineage count is 15.

## 10. Pilot gate

The incremental-value pilot gate passes only if:

1. `g1` accuracy MAE is lower than `g0`;
2. the paired lineage/word-group 95% interval has lower bound above zero;
3. mean source/family/recipe-controlled model-rank Spearman is higher for `g1`;
4. top-1 selection regret is no worse for `g1`.

Order diagnostics compare ordered and symmetric hazard forecasts on sealed exact reverse pairs but are secondary to this gate.

A pass justifies scaling toward the full Phase-2 bank. A failure ends this formulation; it must not be retuned on confirmation outcomes.

## 11. R3 confidence methods

ATC calibrates one confidence threshold per model on its clean evaluation images so the above-threshold fraction matches clean accuracy, then applies it without target labels. DoC predicts target accuracy by adding the target-minus-clean mean-confidence change to clean accuracy. Target labels are used only for scoring.

## 12. Exclusions

Only hash/data corruption, non-finite model output, class-mapping failure, or executable failure may exclude a model/condition. Performance-based exclusion is forbidden. All 30 confirmation models and all 90 confirmation words are required for a complete pilot.

## 13. Freeze record

- Stage-A executable commit: **TBD**
- Stage-B method-freeze commit: **TBD**
- Confirmation result commit: **TBD**
- Phase-2 manifest SHA-256: `7ed83c1074359ac4669b0d8f77d493af093d13cc5bbf88dac55573bb52960b0d`
