# Shift Algebra: NeurIPS-Grade Research and Development Plan

**Status:** prospective plan; no experiments in this document have been run yet
**Research snapshot date:** 2026-10-23
**Target:** a defensible main-track paper, not merely a larger version of the current CIFAR study
**Primary focus:** comparison with current OOD-performance-prediction and compositional-robustness benchmarks, scientific novelty, and an untouched confirmatory evaluation

---

## 0. Executive decision

The current project contains a potentially publishable idea, but it is **not yet directly comparable with the strongest academic benchmark for its neighboring task**.

The central idea is:

> Predict a fixed model's behavior under an unseen **ordered composition** of shifts from primitive-shift measurements and a small pair-calibration set, without evaluating images from the composed target shift.

This is distinct from most out-of-distribution performance prediction (ODP) work, which normally assumes that **unlabeled samples from the target distribution are available**. It is also distinct from compositional-robustness work, which trains a classifier to remain accurate under compositions rather than forecasting the behavior of an already frozen classifier.

That distinction is promising, but the present evidence is too narrow:

- one main CIFAR-10 model and a small external set;
- six mostly synthetic image corruptions;
- words of length at most three in the main test;
- no evaluation on the current 29-dataset/1,444-model ODP-Bench protocol;
- no target-image ODP baselines such as ATC, Agreement, COT/COTT, MaNo, MDE, Nuclear Norm, or Dispersion;
- no across-model ranking or model-selection regret;
- no fully untouched final benchmark after the V1--V3 development cycle;
- custom multivariate BPE that cannot be compared directly with the accuracy-MAE and Spearman metrics used by previous papers.

### Bottom-line plan

The next paper should be framed as:

> **Pre-deployment compositional performance forecasting**: a new information regime and benchmark in which target compositions are unavailable, but their primitive factors have been characterized.

The required work is not just “run more datasets.” It has four parts:

1. **Build a benchmark bridge** to ODP-Bench, ATC/Agreement-on-the-Line, common-corruption benchmarks, and Mason et al.'s exact compositional-corruption protocol.
2. **Freeze and simplify the mathematics** so every exactness claim matches executable code.
3. **Evaluate across many models and natural shifts**, not only across conditions for one classifier.
4. **Create one sealed confirmatory test** that is never inspected until the method, baselines, and analysis are frozen.

A NeurIPS-level go decision should be made only after the gates in Section 17 pass.

---

## 1. The scientific question to own

### 1.1 Primary question

Given:

- a frozen classifier `f`;
- clean reference data;
- primitive shift profiles `(v_X, DeltaB_X)`;
- and a limited set of measured two-shift calibration conditions;

can we predict the model's performance and calibration under an unavailable composition

`X_1 -> X_2 -> ... -> X_m`

including new orders, longer words, new severities, and held-out primitive families?

### 1.2 Information-regime taxonomy

The paper must explicitly separate the following regimes. Comparing methods across different regimes without labeling the advantage would be misleading.

| Regime | Available at prediction time | Representative methods | Purpose |
|---|---|---|---|
| **R0: source-only** | Labeled clean/source validation set; model metadata; no primitive or target images | Accuracy-on-the-Line-style source baseline | Minimum deployment baseline |
| **R1: primitive-only** | R0 + primitive-shift images/statistics and primitive behavior labels; no pair or composed images | additive, mean, max, `L_p` | Cheapest pre-deployment composition forecast |
| **R2: primitive + pair calibration** | R1 + a declared budget of labeled pair conditions; no test-composition images | Shift Algebra `M_B`, `M_A`, sequence baselines | Current project's intended regime |
| **R3: unlabeled composed target available** | R0 + unlabeled images from the actual composed target | ATC, DoC, COT/COTT, MaNo, MDE, Nuclear Norm, Dispersion, Neighbor Invariance, Agreement | Standard ODP regime; information-advantaged upper comparison |
| **R4: labeled composed target** | Labels from the actual target | direct measurement/oracle | Ceiling only |

The headline comparison must be **within R2**. R3 comparisons remain valuable, but the paper must say that they consume information unavailable to R2.

### 1.3 Three intended claims

The final paper should attempt only these claims:

1. **Problem/benchmark claim:** pre-deployment forecasting of ordered shift compositions is a distinct and practically relevant gap not measured by current ODP benchmarks.
2. **method claim:** a signed saturation law plus an explicitly antisymmetric order channel improves target-free prediction over strong statistical and neural sequence baselines.
3. **mechanism claim:** order effects are predictable when the second shift filters, clips, or overwrites structure introduced by the first, and this mechanism transfers beyond one blur/noise example.

Do **not** claim that all real distribution shifts obey a universal algebra, or that the project invented semigroups, `L_p` norms, commutators, or Heisenberg groups.

---

## 2. What current academic work actually measures

### 2.1 ODP-Bench is the principal contemporary comparison

**Yu et al., “ODP-Bench: Benchmarking Out-of-Distribution Performance Prediction,” ICCV 2025** is the benchmark the paper must address directly.

Official repository: <https://github.com/h-yu16/Performance_Prediction>
Paper linked by the official repository: <https://arxiv.org/abs/2510.27263>

The benchmark reports:

- **29 OOD datasets**;
- **1,444 pretrained models**;
- **10 prediction algorithms**;
- synthetic corruption, style, background, data-acquisition, camera-location, demographic, temporal/geographic, domain-generalization, and subpopulation shifts;
- cross-architecture evaluation rather than only comparing checkpoints from one architecture.

The official code exposes the following predictor families:

- ATC;
- DoC;
- COT and COTT;
- Nuclear Norm;
- Dispersion;
- MDE;
- MaNo;
- Neighbor Invariance;
- Agreement.

Its central ranking metric is **Spearman correlation between predicted score and true OOD accuracy across models**. Accuracy prediction error is also important in the underlying performance-prediction literature.

#### Consequence for Shift Algebra

The current `rho_m = 0.929` is a correlation **across composition conditions for one trained classifier**. ODP-Bench Spearman is primarily a correlation **across a heterogeneous model pool for an OOD environment**. These numbers are not directly comparable and must never be placed in the same table as if they measured the same estimand.

Similarly, current `accMAE = 0.0424` means 4.24 percentage points in the project's own 123-condition split. It is useful internally, but it is not an ODP-Bench result until the same model pool, targets, information regime, and aggregation protocol are used.

### 2.2 Accuracy-on-the-Line is both a baseline and a benchmark warning

**Miller et al., “Accuracy on the Line: On the Strong Correlation Between Out-of-Distribution and In-Distribution Generalization,” ICML 2021** showed that ID accuracy can strongly predict OOD accuracy across broad model collections.

Author repository: <https://github.com/millerjohnp/linearfits_app>

A more recent benchmark critique, **“Are Domain Generalization Benchmarks with Accuracy on the Line Misspecified?”** (2025 preprint), argues that strong aggregate ID/OOD lines can indicate that a split fails to stress spurious-correlation robustness and that averaging across splits can hide important failures.

Official repository: <https://github.com/olawalesalaudeen/misspecified_DG_benchmarks>

#### Consequence for Shift Algebra

Every across-model result needs a source-accuracy baseline. A high raw Spearman can be scientifically uninteresting if clean accuracy or architecture identity already supplies the ordering. In addition to raw ODP-style Spearman, report source-controlled/partial rank, source-matched pair ordering, and model-selection regret. Also inspect individual composition families rather than treating one pooled correlation as proof that the benchmark is challenging.

### 2.3 ATC established a strong unlabeled-target baseline

**Garg et al., “Leveraging Unlabeled Data to Predict Out-of-Distribution Performance,” ICLR 2022** introduced Average Thresholded Confidence (ATC).

Official repository: <https://github.com/saurabhgarg1996/ATC_code>

ATC:

1. selects a confidence or entropy threshold on labeled ID validation data so thresholded confidence approximates correctness;
2. applies that threshold to unlabeled OOD target predictions;
3. estimates target accuracy as the fraction of target examples above threshold.

Its testbed includes CIFAR, ImageNet/BREEDS, WILDS/FMoW, RxRx1, CivilComments, and Amazon. It also implements Generalized Disagreement Equality.

#### Consequence for Shift Algebra

ATC has access to the actual unlabeled composed target, whereas the proposed R2 problem does not. Therefore:

- ATC is not a fair R2 baseline;
- it is a required R3 reference;
- the study should report the price of not seeing target images: `R2 error - R3 error`;
- a strong result would show that R2 forecasting is competitive enough to support pre-deployment decisions before target samples exist.

### 2.4 Agreement-on-the-Line changes the unit of evaluation

**Baek et al., “Agreement-on-the-Line: Predicting the Performance of Neural Networks under Distribution Shift,” NeurIPS 2022** predicts OOD performance from agreement among multiple models.

Official repository: <https://github.com/kebaek/Agreement-on-the-line>

Related NeurIPS 2024 work extends agreement-based prediction to foundation models and shows that ensemble diversity matters:

- “Predicting the Performance of Foundation Models via Agreement-on-the-Line”;
- “Test-Time Adaptation Induces Stronger Accuracy and Agreement-on-the-Line.”

#### Consequence for Shift Algebra

A single-model study is insufficient for comparison with this literature. The next benchmark needs a **model bank** so it can ask:

- can the method rank models for a held-out composition?
- does it add information beyond clean/source accuracy?
- can it select the best deployment model?
- is any apparent benefit simply model-family identity?

### 2.5 Mason et al. is the closest public composition protocol

**Mason et al., “Modularity Trumps Invariance for Compositional Robustness,” 2023** studies classifier robustness to ordered corruption compositions.

Official repository: <https://github.com/ianxmason/compositional-robustness>

Its public protocol contains:

- datasets: EMNIST, CIFAR-10, and FaceScrub;
- six corruption primitives: Contrast, Gaussian Blur, Impulse Noise, Invert, Rotate90, and Swirl;
- all 30 ordered pairs;
- sampled permutations for lengths three through six;
- 167 total clean/single/composed conditions;
- accuracy and loss versus number of elemental corruptions;
- elemental/composition invariance scores and Pearson `r`/`p` correlations.

The work trains robust classifiers; it does not forecast frozen-model behavior. Nevertheless, this is the most direct public test of whether Shift Algebra can handle the exact composition words used in previous research.

#### Consequence for Shift Algebra

Reproducing this 167-condition protocol is a **mandatory first external benchmark**, including words of length four, five, and six. The project's present triple-only headline does not meet that bar.

### 2.6 Common-corruption benchmarks define standard robustness metrics

**Hendrycks and Dietterich, ICLR 2019** introduced ImageNet-C/CIFAR-C and ImageNet-P.

Official repository: <https://github.com/hendrycks/robustness>

Standard metrics include:

- clean error/accuracy;
- per-corruption and per-severity accuracy;
- Corruption Error (CE) normalized to a reference model;
- mean Corruption Error (mCE);
- relative mCE;
- mean Flip Rate (mFR) on perturbation sequences;
- mean Top-5 Distance (mT5D).

**CorruptionBenchCV** unifies ImageNet-C, ImageNet-C-bar, ImageNet-P, and ImageNet-3DCC and additionally reports robust accuracy, relative robustness, ECE by severity, mFP, and mT5D:
<https://github.com/nis-research/CorruptionBenchCV>

**3D Common Corruptions (3DCC), CVPR 2022 Oral** provides 20 more realistic geometry-aware corruptions with five severities:
<https://github.com/EPFL-VILAB/3DCommonCorruptions>

#### Consequence for Shift Algebra

The project predicts behavior rather than improving robustness, so mCE is not its loss. However, it must predict the standard quantities researchers care about:

- per-condition top-1 accuracy/error;
- CE/mCE contribution;
- ECE;
- flip-rate or prediction-instability summaries for smoothly varying sequences.

A reader should be able to ask, “Can the forecast recover the final mCE ranking of models without evaluating every composition?”

### 2.7 WILDS, DomainBed, BREEDS, and MetaShift provide natural shifts

Relevant public benchmarks:

- WILDS: <https://github.com/p-lambda/wilds>
- DomainBed: <https://github.com/facebookresearch/DomainBed>
- BREEDS: <https://github.com/MadryLab/BREEDS-Benchmarks>
- MetaShift: <https://github.com/Weixin-Liang/MetaShift>

WILDS contains ten in-the-wild datasets and uses dataset-specific average and worst-group metrics. DomainBed standardizes domain-generalization datasets and model-selection protocols. BREEDS targets subpopulation shift. MetaShift contains 12,868 natural image sets across 410 classes, with explicit context annotations and shift distances.

#### Consequence for Shift Algebra

Natural shifts often do not have a physical order. They must not be forced into a false non-commutative narrative. They should be used in two ways:

1. **commutative factor composition:** forecast intersectional groups such as time × region or context × attribute;
2. **natural-plus-operator composition:** first select a natural domain, then apply a physical corruption.

The first tests whether the symmetric saturation component generalizes. The second tests whether the complete ordered algebra works outside clean CIFAR images.

### 2.8 RobustBench supplies a broad model bank

Official repository: <https://github.com/RobustBench/robustbench>

RobustBench provides standardized model loading and common-corruption evaluation for CIFAR and ImageNet models. It can supply standard, corruption-robust, and adversarially robust checkpoints whose composition sensitivity may differ substantially.

#### Consequence for Shift Algebra

The final benchmark should not contain only a standard ResNet. It should include models with different robustness mechanisms, architectures, and clean accuracies. Otherwise, across-model conclusions are impossible.

### 2.9 Important literature correction

“Compositional Generalization Across Distributional Shifts with Sparse Tree Operations” (NeurIPS 2024) is a neurosymbolic sequence/tree generalization paper covering lexical, structural, length, template, and MCD splits. It is useful as broad compositional-generalization context but **is not a vision corruption-performance-prediction benchmark**. It must not be described as direct evidence that attention pooling solves composed image shifts.

Official project link referenced by the authors: <https://github.com/psoulos/sdtm>

---

## 3. Benchmark and metric comparison matrix

| Work | Actual task | Inputs available at test | Main evaluation unit | Standard metrics | Gap left for this project |
|---|---|---|---|---|---|
| ImageNet-C/P | Measure robustness | Labeled target corruption data | Model × corruption | mCE, relative mCE, mFR, mT5D, accuracy | No forecasting; usually one corruption at a time |
| RobustBench | Standardized robustness ranking | Labeled benchmark | Model | robust accuracy, corruption accuracy | No target-free composition prediction |
| 3DCC | Robustness to realistic shifts | Labeled target corruption data | Model × 3D corruption | task accuracy/error over five severities | Mostly primitive shifts, not order forecasting |
| Mason et al. | Train models robust to compositions | Labeled composed test data | Model × word length/order | accuracy, loss, invariance correlation | Does not predict frozen-model behavior from primitives |
| ATC | Estimate target accuracy | Unlabeled actual target samples | Model × target | accuracy-estimation error | Target images must exist; no factor algebra |
| Agreement/ALine | Estimate/rank target accuracy | Unlabeled target + model ensemble | Models within a target | MAE, correlation, model selection | Requires multiple models and actual target samples |
| ODP-Bench | Benchmark OOD predictors | Unlabeled actual target samples | Heterogeneous models × 29 targets | cross-model Spearman; prediction quality | No ordered composition or pre-deployment track |
| WILDS/DomainBed | Robust training/generalization | Labeled benchmark splits | model × natural domain/group | dataset-specific average/worst-group accuracy | No behavior forecast from primitive factors |
| **Shift Algebra, intended** | Forecast unavailable ordered compositions | primitive profiles + limited pair calibration; no composed target samples | model × composition word | MAE, controlled rank, regret, order/algebra diagnostics | Must now be validated at benchmark scale |

---

## 4. Honest assessment of the current project against those standards

### 4.1 Current strengths

1. **Distinct information regime:** target compositions are not passed through the classifier at prediction time.
2. **Order-aware target:** reverse order is explicitly held out and evaluated.
3. **Multiple behavior coordinates:** the output is more informative than accuracy alone.
4. **Mechanistic ablations:** removing `L_p` and removing the commutator causes different failures.
5. **Primitive holdout:** some tests remove an entire primitive from pair training.
6. **Counterfactual mechanism test:** the high-frequency `H` experiment separates spectral mechanism from raw variance.
7. **Reproducibility artifacts:** code, reports, raw JSON summaries, and audits are committed.

### 4.2 Current weaknesses relative to ODP-Bench

| Dimension | Current project | Contemporary benchmark expectation | Required change |
|---|---|---|---|
| Number of models | essentially one main model per setting | hundreds to 1,444 models | model bank with architecture/training diversity |
| Number of datasets | three small image datasets + synthetic | dozens of natural and synthetic shifts | benchmark bridge, not necessarily all 29 at first |
| Natural shifts | none in the main ordered benchmark | style, background, camera, demographics, time, geography | natural-factor and natural+corruption tracks |
| Across-model ranking | absent | central ODP estimand | per-composition cross-model Spearman and regret |
| Source baseline | mean/additive composition baselines | source accuracy is often very strong | explicit clean/source/family baseline |
| Current ODP methods | absent | ATC, Agreement, COT/COTT, etc. | R3 implementation using official code |
| Composition length | mostly 2 and 3 | Mason protocol reaches 6 | length extrapolation through 6 |
| Final untouched test | no; V1--V3 used the same research environment | sealed confirmatory benchmark | hash/freeze final manifest before labels |
| Metric comparability | custom BPE | accuracy MAE, Spearman, R²/controlled rank, regret | retain BPE as secondary; add standard endpoints |
| Uncertainty | point prediction only | deployment requires confidence/abstention | calibrated intervals and selective forecasting |

### 4.3 Results that must be framed cautiously

- `M_B` is not uniformly best in overall BPE on Fashion-MNIST or CIFAR-100; other methods win those individual tables.
- On the full primitive-holdout average, `M_Lp_only` beats `M_B` (`0.0272` versus `0.0294`). This suggests the simple law can be preferable when interaction evidence is absent.
- Current statistical significance is clustered over composition families from one benchmark, not independent datasets or model families.
- The current 103k-parameter composition module is large relative to 155 aggregate training conditions.
- The matrix core is associative, while the three-shift neural readout includes an explicit two-tree average. Those claims must remain separate.
- The matrix extraction is algebraically equivalent to a closed-form all-pairs computation. The matrix representation supplies closure and an associative state, not an empirical advantage over its exactly equivalent formula.

---

## 5. Proposed benchmark: CompShift-PP

Working name: **CompShift-PP: Compositional Shift Performance Prediction**.

### 5.1 Benchmark object

Every record should be indexed by:

```text
(dataset, model_id, model_family, training_recipe,
 word=(primitive_1,...,primitive_m), severities,
 calibration_role, split_id, random_field_id)
```

Each record stores:

- clean behavior;
- primitive fingerprints and behavior vectors;
- optional pair-calibration measurements;
- ground-truth composed behavior, hidden for sealed tests;
- standard robustness metrics;
- execution and provenance metadata.

### 5.2 Four benchmark tiers

#### Tier A: exact prior-protocol reproduction

Reproduce Mason et al.'s public protocol exactly:

- EMNIST, CIFAR-10, FaceScrub where licensing permits;
- the exact six primitives;
- all ordered pairs;
- the published sampled permutations for lengths 3--6;
- all 167 conditions;
- classification accuracy and loss;
- invariance-score correlations where activations are available.

Primary question: can pair-trained forecasting predict accuracy for lengths 3--6 on a protocol designed independently by prior authors?

#### Tier B: ODP-Comp-CIFAR

Use CIFAR-10 and CIFAR-100 with a diverse model bank.

Development model bank target:

- 30 checkpoints minimum;
- ResNet, WideResNet, DenseNet, VGG, MobileNet, ConvNeXt, ViT/DeiT-like models;
- standard, AugMix/corruption-robust, and adversarially robust training where licenses permit;
- at least three seeds for key families.

Final model bank target:

- 75+ checkpoints;
- at least six architecture families and four training-recipe families.

Shift library:

- brightness/gamma;
- contrast;
- saturation/color;
- Gaussian blur;
- motion or defocus blur;
- Gaussian noise;
- impulse or shot noise;
- pixelation/quantization;
- optional fog and compression as external primitives.

Each stochastic corruption must use a condition-level deterministic random field shared across reverse-order comparisons.

#### Tier C: ODP-Comp-ImageNet

Use ImageNet validation images and public torchvision/timm/RobustBench checkpoints.

Development phase:

- ImageNet-100 or a fixed 100-class subset;
- 20 models;
- 1,000--5,000 images per condition;
- eight core primitives.

Final phase:

- full ImageNet label space;
- 30--50 diverse models initially, expanding only if compute allows;
- ImageNet-C/C-bar-compatible transformations;
- at least four geometry-aware 3DCC primitives where deterministic generation is feasible;
- standard top-1 error, ECE, CE/mCE contribution, and confidence metrics.

The goal is not to regenerate the entire ODP-Bench testbench. It is to build a clean **composition extension** with enough shared models and metrics for direct comparison.

#### Tier D: natural and hybrid compositions

Use two forms of composition:

1. **Natural factor intersections** from WILDS/MetaShift/DomainBed metadata, treated as commutative:
   - FMoW time × region;
   - iWildCam location × time/camera metadata when sample sizes permit;
   - MetaShift context × attribute;
   - Waterbirds background × controlled corruption;
   - DomainNet domain × controlled corruption.
2. **Natural target followed by image operator**, which is ordered:
   - select a natural domain first;
   - apply blur/noise/contrast/etc. second;
   - forecast the hybrid target from natural-domain and operator primitives.

Do not describe a metadata intersection as a physical non-commutative operator.

### 5.3 Word and severity design

For `P` core primitives:

- all singles at five severities;
- all ordered distinct pairs at matched severity;
- a balanced Latin-hypercube sample of cross-severity pairs;
- balanced words of lengths 3, 4, 5, and 6;
- each primitive and ordered adjacency appears equally often within a tolerance;
- reverse words are paired for order analysis;
- repeated-primitive words test semigroup/self-commutation behavior;
- identity insertions test identity consistency.

A word family, not an individual severity point, is the statistical cluster.

### 5.4 Split axes

Create separate, named splits instead of one pooled held-out set:

1. **S-word:** unseen words over known primitives.
2. **S-reverse:** reverse of a one-direction observed pair/word.
3. **S-length:** train on length `<=2`, test separately at lengths 3, 4, 5, 6.
4. **S-severity:** interpolation and extrapolation outside calibration levels.
5. **S-primitive:** all pair/word training conditions containing one primitive are removed.
6. **S-mechanism:** hold out a whole mechanism family, e.g. all blur/filter primitives.
7. **S-architecture:** hold out complete model families.
8. **S-recipe:** hold out a training recipe such as AugMix or adversarial training.
9. **S-dataset:** fit reusable components on some datasets, evaluate on a new dataset.
10. **S-natural:** sealed natural/hybrid targets.

Every headline result must be broken out by split. A single aggregate can hide complete failure on the scientifically important axis.

---

## 6. Baseline suite required for a credible paper

### 6.1 R0 source/model baselines

These are currently missing and are essential because clean/source performance and architecture family can explain much of OOD ranking.

1. clean accuracy only;
2. primitive mean/worst accuracy only;
3. linear `accuracy-on-the-line` regression;
4. model-family fixed effects;
5. source accuracy + parameter count + architecture/training metadata;
6. source and primitive behavior + ridge regression.

The central incremental comparison should be:

```text
g0 = source performance + model family + training recipe

g1 = g0 + compositional predictor score
```

Report `MAE(g0) - MAE(g1)`, controlled/partial rank, and selection-regret improvement.

### 6.2 R1/R2 simple mathematical baselines

- additive behavior;
- mean pooling;
- maximum/dominant-shift envelope;
- product-of-retentions baseline for bounded accuracy;
- five-parameter signed `L_p`;
- `L_p` + scale/offset;
- ridge and elastic-net polynomial features;
- low-rank tensor regression;
- Gaussian process with word/severity kernels and calibrated uncertainty;
- gradient-boosted trees on aggregate primitive features;
- pairwise ANOVA/main-effect model.

### 6.3 R2 neural compositional baselines

- DeepSets;
- Set Transformer;
- ordered Transformer with positional tokens;
- GRU/LSTM fold;
- MLP on concatenated fixed-length words;
- low-rank bilinear recurrent model;
- structured state-space/matrix-product model with matched state dimension;
- path-signature/Chen-series features for ordered words;
- `M_A`, `M_C`, and V1 `M5`;
- parameter-matched versions at small, medium, and full capacity.

All flexible baselines require nested development splits and equal hyperparameter budgets.

### 6.4 R3 ODP baselines using official implementations

Integrate adapters for:

- ATC confidence and entropy;
- DoC/DoE;
- COT and COTT;
- Nuclear Norm;
- Dispersion;
- MDE;
- MaNo;
- Neighbor Invariance;
- Agreement/ALine when an ensemble is available.

Do not silently give these methods composed labels. Use only the source labels and unlabeled target inputs permitted by their protocols.

### 6.5 Oracles

Report separately:

- direct labeled composed measurement;
- direct composed fingerprint decoder `G(E(v_target))`;
- best per-target method selected after seeing target labels;
- triple-oracle `p*` as analysis only.

Oracles must never appear in the ordinary method ranking.

---

## 7. Metrics aligned with previous work

### 7.1 Primary prediction metrics

All accuracy errors must be reported in **percentage points**.

1. **Macro Accuracy MAE:** average absolute error, first within family, then across families/datasets.
2. **Accuracy RMSE:** exposes catastrophic forecasts.
3. **Median absolute error:** robust companion to MAE.
4. **Worst-decile MAE:** safety-relevant tail.
5. **Per-length MAE:** separate lengths 2, 3, 4, 5, 6.

### 7.2 ODP-style ranking metrics

For each target composition/environment, rank models and report:

- Spearman `rho` across all models;
- Spearman within architecture family;
- Spearman within training recipe;
- Kendall `tau`;
- Pearson and `R^2` as secondary diagnostics;
- partial/controlled Spearman after residualizing source accuracy, model family, and training recipe.

For each model, also report the current within-model rank across compositions. Keep the two axes clearly labeled:

```text
rho_model|environment   = rank models for one environment
rho_environment|model   = rank environments for one model
```

### 7.3 Deployment decision metrics

Prediction can look correlated yet make bad decisions. Report:

- top-1 model-selection regret;
- top-`k` regret (`k=3,5`);
- probability of selecting a model within 1 percentage point of the true best;
- pairwise model-ordering accuracy after matching models on source accuracy;
- abstention/coverage curves: error when forecasting only the most certain 100%, 80%, 60%, etc.

### 7.4 Standard robustness metrics

Forecast and score:

- clean accuracy/error;
- per-corruption/per-severity accuracy;
- CE and mCE contribution;
- relative mCE;
- ECE;
- negative log likelihood and Brier score if probabilities are forecast;
- mFR/mT5D-like instability on perturbation trajectories where available;
- WILDS dataset-specific average and worst-group metrics on natural tracks.

### 7.5 Multivariate behavior metrics

Replace raw unscaled BPE as the sole multivariate score.

For each behavior coordinate:

- MAE in its natural units;
- RMSE;
- Spearman;
- normalized MAE using a scale fixed from training data only, e.g. training IQR or standard deviation.

Aggregate with:

- macro average of normalized per-coordinate MAEs;
- multivariate Gaussian negative log likelihood for probabilistic models;
- energy score for non-Gaussian ensembles.

Current BPE remains a continuity metric, not the primary cross-paper endpoint.

### 7.6 Order and algebra metrics

- reverse-order sign accuracy per behavior coordinate;
- commutator-magnitude Spearman;
- reverse-pair MAE;
- antisymmetry residual `||Omega(X,Y)+Omega(Y,X)||`;
- symmetric leakage `||Gamma(X,Y)-Gamma(Y,X)||`;
- identity error;
- repeated-primitive self-commutator;
- **end-to-end** associator, not only matrix-core associator;
- error growth versus word length;
- Lipschitz/Jacobian estimate versus length.

### 7.7 Efficiency metrics

- number of labeled primitive conditions;
- number of labeled pair conditions;
- number of shifted images processed;
- target-composition images required at inference (zero for R2);
- wall time and GPU hours;
- model parameters;
- bytes stored per condition;
- error-versus-calibration-cost curve and area under that curve.

---

## 8. Statistical design

### 8.1 Independent units

The primary inferential units should be:

- dataset;
- model family;
- shift family/word family;
- training recipe.

Severity points and random seeds nested inside one family are not independent replicates.

### 8.2 Required analyses

1. hierarchical bootstrap over dataset -> model family -> word family;
2. paired family-level differences against every primary baseline;
3. mixed-effects model with method fixed effect and dataset/model/word random intercepts;
4. seed-level variance as optimization stability, not the sole inferential sample;
5. Holm correction for the predeclared primary baseline family;
6. false-discovery-rate correction for exploratory analyses;
7. effect sizes and confidence intervals, not only p-values.

### 8.3 Fresh-test discipline

Use three data roles:

- **development:** labels fully visible;
- **validation:** available only to a locked evaluation script;
- **final challenge:** encrypted or held by another person/process until all code hashes are frozen.

Before opening the final set, freeze:

- Git commit;
- container digest;
- model list;
- condition manifest;
- random seeds;
- metrics;
- exclusions;
- baseline hyperparameters;
- statistical script;
- decision gates.

The final set gets one run. Bugs that invalidate it must be disclosed; no same-ID rescue.

---

## 9. Method-development plan: `M_B-v4`

All V1--V3 results must remain reproducible. Implement V4 in new modules rather than modifying historical code in place.

### 9.1 Exact numerical semigroup

Current stabilized `phi` and `phi_inv` are not exact inverses. Replace them with an exactly paired transform, for example:

```text
phi_p(b)     = sign(b) * ((abs(b) + eps)^p - eps^p)
phi_p_inv(u) = sign(u) * ((abs(u) + eps^p)^(1/p) - eps)
```

Add property tests for:

- inverse consistency;
- identity;
- associativity in transformed state;
- finite gradients near zero.

If saturation is a required premise, constrain `p >= 1`, e.g. `p = 1 + softplus(raw_p)`, or clearly permit and interpret `p < 1`.

### 9.2 Separate exact core from heuristic readout

Create two explicitly named models:

1. **`M_B_core`:** accumulate one associative state and decode only once at the end;
2. **`M_B_treeavg`:** optional left/right tree-averaged decoder, labeled as a heuristic ensemble.

Report end-to-end associator for both. Do not use matrix-core associativity to describe a different final readout.

### 9.3 Remove implementation ambiguity

- remove or use the currently unused latent `W_ord`;
- make `LAMBDA_REPR` the actual coefficient rather than an on/off switch;
- centralize order conventions;
- unit-test `X then Y` against image and operator order;
- add schema validation for every behavior vector and severity grid;
- remove duplicate/dead function definitions;
- count shared encoder/decoder and composition parameters separately and jointly.

### 9.4 Architecture-independent fingerprints

The current 518 dimension is tied to a 128-dimensional feature layer. V4 should support arbitrary backbones with a fixed-size sketch:

- layerwise mean/std from 3--4 normalized depth locations;
- fixed random projections with a published seed;
- logit/probability histograms;
- top-1/top-2 margin quantiles;
- feature covariance trace/effective rank;
- pixel and Fourier-band summaries;
- optional CLIP-space summary shared across architectures.

Compare:

1. raw architecture-specific moments;
2. fixed-size moment sketch;
3. logits-only fingerprint;
4. pixels-only fingerprint;
5. representation-only fingerprint.

The headline model must work when a held-out architecture has a different feature dimension.

### 9.5 Reduce sample complexity

Build sizes around the amount of calibration data:

- `M_B-tiny`: <= 5k composition parameters;
- `M_B-small`: <= 20k;
- `M_B-full`: current scale;
- hierarchical/shared model across many classifiers;
- low-rank and monotonic readouts;
- regularization selected without touching final words.

A 103k-parameter module trained on 125 aggregate pair points should no longer be the only headline configuration.

### 9.6 Predict uncertainty

Add:

- deep ensemble or bootstrap over calibration families;
- heteroscedastic output variance;
- conformal calibration on held-out development families;
- out-of-support score from primitive fingerprints;
- explicit abstention when the new primitive is too far from calibration support.

Report empirical coverage and interval width for 80%, 90%, and 95% intervals.

### 9.7 Across-model hierarchical learning

Two modes are required:

- **per-model calibration:** fit one algebra per classifier, matching the current setup;
- **meta-model:** share algebraic parameters across classifiers while conditioning on model fingerprint/training metadata.

Evaluate leave-one-architecture and leave-one-recipe generalization. This is necessary for ODP-style model ranking and practical amortization.

---

## 10. Mathematical/theoretical work package

### 10.1 Prove only implementation-matched theorems

Target theorem set:

1. exact transformed-state associativity with the numerical `phi/phi_inv` actually used;
2. reversal equivariance of the antisymmetric channel;
3. permutation invariance of the symmetric channel;
4. identity and repeated-primitive consistency;
5. length-dependent error bound;
6. identifiability/impossibility result.

### 10.2 Length-dependent error bound

Derive a bound of the form

```text
||B_hat(word) - B(word)||
 <= C1 * m * primitive_error
  + C2 * choose(m,2) * pair_interaction_error
  + C3 * higher_order_remainder
```

under declared Lipschitz and bounded-state assumptions. Measure every bound term empirically. This connects the algebra to length extrapolation and gives a falsifiable reason for failure at lengths 4--6.

### 10.3 Impossibility theorem

Formalize that primitive marginals alone cannot identify arbitrary composition behavior: two worlds can share every primitive profile but have different pair targets. This justifies pair calibration and prevents overclaiming universal zero-shot identifiability.

### 10.4 Representation scope theorem

Characterize the class represented by the potential commutator

```text
Omega(X,Y) = 0.5 * pi_X * pi_Y * (kappa_Y - kappa_X).
```

This imposes a transitive potential order and cannot represent arbitrary cyclic pair preferences. State that limitation and create a synthetic rock-paper-scissors counterexample. Compare:

- potential/transitive commutator;
- unrestricted antisymmetric pair network;
- low-rank skew-symmetric bilinear form.

### 10.5 Matrix-versus-closed-form equivalence

Implement a closed-form extractor that computes `b_Lp`, `Gamma`, and `Omega` without matrices. It should match the matrix product numerically. Treat this as a theorem/property test, not an ablation expected to differ.

The scientific value of the matrix form is:

- a closed associative state representation;
- streaming composition;
- exact order bookkeeping;
- extensibility to richer group blocks.

It is not evidence of predictive superiority over an algebraically identical formula.

### 10.6 External proof audit

Before submission, ask an independent mathematically trained reviewer to check:

- every proposition against code;
- norm-inequality directions;
- numerical stabilizers;
- scope of associativity;
- identifiability assumptions;
- whether a “Lie” or “Heisenberg” label is mathematically precise rather than decorative.

---

## 11. Novelty program

### 11.1 Defensible novelty statement

Use this provisional statement:

> Existing OOD performance prediction estimates performance after unlabeled target samples become available, while compositional-robustness work measures or improves a classifier on observed composition data. We study the earlier pre-deployment problem: forecasting an unavailable ordered target composition from characterized primitive shifts and limited pair calibration. We contribute a benchmark protocol and a structured saturation-plus-order predictor for that regime.

### 11.2 Claims to avoid

Do not write:

- “the first algebra for distribution shifts”;
- “the first non-commutative neural model”;
- “proves model degradation follows a universal law”;
- “dataset independent” without saying parameters are recalibrated;
- “zero-shot” without specifying that primitive behavior labels and training pairs are used;
- “exact associativity” for the heuristic tree-averaged end-to-end readout;
- “beats ODP-Bench” unless the official protocol is actually run.

### 11.3 Formal literature audit before experiments are frozen

Search at minimum:

- Semantic Scholar;
- OpenAlex;
- Google Scholar;
- OpenReview;
- NeurIPS/ICML/ICLR/CVPR/ICCV proceedings;
- forward and backward citations of ATC, Agreement-on-the-Line, ODP-Bench, ImageNet-C, and Mason et al.

Search strings:

```text
"compositional shift" performance prediction
"compound corruption" accuracy estimation
"ordered corruption" model performance
"multi-corruption" pre-deployment forecast
"distribution shift composition" model evaluation
"unseen corruption combinations" accuracy prediction
"OOD performance prediction" target unavailable
"performance prediction under dataset shift" composition
semigroup neural operator distribution shift
non-commutative corruption model
```

Create `literature/CLAIM_MATRIX.csv` with columns:

```text
paper, venue, year, task, target_samples_available,
ordered_compositions, primitive_only_forecast, pair_calibration,
models, datasets, metrics, code_url, exact_overlap, notes
```

Novelty remains a working hypothesis until this audit is refreshed immediately before submission.

### 11.4 Paper contribution hierarchy

The safest hierarchy is:

1. new benchmark/problem protocol;
2. strong empirical finding about saturation and order under this protocol;
3. structured model;
4. mathematical analysis of that structure.

If `M_Lp_only` remains best on broad new tests, the paper can still succeed as a benchmark and empirical-law paper, with the Heisenberg term restricted to genuinely ordered mechanisms.

---

## 12. Experiment matrix

| ID | Question | Datasets | Models | Train information | Held-out axis | Primary metric |
|---|---|---|---|---|---|---|
| E0 | Does V4 exactly preserve intended algebra? | synthetic/property tests | n/a | generated | identities/reversal/parentheses | numerical invariant error |
| E1 | Can the method reproduce an independent composition protocol? | Mason EMNIST/CIFAR/FaceScrub | ERM, Contrastive, Modular + standard | singles/pairs | lengths 3--6 | acc MAE by length |
| E2 | Does it scale across model families? | CIFAR-10/100 | 75+ | R1/R2 | architecture/recipe | cross-model Spearman, regret |
| E3 | Does it scale to ImageNet? | ImageNet-C/C-bar-compatible | 30--50 | R1/R2 | words/length/severity | acc MAE, mCE forecast |
| E4 | Does order modeling transfer mechanistically? | CIFAR/ImageNet/3DCC subset | diverse | selected bidirectional pairs | reverse mechanisms | sign, reverse MAE |
| E5 | Does it work on natural factors? | WILDS/MetaShift/DomainBed | 20+ | commutative factors | intersection groups | average/worst-group MAE |
| E6 | How much is lost without composed target samples? | E2--E5 | same | R2 vs R3 | same final targets | MAE gap to ATC/Agreement/etc. |
| E7 | Is improvement incremental beyond source/model identity? | all | same | controlled regression | architecture/recipe | delta MAE, partial rho |
| E8 | Is uncertainty trustworthy? | all | ensembles | calibration families | final families | coverage, width, selective risk |
| E9 | Which component matters? | all development tiers | same | fixed | all | multi-axis ablation panel |
| E10 | Untouched confirmation | sealed subset | held-out families | frozen | dataset+architecture+primitive | preregistered gate |

---

## 13. Ablation matrix

Every row must use the same split and tuning budget.

| Component question | Comparison |
|---|---|
| Is saturation needed? | `p=1`, learned `p>=1`, max pool, product-of-retentions |
| Is `L_p` enough? | `M_Lp_only` vs full model by order/non-order split |
| Is the symmetric interaction needed? | zero `Gamma`; linear pair ANOVA; learned symmetric MLP |
| Is order needed? | zero `Omega`; shuffled order; positionless sequence model |
| Is potential transitivity useful? | potential `Omega` vs unrestricted antisymmetric network |
| Does the matrix form add computation or only structure? | matrix vs exactly equivalent closed form |
| Does tree averaging help? | exact core vs heuristic tree average |
| Are 518 raw coordinates necessary? | fixed sketch, logits-only, pixels-only, features-only |
| Is behavior supervision necessary? | representation-only; primitive-only decoder |
| Is capacity driving results? | 5k/20k/100k parameter curves and matched baselines |
| Does pair coverage matter? | 0/10/25/50/75/100% pair budget |
| Does bidirectional coverage matter? | spanning tree, random edges, mechanism-balanced edges |
| Is support fallback doing the work? | fallback on/off with distance-stratified reporting |

---

## 14. Compute and data plan

The current 2-CPU sandbox is sufficient for code design and small verification, not the proposed final benchmark.

### 14.1 Staged compute

#### Stage 1: CPU/small GPU

- property tests;
- Mason protocol on small models;
- CIFAR development bank;
- method/baseline integration.

Estimated order: 200--500 GPU-hours depending on retraining versus public checkpoints.

#### Stage 2: ImageNet screening

- 20 models;
- 1,000 images per condition;
- approximately 100 development composition conditions;
- streaming feature/behavior summaries.

Use this only for engineering and variance estimates, not final claims.

#### Stage 3: frozen final

- 30--50 models;
- 5,000 or more images per condition where confidence intervals require it;
- sealed composition manifest;
- distributed inference.

A realistic budget is multiple modern GPUs for one to three weeks. Record actual throughput and cost rather than promising an exact number now.

### 14.2 Storage discipline

Do not save every intermediate feature tensor. Stream and store:

- aggregate feature sketches;
- behavior statistics;
- top-k probabilities plus required scalar scores where sufficient;
- per-condition provenance and checksums.

Some R3 methods require full probability matrices. Compute them in one pass and store compressed artifacts only when licensing and disk allow. Document any approximation, because COT-like methods may require complete class probabilities.

### 14.3 Licensing/provenance

For every external asset record:

- official URL;
- license;
- checksum/version/commit;
- whether redistribution is allowed;
- model-card preprocessing;
- class mapping;
- transformation implementation and seed.

Do not commit restricted datasets or large checkpoints to Git.

---

## 15. Software work packages

### WP1: immutable V1--V3 preservation

- tag result schemas;
- add regression tests for `reproduce.py --verify`;
- never overwrite historical JSON;
- document known mathematical/code qualifications.

### WP2: V4 core

Proposed files:

```text
src_v4/
  algebra.py
  fingerprints.py
  behavior.py
  models.py
  uncertainty.py
  support.py
  losses.py
```

### WP3: benchmark schema

```text
benchmark/
  schema.py
  manifests/
  splitters.py
  corruption_registry.py
  natural_factor_registry.py
  provenance.py
```

### WP4: external adapters

```text
adapters/
  mason_compositions.py
  robustbench_models.py
  odpbench_predictors.py
  wilds_domains.py
  domainbed_domains.py
  metashift_contexts.py
```

### WP5: baselines

```text
baselines/
  source_family.py
  statistical.py
  sequence.py
  odp_target_available.py
```

### WP6: evaluation

```text
analysis_v4/
  prediction_metrics.py
  rank_metrics.py
  regret.py
  algebra_metrics.py
  hierarchical_bootstrap.py
  mixed_effects.py
  tables.py
```

### WP7: preregistration and manifests

```text
protocol/
  preregistration.md
  final_manifest.json
  metric_contract.json
  exclusions.json
  compute_accounting.csv
```

All result-producing commands must be callable from one runner and emit machine-readable JSON plus a human-readable report.

---

## 16. Timeline

### Phase 0 — claim and code audit (Weeks 1--3)

- finish literature claim matrix;
- implement V4 mathematical property tests;
- fix inverse, coefficient, dead-parameter, and order-convention issues;
- define information regimes and metric contract;
- publish a current-project limitations note.

**Gate:** no unresolved mismatch between theorem, code, and stated metric.

### Phase 1 — independent composition bridge (Weeks 4--7)

- reproduce Mason condition manifest;
- run words through length six;
- implement simple/statistical/sequence baselines;
- report length extrapolation.

**Gate:** full model must beat source and neural baselines on at least one order-sensitive split; otherwise simplify the paper around `L_p` and benchmark findings.

### Phase 2 — CIFAR model bank (Weeks 8--13)

- collect/train 30 development then 75+ final-eligible checkpoints;
- run ODP-Comp-CIFAR;
- add across-model ranking and regret;
- integrate R3 ODP methods.

**Gate:** compositional score must add value beyond source accuracy/model family with a family-clustered CI excluding zero.

### Phase 3 — ImageNet and realistic corruptions (Weeks 14--20)

- screening run;
- freeze core transformations;
- use ImageNet-C/C-bar/3DCC-compatible shifts;
- forecast mCE contributions and accuracy;
- evaluate held-out architectures.

**Gate:** no catastrophic loss versus the best R2 baseline; target relative improvement defined in Section 17.

### Phase 4 — natural/hybrid shifts (Weeks 21--25)

- WILDS/MetaShift/DomainBed factor protocol;
- commutative intersections and natural+operator words;
- average/worst-group metrics.

**Gate:** at least two natural/hybrid datasets show positive controlled improvement; otherwise restrict the paper's scope to physical corruptions.

### Phase 5 — method freeze and preregistration (Weeks 26--27)

- choose one primary model and one small variant;
- freeze all baselines/hyperparameters;
- generate final manifest and hashes;
- independent code/statistics review.

### Phase 6 — one-shot final evaluation (Weeks 28--31)

- run sealed test once;
- generate tables automatically;
- no target-guided model changes;
- record failures and exclusions.

### Phase 7 — paper and artifact (Weeks 32--36)

- write an 8--9 page main paper plus appendix;
- package code, manifests, and small artifacts;
- run anonymous reproducibility test from a clean machine;
- perform final citation/novelty refresh.

---

## 17. Predeclared go/no-go gates

These thresholds are prospective and must be frozen before final evaluation.

### G1: accuracy forecasting

Across sealed dataset × model-family × word-family clusters:

- at least **15% relative reduction in macro accuracy MAE** versus the best R2 non-algebraic baseline;
- 95% hierarchical-bootstrap CI for improvement excludes zero;
- no more than one benchmark tier shows material harm greater than 5% relative.

### G2: incremental value

Against `g0 = source performance + model family + training recipe`:

- positive held-out `MAE(g0) - MAE(g1)`;
- controlled/partial Spearman improvement with CI excluding zero;
- model-selection regret no worse than `g0` and improved on at least half of target families.

### G3: order

On preregistered genuinely non-commutative mechanisms:

- reverse-order sign accuracy >= 80%;
- reverse-order MAE lower than all commutative R2 baselines;
- antisymmetry residual passes a numerical tolerance;
- success occurs on at least two datasets and two model families.

### G4: length

- error reported separately through length six;
- full model beats left-fold MLP/GRU and `L_p` on order-sensitive length-4--6 words;
- if error grows faster than the preregistered theoretical bound, the long-word claim fails.

### G5: uncertainty

For nominal 90% intervals:

- empirical family-level coverage in `[0.85, 0.95]`;
- selective forecasting curve improves monotonically as low-confidence predictions are rejected.

### G6: breadth

Final evidence must include:

- one exact prior composition protocol;
- CIFAR-10 and CIFAR-100 model banks;
- one ImageNet-scale setting;
- at least two natural or natural+operator settings;
- at least six architecture families overall.

### Decision outcomes

- **All G1--G6 pass:** main-track submission with method + benchmark claims.
- **G1/G2 pass but G3 fails:** remove non-commutative headline; submit saturation/composition forecasting paper.
- **G3 passes only on synthetic/CIFAR:** scope claim to low-level physical corruptions; do not claim natural-shift algebra.
- **`L_p` equals/beats full model broadly:** headline the simple law and benchmark; retain Heisenberg as an order-specific extension.
- **R2 is far worse than R3 but useful before deployment:** frame as a cost/information tradeoff, not SOTA ODP.
- **G1/G2 fail:** do not send as a positive method paper; release the benchmark and negative characterization.

---

## 18. Main paper structure

1. **Introduction:** pre-deployment versus target-available performance prediction.
2. **Problem setup:** explicit R0--R4 information regimes.
3. **Benchmark:** CompShift-PP and independently sourced protocols.
4. **Method:** signed saturation state, symmetric interaction, antisymmetric order state.
5. **Theory:** exact properties, representational limits, and error bound.
6. **Experiments:** source-controlled across-model and across-environment evaluation.
7. **Mechanism:** filter/write counterfactuals and cyclic-order limitation.
8. **Limitations:** labels for primitive behavior, calibration cost, summary-statistic sufficiency, scope of natural factors.
9. **Broader impact:** safer pre-deployment model selection, with warning that forecasts cannot replace direct safety evaluation.

Headline tables:

- Table 1: benchmark comparison and information regimes;
- Table 2: accuracy MAE/RMSE by tier and length;
- Table 3: ODP-style cross-model rank and regret;
- Table 4: component ablations by symmetric/order split;
- Table 5: natural/hybrid results;
- Figure 1: task diagram showing no composed target access;
- Figure 2: error versus calibration cost;
- Figure 3: error versus word length;
- Figure 4: order counterfactuals;
- Figure 5: uncertainty/abstention curves.

---

## 19. Risk register

| Risk | Why it matters | Mitigation | Honest fallback |
|---|---|---|---|
| Source accuracy/model family explains results | Common in OOD prediction | `g0/g1`, controlled rank, matched pairs | Report no incremental value |
| `L_p_only` remains strongest | Full algebra may be unnecessary | split order-sensitive and commutative targets | simple-law paper |
| Matrix form appears decorative | closed form is equivalent | prove equivalence; emphasize streaming state | remove “matrix advantage” language |
| Natural factors are not operators | invalid non-commutative claim | commutative factor track + hybrid operator track | scope to physical shifts |
| Iterative test overfitting | current V1--V3 test is no longer pristine | sealed final manifest | label old results exploratory |
| Large model overfits aggregate conditions | 103k params vs few pair points | tiny/small variants, model bank, nested splits | use parametric law |
| ODP methods dominate | they see actual target images | separate R2 and R3 | report information-cost frontier |
| New primitive outside support | interaction cannot be identified | uncertainty and abstention | decline to predict |
| Cyclic order effects | potential model is transitive | unrestricted skew baseline and synthetic cycle | state representational limit |
| Mixed-unit BPE is misleading | drift can dominate | standardized/per-coordinate metrics | retire BPE headline |
| Compute becomes prohibitive | ImageNet model × word product is large | screening, streaming summaries, balanced subset | narrower but sealed benchmark |
| Dataset/model licensing blocks release | harms reproducibility | manifests and scripts, no redistribution | release derived small summaries only |

---

## 20. Immediate next actions

### First 30 days

1. Create `literature/CLAIM_MATRIX.csv` and complete the formal search.
2. Add `KNOWN_LIMITATIONS.md` separating matrix-core and end-to-end claims.
3. Create `src_v4/algebra.py` with exact inverse and property tests.
4. Implement `M_B_core`, `M_B_treeavg`, and exact closed-form equivalence.
5. Implement source/model-family baseline `g0`.
6. Add standard accuracy MAE/RMSE/Spearman/regret metrics.
7. Port Mason's fixed composition manifest without changing it.
8. Draft the CompShift-PP JSON schema and split manifest.
9. Estimate variance and compute on a 5-model CIFAR pilot.
10. Freeze Phase-1 hypotheses before running the full Mason bridge.

### First concrete deliverables

```text
literature/CLAIM_MATRIX.csv
KNOWN_LIMITATIONS.md
src_v4/algebra.py
tests_v4/test_algebra_properties.py
benchmark/schema.py
benchmark/manifests/mason_exact.json
baselines/source_family.py
analysis_v4/prediction_metrics.py
analysis_v4/regret.py
protocol/phase1_preregistration.md
```

---

## 21. Source register used for this plan

The following sources were inspected through their public official GitHub repositories on 2026-10-23 unless noted otherwise.

| Source | URL | Fact used |
|---|---|---|
| ODP-Bench official repository | <https://github.com/h-yu16/Performance_Prediction> | ICCV 2025 benchmark; predictor implementations; datasets/models described in paper/repository |
| Accuracy-on-the-Line author repository | <https://github.com/millerjohnp/linearfits_app> | source accuracy as a strong OOD-performance baseline |
| DG benchmark misspecification repository | <https://github.com/olawalesalaudeen/misspecified_DG_benchmarks> | warning that aggregate ID/OOD lines and pooled splits can hide benchmark failures |
| ATC official repository | <https://github.com/saurabhgarg1996/ATC_code> | ATC algorithm, datasets, GDE comparison, target-unlabeled protocol |
| Agreement-on-the-Line official repository | <https://github.com/kebaek/Agreement-on-the-line> | NeurIPS 2022 performance-prediction framing |
| Mason compositional robustness | <https://github.com/ianxmason/compositional-robustness> | six primitives, 167 conditions, lengths 1--6, accuracy/loss/invariance analysis |
| ImageNet-C/P official repository | <https://github.com/hendrycks/robustness> | mCE, mFR, mT5D and benchmark conventions |
| RobustBench | <https://github.com/RobustBench/robustbench> | standardized model zoo and corruption evaluation |
| CorruptionBenchCV | <https://github.com/nis-research/CorruptionBenchCV> | unified C/C-bar/P/3DCC metrics including ECE and relative robustness |
| 3D Common Corruptions | <https://github.com/EPFL-VILAB/3DCommonCorruptions> | 20 realistic 3D corruptions and five severities |
| WILDS | <https://github.com/p-lambda/wilds> | ten natural-shift datasets and standardized evaluators |
| DomainBed | <https://github.com/facebookresearch/DomainBed> | domain-generalization datasets, algorithms, model-selection protocols |
| BREEDS | <https://github.com/MadryLab/BREEDS-Benchmarks> | ImageNet subpopulation-shift benchmark |
| MetaShift | <https://github.com/Weixin-Liang/MetaShift> | 12,868 natural image sets, 410 classes, explicit contexts and distances |
| AugMix | <https://github.com/google-research/augmix> | composed augmentation context; ImageNet-C/P robustness and uncertainty |
| Sparse DTM author project | <https://github.com/psoulos/sdtm> | correction that this is neurosymbolic sequence/tree compositional generalization, not the direct vision benchmark |

### Research limitation

This plan is a deep repository-based review, not a claim of an exhaustive literature search. GitHub search can miss closed-source papers, papers with no code, recent indexing, and differently worded work. The formal multi-index literature audit in Section 11.3 remains mandatory before asserting novelty.

---

## 22. Final recommendation

Do not submit the current V3 repository unchanged as a NeurIPS method paper. Use it as a strong pilot.

The fastest credible path is:

1. repair and simplify the implementation;
2. reproduce Mason's exact length-6 composition benchmark;
3. build a multi-model CIFAR composition benchmark with ODP metrics and source-controlled baselines;
4. add one ImageNet-scale and two natural/hybrid settings;
5. compare R2 target-free forecasting with R3 target-available ODP methods;
6. run one sealed final test;
7. frame novelty around the **pre-deployment composition problem and benchmark**, with the algebra as the structured solution rather than as a universal physical law.

If those experiments pass the predeclared gates, the work would have a substantially stronger NeurIPS case: a clearly differentiated problem, a benchmark that connects two previously separate literatures, a mechanism-driven method, and evidence measured in the metrics previous papers actually use.
