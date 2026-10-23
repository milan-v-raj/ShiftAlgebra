# Phase 2 Pilot Final Report — CIFAR Model Bank

**Protocol:** `V4-P2-CIFAR-BANK-PILOT-001`
**Stage-A executable commit:** `63f257b777aa8cf1b63fca9cd5f8abea30fc145e`
**Stage-B method-freeze commit:** `9dd98357ca8af2e6122e399be66046ca659a3566`
**Status:** one-shot CPU pilot complete; preregistered incremental-value gate passed; full Phase 2 remains incomplete

## 1. Executive result

The Phase-2 CPU pilot produced the first positive prospective V4 result.

On 30 confirmation snapshots from 15 independently trained lineages and 90 sealed length-3--6 words, adding frozen composition forecasts to the source/model/word baseline reduced accuracy MAE from **3.304pp (`g0`) to 1.737pp (`g1`)**, a **47.4% relative reduction**.

The lineage/reverse-word-group clustered improvement was **1.529pp**, with preregistered 95% bootstrap interval **[0.981, 2.191]pp**. All four pilot gates passed.

Across-model deployment metrics also improved:

- raw model-rank Spearman: **0.242 -> 0.665**;
- source/family/recipe-controlled Spearman: **-0.235 -> 0.657**;
- top-1 model-selection regret: **2.158pp -> 0.489pp**;
- selecting a model within 1pp of oracle: **44.4% -> 78.9%**.

This result supports the narrower claim that composition measurements add value beyond source/model metadata in this small CIFAR model bank. It does not complete Phase 2 or restore the non-commutative headline: the symmetric hazard ablation slightly beat the ordered hazard model, and the order diagnostic failed.

## 2. Frozen design

### 2.1 Model bank

Each split contains:

- five lightweight architecture families;
- three training recipes;
- 15 independent training lineages;
- epoch-one and epoch-two snapshots from each lineage;
- 30 deployment candidates total.

Development and confirmation use disjoint seeds. Epoch snapshots are clustered within lineage and are not treated as independent training replicates.

Confirmation clean accuracy ranges from **29.8% to 53.5%**, with mean **42.2%**. These intentionally cheap two-epoch CPU models create ranking diversity but are much weaker than modern CIFAR models.

### 2.2 Condition seal

The fixed 167 Mason words use the frozen Phase-1B CIFAR homotopy strengths:

- 37 clean/single/pair calibration conditions;
- 40 method-development words, ten per length;
- 90 confirmation words.

Reverse words were assigned together. Development used 500 CIFAR test images per condition; confirmation used a disjoint 1,000 images per condition.

### 2.3 Two-stage chronology

1. Bank identities, seeds, word roles, methods, metrics, and gates were committed and pushed before Phase-2 outcomes.
2. Development lineages were trained and evaluated.
3. The scaling exponent, ridge coefficients, feature normalization, and complete q-grid search were committed and pushed.
4. Confirmation lineages were then independently trained and evaluated once.
5. The frozen forecast and analysis scripts generated this result without post-confirmation tuning.

## 3. Extensive hazard development

The proposed accuracy state uses normalized clean-to-chance retention and hazard:

```text
r(a) = clip((a-0.1)/(a_clean-0.1), 0.02, 2)
h(a) = -log r(a)
```

For pair residual `R_ij = h_ij - h_i - h_j`, a length-m word uses

```text
h_hat = sum h_i + N_pairs^(q-1) sum_(i before j) R_ij.
```

The frozen development grid selected **q=0.6**. Thus the pair correction grows faster than an average (`q=0`) but slower than a full pair sum (`q=1`).

Development-only MAE was 1.513pp for ordered hazard and 0.951pp for `g1`. These values were recorded before confirmation and were not the final endpoint.

## 4. Confirmation accuracy forecasting

Methods are sorted by sealed accuracy MAE. R3 methods see unlabeled images from the actual target and are reported separately as information advantaged.

| Method | Regime | Accuracy MAE | RMSE |
|---|---|---:|---:|
| **`g1` metadata + composition** | R2 meta-calibrated | **1.737pp** | 2.077pp |
| Symmetric hazard, q=0.6 | R2 | 2.089pp | 2.574pp |
| Ordered hazard, q=0.6 | R2 | 2.093pp | 2.577pp |
| `g0` source/model/word | R0 + word metadata | 3.304pp | 4.144pp |
| Product retention | R1 | 5.226pp | 6.975pp |
| Accuracy-space pair residual | R2 | 7.265pp | 9.452pp |
| Maximum primitive damage | R1 | 12.033pp | 13.325pp |
| Primitive mean | R1 | 15.834pp | 16.937pp |
| DoC | R3 target images | 19.247pp | 21.342pp |
| ATC | R3 target images | 20.495pp | 25.317pp |

ATC and DoC perform poorly on this bank because the tiny two-epoch models are strongly and inconsistently miscalibrated. These are clean implementations of the declared confidence rules, not evidence against official ODP methods on modern model banks.

### 4.1 MAE by architecture family for `g1`

| Family | MAE |
|---|---:|
| Shallow CNN | 1.439pp |
| Depthwise | 1.613pp |
| Patch convolution | 1.628pp |
| Pointwise-heavy | 1.756pp |
| Residual | 2.248pp |

All five families remain below 2.25pp, although family sample sizes are small.

### 4.2 MAE by word length for `g1`

| Length | MAE |
|---:|---:|
| 3 | 1.641pp |
| 4 | 1.541pp |
| 5 | 1.768pp |
| 6 | 2.046pp |

Error increases moderately at length six but does not reproduce the catastrophic Phase-1B magnitude failure.

### 4.3 Mean true accuracy by length

| Length | Mean accuracy |
|---:|---:|
| 3 | 24.732% |
| 4 | 20.688% |
| 5 | 18.708% |
| 6 | 17.713% |

The bank remains above the 10% chance level on average, but long words are still difficult.

## 5. Across-model ranking and selection

| Method | Model-rank Spearman | Controlled Spearman | Top-1 regret | Top-3 regret | Within-1pp selection |
|---|---:|---:|---:|---:|---:|
| **`g1`** | **0.665** | 0.657 | 0.489pp | 0.214pp | 78.9% |
| Symmetric hazard | 0.610 | **0.683** | 0.744pp | 0.238pp | 72.2% |
| Ordered hazard | 0.607 | 0.681 | 0.697pp | 0.230pp | 73.3% |
| `g0` | 0.242 | -0.235 | 2.158pp | 1.907pp | 44.4% |
| Accuracy pair residual | 0.526 | 0.583 | **0.378pp** | 0.250pp | **81.1%** |

`g1` is best in MAE and raw rank, but not every deployment metric: the accuracy-space pair residual has lower top-1 regret despite much worse calibrated MAE. This illustrates why accuracy error, ranking, and selection must all be reported.

`g1` has strictly lower per-environment selection regret than `g0` on 63.3% of sealed words and is no worse on 65.6%.

## 6. Incremental-value inference and gate

The frozen 10,000-draw paired bootstrap resampled reverse-word groups and training lineages, retaining epoch snapshots within lineage.

For `g0 absolute error - g1 absolute error`:

- clustered point: **1.529pp**;
- 95% interval: **[0.981, 2.191]pp**.

| Pilot criterion | Result |
|---|---|
| `g1` MAE lower than `g0` | PASS |
| Clustered 95% interval lower bound above zero | PASS |
| `g1` controlled rank higher than `g0` | PASS |
| `g1` top-1 regret no worse than `g0` | PASS |
| **Overall incremental pilot gate** | **PASS** |

## 7. Order result

The ordered component did not pass its secondary diagnostic:

- ordered reversal-difference MAE: **0.582pp**;
- symmetric reversal-difference MAE: **0.570pp**;
- ordered sign accuracy on 20 measurable model/reverse pairs: **65%**.

The symmetric model is marginally better in both primary MAE and reversal MAE. Therefore the pilot's positive result comes from the extensive retention/hazard representation and meta-calibration, not demonstrated non-commutative order modeling.

This is consistent with the Phase-1B conclusion: order signal exists in some settings, but it is not yet broad or reliable enough for a headline claim.

## 8. What the pilot establishes

Supported within this pilot:

1. primitive/pair composition measurements add substantial predictive information beyond source accuracy, architecture, recipe, epoch, parameter count, length, and primitive identities;
2. a sub-extensive pair scaling law learned on separate models/words transfers to independent seeds and sealed words;
3. the resulting score improves model ranking and deployment selection as well as MAE;
4. across-model evaluation changes the conclusion compared with one-model length extrapolation.

Not established:

1. completion of the roadmap's final Phase-2 bank;
2. transfer to CIFAR-100, modern high-accuracy architectures, robust training, or public checkpoints;
3. architecture-family holdout, since all five families occur in development and confirmation;
4. superiority to official ODP-Bench implementations;
5. a positive non-commutative order claim;
6. statistical independence of the two epoch snapshots within a lineage.

## 9. Decision

**CPU pilot gate: PASS. Full Phase 2: IN PROGRESS.**

The result justifies scaling to the roadmap's final-eligible bank, but it cannot substitute for it. The next Phase-2 work should add:

- at least 75 final checkpoints;
- six or more architecture families;
- four or more training-recipe families, including robust training;
- CIFAR-100;
- stronger/high-accuracy models;
- architecture- and recipe-holdout splits;
- official R3 ODP adapters where licensing and compute permit.

The frozen pilot remains a prospective positive result and must not be rerun or tuned on the confirmation outcomes.

## 10. Reproduction

```bash
python v4/phase2/verify_phase2.py
```

Main aggregate artifact: `v4/phase2/results/PHASE2_PILOT_ANALYSIS.json`, SHA-256 `3a208af4c3ddd1c309187989d735f315973bc42da2954189564b581b131f058e`.
