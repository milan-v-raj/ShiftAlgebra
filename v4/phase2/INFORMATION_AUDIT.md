# Phase 2 Pilot Information-Leak and Correctness Audit

**Protocol:** `V4-P2-CIFAR-BANK-PILOT-001`
**Audit timing:** after one-shot confirmation, before result commit
**Conclusion:** no detected confirmation-to-fit leak or result-changing implementation defect

## 1. Freeze chronology

- Stage-A executable commit `63f257b777aa8cf1b63fca9cd5f8abea30fc145e` fixed the bank, word split, model, metrics, and gates before Phase-2 outcomes.
- Development used only seed-221xx lineages and the 40 declared method-development words.
- `FROZEN_METHOD.json` and all development artifacts were committed as `9dd98357ca8af2e6122e399be66046ca659a3566`.
- The method-freeze record was pushed before confirmation training began.
- Confirmation uses separately initialized seed-321xx lineages.
- Selected `q=0.6`, both ridge models, feature means/scales, category levels, and ridge alpha remained unchanged during confirmation.

## 2. Word and model separation

The manifest contains exactly:

- 30 development snapshots in 15 lineages;
- 30 confirmation snapshots in 15 lineages;
- five architecture families and three recipes in each split;
- 37 calibration conditions;
- 40 method-development words;
- 90 confirmation words.

Exact reverse pairs share one word role. Confirmation scoring uses only rows marked `confirmation`. The confirmation behavior file also contains calibration/development-role conditions because all 167 conditions were evaluated uniformly, but no confirmation parameter is fitted after that file is created.

Earlier Phase-1/1B experiments evaluated the same word identities on different classifiers. This pilot therefore seals new model outcomes, not globally novel word names. The protocol and report state this limitation.

## 3. Image separation

- Development behaviors use 50 images/class at test offset 0: 500 images.
- Confirmation behaviors use 100 images/class at test offset 100: 1,000 images.
- Stable class-local indexing makes these sets disjoint.
- Model training uses the declared 40,000-image CIFAR training split, not either behavior partition.

## 4. R1/R2 access audit

For each model, `forecast_one_model` constructs all ordinary forecasts from:

- identity accuracy;
- six single accuracies;
- all 30 directed pair accuracies;
- the frozen scalar `q`;
- frozen development ridge parameters for `g0`/`g1`.

The model's length-3--6 truth is copied into output only for scoring. It is not passed into product retention, pair residual, hazard, or ridge prediction functions.

The q-grid search and ridge fitting run only in `freeze_development`, filtered to `word_role == method_development` on the development model bank. `forecast_confirmation` only deserializes and applies the frozen values; it has no fitting call.

## 5. R3 label-access audit

ATC and DoC consume actual target logits and are correctly labeled R3.

- ATC threshold is calibrated from clean confidence and clean correctness.
- Target ATC prediction is the fraction of target confidences above that frozen threshold.
- DoC uses clean accuracy plus target-minus-clean mean confidence.
- Target labels are used to accumulate the separate ground-truth accuracy field, but neither R3 formula reads target correctness or labels.

The evaluator computes predictions and truth in one process, so separation is enforced by code paths rather than cryptographic process isolation. A future full benchmark should write unlabeled predictions before opening labels.

## 6. Determinism and model accounting

- Python, NumPy, and Torch seeds are set per lineage.
- DataLoader shuffling and augmentations use the lineage generator.
- Horizontal flips and Gaussian augmentation are generator controlled.
- All checkpoint hashes are recorded in split training JSON files.
- Epoch-one and epoch-two snapshots share a lineage and are retained together in the paired bootstrap.
- No checkpoint or condition was excluded for performance.

The 30 confirmation snapshots correspond to 15 independent optimization runs, not 30 independent models. Reports use both numbers.

## 7. Metric recomputation

The final analysis reads exactly 2,700 sealed rows (`90 words × 30 snapshots`). Independent checks confirm:

- `g0` MAE 3.303592pp;
- `g1` MAE 1.736934pp;
- direct reduction 1.566658pp;
- lineage/word-group macro bootstrap point 1.529409pp;
- 10,000-draw 95% interval `[0.981429, 2.190969]`pp;
- raw model-rank Spearman 0.242 for `g0` and 0.665 for `g1`;
- top-1 regret 2.158pp for `g0` and 0.489pp for `g1`.

The clustered point differs slightly from the direct pooled MAE reduction because its estimand gives word groups and training lineages explicit cluster weighting.

## 8. Findings that limit interpretation

No implementation error was found, but the following are material:

1. Confirmation models are low-accuracy two-epoch CPU models (29.8--53.5% clean accuracy).
2. All five architecture and three recipe families occur in both development and confirmation; this is seed/lineage transfer, not family holdout.
3. The bank has only 15 independent lineages per split.
4. The symmetric hazard model slightly beats the ordered model in MAE and reversal error; the positive result is not evidence for the order term.
5. The pair-accuracy method has lower top-1 selection regret than `g1` despite worse MAE, so no single method dominates all deployment metrics.
6. ATC and DoC are simplified declared references on poorly calibrated tiny models, not an official ODP-Bench reproduction.
7. CIFAR-100, robust recipes, public modern models, and 75+ final checkpoints remain absent.

## 9. Verification

- Phase-2 tests: 4/4 pass.
- Both behavior files contain exactly 5,010 rows (`30 × 167`).
- Development and confirmation sample counts are uniformly 500 and 1,000 respectively.
- Confirmation forecast contains exactly 3,900 target rows and 2,700 sealed rows.
- Phase-1B, Phase-1, Phase-0, and historical V1--V3 regression checks remain required before final commit.
- Compile and whitespace checks pass after finalization.

The pilot result should remain frozen and should not be rerun as a rescue if later full-scale tests disagree.
