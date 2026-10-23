# Phase 1B Preregistration — Severity-Calibrated Mason Extension

**Protocol ID:** `V4-P1B-CALIBRATED-001`
**Status:** STAGE B — STRENGTHS FROZEN; PHASE-1B PAIRS AND TARGETS UNOPENED
**Date:** 2026-10-23
**Parent protocol:** `V4-P1-MASON-001` (completed negative result; not replaced)

## 1. Motivation and non-rescue boundary

Phase 1 found that the published fixed severities collapse CIFAR-10 and EMNIST accuracy toward chance. A maximum-absolute primitive envelope consequently beat Shift Algebra. That result remains final and will not be overwritten, reinterpreted, or tuned away.

Phase 1B asks a new, narrower question:

> With the word list, classifiers, information regime, forecasters, and confirmation image partition held fixed, does matching each single primitive to a mild accuracy effect restore enough target dynamic range for composition forecasting to be informative?

This is a new intervention benchmark, not a rerun of Phase 1. Severity selection may use only clean and single-primitive outcomes on the old Phase-1 development partition. It may not use any ordered pair or length-3--6 Phase-1B outcome. The Phase-1 confirmation outcomes at the original severities are known and motivate this extension, but they do not select Phase-1B strengths.

## 2. Fixed datasets, models, words, and image access

- Datasets: CIFAR-10 and EMNIST Balanced, loaded and hash-validated exactly as in Phase 1.
- Word manifest: `v4/benchmark/manifests/mason_exact.json`, SHA-256 `d2be1ca5c899659482204630d1f2b755587b71ee994372a7b716e5c5ad591264`.
- Words per dataset: identity 1, singles 6, all ordered distinct pairs 30, and 130 targets of lengths 3--6.
- CIFAR classifier configuration: SmallCNN width 24, seed 1301, 8 epochs.
- EMNIST classifier configuration: SmallCNN width 16, seed 1301, 5 epochs.
- CIFAR checkpoint SHA-256: `9f5815dd3ab43edd5d09636e8d2688fa9ab8aaae346143ec6d5184bd99944848`.
- EMNIST checkpoint SHA-256: `05c49cd3678ee537accac6380d33002e4d8f512e5eb93093b18f0701f815ca53`.

Both hashes exactly reproduce the checkpoint identities bound in the completed Phase-1 freeze.

Phase-1B calibration partitions:

- CIFAR-10: 100 images/class, class-local offset 0 (1,000 images);
- EMNIST: 40 images/class, class-local offset 0 (1,880 images).

Phase-1B one-shot target partitions:

- CIFAR-10: 900 images/class, class-local offset 100 (9,000 images);
- EMNIST: 360 images/class, class-local offset 40 (16,920 images).

The target partitions reuse the base images used by the completed Phase-1 confirmation, which must be disclosed. However, those images have never been evaluated under the Phase-1B homotopy strengths at Stage A. No claim of a new independent image sample will be made. Crucially, no Phase-1B pair or target outcome from these images may inform severity or method selection.

## 3. Severity operator

For original Phase-1 primitive `T_j` and attenuation `alpha_j`, define

```text
H[j, alpha_j](x) = (1 - alpha_j) x + alpha_j T_j(x).
```

Properties:

- `alpha=0` is exactly identity;
- `alpha=1` is exactly the Phase-1 operator;
- all primitives, including fixed Invert and Rotate90, share one declared scale;
- for a word, each attenuated operator is applied chronologically to the output of the preceding operator;
- impulse-noise fields retain seed 2778 and batching-independent sample IDs.

This homotopy is an experimental severity definition. It is not claimed to be the canonical perceptual parameterization of each corruption.

## 4. Single-only calibration rule

Candidate grid, fixed for every primitive and dataset:

```text
0.000, 0.025, ..., 0.500, 0.600, 0.700, 0.800, 0.900, 1.000
```

For clean accuracy `a0`, class chance `c`, and candidate single accuracy `a(j, alpha)`, compute

```text
normalized_drop(j, alpha) = (a0 - a(j, alpha)) / (a0 - c).
```

For each dataset and primitive independently:

1. select the candidate minimizing `abs(normalized_drop - 0.08)`;
2. break an exact tie toward smaller `alpha`;
3. report whether the selected drop lies in the diagnostic band `[0.04, 0.12]`;
4. freeze the selected values even if one misses the band—no grid, target, or tie-break revision is allowed.

The calibration executable is structurally restricted to clean and one primitive at a time. It does not accept a word manifest and cannot evaluate pairs or targets.

### Frozen calibration output

| Dataset | Contrast | Blur | Impulse | Invert | Rotate90 | Swirl |
|---|---:|---:|---:|---:|---:|---:|
| CIFAR-10 | 0.450 | 0.350 | 0.100 | 0.175 | 0.225 | 0.225 |
| EMNIST | 0.250 | 0.275 | 0.150 | 0.050 | 0.150 | 0.150 |

All CIFAR selected drops are inside `[0.04, 0.12]`. EMNIST Invert has normalized drop `0.1332`, outside the diagnostic band, because the frozen grid jumps across the target between adjacent candidates. In accordance with the preregistration, it remains frozen at the nearest candidate (`alpha=0.050`); the grid was not refined.

## 5. Two-stage seal

### Stage A: this preregistration

Before single-only calibration:

- freeze the severity operator, candidate grid, target drop, tie break, models, partitions, methods, metrics, statistical script, and gates;
- commit executable code and checkpoint hashes.

### Stage B: calibrated-strength freeze

After clean/single calibration, but before any Phase-1B pair or target evaluation:

- commit both complete candidate tables;
- commit `v4/phase1b/FROZEN_STRENGTHS.json` with selected strengths, calibration hashes, model hashes, and Stage-A commit;
- update only the status/freeze-record fields in this document;
- make no method or gate changes.

Only after the Stage-B commit may the confirmation pipeline run.

## 6. Information regime and unchanged forecasters

The primary regime remains R2:

Available during per-dataset fitting:

- clean behavior;
- six single-primitive behavior vectors;
- all 30 ordered-pair behavior vectors.

Unavailable during fitting:

- all 130 length-3--6 target behaviors.

The Phase-1 forecasters and hyperparameters are reused unchanged:

1. physically clipped additive;
2. primitive mean;
3. maximum-absolute primitive envelope;
4. pair-fitted signed `L_p` (`p>=1`);
5. pair-feature ridge;
6. arbitrary directed-pair residual extrapolation;
7. five-seed potential Shift Algebra, with `p` fit first and frozen before interaction fitting.

The only Phase-1 forecasting-code additions are protocol labeling and emission of per-target scored predictions for the frozen bootstrap. They do not alter fitting or prediction equations. Potential fitting remains 2,500 steps; `L_p` fitting remains 1,500 steps. Accuracy predictions are clipped to `[0,1]`, with clipping frequency disclosed.

## 7. Primary and secondary endpoints

### Primary

Dataset-macro accuracy MAE in percentage points over the 130 target words, with CIFAR-10 and EMNIST weighted equally. Potential Shift Algebra is compared against the best observed non-algebraic R2 method from the six declared alternatives.

### Secondary

- MAE by dataset and word length;
- RMSE and worst-decile MAE;
- normalized five-behavior MAE;
- physical-clipping fraction;
- mean target accuracy by length;
- reversal-difference MAE on every target word whose exact reverse also occurs;
- measurable reversal-sign accuracy (truth difference at least 1pp for CIFAR or 0.5pp for EMNIST).

## 8. Dynamic-range validity guard

For each dataset, with clean accuracy `a0` and chance `c`:

1. mean target accuracy at each of lengths 3, 4, 5, and 6 must exceed `c + 0.15(a0-c)`;
2. at most 10% of all target conditions may have accuracy at or below `c + 0.02`.

If either dataset fails, Phase 1B is declared dynamically uninformative and cannot pass a method gate, regardless of method ranking. The result is still reported.

## 9. Inference

Use 10,000 paired hierarchical bootstrap draws with seed 15101. The hierarchy is:

1. resample datasets;
2. within dataset, retain separate lengths 3--6;
3. within length, resample unordered primitive-set families;
4. retain and average every available order in a sampled family.

The bootstrap estimand is comparator MAE minus potential-algebra MAE; positive values favor Shift Algebra. The comparator is the non-algebraic method with the lowest observed primary macro MAE. Report the clustered point estimate and percentile 95% interval. This two-dataset interval is a protocol diagnostic, not a broad population claim.

## 10. Frozen gates

A **directional method success** requires all of:

1. dynamic-range validity passes on both datasets;
2. potential Shift Algebra has lower primary macro MAE than the best non-algebraic R2 baseline;
3. the clustered 95% interval for baseline-minus-potential improvement has lower bound above zero;
4. on neither dataset is potential MAE more than 5% worse than that dataset's best non-algebraic baseline.

A **strong method success** additionally requires at least 15% relative macro-MAE reduction.

The **order-channel gate** requires potential Shift Algebra to beat signed `L_p` on reversal-difference MAE across all available reverse pairs. The overall Phase-1B method gate passes only when both strong success and the order-channel gate pass.

No gate, method, threshold, grid, exclusion, or endpoint may change after Stage A.

## 11. Exclusions and failures

A run may stop only for non-finite outputs, corrupt/missing source assets, hash mismatch, class mapping failure, or executable failure. No condition may be excluded for poor performance. A runtime failure may resume from a complete behavior CSV, but the existing CSV must not be overwritten silently.

If the method gate fails, retain Phase 1B as a negative result and proceed, if scientifically useful, to Phase 2 model-bank work rather than another same-data severity rescue.

## 12. Frozen commands

Stage A verification:

```bash
python v4/phase1b/verify_phase1b.py
```

Single-only calibration after the Stage-A commit:

```bash
python -m v4.phase1b.calibration --dataset cifar10
python -m v4.phase1b.calibration --dataset emnist
python -m v4.phase1b.freeze_strengths
```

One-shot confirmation only after the Stage-B commit:

```bash
python -m v4.phase1b.pipeline --dataset cifar10
python -m v4.phase1b.pipeline --dataset emnist
python -m v4.phase1b.analyze
```

## 13. Freeze record

- Stage-A executable commit: `5ff297827e46fd0fde9b5d8298df13409d36fc74`
- Stage-B strength-freeze commit: `44f55c1b0081d16339a6dd22649dc4abf7a60a46`
- Confirmation result commit: **TBD**
- Frozen strength file SHA-256: `f851fc920cc2d1ff677122b5de7f2fde0ab229acfcd197c8fef40913fd7c8772`
