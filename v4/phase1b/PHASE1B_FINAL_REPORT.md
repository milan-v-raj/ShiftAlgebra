# Phase 1B Final Report — Severity-Calibrated Mason Extension

**Protocol:** `V4-P1B-CALIBRATED-001`
**Parent result:** `V4-P1-MASON-001` (unchanged)
**Stage-A executable commit:** `5ff297827e46fd0fde9b5d8298df13409d36fc74`
**Stage-B severity freeze:** `44f55c1b0081d16339a6dd22649dc4abf7a60a46`
**Status:** one-shot confirmation complete; dynamic-range and method gates failed

## 1. Executive result

Phase 1B did not produce a positive method result.

Severity calibration substantially weakened every primitive and invalidated the Phase-1 maximum-envelope explanation: the max envelope went from the best Phase-1 baseline to one of the worst Phase-1B methods. Nevertheless, effects compounded much faster from pairs to length 3--6 than any frozen forecaster predicted. The arbitrary directed-pair residual baseline won on both datasets.

Primary dataset-macro accuracy MAE:

- pair residual: **18.837 percentage points**;
- potential Shift Algebra: **20.575pp**;
- difference: potential is **1.737pp / 9.22% worse**;
- paired word-family bootstrap for `pair residual - potential`: **-1.737pp**, 95% CI **[-3.277, -0.380]**.

The preregistered dynamic-range guard also failed: CIFAR length-six accuracy and EMNIST length-four through length-six accuracy fell below their clean-to-chance floors. Thus Phase 1B cannot support a method-success claim even independently of the adverse method ranking.

One component did work: potential Shift Algebra beat signed `L_p` on the small reversal diagnostic, with 2.208pp versus 3.109pp reversal-difference MAE and 100% versus 16.7% sign accuracy on six measurable pairs. This passes the secondary order-channel gate but not the overall method gate.

## 2. Design and freeze chronology

Phase 1B was implemented as a two-stage seal:

1. The severity operator, strength-selection rule, forecasters, endpoints, bootstrap, validity guard, and gates were committed before Phase-1B measurements.
2. Strength calibration used only clean and single-primitive outcomes on the old development partitions.
3. Complete candidate tables and selected strengths were committed before any Phase-1B pair or target composition was evaluated.
4. The frozen confirmation partitions were then processed once.

The confirmation used the exact Phase-1 checkpoint hashes and the same independently generated 167-word manifest. It reused the Phase-1 base image partition, which is disclosed; it is not a new independent image sample. The Phase-1B transformed pair and target conditions had not been evaluated before the Stage-B freeze.

| Dataset | Confirmation images | Clean accuracy | Checkpoint SHA-256 prefix |
|---|---:|---:|---|
| CIFAR-10 | 9,000 | 77.844% | `9f5815dd3ab4...` |
| EMNIST Balanced | 16,920 | 87.388% | `05c49cd3678e...` |

No conditions were excluded and no non-finite outputs occurred.

## 3. Frozen severity calibration

Each endpoint primitive `T` was replaced by the preregistered homotopy

```text
H[T, alpha](x) = (1-alpha)x + alpha T(x).
```

For each dataset and primitive, the selected grid value was nearest to a normalized clean-to-chance single-primitive accuracy drop of 0.08.

### 3.1 Selected strengths

| Dataset | Contrast | Gaussian blur | Impulse noise | Invert | Rotate90 | Swirl |
|---|---:|---:|---:|---:|---:|---:|
| CIFAR-10 | 0.450 | 0.350 | 0.100 | 0.175 | 0.225 | 0.225 |
| EMNIST | 0.250 | 0.275 | 0.150 | 0.050 | 0.150 | 0.150 |

### 3.2 Achieved normalized single drops

| Dataset | Contrast | Gaussian blur | Impulse noise | Invert | Rotate90 | Swirl |
|---|---:|---:|---:|---:|---:|---:|
| CIFAR-10 | 0.086 | 0.077 | 0.098 | 0.094 | 0.086 | 0.080 |
| EMNIST | 0.084 | 0.077 | 0.100 | 0.133 | 0.066 | 0.073 |

All CIFAR values fell inside the diagnostic `[0.04, 0.12]` band. EMNIST Invert missed it at 0.133 because adjacent values on the frozen grid straddled the target. As preregistered, the nearest candidate was retained rather than refining the grid.

The complete candidate tables are committed as `results/cifar10-calibration.json` and `results/emnist-calibration.json`.

## 4. Dynamic-range result

### 4.1 Mean target accuracy

| Length | CIFAR-10 | EMNIST |
|---:|---:|---:|
| 3 | 41.179% | 22.639% |
| 4 | 29.429% | 11.661% |
| 5 | 21.498% | 8.244% |
| 6 | 18.248% | 6.396% |

This is materially less saturated than Phase 1, whose corresponding length-three means were 10.15% and 2.35%. However, mild singles still produced a steep composition collapse.

The preregistered minimum mean was `chance + 0.15(clean-chance)`:

- CIFAR-10 floor: 20.177%; length six failed at 18.248%.
- EMNIST floor: 14.917%; lengths four, five, and six failed.

No target fell within two absolute percentage points of chance, so the old literal chance-floor pathology was reduced. The new failure is different: accuracy retains some distance from chance but drops too rapidly with composition length for the benchmark's declared dynamic-range criterion.

**Dynamic-range validity gate: FAIL.**

## 5. Primary accuracy forecasting

Accuracy MAE in percentage points; lower is better:

| R2 method | CIFAR-10 | EMNIST | Dataset-macro |
|---|---:|---:|---:|
| **Pair residual** | **17.202** | **20.473** | **18.837** |
| Pair-feature ridge | 17.295 | 20.475 | 18.885 |
| Potential Shift Algebra | 17.765 | 23.385 | 20.575 |
| Additive | 25.152 | 42.280 | 33.716 |
| Learned signed `L_p` | 25.191 | 42.325 | 33.758 |
| Max-absolute envelope | 42.849 | 64.231 | 53.540 |
| Primitive mean | 43.723 | 67.037 | 55.380 |

Potential is 3.27% worse than the per-dataset best baseline on CIFAR and 14.23% worse on EMNIST. The latter violates the frozen maximum-5%-harm condition.

Every method had positive signed accuracy error on every target condition, so MAE equals mean signed bias in these results: all forecasters systematically overestimated target accuracy. Pair residual reduced but did not solve this underestimation of accumulated damage.

### 5.1 Length-specific comparison

| Dataset | Length | Pair residual | Potential algebra |
|---|---:|---:|---:|
| CIFAR-10 | 3 | **12.068** | 12.648 |
| CIFAR-10 | 4 | **18.684** | 19.233 |
| CIFAR-10 | 5 | **20.971** | 21.521 |
| CIFAR-10 | 6 | **18.797** | 19.362 |
| EMNIST | 3 | **20.922** | 23.831 |
| EMNIST | 4 | **24.766** | 27.670 |
| EMNIST | 5 | **20.740** | 23.658 |
| EMNIST | 6 | **15.313** | 18.233 |

Pair residual wins at every dataset-length cell.

### 5.2 Bootstrap and gates

The frozen 10,000-draw paired hierarchy resampled dataset, length, and unordered word family while retaining available orders. For `pair residual MAE - potential MAE`:

- clustered point: -1.737pp;
- percentile 95% interval: `[-3.277, -0.380]`pp.

Negative values favor pair residual. The interval excludes zero in the direction adverse to Shift Algebra.

| Gate | Result |
|---|---|
| Dynamic range valid on both datasets | FAIL |
| Potential beats best non-algebraic R2 baseline | FAIL |
| Improvement CI lower bound above zero | FAIL |
| No dataset harmed by more than 5% | FAIL |
| At least 15% relative improvement | FAIL |
| Order channel beats `L_p` | **PASS** |
| Overall Phase-1B method gate | **FAIL** |

## 6. Five-behavior and physical-support diagnostics

Normalized five-behavior MAE:

| Method | CIFAR-10 | EMNIST | Dataset-macro |
|---|---:|---:|---:|
| **Pair residual** | **5.887** | **5.966** | **5.927** |
| Pair-feature ridge | 5.957 | 5.994 | 5.976 |
| Potential Shift Algebra | 6.059 | 6.138 | 6.098 |

Unlike Phase 1, no method required physical clipping of an accuracy prediction. Phase 1B therefore removed the earlier out-of-range extrapolation symptom. The failure remains after that repair: the predictions are physically valid but too optimistic.

## 7. Order diagnostic

The independently fixed target list contains five exact reverse pairs per dataset, ten total. Six pass the preregistered measurable-effect thresholds.

| Method | Reversal-difference MAE | Measurable sign accuracy |
|---|---:|---:|
| Signed `L_p` | 3.109pp | 16.7% |
| **Potential Shift Algebra** | **2.208pp** | **100.0%** |

The order component captures useful directionality on this small diagnostic. This does not rescue primary accuracy forecasting: its absolute-magnitude errors are much larger and the order sample is only ten pairs.

## 8. Mechanistic interpretation

Phase 1B separates two effects that Phase 1 could not:

1. **The Phase-1 max-envelope victory was severity dependent.** Once singles were mild and matched, the max envelope became grossly optimistic and ranked near last.
2. **Severity saturation was not the only problem.** The frozen composition laws still failed because higher-order damage accumulated faster than pair-calibrated extrapolation predicted.

The implementation's pair-residual and potential interactions are averaged over constituent pairs. That stabilization was appropriate against the exploding forecasts diagnosed during Phase-1 development, but it makes the residual contribution approximately intensive rather than extensive as word length grows. Phase 1B exhibits the opposite regime: interaction damage compounds with length, so pair-averaged corrections are too small. This interpretation is post-result diagnosis, not a new fitted result.

A useful future model would need a preregistered length/extensivity law learned without length-3--6 target labels—for example from repeat-depth controls, mechanistic retention states, or a separate development benchmark. It must be tested on new conditions or a new model bank, not retrofitted and rescored as Phase 1B.

## 9. Scientific conclusion

Phase 1B yields another informative negative result:

- calibrated singles do not guarantee calibrated long compositions;
- a dominant-shift envelope is not universally strong;
- pair fit does not identify how interaction magnitude scales with word length;
- the current order channel has signal, but magnitude forecasting dominates total error;
- the current potential Shift Algebra is not supported as the primary long-word accuracy forecaster on this bridge.

The next defensible step is not Phase 1C on these same target outcomes. Proceed to Phase 2 only with a revised, prospectively frozen extensivity model developed on separate data/models, while retaining both Phase 1 and Phase 1B as transparent scope boundaries.

## 10. Reproduction and artifacts

Verification:

```bash
python v4/phase1b/verify_phase1b.py
```

Frozen aggregate result:

- `results/PHASE1B_ANALYSIS.json`, SHA-256 `c52507a02e251f8a0938f1a4faaec834be4de1f84d01de213464ac693d38709d`.

Per-dataset CSV/JSON behavior, fingerprint, forecast, and run records are committed in `v4/phase1b/results/`. Data and checkpoint binaries remain ignored; their exact hashes are recorded in the protocol and run artifacts.
