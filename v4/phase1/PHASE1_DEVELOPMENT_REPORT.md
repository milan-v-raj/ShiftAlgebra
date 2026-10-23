# Phase 1 Development Report

**Protocol:** `V4-P1-MASON-001`
**Status:** development-only; non-overlapping confirmatory images remain sealed
**Date:** 2026-10-23

## 1. What was run

The exact 167-word Mason composition list was evaluated on independently trained frozen classifiers:

| Dataset | Model | Development samples | Clean accuracy | Calibration | Targets |
|---|---|---:|---:|---:|---:|
| CIFAR-10 | SmallCNN width 24, seed 1301 | 100/class = 1,000 | 76.10% | 6 singles + all 30 ordered pairs | 130 words, lengths 3--6 |
| EMNIST Balanced | SmallCNN width 16, seed 1301 | 40/class = 1,880 | 88.40% | 6 singles + all 30 ordered pairs | 130 words, lengths 3--6 |

Target-word images and labels were used only to score frozen predictions. Forecast fitting consumed primitive and pair behavior labels only.

The implementation is protocol-compatible but clean-room; it does not claim bitwise equality with the unlicensed upstream corruption code.

## 2. Primary development result

Accuracy MAE in percentage points after applying the standard physical constraint that predicted accuracy lies in `[0,1]`:

| R2 method | CIFAR-10 | EMNIST |
|---|---:|---:|
| **Max-absolute primitive envelope** | **4.570** | **0.351** |
| Pair-residual extrapolation | 6.867 | 2.192 |
| Potential Shift Algebra | 7.181 | 2.193 |
| Pair-feature ridge | 7.148 | 2.195 |
| Learned signed `L_p` | 7.317 | 2.195 |
| Additive, physically clipped | 9.053 | 2.195 |
| Mean primitive | 14.764 | 2.176 |

The proposed potential algebra does **not** win this development bridge. The dominant-primitive baseline wins decisively on both datasets.

## 3. Why: the published default corruptions saturate accuracy

### CIFAR-10

| Word length | Mean accuracy | Conditions within ±2pp of chance (10%) |
|---:|---:|---:|
| 1 | 23.82% | 16.7% |
| 2 | 13.10% | 50.0% |
| 3 | 10.16% | 55.0% |
| 4 | 9.21% | 73.3% |
| 5 | 8.48% | 63.3% |
| 6 | 7.99% | 50.0% |

### EMNIST Balanced

| Word length | Mean accuracy | Conditions within ±2pp of chance (2.13%) |
|---:|---:|---:|
| 1 | 4.37% | 66.7% |
| 2 | 2.39% | 93.3% |
| 3 | 2.21% | 100% |
| 4 | 2.23% | 100% |
| 5 | 2.16% | 100% |
| 6 | 2.18% | 100% |

Once one severe primitive has reduced performance to chance, a maximum/dominant-shift envelope is almost optimal. There is little remaining accuracy signal for composition algebra to forecast.

The same saturation suppresses order effects. Among reverse words available in the 167-word set, the mean absolute accuracy difference is approximately 1.72pp on CIFAR-10 and 0.20pp on EMNIST for length three or greater.

## 4. Important implementation finding

The first development implementation jointly learned `p` and interaction parameters. Pair interactions compensated for a nearly additive base, causing catastrophic length extrapolation. This was fixed without using final images by enforcing the planned two-stage procedure:

1. fit `p` on pair calibration only;
2. freeze `p`;
3. fit symmetric/order residuals on pairs only.

Accuracy predictions are also physically clipped to valid absolute accuracy `[0,1]` for every method, and the clipping fraction is disclosed. This is now frozen before confirmatory evaluation.

## 5. Interpretation

This result does **not** show that composition forecasting is impossible. It shows that Mason's fixed-severity protocol is a poor primary accuracy-forecast benchmark for these clean classifiers because it intentionally applies strong semantic/geometric corruptions and reaches chance rapidly.

It remains valuable as:

- an independent length-six stress test;
- a falsification test for uncontrolled long-word extrapolation;
- evidence that a dominant-shift baseline is mandatory;
- motivation for the planned severity-calibrated benchmark where target accuracies do not collapse.

## 6. Decision under the development plan

The Phase-1 development gate is currently **not met**: Shift Algebra does not beat the best target-free baseline.

Required response:

1. run the already-defined non-overlapping confirmatory image split once with the now-frozen methods;
2. report the result even if it remains negative;
3. retain Mason as an external saturation stress test, not the sole headline benchmark;
4. proceed afterward to a preregistered severity-calibrated extension where primitive degradation is matched and accuracy retains dynamic range;
5. do not tune again on the Mason confirmatory split.

## 7. Reproduction

```bash
python3 v4/phase1/verify_phase1.py
```

Raw compact artifacts are in `v4/phase1/results/`. Large datasets and classifier checkpoints remain ignored.
