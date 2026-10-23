# Phase 1 Final Report — Independent Length-Six Bridge

**Protocol:** `V4-P1-MASON-001`
**Manifest:** 167 Mason words per dataset
**Executable freeze:** `b761987bc691db191bd2617ad05382f67a68ba18`
**Freeze record commit:** `aa9fbba`
**Status:** one-shot confirmation complete; Phase-1 method gate failed

## 1. Confirmatory design

The confirmation used images disjoint from all development measurements:

| Dataset | Frozen classifier | Confirmation sample | Clean accuracy |
|---|---|---:|---:|
| CIFAR-10 | SmallCNN width 24, seed 1301 | 900/class, 9,000 total | 77.84% |
| EMNIST Balanced | SmallCNN width 16, seed 1301 | 360/class, 16,920 total | 87.39% |

For each dataset, all six primitive behaviors and all 30 ordered-pair behaviors were calibration inputs. Predictions for the 130 length-3--6 words were generated without using images or labels from those target words during fitting. Target labels were opened only to score the frozen predictions.

The checkpoint, transformations, methods, metric definitions, physical clipping rule, partitions, and commands were committed in `FINAL_FREEZE.md` before confirmation.

## 2. Confirmatory accuracy forecasting

Accuracy MAE in percentage points; lower is better:

| R2 method | CIFAR-10 | EMNIST | Dataset-macro |
|---|---:|---:|---:|
| **Max-absolute primitive envelope** | **4.664** | **0.457** | **2.561** |
| Pair-residual extrapolation | 6.652 | 2.262 | 4.457 |
| Pair-feature ridge | 7.015 | 2.270 | 4.642 |
| Potential Shift Algebra | 7.081 | 2.267 | 4.674 |
| Learned signed `L_p` | 7.214 | 2.270 | 4.742 |
| Additive, physically clipped | 8.973 | 2.270 | 5.621 |
| Mean primitive | 14.714 | 2.061 | 8.388 |

Potential Shift Algebra is 82.5% worse in macro accuracy MAE than the max-envelope baseline. The preregistered Phase-1 gate therefore fails.

The result replicates the development ranking. Confirmatory-minus-development MAE changes were small: +0.094pp for max and -0.101pp for potential on CIFAR-10; +0.106pp for max and +0.074pp for potential on EMNIST.

## 3. Length breakdown

### CIFAR-10 accuracy MAE

| Length | Max envelope | Pair residual | Potential algebra |
|---:|---:|---:|---:|
| 3 | 5.874 | **4.046** | 5.863 |
| 4 | **4.978** | 7.266 | 7.184 |
| 5 | **4.173** | 8.403 | 7.921 |
| 6 | **3.229** | 7.760 | 7.760 |

Pair interactions help on CIFAR triples, but their extrapolation worsens as words lengthen. The envelope improves with length because targets collapse further toward chance.

### EMNIST accuracy MAE

| Length | Max envelope | Pair residual | Potential algebra |
|---:|---:|---:|---:|
| 3 | **0.477** | 2.323 | 2.340 |
| 4 | **0.464** | 2.295 | 2.295 |
| 5 | **0.448** | 2.198 | 2.198 |
| 6 | **0.432** | 2.210 | 2.210 |

## 4. Confirmed saturation diagnosis

### Mean target accuracy

| Length | CIFAR-10 | EMNIST |
|---:|---:|---:|
| 1 | 23.69% | 4.33% |
| 2 | 13.23% | 2.55% |
| 3 | 10.15% | 2.35% |
| 4 | 9.18% | 2.29% |
| 5 | 8.40% | 2.20% |
| 6 | 7.76% | 2.21% |

Chance is 10% on CIFAR-10 and 2.13% on EMNIST. Every EMNIST length-3--6 condition is within two percentage points of chance. On CIFAR-10, 50%, 70%, 60%, and 46.7% of conditions are within two points of chance at lengths 3, 4, 5, and 6 respectively.

Available reverse words show little surviving accuracy-order signal at length three or greater: mean absolute reversal differences are 1.77pp on CIFAR-10 and 0.25pp on EMNIST.

This independently confirms the development diagnosis: under the published fixed severities, one destructive primitive usually dominates and longer words reach an accuracy floor. A dominant-primitive envelope is therefore the correct strong baseline.

## 5. Multidimensional behavior

The max envelope also has the best normalized five-behavior MAE:

| Method | CIFAR-10 | EMNIST | Dataset-macro |
|---|---:|---:|---:|
| **Max envelope** | **0.900** | **0.696** | **0.798** |
| Pair residual | 0.973 | 0.793 | 0.883 |
| Potential algebra | 1.048 | 0.869 | 0.958 |

Thus the failure is not solely an artifact of accuracy clipping. Interaction models also fail to beat the envelope on the broader behavior vector.

Physical clipping remains diagnostically important: approximately 48.5% of CIFAR and 97.7% of EMNIST potential-algebra target predictions required accuracy clipping, versus 0% for the max envelope. This is evidence that pair-calibrated interaction extrapolation is out of support at these lengths.

## 6. Scientific conclusion

Phase 1 provides a useful negative result:

1. Shift Algebra does not beat a dominant-primitive baseline on this independent protocol.
2. Pair fit alone does not guarantee stable extrapolation through length six.
3. The Mason fixed-severity suite is valuable as a saturation and support stress test, but it is not an informative primary benchmark for fine-grained accuracy forecasting once most targets are at chance.
4. Future experiments must include maximum/dominant-shift baselines and disclose physically invalid predictions.
5. The next benchmark should preserve dynamic range through severity calibration while retaining the independently fixed word list and reversal structure.

No tuning or same-protocol rescue will be performed on the confirmatory partitions.

## 7. Decision

**Phase-1 method gate: FAIL.**

Proceed to a separately preregistered Phase 1B severity-calibrated extension, not a modification of this result. The external Mason result remains in the paper as a transparent falsification test and scope boundary.

## 8. Verification

```bash
python3 v4/phase1/verify_phase1.py
```

The verifier checks all 167 conditions per run, six 301-dimensional primitive fingerprints, the 30-pair/130-target access contract, and all Phase-1 unit tests. Raw aggregate CSV/JSON artifacts are committed under `v4/phase1/results/`; datasets and checkpoints are not.
