# Phase 2 Development Report — Before Confirmation

**Protocol:** `V4-P2-CIFAR-BANK-PILOT-001`
**Status:** development complete; confirmation models and outcomes unopened

## Development bank

- 15 independently trained lineages;
- two epoch snapshots per lineage, 30 model checkpoints;
- five architecture families;
- three training recipes;
- 500 disjoint CIFAR-10 test images per condition;
- 37 clean/single/pair calibration conditions;
- 40 declared method-development words, ten per length 3--6.

## Frozen extensive scaling

The fixed q-grid search selected `q=0.6`. Development ordered-hazard MAE by nearby q:

| q | Accuracy MAE |
|---:|---:|
| 0.4 | 1.745pp |
| 0.5 | 1.557pp |
| **0.6** | **1.513pp** |
| 0.7 | 1.637pp |
| 0.8 | 1.918pp |

This lies between an averaged pair correction (`q=0`) and a fully extensive pair sum (`q=1`).

## Development-only ranking of accuracy forecasts

These values are training/development diagnostics and are not Phase-2 results:

| Method | Development MAE |
|---|---:|
| `g1` source/word + composition | **0.951pp** |
| Symmetric hazard, q=0.6 | 1.509pp |
| Ordered hazard, q=0.6 | 1.513pp |
| `g0` source/word metadata | 1.533pp |
| Product retention | 4.686pp |
| Accuracy pair residual | 7.302pp |
| Maximum primitive damage | 12.411pp |
| Primitive mean | 16.524pp |
| DoC (R3) | 21.522pp |
| ATC (R3) | 24.111pp |

The `g0` and `g1` ridge coefficients, feature scales, q-grid, and selected q are frozen in `FROZEN_METHOD.json`. No result here may be used for another manual method change. The next action is one-shot training/evaluation of the independent confirmation lineages followed by the already committed analysis.
