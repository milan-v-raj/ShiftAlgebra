# Phase 1 One-Shot Confirmation Freeze

**Freeze date:** 2026-10-23
**Protocol:** `V4-P1-MASON-001`
**Executable code commit:** `b761987bc691db191bd2617ad05382f67a68ba18`
**Manifest SHA-256:** `d2be1ca5c899659482204630d1f2b755587b71ee994372a7b716e5c5ad591264`

## Frozen classifiers

| Dataset | Model | Checkpoint SHA-256 |
|---|---|---|
| CIFAR-10 | `smallcnn-w24-s1301`, 8 epochs | `9f5815dd3ab43edd5d09636e8d2688fa9ab8aaae346143ec6d5184bd99944848` |
| EMNIST | `smallcnn-w16-s1301`, 5 epochs | `05c49cd3678ee537accac6380d33002e4d8f512e5eb93093b18f0701f815ca53` |

Checkpoint binaries remain ignored; hashes bind the exact frozen models.

## Development artifacts used for method decisions

| Artifact | SHA-256 |
|---|---|
| CIFAR behavior | `3fdf2f5c97f5b8ed63d679e2957349dda4724c1fd68ed788bc0b7d78855ea6e1` |
| CIFAR forecast | `ccbf575ac5a12f2aaf72ad7dd6e20ed03e8a720be570dd212dc72412d45e0e1e` |
| EMNIST behavior | `afb3c4bd6bfde8fca1b73a68cfc65b51b08856de66d58bfa6248d4ff8ba74476` |
| EMNIST forecast | `1b13b6e8801e478008bf07a597a0bc9a6ea28744814eefecdff8ead79139c25d` |

## Sealed image partitions

- CIFAR-10 confirmation: 900 images/class beginning at class-local offset 100; 9,000 total.
- EMNIST confirmation: 360 images/class beginning at class-local offset 40; 16,920 total.
- These partitions are disjoint from development by construction and by tested stable indices.

## Frozen methods

- additive with physical accuracy clipping;
- primitive mean;
- maximum-absolute primitive envelope;
- pair-fitted signed `L_p`;
- pair-feature ridge;
- arbitrary directed-pair residual extrapolation;
- five-seed potential Shift Algebra ensemble with pair-fitted `p` frozen before interaction fitting.

All methods use the same six primitive behavior vectors and all 30 pair labels measured on the confirmation images. No length-3--6 label enters fitting.

## Frozen primary endpoint

Macro accuracy MAE in percentage points over the 130 words of lengths three through six. Per-length MAE, normalized five-behavior MAE, RMSE, worst-decile MAE, and physical-clipping fraction are secondary.

The development winner was the maximum-absolute primitive envelope. Confirmation must report that baseline unchanged. The Phase-1 gate passes only if potential Shift Algebra beats the best non-algebraic R2 baseline; otherwise the failure and saturation diagnosis stand.

## One-shot commands

```bash
python3 -m v4.phase1.pipeline \
  --dataset cifar10 --epochs 8 --width 24 --seed 1301 \
  --dev-per-class 900 --dev-offset 100 --split-name confirm \
  --train-batch-size 256 --eval-batch-size 256

python3 -m v4.phase1.pipeline \
  --dataset emnist --epochs 5 --width 16 --seed 1301 \
  --dev-per-class 360 --dev-offset 40 --split-name confirm \
  --train-batch-size 512 --eval-batch-size 512
```

No method, transformation, metric, checkpoint, or exclusion may change after this freeze. A runtime-invalid run may be diagnosed, but must not be silently overwritten under this protocol ID.
