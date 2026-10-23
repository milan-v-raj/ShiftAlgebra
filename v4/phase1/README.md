# Phase 1 — Independent Mason Protocol Bridge

**Status:** one-shot confirmation complete; method gate failed
**Protocol:** `V4-P1-MASON-001`

See [`PHASE1_FINAL_REPORT.md`](PHASE1_FINAL_REPORT.md) for the frozen result and saturation diagnosis.

## Aim

Train only on the six primitive behaviors and all 30 ordered pairs, then forecast the 130 independently sampled words of lengths three through six without using images from those target compositions at prediction time.

## Current implementation

- clean-room CIFAR-10 and EMNIST Balanced loaders;
- canonical CIFAR batch-hash verification;
- Mason-compatible class-stable training/validation splits;
- vectorized fixed-severity Contrast, Gaussian Blur, Impulse Noise, Invert, Rotate90, and Swirl;
- batching-independent stochastic noise fields;
- frozen small-CNN training;
- behavior measurement: accuracy, margin, NLL, ECE, and representation drift;
- architecture-independent fingerprints for all six primitives;
- R2 forecasters: additive, mean, max, learned `L_p`, pair-feature ridge, pair-residual, and potential Shift Algebra;
- separate development subset so the remaining test images can stay sealed.

## Development commands

```bash
# CIFAR-10: 100 test images/class (1,000 total)
python3 -m v4.phase1.pipeline --dataset cifar10 --dev-per-class 100

# EMNIST: 40 test images/class (1,880 total)
python3 -m v4.phase1.pipeline --dataset emnist --dev-per-class 40
```

Large datasets and checkpoints remain under ignored `data/` and `checkpoints_v4/`. Compact CSV/JSON results are written to `v4/phase1/results/`.

## Interpretation

These development results may identify implementation errors and select a frozen method configuration. They are not the final Phase-1 claim. Final evaluation must use the non-overlapping remainder of each test set after configuration and analysis code are committed.
