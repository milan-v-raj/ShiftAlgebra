# Phase 3A — Realistic-Corruption Composition Confirmation

Phase 3A prospectively tests whether the primary order-blind symmetric composition method transfers to six deterministic common-corruption categories on CIFAR-100.

The operators are clean-room **ImageNet-C-style categories**, not official ImageNet-C implementations and not severity-equivalent to that benchmark.

## Frozen design

- Corruptions: brightness, Gaussian noise, JPEG-style block-DCT quantization, motion blur, pixelation, and shot-noise approximation.
- Development: reconstructed Phase-2H models, CIFAR-100 class offsets 80--89, and 60 development targets/30 reversal groups.
- Confirmation: 18 fresh lineages/36 snapshots, offsets 90--99, and 60 disjoint targets/30 reversal groups.
- Calibration per model: identity, six singles, and all 30 directed distinct-primitive pairs.
- Behavior vector: margin, feature drift, ECE, balanced accuracy, overall accuracy, and mean confidence.
- Primary: fitted symmetric order-blind method.
- Comparators: metadata-only ridge, independent retention, envelope, and unfitted symmetric-pair formulas.

See [`../protocol/phase3a_realistic_corruptions_preregistration.md`](../protocol/phase3a_realistic_corruptions_preregistration.md) for gates and interpretation boundaries.

## Result

**The all-required gate failed 6/7.** The fitted symmetric method improved all six coordinate MAEs, reduced macro normalized MAE from 0.576501 to 0.471656, produced a hierarchical improvement CI of [0.050407, 0.154644], improved controlled rank from 0.191236 to 0.613512, and had positive mean improvement in all six families. However, accuracy top-1 selection regret worsened from 0.1083pp to 0.1367pp, so the strict no-worse criterion failed. The unfavorable decision is retained without tuning.

See [`PHASE3A_FINAL_REPORT.md`](PHASE3A_FINAL_REPORT.md) and [`INFORMATION_AUDIT.md`](INFORMATION_AUDIT.md).

## Prospective execution order

Run from the repository root with the CPU environment. Each numbered outcome boundary is committed before the next one is opened.

```bash
# 1. Freeze manifest before any Phase-3A behavior.
.venv/bin/python -m v4.phase3a.manifest

# 2. Reconstruct Phase-2H checkpoints if ignored artifacts are absent.
.venv/bin/python -m v4.phase2h.train_bank

# 3. Open development behavior only, then fit and freeze.
.venv/bin/python -m v4.phase3a.evaluate_bank --split development
.venv/bin/python -m v4.phase3a.freeze_methods

# 4. Train/hash fresh confirmation checkpoints; commit the pre-outcome ledger.
.venv/bin/python -m v4.phase3a.train_bank

# 5. Open confirmation behavior exactly once, forecast, analyze, and verify.
.venv/bin/python -m v4.phase3a.evaluate_bank --split confirmation
.venv/bin/python -m v4.phase3a.forecast
.venv/bin/python -m v4.phase3a.analyze
.venv/bin/python -m v4.phase3a.robustness
.venv/bin/python -m v4.phase3a.verify

# Tests
.venv/bin/python -m unittest v4.phase3a.test_phase3a -v
```

Checkpoints and dataset caches are ignored binary artifacts. Compact manifests, hashes, CSV behavior tables, forecasts, analyses, and reports are retained in Git.
