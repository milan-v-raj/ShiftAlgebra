# Phase 2G — Strict No-Refit CIFAR-100 Architecture Transfer

**Protocol:** `V4-P2G-CIFAR100-NOREFIT-ARCH-001`

**Status:** prospective protocol; confirmation behavior unopened

Phase 2G transfers the complete Phase-2F six-coordinate method byte-for-byte to six unseen lightweight architecture families. It changes no exponent, coefficient, scale, feature definition, or full/magnitude/random pair design. New family indicators are all zero under the frozen Phase-2F feature schema.

The bank has 18 independent lineages and 36 snapshots spanning residual, depthwise-separable, ConvNeXt-style, grouped/shuffle, dilated-pyramid, and gated-convolution models. It uses fresh CIFAR-100 test offsets 40--59 and sixty fresh target words while retaining all six metrics: margin, feature drift, ECE, balanced accuracy, accuracy, and mean confidence.

This is a strict architecture-transfer experiment within CIFAR-100, not an independent-dataset test.

```bash
python3 v4/phase2g/verify_phase2g.py
python3 -m unittest discover -s v4/phase2g/tests -v
```
