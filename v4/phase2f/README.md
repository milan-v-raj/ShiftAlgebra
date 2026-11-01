# Phase 2F — CIFAR-100 Full-Vector Benchmark

**Protocol:** `V4-P2F-CIFAR100-FULLVECTOR-CPU-001`  
**Status:** complete; **full-vector gate passed (5/5)** and **magnitude-selector gate passed (4/4)**

Phase 2F moves to canonical CIFAR-100 and restores the complete six-coordinate behavior output: margin, feature drift, ECE, balanced accuracy, overall accuracy, and confidence. It uses six architecture families, 36 independent lineages across development/confirmation, disjoint images and words, development-only method fitting, and one sealed confirmation.

Balanced and overall accuracy are retained separately for historical compatibility even though balanced CIFAR-100 partitions make them numerically identical.

On sealed confirmation, full composition reduced macro normalized MAE from **0.554 to 0.254**, improved all six coordinates, raised macro controlled rank from **0.302 to 0.811**, and passed its hierarchical gate. Magnitude five-pair calibration reached **0.271** versus **0.338** expected-random MAE and also passed. See `PHASE2F_FINAL_REPORT.md` and `INFORMATION_AUDIT.md`.

```bash
python3 v4/phase2f/verify_phase2f.py
python3 -m v4.phase2f.analyze
python3 -m v4.phase2f.robustness
```
