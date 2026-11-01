# Phase 2F — CIFAR-100 Full-Vector Benchmark

**Protocol:** `V4-P2F-CIFAR100-FULLVECTOR-CPU-001`  
**Status:** protocol and banks frozen; no behavior outcome opened

Phase 2F moves to canonical CIFAR-100 and restores the complete six-coordinate behavior output: margin, feature drift, ECE, balanced accuracy, overall accuracy, and confidence. It uses six architecture families, 36 independent lineages across development/confirmation, disjoint images and words, development-only method fitting, and one sealed confirmation.

Balanced and overall accuracy are retained separately for historical compatibility even though balanced CIFAR-100 partitions make them numerically identical.
