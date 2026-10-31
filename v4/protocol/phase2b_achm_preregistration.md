# Phase 2B ACHM CPU Pilot — Outcome-Blind Protocol

**Protocol:** `V4-P2B-ACHM-CPU-001`  
**Status:** method and fresh-test design to be frozen before any Phase-2B model is trained  
**Compute:** CPU only, two PyTorch threads  
**Scope:** a fresh CPU pilot, not the roadmap's final 75+ model Phase-2 benchmark

## Scientific question

Does an Adaptive Composition Hazard Mixture (ACHM), fit only on the previously declared Phase-2 development models and words, transfer to new training lineages and completely new length-3--6 words better than the frozen ordered extensive-hazard law?

## Information boundary

Method development may read only rows with roles `calibration` and `method_development` from `v4/phase2/results/development-behavior.csv`. It must not load the opened Phase-2 confirmation file or any Phase-2B outcome. The source development CSV hash is recorded in `PHASE2B_MANIFEST.json`.

The method idea was motivated after the earlier Phase-2 result was known. Consequently, the earlier confirmation is not new evidence for ACHM. Only the fresh Phase-2B bank is a confirmation test.

## Fresh confirmation

- CIFAR-10 training set, new deterministic seeds `42100` onward;
- 15 independent lineages: five lightweight architecture families by three recipes;
- epoch-1 and epoch-2 snapshots, giving 30 models but 15 inferential lineages;
- 1,000 balanced test images from per-class offsets 200--299, disjoint from Phase-2 development and confirmation behavior partitions;
- 37 calibration conditions: identity, six singles, and 30 ordered pairs;
- 64 fresh words: 16 each at lengths 3, 4, 5, and 6;
- every fresh word is absent from the 167-word Phase-2 protocol and is paired with its reversal.

## Frozen experts

Using clean-to-chance hazard `h(a)=-log((a-c)/(a0-c))`, clipped exactly as in Phase 2:

1. dominant shift: maximum primitive hazard;
2. independent retention: sum of primitive hazards;
3. symmetric pair compounding;
4. ordered pair compounding.

Pair compounding keeps the already developed exponent `q=0.6`. No Phase-2B outcome may alter it.

## Adaptive gate

A four-way softmax gate forms a convex mixture in hazard space. Inputs are available before the composed target: clean/model metadata, length, primitive hazard summaries, measured pair residual summaries, pair asymmetry, calibration coverage, and expert disagreement. The output remains bounded after converting hazard back to accuracy.

The gate is deliberately small. Its L2 coefficient is selected from `{0, 1e-4, 1e-3, 1e-2}` by five-fold leave-lineage-group development validation. The selected gate is then refit to the 15 Phase-2 development lineages and frozen.

## Active pair calibration

Secondary budget variants measure 0, 5, 10, or 15 unordered pair groups (0, 10, 20, or 30 directed pair conditions). Pair groups are greedily selected from development-only interaction magnitude and word frequency with a primitive-coverage bonus. Unmeasured residuals are set to zero, a conservative independence assumption. Each budget receives a separately cross-validated gate before fresh confirmation.

## Primary comparator and gate

The frozen primary comparator is the ordered hazard expert, because it was the Phase-2 method selected before the original confirmation. ACHM passes only if all four conditions hold on fresh confirmation:

1. lower accuracy MAE than ordered hazard;
2. 10,000-draw bootstrap CI for the paired improvement, clustered by reverse-word group and training lineage, excludes zero;
3. higher source/family/recipe-controlled model-rank Spearman;
4. top-1 model-selection regret is no worse.

Comparisons with symmetric hazard, the prior `g0`/`g1`, active budgets, uncertainty, and order diagnostics are secondary and must all be reported.

## Failure policy

No rerun, hyperparameter rescue, word replacement, or seed replacement is allowed after fresh outcomes are opened. A failed gate is retained and reported. ACHM does not establish an order claim unless it also beats the symmetric reversal baseline; this is diagnostic rather than a rescue criterion.
