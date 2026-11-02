# Phase 3A Preregistration — Realistic-Corruption Composition Bridge

**Protocol:** `V4-P3A-CIFAR100-REALISTIC-CORRUPTIONS-001`

## Question

Does the now-primary order-blind composition-aware forecasting methodology transfer from the original geometric/artifact primitive set to a new set of realistic common-corruption categories?

This phase uses deterministic clean-room operators inspired by ImageNet-C categories. It is not an official ImageNet-C evaluation and makes no bitwise or severity-equivalence claim.

## New corruption vocabulary

Six primitives are frozen before behavior outcomes:

1. brightness offset;
2. Gaussian noise;
3. JPEG-style block-DCT quantization;
4. diagonal motion blur;
5. pixelation;
6. signal-dependent shot-noise approximation.

Parameters and deterministic primitive-specific random-field seeds are stored in `PHASE3A_MANIFEST.json`. Random fields are keyed by canonical sample ID, so results are independent of batching and use the same primitive field under word reversal.

## Development and confirmation boundary

Development:

- the already-trained Phase-2H bank, used only as a model source;
- CIFAR-100 test offsets 80--89, ten images/class;
- sixty development words in thirty reversal groups;
- development calibration and target outcomes may select metric-specific exponents, scales, and ridge coefficients.

Confirmation:

- 18 newly seeded lineages and 36 snapshots across six families and three recipes;
- CIFAR-100 offsets 90--99, disjoint from all prior V4 partitions;
- sixty disjoint confirmation words in thirty reversal groups;
- confirmation behavior remains sealed until the development behavior, fitted method, and confirmation checkpoint hashes are committed.

Every behavior and forecast contains margin, feature drift, ECE, balanced accuracy, overall accuracy, and mean confidence.

## Methods

The primary method is order-blind symmetric composition. It receives clean behavior, six singles, and all thirty directed pair measurements. Directed pair residuals are averaged before feature construction. The fitted model uses metadata plus symmetric-pair, independent-retention, and envelope features.

Comparators are:

- metadata-only ridge `g0`;
- independent-retention formula;
- envelope formula;
- unfitted symmetric-pair formula.

All methods are target-free at confirmation inference. No composed target image or outcome enters prediction.

## Primary all-required gate

All seven criteria must pass:

1. symmetric fitted macro normalized MAE is below metadata-only `g0`;
2. the architecture-family/lineage/reversal hierarchical 95% interval for normalized improvement excludes zero;
3. at least four of six raw coordinate MAEs improve;
4. symmetric fitted macro normalized MAE is below both independent and envelope formulas;
5. macro controlled model-rank Spearman improves over `g0`;
6. accuracy top-1 selection regret is no worse than `g0`;
7. at least five of six architecture-family mean improvements are positive.

Ten thousand deterministic bootstrap draws resample architecture family, lineage within family, and reversal group as a crossed axis. Epoch snapshots remain clustered within lineage.

## Outcome handling

The gate is retained exactly whether positive or negative. Balanced and overall accuracy remain separately reported; a labeled post-outcome sensitivity may remove balanced accuracy but cannot replace the primary decision.

## Interpretation boundary

Passing supports transfer of composition-aware forecasting to these common-corruption categories on CIFAR-100. It does not establish official ImageNet-C performance, ImageNet transfer, natural-shift transfer, modern-checkpoint validity, or superiority to target-access ODP methods. Failing narrows the positive evidence to the earlier transformation vocabulary.
