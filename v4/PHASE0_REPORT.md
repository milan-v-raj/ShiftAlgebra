# Phase 0 Progress Report: Claim and Code Audit

**Date:** 2026-10-23
**Status:** in progress; code/audit foundation complete, external freezes pending
**Experimental result status:** no new benchmark result has been produced

## Completed in this increment

### 1. Isolated development workspace

Created `v4/` so prospective work cannot silently alter V1--V3 implementations or committed result artifacts.

### 2. Exact numerical algebra core

Implemented `v4/src/algebra.py` with:

- a signed-power transform and its actual exact numerical inverse;
- enforced `p >= 1`;
- an associative streaming sufficient state;
- explicit chronological convention `T_Xm ... T_X1`;
- independent closed-form all-pairs extraction;
- literal 3x3 block-matrix extraction;
- identity, reversal, and repeated-primitive handling.

The streaming, closed-form, and literal-matrix paths agree in property tests through word length six.

### 3. Prospective V4 model

Implemented `MBCore` and `LpOnly` in `v4/src/models.py`:

- one decode after full state composition;
- `p = 1 + softplus(raw_p)`;
- small non-zero residual initialization;
- behavior base plus learned residual.

No claim is made yet about benchmark performance.

### 4. Architecture-independent fingerprint

Implemented `v4/src/fingerprints.py`:

- deterministic fixed-size random-projection sketches at four normalized depth slots;
- paired clean/shifted representation displacement;
- fixed-size confidence, margin, entropy, and probability-displacement summaries;
- fixed-size pixel and high-frequency summaries;
- output dimension independent of feature width, layer count, class count, and channel count.

### 5. Order-model limitation control

Implemented `v4/src/order_models.py`:

- historical transitive potential effect;
- low-rank skew-bilinear alternative;
- flexible antisymmetrized pair network;
- explicit rock-paper-scissors counterexample.

The counterexample confirms that a scalar potential cannot represent a three-cycle, while a skew-bilinear model can.

### 6. Comparable evaluation metrics

Implemented `v4/analysis/metrics.py`:

- accuracy MAE/RMSE/median/worst-decile error in percentage points;
- separate model-given-environment and environment-given-model Spearman axes;
- partial Spearman with controls;
- top-k model-selection regret and near-oracle selection rate;
- macro family MAE;
- training-only IQR normalization for multivariate behavior;
- interval coverage and width.

### 7. Source/model baseline

Implemented `v4/baselines/source_family.py`, an R0 ridge baseline using only:

- source behavior;
- model family;
- training recipe.

Its information contract explicitly forbids composed target images and labels.

### 8. Benchmark schema and exact Mason manifest

Implemented a versioned R0--R4 schema and generated the pinned Mason protocol:

- upstream commit: `fc62a92ccd11f64eec26c7a019e9e0957fd9bd33`;
- seed: `2778`;
- 167 unique words per dataset;
- 501 dataset-conditions across EMNIST, CIFAR-10, and FaceScrub;
- length counts: identity 1, singles 6, pairs 30, triples 40, and 30 each at lengths 4, 5, and 6;
- manifest SHA-256: `d2be1ca5c899659482204630d1f2b755587b71ee994372a7b716e5c5ad591264`.

### 9. Claim audit artifacts

Created:

- `v4/literature/CLAIM_MATRIX.csv` with 16 directly relevant works;
- `v4/KNOWN_LIMITATIONS.md`;
- `v4/protocol/phase1_preregistration.md`.

The preregistration remains a draft because checkpoint, license, container, and hyperparameter fields are intentionally not invented.

## Verification

Command:

```bash
.venv/bin/python v4/verify_phase0.py
```

Result:

- 38 tests passed;
- exact Mason count passed;
- manifest round-trip/hash passed;
- claim-matrix schema passed;
- R2 information-access guards passed;
- algebraic inverse/identity/associativity/reversal/equivalence passed;
- fingerprint dimensional invariance passed;
- cyclic-order limitation test passed.

## Remaining before Phase 0 can close

1. Run the literature audit through external scholarly indexes, not only public repositories.
2. Resolve Phase-1 dataset and checkpoint licenses, especially FaceScrub.
3. Pin checkpoint URLs/hashes and preprocessing.
4. Freeze baseline hyperparameter grids and statistical implementation.
5. Run the five-model CIFAR development-only compute/variance pilot.
6. Add container/environment lock and record its digest.

Phase 1 benchmark inference must not begin until items 2--4 are complete and the preregistration receives a frozen commit ID.
