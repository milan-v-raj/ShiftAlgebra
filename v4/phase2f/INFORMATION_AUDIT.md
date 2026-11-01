# Phase 2F Information and Correctness Audit

**Protocol:** `V4-P2F-CIFAR100-FULLVECTOR-CPU-001`
**Outcome:** both preregistered gates passed; no confirmation leak detected

## Chronology

1. `0c6d846` froze metrics, words, images, architectures, development procedure, and gates before Phase-2F behavior outcomes.
2. `4e65720` corrected only the path used to record the test-pickle transport commit; this preceded all training and behavior outcomes.
3. `3e2e1ca` recorded 36 development and 36 confirmation snapshots with checkpoint hashes before any behavior evaluation.
4. Development behavior was evaluated and used exactly as allowed.
5. `af5f5d2` committed the development CSV, selected exponents, normalization scales, five-pair design, twenty random designs, and all ridge coefficients before confirmation behavior was opened.
6. Confirmation was evaluated once. The only post-outcome correction needed to complete the preregistered analysis converted NumPy boolean/rank scalar types to JSON-compatible Python values; predictions, metrics, gates, and intervals were unchanged.
7. Later audit-only changes added stricter forecast/hash/bounds assertions, the declared balanced-accuracy sensitivity, and documentation. They do not fit or alter any forecast.

## Dataset provenance

- image transport commit: `b105b30e79972558f19db69658f34b9d0a9b009b`;
- canonical test-pickle transport commit: `7113a884221682cf835bd6009d3bb0ffef134241`;
- canonical test pickle MD5: `f0ef6b0ae62326f3e7ffdfab6717acfc`;
- 10,000/10,000 transported test PNGs pixel-match the canonical test pickle;
- dataset files and checkpoints are not redistributed.

The training class folders come from the stated official-derived transport but cannot be byte-compared to the canonical 161MB train pickle through the available transport. This is retained as a provenance limitation.

## Information access

Development access was limited to development lineages, image offsets 0--19, calibration conditions, and development words. It determined:

- per-coordinate `q` values: margin 0.8, feature drift 0.8, ECE 0.7, balanced accuracy 0.6, accuracy 0.6, confidence 0.8;
- development standard deviations;
- `g0` and composition-aware ridge coefficients;
- magnitude pair groups;
- twenty deterministic random controls.

Confirmation lineages, image offsets 20--39, and confirmation words were not loaded by method freezing. Forecasting verifies the frozen development behavior hash. During confirmation forecasting, target-word values are copied only into `true_*` output columns; feature construction uses the frozen model, target-word identity/metadata, and permitted clean/single/pair calibration conditions. No target coordinate enters a forecast feature or coefficient fit.

## Complete metric accounting

Every behavior and forecast row includes all six required coordinates. The confirmation artifact contains:

- 6 families;
- 18 independent lineages and 36 snapshots;
- 97 evaluated conditions per snapshot;
- 3,492 behavior rows;
- 60 target words and 2,160 forecast rows;
- 30 reversal groups;
- full, magnitude-five, and twenty random prediction vectors.

Balanced and overall accuracy differ by at most floating-point accumulation because each class contributes equally. The duplication was declared before outcomes. A post-outcome analysis excluding balanced accuracy preserves both positive conclusions.

## Statistical accounting

Primary inference resamples architecture families, lineages within sampled families, and reversal groups. Epoch snapshots stay inside lineages. Development standard deviations normalize heterogeneous coordinate units. Raw coordinate errors are always retained.

All six coordinates improve, all six family mean aggregate improvements are positive, and the primary interval excludes zero. The magnitude selector also passes all four criteria against the expected random strategy.

## Executed verification

The verifier checks disjoint lineages, target words, and image offsets; test provenance; the development-behavior freeze hash; confirmation behavior and forecast hashes; complete finite six-coordinate forecasts; natural bounds; recomputed per-coordinate MAEs; both gate records; and the unique-coordinate sensitivity. It passes 6/6. All five Phase-2F unit tests pass. Rerunning the forecast, primary analysis, and robustness analysis reproduced byte-identical artifact hashes.

## Limitations

- Models have low clean CIFAR-100 accuracy because they are tiny two-epoch CPU models.
- The same synthetic transformation system is used as in CIFAR-10.
- Full-vector coefficients were fit on CIFAR-100 development outcomes; this is not strict no-refit transfer from CIFAR-10.
- Feature centroid drift depends on learned representation scale and is normalized only for aggregate analysis.
- Accuracy and balanced accuracy are redundant under a balanced partition.
- No natural shift or modern public checkpoint is included.
