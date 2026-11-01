# Phase 2H Information and Correctness Audit

**Protocol:** `V4-P2H-CIFAR100-ORDER-ABLATION-001`

**Outcome:** order-awareness gate failed 5/6; no confirmation leak detected

## Prospective chronology

1. Commit `ba4b36d` froze the question, nested methods, coefficients, method hash, architectures, seeds, images, words, inference, and all-required gate before Phase-2H model training or behavior outcomes.
2. The frozen method was fitted only from committed Phase-2F development behavior. Phase-2F confirmation and Phase-2G outcomes motivated the question but were not loaded by fitting code.
3. Eighteen new lineages and 36 snapshots were trained. Commit `a2ec89b` recorded all checkpoint hashes while Phase-2H behavior remained unopened.
4. Confirmation behavior was evaluated exactly once on all 97 frozen conditions.
5. Forecasting and primary analysis used code already committed in `ba4b36d`.
6. Post-outcome changes were limited to the declared unique-coordinate sensitivity, stricter verification assertions, reports, and synthesis documents. No model, prediction, primary statistic, or gate was changed.

## Development boundary

Both methods were fitted from:

- `v4/phase2f/results/development-behavior.csv`;
- SHA-256 `8c3289ac47c01267751a2f3d8104d1940f04e20b8fa0cec4dea0c2baef7b90a7`.

The frozen Phase-2H method SHA-256 is:

`9c4eef64aa2e1812306c6df3ba7c90e5f8a7bbf02f97eeece6603509d31e375e`.

The freeze script names only the Phase-2F development artifact as a behavior source. It cannot load Phase-2F confirmation, Phase-2G, or Phase-2H outcomes.

## Equal-budget nesting

Both composition methods receive exactly the same identity, six single-primitive, and thirty directed pair measurements. Both include identical metadata, clean behavior, primitive counts, symmetric composition, independent retention, and envelope features.

The symmetric method averages directed residuals before feature construction. The order-aware feature vector is the complete symmetric vector plus exactly one `ordered - symmetric` value. Every metric uses the same ridge penalty. Tests verify that the methods are nested by one feature and that symmetric features are identical under complete word reversal.

This is an incremental order-information test. The order-aware method has one additional fitted coefficient per output coordinate; that extra degree of freedom is part of the preregistered hypothesis and is evaluated only out of sample.

## Fresh confirmation boundary

- New seeds 68100--68602 define 18 lineages absent from prior banks.
- CIFAR-100 offsets 60--79 are disjoint from Phase-2F offsets 0--39 and Phase-2G offsets 40--59.
- Sixty target words are absent from all prior frozen banks.
- Targets form thirty exact reversal pairs at lengths four through six.
- Six architecture families and three recipes are represented.
- Both epoch snapshots remain clustered inside each lineage.

Canonical test provenance remains MD5 `f0ef6b0ae62326f3e7ffdfab6717acfc`, with all 10,000 transported test images pixel-matched.

## Forecast information path

Forecasting loads the frozen method by hash. For each held-out target word, it constructs features from permitted calibration conditions and model/word metadata. Target measurements are copied only to `true_*` columns after prediction inputs are constructed. Forecasting contains no optimizer, fit, selector, or coefficient update.

The committed forecast contains three complete prediction vectors: metadata-only `g0`, symmetric composition, and order-aware composition.

## Complete metric accounting

Every behavior and forecast row retains:

1. margin;
2. feature drift;
3. ECE;
4. balanced accuracy;
5. accuracy;
6. mean confidence.

Artifacts contain 6 families, 18 lineages, 36 snapshots, 97 behavior conditions, 3,492 behavior rows, 60 target words, 30 reversal groups, and 2,160 forecast rows.

Order awareness improves four raw coordinate MAEs and worsens two. Symmetric composition remains better in macro controlled rank. No metric is omitted from the preregistered decision.

## Statistical accounting

The primary normalized-error difference is `symmetric - order-aware`; positive values favor order. Inference resamples architecture family, lineage within sampled family, and reversal group. Epoch snapshots remain within lineage/group cells. Ten thousand draws use the frozen seed.

The aggregate MAE interval excludes zero, but this does not override the failed rank criterion. The all-required rule was frozen before outcomes and is applied literally.

Reversal-contrast analysis uses one signed forward-minus-reverse difference per model, group, and coordinate. The symmetric model's predicted contrast is zero by construction. The order-aware model's contrast error is compared on exactly the same groups.

## Post-outcome redundancy sensitivity

Balanced and overall accuracy are effectively duplicated under the balanced partition. The declared sensitivity removes balanced accuracy and recomputes all applicable MAE, rank, regret, family, and contrast criteria. The rank criterion still fails, so the scientific decision is unchanged.

## Executed verification

The Phase-2H verifier checks:

- manifest and reversal-pair counts;
- frozen method hash and one-feature nesting;
- equal calibration budget;
- fresh image partition and canonical provenance;
- the pre-outcome checkpoint hash ledger and local checkpoint bytes;
- behavior and forecast hashes;
- finite bounded predictions for all six coordinates;
- exact symmetric reversal invariance;
- recomputed raw MAEs and retained failed gate;
- unique-coordinate sensitivity.

The verifier passes 7/7 and all seven Phase-2H unit tests pass. Rerunning the forecast, primary analysis, and sensitivity analysis reproduced byte-identical hashes before final commit.

## Limitations

- The order-aware method has one more fitted coefficient per coordinate than the symmetric model.
- The experiment uses the same six synthetic primitives and CIFAR-100 as Phases 2F/2G.
- Architecture families are reused from Phase 2G, although lineages, seeds, images, and words are fresh.
- Models are tiny and trained for only two CPU epochs.
- Pair calibration is available; this is not zero-calibration forecasting.
- Accuracy order effects are very small, and balanced accuracy duplicates accuracy.
- No natural shift, independent dataset, ImageNet model, or modern public checkpoint is included.
