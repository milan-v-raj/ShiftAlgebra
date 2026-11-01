# Phase 2G Information and Correctness Audit

**Protocol:** `V4-P2G-CIFAR100-NOREFIT-ARCH-001`

**Outcome:** strict no-refit full-vector and magnitude gates passed; no confirmation leak detected

## Prospective chronology

1. Commit `c50b04e` froze all code, six architectures, seeds, recipes, images, words, method hashes, inference, and gates before model training or behavior outcomes.
2. The 18-lineage/36-snapshot bank was trained. Commit `1a0d695` recorded every checkpoint hash and training summary while confirmation behavior remained unopened.
3. Confirmation behavior was then evaluated exactly once on all 97 frozen conditions.
4. Forecasting and primary analysis used the code already committed in `c50b04e`.
5. Post-outcome work was limited to the declared balanced-accuracy sensitivity, stricter verification assertions, result documentation, and repository synthesis updates. It did not alter forecasts or primary gates.

## Frozen-method identity

The experiment transferred `v4/phase2f/FROZEN_FULL_VECTOR_METHODS.json` with SHA-256:

`243761ddd3749dcc0149e83cca7e40241d43a3800320d8f777bac624b1f5827e`

That artifact points to Phase-2F development behavior SHA-256:

`8c3289ac47c01267751a2f3d8104d1940f04e20b8fa0cec4dea0c2baef7b90a7`

The verifier checks both hashes. Phase 2G contains no fit function and does not write a replacement method file. It loads the Phase-2F `q` values, feature means/scales, coefficients, target scales, full design, magnitude design, and twenty random designs directly.

## Architecture boundary

The Phase-2F family indicators are VGG, DenseNet, Inception, Squeeze, Mixer, and Attention. Phase 2G uses residual, depthwise-separable, ConvNeXt-style, grouped/shuffle, dilated-pyramid, and gated-convolution models. The sets are disjoint. Because feature construction retains the frozen Phase-2F schema, all six old-family indicators are zero for every Phase-2G model. Known recipe indicators, parameter count, epoch, word length, and primitive counts remain available exactly as preregistered.

Each architecture has three independently seeded recipe lineages and two epoch snapshots. Statistical inference keeps epochs inside lineages.

## Target and image boundary

- Phase-2F development used CIFAR-100 test offsets 0--19 per class.
- Phase-2F confirmation used offsets 20--39.
- Phase-2G uses offsets 40--59.
- All sixty Phase-2G target words are absent from every prior frozen word bank and form thirty complete reversal groups.
- Calibration remains identity, six singles, and thirty directed distinct pairs.

Dataset provenance remains tied to the canonical test-pickle MD5 `f0ef6b0ae62326f3e7ffdfab6717acfc`, with all 10,000 transported test images pixel-matched.

## Forecast information path

For each held-out model and target word, forecasting uses only:

- the clean condition;
- single-primitive calibration;
- pair calibration permitted by the frozen design;
- model parameter count, epoch, recipe, unknown-family encoding, word length, and primitive counts;
- the byte-frozen Phase-2F method.

Target-word measurements are copied only to `true_*` result columns. They do not enter component construction, normalization, coefficients, pair selection, or predictions. Forecasting has no optimizer or development step.

## Complete metric accounting

Every behavior row and every forecast include:

1. margin;
2. feature drift;
3. ECE;
4. balanced accuracy;
5. accuracy;
6. mean confidence.

The sealed artifacts contain 6 families, 18 lineages, 36 snapshots, 97 behavior conditions, 3,492 behavior rows, 60 target words, 30 reversal groups, and 2,160 forecast rows. They retain frozen `g0`, full, magnitude-five, and twenty random prediction vectors.

All six raw coordinate MAEs improve. All six architecture-family aggregate improvements are positive. The full-vector gate passes 6/6 and the magnitude gate passes 4/4.

## Statistical accounting

Primary inference resamples architecture families, lineages within sampled families, and reversal groups as a crossed target axis. Epoch snapshots are averaged inside each lineage/group cell. The Phase-2F development standard deviations remain frozen as coordinate normalizers. Raw errors remain separately reported.

Balanced and overall accuracy are redundant on this class-balanced partition. The preregistered analysis retains both. A labeled post-outcome sensitivity removes balanced accuracy and recomputes every applicable MAE, rank, family-sign, regret, and interval criterion; both gates remain passed.

## Executed verification

The verifier checks:

- manifest counts and held-out family identity;
- frozen method and development hashes;
- fresh image partition and canonical provenance;
- the pre-outcome 36-checkpoint hash ledger, and checkpoint bytes when locally present;
- behavior and forecast hashes and row accounting;
- finite, naturally bounded predictions for all six coordinates;
- recomputed per-coordinate MAEs;
- complete primary and magnitude gate records;
- unique-coordinate sensitivity gates.

The Phase-2G verifier passes 7/7 and all six Phase-2G unit tests pass. Rerunning the forecast, primary analysis, and robustness analysis reproduced byte-identical artifact hashes before final commit.

## Limitations

- Dataset and transformations are shared with Phase 2F; this is not independent-dataset transfer.
- Recipes were present during Phase-2F fitting; only architecture families are held out.
- Models are tiny, trained for two CPU epochs, and have low CIFAR-100 accuracy.
- Per-model pair calibration is available; this is not zero-calibration forecasting.
- Feature drift is representation-scale dependent, although normalization remains frozen.
- No natural shift, realistic corruption suite, ImageNet data, or modern public checkpoint is tested.
