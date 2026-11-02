# Phase 3A Information-Flow and Correctness Audit

**Protocol:** `V4-P3A-CIFAR100-REALISTIC-CORRUPTIONS-001`  
**Audit timing:** after the one-shot confirmation outcome  
**Result:** no confirmation-to-fit leakage found; automated verification PASS (35/35)

## 1. Prospective artifact chronology

The Git history preserves three boundaries before confirmation outcomes:

| Commit | Boundary |
|---|---|
| `d32633d` | corruptions, parameters, partitions, model roster, target words, primary method, comparators, metrics, gates, analysis, and verification code frozen before any Phase-3A behavior |
| `98d4a13` | development behavior and all metric-specific exponents, scales, and ridge coefficients frozen before fresh confirmation training or behavior |
| `baec63e` | all 36 confirmation checkpoint hashes committed with `behavior_opened: false` before confirmation evaluation |

The confirmation behavior and analyses were produced only after `baec63e`. The failed gate was retained without changing any frozen component.

## 2. Partition audit

- Development images: ten CIFAR-100 test images per class at canonical within-class offsets 80--89.
- Confirmation images: offsets 90--99.
- The verifier confirms these sets are disjoint.
- Development and confirmation each contain 60 target words in 30 reversal groups.
- The target-word sets are disjoint.
- Both splits reuse the same *types* of allowed calibration conditions—identity, six singles, and 30 directed pairs—but their measured values come from disjoint models and images.
- Development and confirmation model IDs are disjoint: reconstructed Phase-2H seeds 68100--68602 versus fresh Phase-3A seeds 69100--69602.
- The confirmation bank contains six families, three recipes, 18 lineages, and two within-lineage epoch snapshots.

## 3. Fitting dataflow

`freeze_methods.py` reads only `results/development-behavior.csv`. It derives:

- development-only coordinate scales;
- one exponent `q` per behavior coordinate;
- metadata/clean `g0` ridge coefficients;
- symmetric-composition ridge coefficients.

The frozen file records the development CSV SHA-256. `forecast.py` refuses to run if that hash changes.

The inherited `g0` label “metadata-only” needs an exact qualification: its inputs include clean-reference behavior plus model/word metadata, but no single-corruption or pair-corruption behavior. This clean-reference scalar is treated as model metadata in the V4 baseline convention. The primary method adds only the frozen symmetric, independent, and envelope composition coordinates derived from allowed calibration behavior.

## 4. Confirmation prediction dataflow

For a target word, `method.components` uses:

- clean behavior;
- six single-corruption behaviors;
- all thirty directed pair behaviors, averaged into 15 symmetric pair residuals;
- word multiset/length and model metadata.

It does **not** use the target's measured six-coordinate behavior to build any prediction feature. `make_rows` copies target behavior into a separate `target` field solely for later scoring/output. `apply` consumes only the frozen feature vector and frozen coefficient map.

The confirmation evaluator writes calibration and target rows in one behavior CSV before `forecast.py` runs. Therefore the software does not provide cryptographic pre-outcome prediction isolation. The protection is instead: (1) a committed frozen method and source hash, (2) no target coordinate in the prediction feature path, (3) exact code/dataflow inspection, and (4) independent post-outcome arithmetic checks. No target-to-feature path was found.

## 5. Corruption correctness

The six clean-room operators were checked as follows:

- inputs and outputs are finite RGB CIFAR tensors;
- all outputs remain in [0,1];
- each operator is deterministic;
- Gaussian and shot-noise fields are keyed by canonical sample ID and primitive-specific frozen seeds, making them batching-independent;
- chronological words execute left to right;
- JPEG-style quantization uses an orthonormal 8x8 block DCT and deterministic quantization/dequantization;
- motion blur, pixelation, brightness, and noise parameters are frozen in the manifest;
- the primary composition calculation canonically sorts the word before accumulation, giving bit-exact order blindness, not merely theoretical symmetry.

These are category proxies only. Nothing in the code establishes official ImageNet-C equivalence.

## 6. Metric and statistical correctness

All behavior and forecast rows contain margin, feature drift, ECE, balanced accuracy, overall accuracy, and mean confidence.

- Feature drift is the norm between each corrupted feature centroid and that model's clean centroid.
- Ten-bin ECE is accumulated over the full 1,000-image partition.
- Balanced accuracy and overall accuracy are both retained. They are effectively identical because every class contributes ten examples.
- Native-coordinate MAEs are reported separately.
- Macro MAE uses development-only target standard deviations, including an explicit floor.
- Controlled model-rank Spearman residualizes ranked parameter count, epoch, family, and recipe controls within each word.
- Accuracy-selection regret selects a model using predicted accuracy and scores loss against the true best model for each word.
- The 10,000-draw interval resamples architecture family, lineage within sampled family, and reversal group as a crossed axis. Both epoch snapshots and both reversal orientations remain clustered inside their cells.

The strengthened verifier independently recomputed every per-metric MAE and every macro normalized MAE directly from the forecast CSV, then reconstructed all seven gate booleans. All checks matched exactly.

## 7. Hash and roster audit

The verifier checks:

- all 36 ledgered checkpoint files against SHA-256;
- exact confirmation roster agreement;
- behavior CSV against its evaluation record;
- forecast CSV against its forecast record;
- frozen method against the hash used by the forecast;
- frozen development behavior against the method's recorded source hash;
- 3,492 unique behavior keys and 2,160 unique target forecast keys;
- finite values for every coordinate and method;
- exact equality between forecast targets and measured behavior targets.

All checks pass.

## 8. Reversal audit

Each model has 30 groups with exactly two reverse-order targets. All five order-blind methods have exactly zero predicted reversal gap in all six coordinates. Real targets have nonzero reversal gaps. This confirms both that the primary implementation is truly order-blind and that it necessarily leaves order-dependent signal unexplained.

No order-aware correction was fitted after seeing Phase-3A outcomes, consistent with the decision not to spend further compute tuning order awareness.

## 9. Remaining limitations, not leaks

1. Checkpoint families are tiny, two-epoch CPU models rather than modern production checkpoints.
2. Development and confirmation share CIFAR-100, architecture-family definitions, recipe definitions, and corruption code.
3. Confirmation words are protocol-public before fitting; only their outcomes are sealed.
4. Calibration is substantial: identity, all singles, and all directed pairs for every model.
5. No image-level bootstrap is included; uncertainty targets the declared family/lineage/reversal sampling structure.
6. Balanced and overall accuracy duplicate one coordinate on balanced partitions.
7. The strict selection-regret gate failed and cannot be rescued by the favorable error/rank results.
8. The clean-room corruptions do not support official ImageNet-C claims.

## 10. Audit conclusion

No evidence of confirmation-outcome leakage, model-roster contamination, image overlap, target-word overlap, changed development behavior, checkpoint substitution, omitted metric, arithmetic mismatch, or gate reinterpretation was found. Verification success establishes artifact and calculation integrity; it does not convert the failed scientific gate into a pass.
