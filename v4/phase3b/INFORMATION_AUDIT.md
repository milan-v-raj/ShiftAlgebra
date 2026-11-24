# Phase 3B Information-Flow and Correctness Audit

**Protocol:** `V4-P3B-CIFAR10-REALISTIC-REPLICATION-001`

**Audit timing:** after the one-shot confirmation outcome

**Result:** no confirmation-to-fit leakage found; automated verification PASS (40/40)

## 1. Prospective artifact chronology

The pushed Git history preserves three boundaries before confirmation outcomes:

| Commit | Boundary |
|---|---|
| `603e2b5` | dataset contract, byte-identical corruption code, parameters, fresh seeds, partitions, model rosters, target words, primary method, comparators, metrics, gates, analysis, and verification code frozen before Phase-3B behavior |
| `18282c5` | 3,492 development behavior rows and every metric-specific scale, exponent, and ridge coefficient frozen before confirmation training or behavior |
| `c7d5b57` | all 36 confirmation checkpoint hashes pushed with `behavior_opened: false` before confirmation evaluation |

Confirmation behavior was opened only after `c7d5b57`. The failed criterion and 6/7 outcome were retained without modifying any frozen component.

## 2. Dataset and partition audit

- The loader verifies canonical CIFAR-10 batch hashes and exposes 40,000 train, 10,000 validation, and 10,000 test examples across ten classes.
- Development uses 200 test images per class at canonical within-class offsets 600--799: 2,000 images total.
- Confirmation uses offsets 800--999: another 2,000 images.
- These Phase-3B partitions are disjoint from each other and from Phase-2 through Phase-2E bank partitions.
- They are not claimed globally untouched: historical Phase 1 used much of the CIFAR-10 test set.
- Development and confirmation each contain 60 target words in 30 exact reversal groups; their target sets are disjoint.
- All 120 Phase-3B targets are disjoint from all 120 Phase-3A targets.
- Both splits use the same allowed calibration condition types—identity, six singles, and all 30 directed distinct-primitive pairs—but obtain their values from disjoint model and image banks.
- Development and confirmation model IDs and seeds are disjoint: seeds 73100--73602 versus 74100--74602.
- Each bank contains six families, three recipes, 18 lineages, and two within-lineage epoch snapshots.

## 3. Development fitting dataflow

`freeze_methods.py` reads only `results/development-behavior.csv`. It derives:

- development-only coordinate scales;
- one exponent `q` per behavior coordinate;
- metadata/clean `g0` ridge coefficients;
- symmetric-composition ridge coefficients.

The frozen file records development CSV SHA-256 `b89080a1b122194c3a68cd13eda2f3d344b7749f8fd57466146f079383b082e2`. `forecast.py` refuses to run if that source changes. The frozen method SHA-256 is `f71dea2ceba0fcd5c9c84fb342449135fad3da1f2257fd45786d924050037526`.

The inherited `g0` name requires precision: its inputs include clean-reference behavior plus model and word metadata, but no single-corruption or pair-corruption behavior. The primary method adds only frozen symmetric, independent, and envelope coordinates derived from allowed calibration behavior.

## 4. Confirmation prediction dataflow

For each target word, `method.components` uses:

- clean behavior;
- six single-corruption behaviors;
- all 30 directed pair behaviors, averaged into 15 symmetric pair residuals;
- the word multiset and length;
- static model metadata.

It does **not** use measured target behavior to construct any prediction feature. `make_rows` carries the target into a separate field solely for later scoring and output; `apply` consumes only the frozen feature vector and coefficient map.

The confirmation evaluator writes calibration and target rows into one behavior CSV before `forecast.py` runs, so there is no cryptographically isolated prediction service. Protection instead comes from: (1) the method and source hash pushed before confirmation training, (2) the checkpoint ledger pushed before outcomes, (3) no target coordinate in the feature path, (4) fail-closed hash checks, and (5) independent post-outcome arithmetic verification. No target-to-feature path was found.

## 5. Corruption implementation audit

Phase 3B copies the Phase-3A corruption source byte-for-byte. Both files have SHA-256 `320e2b14302d051c6f34579bb0d64ac572bbeeccbf83a23198165073d28f2a60`.

The test and verification suites establish that:

- inputs and outputs are finite RGB CIFAR tensors bounded in [0,1];
- every primitive is deterministic;
- Gaussian and shot-noise fields depend on canonical sample ID and frozen primitive seeds, so batching does not alter them;
- chronological words execute left to right;
- JPEG-style quantization uses an orthonormal 8x8 block DCT and deterministic quantization/dequantization;
- motion blur, pixelation, brightness, and noise parameters match the prospective manifest;
- the primary composition feature path canonically sorts the word before accumulation, producing bit-exact order blindness.

These are category proxies. No code or result establishes equivalence to official CIFAR-C or ImageNet-C implementations or severities.

## 6. Metric and statistical audit

Every behavior and forecast row contains margin, feature drift, ECE, balanced accuracy, overall accuracy, and mean confidence.

- Feature drift is the norm between a corrupted feature centroid and that model's clean centroid.
- Ten-bin ECE is accumulated over the full 2,000-image partition.
- Balanced and overall accuracy are effectively identical because each class contributes 200 examples.
- Native-coordinate MAEs are reported separately.
- Macro MAE normalizes with development-only target standard deviations and an explicit floor.
- Controlled model-rank Spearman residualizes ranked parameter count, epoch, family, and recipe controls within each word.
- Accuracy-selection regret selects a model using forecast accuracy and scores loss against the truly best model for each word.
- The 10,000-draw interval resamples architecture family, lineage within sampled family, and reversal group as a crossed axis. Both epochs and reversal orientations remain clustered within their cells.

The verifier independently recomputes every per-metric and macro MAE from the forecast CSV and reconstructs all seven primary gates. Every value and boolean matches the recorded analysis.

## 7. Hash, roster, and arithmetic audit

The verifier checks:

- all 36 ledgered checkpoint identities and any locally available checkpoint files against SHA-256;
- exact confirmation roster agreement;
- confirmation behavior CSV SHA-256 `0d3212ea773ed765624ea4cab7f284ede14e56ea2d510d27760c5deea88d9b3b` against its evaluation record;
- confirmation forecast CSV SHA-256 `519c53ca5053a16d91d06afbd59c23f8f460331683728e8015d25de451fd4340` against its forecast record;
- frozen method identity against the method hash used for forecasting;
- frozen development behavior against the method's source hash;
- 3,492 unique confirmation behavior keys and 2,160 unique target forecast keys;
- finite values for every coordinate and method;
- exact equality between forecast targets and measured target behavior;
- all five methods and all six metrics;
- independent recalculation of all errors and all seven gate decisions.

All 40 checks pass.

## 8. Replication and reversal audit

The Phase-3B analysis reads the completed Phase-3A analysis only after the Phase-3B gate is computed. This comparison is descriptive and cannot alter Phase-3B predictions or gates. The two phases agree on all six metric-improvement directions and all seven gate directions; both fail only accuracy regret.

Every model has 30 groups containing exactly two reverse-order targets. All five order-blind methods have zero predicted reversal gap in every coordinate. Real target gaps are nonzero. This verifies exact implementation symmetry and demonstrates the order-dependent signal the primary method necessarily omits.

No order-aware correction was fitted after either Phase-3A or Phase-3B outcomes. WCON remains outside this protocol as a possible separate V5 research program.

## 9. Remaining limitations, not leaks

1. The six families are tiny, two-epoch CPU models rather than modern production checkpoints.
2. CIFAR-10 differs from Phase-3A's CIFAR-100, but the studies share architecture/recipe templates and byte-identical corruption code; this is not independent corruption-implementation evidence.
3. Some Phase-3B images appeared in historical Phase 1, so they are not globally untouched across the repository.
4. Confirmation target words are protocol-public before fitting; only their outcomes are sealed.
5. Calibration is substantial: identity, every single, and every directed pair for every model.
6. No image-level bootstrap is included; uncertainty follows the preregistered family/lineage/reversal structure.
7. Balanced and overall accuracy duplicate one coordinate on balanced partitions.
8. The order-blind primary method cannot forecast true reversal differences.
9. Calibration and target measurements share one evaluator and CSV before forecasting; chronology and audited code flow provide isolation rather than a separate prediction service.
10. The strict selection-regret criterion failed and cannot be rescued by favorable error, rank, family, replication-agreement, or robustness results.
11. Clean-room categories do not support official CIFAR-C, ImageNet-C, or natural-shift claims.

## 10. Audit conclusion

No evidence was found of confirmation-outcome leakage, checkpoint substitution, model-roster contamination, within-Phase-3B image overlap, development/confirmation target overlap, Phase-3A target reuse, changed development behavior, corruption-source drift, omitted metrics, arithmetic mismatch, or gate reinterpretation. Verification establishes artifact and calculation integrity; it does not convert the failed scientific gate into a pass.
