# Known Limitations and Claim Boundaries

**Applies to:** historical V1--V3 evidence and prospective V4 work
**Status:** audit document; limitations are not silently repaired retroactively
**Last updated:** 2026-10-23

## 1. Scope of the evidence

1. The main V3 result is a six-primitive CIFAR-10 corruption experiment, not a universal law of distribution shift.
2. Fashion-MNIST, CIFAR-100, and ViT-PatchMixer are useful stress tests but do not constitute the broad multi-model, multi-environment protocol used by ODP-Bench.
3. The project has not yet evaluated natural temporal, geographic, demographic, camera, or subpopulation shifts.
4. V4 now tests length-four through length-six words on one frozen classifier each for CIFAR-10 and EMNIST. Both Phase 1 and severity-calibrated Phase 1B failed their method gates, so this evidence limits rather than extends the positive V3 claim.
5. The V1--V3 research loop repeatedly inspected related benchmark families. These results are development evidence, not an untouched final confirmation.

## 2. Metric comparability

1. Historical BPE is a project-specific, mixed-unit behavior loss. It is not directly comparable with accuracy MAE, mCE, or ODP-Bench Spearman.
2. Current `rho_m` ranks composition environments for a model. ODP-Bench primarily ranks models for an environment. These are different estimands.
3. Current statistical clusters are composition families from a small number of trained classifiers; they are not independent datasets or architecture families.
4. High raw across-model rank can be explained by source accuracy, architecture, or training recipe. V4 therefore requires source-controlled and within-family analyses.
5. Statistical significance does not establish practical deployment value. Model-selection regret and calibration cost must also be measured.

## 3. Information-access boundary

1. Historical Shift Algebra uses primitive behavior labels and pair calibration labels.
2. “Zero-shot” means no images from the held-out *composed* target are run through the classifier; it does not mean no shift labels or no calibration data are used.
3. ATC, Agreement, COT/COTT, and related ODP methods normally see unlabeled samples from the actual target. They belong to a different, information-advantaged regime.
4. Comparisons must be labeled R0--R4 according to `v4/benchmark/schema.py`.

## 4. Mathematical boundaries

1. Matrix multiplication, non-commutativity, associativity, semigroups, commutators, and Heisenberg groups are established mathematics and are not novelty claims.
2. The historical numerical `phi`/`phi_inv` pair was not exactly inverse because the stabilizer was placed differently. V4 fixes this in a new module; historical code remains frozen for reproduction.
3. The V3 matrix core is associative. Its triple readout additionally averages recursive decoding trees. Matrix-core associativity must not be used to claim that every historical end-to-end path is one exact homomorphism.
4. Matrix extraction and the all-pairs closed form are algebraically equivalent. Matrix form provides a streaming closed state and order bookkeeping, not empirical superiority over the equivalent formula.
5. The potential commutator imposes a transitive latent ordering. It cannot represent arbitrary cyclic rock-paper-scissors order effects.
6. Primitive marginals alone cannot identify arbitrary pair behavior. Pair calibration or structural assumptions are necessary.
7. Error can grow with word length because omitted third- and higher-order interactions need not vanish.
8. Natural metadata intersections are generally commutative factors, not physical ordered operators.

## 5. Model and optimization boundaries

1. The current 518-dimensional fingerprint is tied to a 128-dimensional feature representation and is not architecture independent.
2. The historical full model has about 103k composition parameters but only about 125 aggregate pair conditions in its main fit. Parameter-count and family-level sample size must both be disclosed.
3. `M_Lp_only` beats `M_B` on the mean of the three historical primitive-holdout experiments. Interaction modeling is not uniformly beneficial.
4. `M_B` is not the lowest-BPE method in every external dataset table.
5. Zero-initializing every directional generator and readout can create a zero-gradient saddle. V4 uses small non-zero output weights.
6. Historical `LAMBDA_REPR=0.2` acts as an enable flag while executable training uses a coefficient of `0.5`. V4 coefficients must be single-source and tested.
7. Historical code contains an unused or ambiguously used `W_ord` pathway in some models. V4 must not retain dead scientific parameters.
8. Learned exponents require a declared domain. V4 constrains `p >= 1`; any future relaxation must be preregistered and interpreted.
9. Phase-1 pair residual and potential interactions are normalized by pair count. Phase 1B shows that this intensive correction can systematically underestimate extensive higher-order damage, even when every single primitive is mild.

## 6. Data and mechanism boundaries

1. Synthetic image corruptions provide controlled operators but may not represent deployment shifts.
2. Some stochastic corruptions can create artificial order differences when reverse words do not share their random field. V4 requires paired deterministic fields for reversal tests.
3. Apparent order effects can arise from clipping, interpolation, or implementation details. Those are valid operator mechanisms only if declared and reproduced across libraries/settings.
4. The counterfactual `H` study supports one filter/write mechanism; it does not prove all order effects are spectral.
5. FaceScrub and other external datasets require a fresh license and redistribution review before use.
6. Robust-model checkpoints differ in preprocessing and label mappings. Provenance must be recorded per model.

## 7. Reproducibility boundaries

1. The historical experiments are reproducible from committed summaries and runners, but large checkpoints/data are intentionally not in Git.
2. No LaTeX engine is installed in the current sandbox, so `MATHEMATICAL_TUTORIAL.tex` received structural rather than PDF-compilation validation.
3. Future final manifests must include source commits, checksums, random fields, model preprocessing, and exclusions.
4. A sealed final benchmark must be evaluated once after code and analysis hashes are frozen.

## 8. Current safe claim

The strongest claim supported today is:

> On the audited V3 synthetic-corruption testbed, a signed-power saturation component and an antisymmetric order component improve prediction of held-out composed-shift behavior over the tested baselines, without using images from those held-out compositions at prediction time.

The independently frozen V4 evidence does **not** support superiority for length-six extrapolation: max-envelope wins the severe Phase-1 protocol, and pair residual wins severity-calibrated Phase 1B. Broader claims about natural shifts, cross-architecture ranking, or superiority to ODP-Bench methods remain prospective.
