# Shift Algebra — Detailed Experimental Plan

**Purpose.** Test the four central hypotheses of the Shift Algebra proposal
(manuscript: `CSMLResearch (1).pdf`) in a controlled, pre-registered,
reproducible way. This plan is written so that every claim in the paper can
be mapped to at least one experiment, and every experiment to a decision rule.

**Hypotheses under test**

| ID | Hypothesis | Falsified if |
|----|-----------|--------------|
| H1 | Explicit shift composition yields a smaller compositional generalization gap (CGG) than directly learning each observed distribution. | No significant CGG improvement on ≥2 of 3 benchmarks. |
| H2 | Interaction-aware composition beats purely additive composition when primitive shifts interact non-linearly. | Additive baseline ties or wins on the interacting tiers. |
| H3 | Order-aware composition improves prediction on non-commutative transformations. | Order pathway gives no benefit on held-out reversed pairs. |
| H4 | Behavior-supervised shift representations predict unseen model degradation better than reconstruction-only representations. | BPE not significantly lower; rank correlation with true degradation not high. |

Research questions RQ1–RQ6 map onto these: RQ1→H1, RQ2→H1/H2, RQ3→H2,
RQ4→H4, RQ5→H3 + commutator-recovery analysis, RQ6→Regime IV / natural benchmark.

---

## 1. Test-time information protocol (leakage control — critical)

This is the single most important design rule. The zero-shot claim is only
valid if the model never sees the target shifted distribution at test time.

**At test time, for a held-out composition S = {A₁…Aₘ}, a model may receive:**
- the primitive embeddings z_{Aᵢ}, computed from *training-time* samples of each
  primitive-shifted distribution D_{Aᵢ};
- the identity/indices of the primitives in S (and their order, where order is
  part of the task);
- statistics of the reference distribution D₀;
- the architecture and weights of the fixed base model f_θ (needed to define
  feature-space encodings of primitives).

**A model may NOT receive:** samples or summary statistics from D_S, from any
superset of S, or from any distribution "close to" D_S.

**Controls that enforce this:**
1. **Transductive upper bound (not zero-shot, reported separately):** a variant
   that *does* receive D_S summary statistics. Quantifies how much leakage is
   worth. If the zero-shot method approaches this bound, the claim is strong;
   if the gap is huge, the benchmark is too hard to be informative.
2. **Oracle:** full access to D_S samples; establishes the ceiling.
3. **Contamination audit:** for every held-out composition, verify that no
   training sample was generated with a superset of its primitives. Implement
   as an automated check over the data-generation log (every generated sample
   records its exact corruption set; assert set-inclusion disjointness).
4. **Memorization control:** a variant trained with a free embedding per
   *observed* composition and a shared "unknown" embedding for held-out ones.
   It must fail; if it doesn't, the held-out test does not test composition.
5. **Label-permutation control:** train with composition labels shuffled. Must
   collapse to chance-level composition prediction. Validates that the task
   signal is the composition structure, not spurious cues.

**Baseline fairness:** every baseline receives exactly the same test-time
inputs as Shift Algebra (primitive embeddings + D₀ statistics). The
"no-composition" baseline consumes the same primitive embeddings but replaces
the composition operator with a fixed readout (mean-pooling), so any
performance difference is attributable to the composition mechanism alone.

---

## 2. Benchmarks

Three benchmarks, ordered by control. A hypothesis is only claimed if it holds
on the benchmarks where its assumptions are met, and the manuscript must
report per-benchmark results separately (no averaging away failures).

### 2.1 Benchmark S — Synthetic controlled (primary)

Three tiers with **known ground truth**, so interaction terms γ and
commutators κ are measurable exactly. Data are cheap; this is where H2/H3 are
decided.

**Base task (all tiers):** 10-class classification, input x ∈ R^d (d = 20),
base model = fixed MLP (2 hidden layers, 128 units), trained once on clean
data and frozen. Behavior vector B(f, D) = (accuracy, per-class error,
ECE, penultimate feature drift ‖E[f(x_D)] − E[f(x₀)]‖, mean confidence).

- **Tier S1 — additive control (sanity).** x = (x₁,…,x₂₀), xᵢ ~ N(µᵢ, σᵢ²),
  independent. Primitives: A: µ₁ += α; B: σ₁² ×= β; C: µ₂ += γ; D: σ₃² ×= δ.
  Effects on the MLP are near-additive. **Expected outcome:** additive
  baseline ≈ Shift Algebra. If Shift Algebra is *worse* here, the harness or
  training is broken. This tier cannot confirm H2; it validates the pipeline.
- **Tier S2 — interacting (tests H2).** Introduce designed interactions with
  measurable γ:
  - *Shared-pathway interaction:* primitives A and B both perturb feature x₁
    (A adds noise, B scales it). Degradation compounds super-additively on the
    x₁ pathway, so true γ_AB ≠ 0 in behavior space.
  - *Correlated-feature interaction:* x₁, x₂ correlated with ρ; A rotates the
    (x₁,x₂) plane, B rescales x₂. Composition effect is non-additive in any
    fixed basis.
  - *True triple interaction:* threshold/gating corruption that only bites when
    three primitives co-occur (e.g., accuracy collapses only when noise,
    scale, and mask are all present), giving γ_ABC ≠ 0 with γ_AB = γ_AC =
    γ_BC ≈ 0. This is the sharpest test of higher-order composition.
  Ground-truth γ for any subset S is computed directly in **behavior space**:
  γ^B_S = ∆B_S − Σ_{i∈S} ∆B_{Aᵢ} − Σ_{pairs} γ^B_pairs (inclusion–exclusion).
  Behavior space is model-anchored and identifiable, unlike latent-space γ.
- **Tier S3 — non-commutative (tests H3).** Order-dependent transformations
  with exactly computable commutators:
  - quantize-then-Gaussian-noise vs. noise-then-quantize (classical
    non-commutativity; true KL(D_{A∘B} ‖ D_{B∘A}) > 0 computable in closed
    form for scalar Gaussian case);
  - nonlinearity-then-noise vs. noise-then-nonlinearity through the frozen MLP
    (ReLU makes these differ);
  - crop-then-blur vs. blur-then-crop on a small image patch variant.
  Ground-truth commutator: κ_true(A,B) = KL(D_{A∘B} ‖ D_{B∘A}) (or MMD for
  the image case).

**Pilot gate (mandatory):** before any method comparison, verify on S2/S3
that (a) the additive baseline fails where γ ≠ 0, and (b) a commutative model
fails on S3. If the designed interactions are too weak, redesign the tier —
do not proceed to method comparison on a non-discriminating benchmark.

**Shift encoder input (synthetic):** summary statistics of D₀ and D_A from
n = 5,000 samples: mean vector, covariance, skewness/kurtosis of each
coordinate, plus base-model penultimate embeddings aggregated the same way
(feature-space encoding, Eq. 24 of the manuscript).

### 2.2 Benchmark V — Vision (CIFAR-10, controlled corruptions)

Tests H1, H2, H4 at scale with a realistic model.

- **Data:** CIFAR-10. Base model: ResNet-18 trained once on clean CIFAR-10
  (fixed seed, shared by all methods). All corruption applied at evaluation
  only; training data for the shift-algebra modules consists of *statistics of
  corrupted evaluation pools*, never of retrained base models.
- **Primitive library (K = 6),** each with 5 severity levels:
  gaussian_noise (N), brightness (B), contrast (C), defocus_blur (D),
  pixelate (P), saturation (S).
- **Composition splits (pre-generated, fixed, hash-released):**
  - Regime II: all 6 primitives + all 15 unordered pairs observed; **all 20
    triples held out** (C(6,3) = 20).
  - Order test: additionally observe 8 *ordered* pairs (e.g., N∘B, B∘D, …) and
    **hold out their 8 reverses** (B∘N, D∘B, …).
  - Regime III: additionally observe 6 of the 20 triples; hold out the other 14.
  - Regime I: no compositions observed on this benchmark (see §4 for how C is
    trained — transfer from synthetic).
- **Contamination control:** every corrupted sample logs its corruption set;
  automated assertion that no training pool contains a superset of any held-out
  composition. Also verify severity levels are matched so a held-out triple is
  not approximated by a seen pair at higher severity.
- **Behavior vector:** top-1 accuracy, per-class accuracy, ECE, penultimate
  feature drift, mean confidence — all as deltas ∆B vs. clean.

### 2.3 Benchmark N — Natural distribution shift (transfer, RQ6)

Tests whether the mechanism transfers where primitives are not cleanly
controllable.

- **Setting:** WILDS-Camelyon17 (hospital-site subpopulation shift) and/or
  Office-Home (domain shift). Primitives = domain/site shifts; compositions =
  held-out site combinations or cross-dataset transfer of the composition
  operator.
- Because primitive identity is uncertain here, report **shift-identity
  uncertainty** explicitly (as the manuscript's limitations require) and treat
  this benchmark as secondary: a hypothesis is not falsified by failure here,
  but RQ6 is only answered positively by a successful transfer.

---

## 3. Methods to implement

All methods share the same shift encoder interface: z_A = E_φ(stats(D₀),
stats(D_A)). Only the composition mechanism and supervision differ, so
comparisons isolate the composition operator.

| # | Method | Composition operator | Role |
|---|--------|---------------------|------|
| M0 | No-composition | none — mean-pooled primitive embeddings → behavior decoder | H1 baseline |
| M1 | Additive | ẑ_{A∘B} = z_A + z_B | H2 baseline (key) |
| M2 | MLP composition | unconstrained MLP([z_A; z_B]) | capacity-matched baseline |
| M3 | Attention composition | cross-attention over primitive embeddings | capacity-matched baseline |
| M4 | CFA-style | compositional feature alignment adapted to shift embeddings | literature baseline |
| M5 | **Shift Algebra (full)** | C(z_A,z_B) = S(z_A,z_B) + λR(z_A,z_B), R antisymmetric (Eq. 30–33) | proposed method |
| M6 | Shift Algebra w/o order | M5 with λ = 0 | H3 ablation |
| M7 | Shift Algebra w/o interaction | M5 with γ-pathway removed | H2 ablation |
| M8 | Reconstruction-only | M5 trained without L_beh | H4 ablation |
| M9 | Task-specific adaptation | base model fine-tuned on available shifted data | practical baseline |
| M10 | Oracle | sees D_S samples | upper bound |
| M11 | Transductive | sees D_S statistics (not zero-shot) | leakage-value bound |

**Shift encoders (ablation dimension):** (a) statistics encoder (MLP over
moments), (b) feature-delta encoder (Eq. 24: per-sample embedding deltas,
aggregated), (c) paired-sample set encoder (Eq. 15, attention pooling).

**Behavior decoder G_ω:** 2-layer MLP mapping ẑ_S → ∆B̂_S (5-dim behavior
vector). Same architecture across all methods.

---

## 4. Training regimes (with the Regime I fix)

The manuscript's Regime I ("compose without seeing any composite") leaves the
composition operator C without training signal. Resolution:

- **Regime I (revised):** C is trained on **Benchmark S** (where compositions
  are cheap to generate) on a *disjoint* set of compositions, then transferred
  frozen/fine-tuned to the target benchmark, where only primitives are
  observed. This tests whether the composition *mechanism* transfers — which
  is the honest version of the regime's intent.
- **Regime II:** primitives + pairs observed; triples held out (Benchmark V
  main setting; Benchmark S tiers S2/S3).
- **Regime III:** primitives + pairs + 6 triples observed; 14 triples held out.
- **Regime IV:** learn everything on Benchmark S or V; test on Benchmark N
  (RQ6).

**Loss** (as in manuscript Eq. 38): L = λ₁L_repr + λ₂L_assoc + λ₃L_beh +
λ₄L_order + λ₅L_reg, with:
- L_repr = ‖ẑ_{A∘B} − z_{A∘B}‖² over observed compositions (+ cosine term);
- L_assoc = ‖C(z_A, C(z_B, z_C)) − C(C(z_A, z_B), z_C)‖² — imposed **only on
  tiers where interventions are functional on a shared state space** (S3, V);
  never on stateful/natural settings (manuscript's own caveat, kept);
- L_beh = ‖G(ẑ_S) − ∆B_S‖² over observed compositions;
- L_order = ‖(ẑ_{A∘B} − ẑ_{B∘A}) − (z_{A∘B} − z_{B∘A})‖² on observed
  ordered pairs;
- L_reg = weight decay + embedding-norm penalty (gauge regularizer, see §6).

---

## 5. Evaluation metrics

Primary error for CGG/RCE is **behavior-prediction error (BPE)** — predicting
model degradation is the deployment-relevant quantity — with **accuracy on the
composed distribution** as secondary.

| Metric | Definition | Notes |
|--------|-----------|-------|
| CGG | E_unseen − E_seen | lower better; computed for BPE and for accuracy |
| RCE | (E_unseen − E_seen)/(E_oracle − E_seen + ε) | fraction of oracle gap remaining |
| BPE | (1/N)Σ‖∆B̂_i − ∆B_i‖² over held-out compositions | primary |
| Rank-ρ | Spearman correlation between predicted and true degradation across held-out compositions | what matters for risk triage; target ≥ 0.8 for H4 |
| IRE | ‖γ̂_AB − γ_AB‖², with γ defined in **behavior space** (identifiable) | H2 structure test |
| OSI-abs | ‖ ‖ẑ_{A∘B} − ẑ_{B∘A}‖ − ‖z_{A∘B} − z_{B∘A}‖ ‖ | replaces the manuscript's ratio form, which is gameable by scaling and unstable near ε |
| Commutator-ρ | Spearman correlation between learned κ(A,B) ranking and true KL(D_{A∘B}‖D_{B∘A}) ranking | RQ5 |
| Probe-R² | R² of linear probes from z_A onto known shift parameters (α, β, ρ, severity) | representation quality |
| Assoc-violation | ‖C(C(z_A,z_B),z_C) − C(z_A,C(z_B,z_C))‖ on held-out triples, zero-shot | algebraic structure check |

---

## 6. Identifiability and structure-recovery analyses

(Addresses the gauge-freedom critique: algebraic claims must be anchored or
behavioral.)

1. **Behavior-space anchoring:** all interaction/commutator ground truths are
   defined via ∆B or KL between distributions — quantities with no gauge
   freedom. Latent-space γ is only ever evaluated *through* the behavior
   decoder.
2. **Gauge regularizer:** penalize ‖z_A‖ drift and encourage primitive
   embeddings of *identical* shifts (same parameters, different seeds) to
   coincide; report embedding stability across seeds.
3. **Linear probes** (§5) test whether the latent space encodes known
   structure linearly — a necessary (not sufficient) condition for the
   "algebraic structure" claim.
4. **Commutator recovery** against exact ground truth on S3 and against
   measured accuracy differences on V.
5. **Zero-shot associativity** on held-out triples: a genuine composition
   operator should approximately associate even where never trained.

---

## 7. Statistical protocol and pre-registered decision rules

- **Seeds:** 5 training seeds per method per benchmark; all splits fixed and
  hash-released before any method is run.
- **Unit of analysis:** one held-out composition × one seed. Report mean ±
  95% CI over compositions; test with paired Wilcoxon signed-rank (method vs.
  baseline, paired by composition), Bonferroni-corrected across the 4
  hypotheses (α = 0.05).
- **Decision rules (pre-registered):**
  - **H1 confirmed** iff Shift Algebra (M5) has significantly lower CGG than
    M0 on ≥ 2 of 3 benchmarks (p < 0.05, corrected) and never significantly
    worse.
  - **H2 confirmed** iff M5 beats M1 (additive) on S2 and V-Regime-II by a
    relative RCE reduction with 95% CI excluding 0; **falsified** if M1 ties
    or wins on both.
  - **H3 confirmed** iff M5 beats M6 (λ = 0) on the 8 held-out reversed pairs
    and S3 (OSI-abs and BPE), and commutator-ρ ≥ 0.7 on S3.
  - **H4 confirmed** iff M5 beats M8 (reconstruction-only) in BPE on held-out
    compositions and rank-ρ ≥ 0.8 on V.
  - **Project-level falsification** (from the manuscript): additive ties
    everywhere; no consistent unseen-composition benefit; latent space cannot
    predict behavior; success traceable to label memorization (memorization
    control succeeds); order sensitivity unrecoverable on S3. Any of these →
    report as a negative result with the same rigor.
- **Exploratory analyses** (latent-dim sweep, encoder-capacity sweep,
  severity generalization) are labeled exploratory and not used for
  hypothesis decisions.

---

## 8. Ablation matrix

Core ablations (manuscript §IX), run on S2, S3, and V-Regime-II:

1. Remove interaction terms (M7) → isolates H2's mechanism.
2. Remove order pathway (M6) → isolates H3's mechanism.
3. Remove behavior supervision (M8) → isolates H4's mechanism.
4. No pairwise training (Regime I-revised) → tests pure primitive→triple transfer.
5. Latent dimension d ∈ {16, 32, 64, 128, 256}.
6. Encoder capacity: statistics vs. feature-delta vs. paired-sample set encoder.
7. λ (order strength) sweep: 0, 0.1, 0.3, 1.0, 3.0.
8. Loss-term ablation: drop L_assoc, drop L_order, drop L_reg — measures each
   term's contribution and their interactions.

Full cross of {interaction on/off} × {order on/off} × {behavior supervision
on/off} on V-Regime-II (8 cells × 5 seeds) to detect interactions between
mechanisms.

---

## 9. Experiment schedule and compute

| Phase | Content | Gate to proceed |
|-------|---------|-----------------|
| P0 (wks 1–3) | Harness: data generators with corruption-set logging, split generator + hashes, base models (MLP, ResNet-18), encoder/decoder implementations, contamination audit, memorization & permutation controls | S1 sanity: additive ≈ M5; controls behave (memorization fails, permutation collapses) |
| P1 (wks 4–6) | Benchmark S tiers S1–S3; pilot interaction-strength check; M0–M8 on S | Pilot: additive fails on S2, commutative model fails on S3 |
| P2 (wks 7–12) | Benchmark V: all regimes, M0–M11, 5 seeds | H1/H2/H4 readout; rank-ρ ≥ 0.8 check |
| P3 (wks 13–16) | Benchmark N + Regime IV transfer (RQ6); severity generalization | Transfer result reported either way |
| P4 (wks 17–20) | Ablation matrix (§8), structure-recovery (§6), theory validation (§10), stats, writeup | All decision rules resolved |

**Compute estimate:** base-model pretraining ~1 GPU-hour; corruption-pool
generation and inference dominate (~50–100 GPU-hours total); shift-algebra
modules are small (encoders over summary statistics). Synthetic tiers run on
CPU. Total well under ~200 GPU-hours — the plan is deliberately cheap so it
can be repeated.

---

## 10. Theory validation (supports the conditional theorem)

- Empirically estimate the Lipschitz constant L of the learned C (max Jacobian
  norm over training pairs); test whether L < 1 regions exist in latent space.
- Plot zero-shot BPE vs. composition order |S| (2, 3, 4) and compare the
  compounding curve's shape to the bound in manuscript Eq. 71.
- If compounding is exponential in practice, report it — it bounds the method's
  regime of usefulness and motivates the low-rank/tensor interaction variant
  (stretch item below).

## 11. Stretch items (only if core plan succeeds)

- Tensor/low-rank interaction parameterization of γ_S for higher orders.
- Uncertainty-calibrated behavior prediction (predict a distribution over
  ∆B, report calibration) — directly useful for pre-deployment risk triage.
- Additional natural benchmarks (speech: speaker×noise×channel; medical:
  demographic×acquisition).

---

## 12. Reproducibility and artifact release

For every benchmark, release (per manuscript §XV, made concrete):
1. primitive intervention definitions (code + parameter grids);
2. the composition graph and exact held-out sets (split generator + SHA-256);
3. random seeds (data, training, model init);
4. model configurations (YAML) and training checkpoints;
5. evaluation scripts and per-composition metrics (CSV);
6. the pre-registration document (this plan, frozen before P1);
7. contamination-audit output and control-variant results.

The split is generated before any model development and never regenerated.

---

## 13. Risk register

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Synthetic interactions too weak to discriminate | Medium | Mandatory pilot gate (P1); redesign tiers before method comparison |
| Held-out compositions accidentally approximated in training | Medium | Automated set-inclusion audit + severity matching check |
| Latent algebra unidentifiable / structure claims vacuous | High (known) | Behavior-space ground truths, gauge regularizer, probes; algebraic claims evaluated behaviorally only |
| Regime I ambiguity | Resolved | Revised definition: C trained on disjoint synthetic compositions, transferred (§4) |
| Additive baseline too strong on vision corruptions | Medium | Designed interacting primitives; report per-benchmark; H2 claimed only where assumptions hold |
| Behavior prediction target too noisy (accuracy non-smooth) | Medium | Predict smooth components (feature drift, ECE, logit stats) alongside accuracy; rank-ρ as primary H4 readout |
| Compute overrun | Low | Cheap design; synthetic-first ordering; gates stop early |

---

## 14. Mapping: paper claim → experiment

| Paper claim | Experiment |
|-------------|------------|
| H1 (composition < direct learning) | M5 vs. M0, CGG, all benchmarks, §7 rule |
| H2 (interaction-aware > additive) | M5 vs. M1 on S2 + V; IRE vs. behavior-space γ |
| H3 (order-aware helps) | M5 vs. M6 on S3 + 8 reversed pairs; OSI-abs, commutator-ρ |
| H4 (behavior supervision helps) | M5 vs. M8; BPE + rank-ρ |
| RQ1 (shifts representable composably) | Probe-R², zero-shot associativity, CGG on S |
| RQ2 (composition generalizes to higher orders) | Regime II/III held-out triples, BPE |
| RQ3 (= H2) | as H2 |
| RQ4 (= H4) | as H4 |
| RQ5 (commutativity detectable) | commutator-ρ vs. KL ground truth on S3 |
| RQ6 (transfer across datasets) | Regime IV, Benchmark N |
| Eq. 71 bound direction | §10 compounding curves |

*End of plan.*
