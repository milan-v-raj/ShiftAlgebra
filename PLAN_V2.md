# Shift Algebra v2 — Revised Research Plan ("making the idea work")

**Status:** plan (pre-registered before any v2 experiment is run).
**Basis:** v1 experiment results (`RESULTS.md`), post-hoc audit (`AUDIT.md`),
and a targeted literature review (citations in §2).
**Goal:** turn the v1 negative result into a method + benchmark on which the
core idea — *predict the effect of an unseen shift composition from observed
primitive shifts* — can actually be tested and, if true, demonstrated.

---

## 1. Diagnosis: why v1 failed (evidence-linked)

| # | Failure | Evidence from v1 | Root cause |
|---|---------|------------------|------------|
| D1 | MLP pair-operator overfit | M5 BPE 0.962 vs M3 (attention) 0.236, M0 (mean-pool) 0.381; M5 significantly worse than M0/M2/M3 (p ≤ 0.046) | A 2-input MLP trained on 25 pairs memorizes pair-specific structure instead of a general composition rule |
| D2 | Composition trained in an unanchored latent space | H4 partial: behavior supervision helped (M5 ≫ M8) but the representation itself was trained by embedding reconstruction (gauge-ambiguous) | The "algebra" lived in a learned embedding space with no identifiable anchor; composition quality was never measured where it matters (behavior space) |
| D3 | Order pathway learned *an* order effect with the *wrong direction* | Measured ∆acc(B∘C) − ∆acc(C∘B) = **−6.2pp**; M5 predicted **+21.2pp** (sign flipped); commutator-ρ = 0.10 | The antisymmetric bilinear form R(z_A,z_B) = W(z_A⊙Uz_B − z_B⊙Uz_A) is unconstrained in *direction* — it can encode any order effect, and with no ordered-pair supervision in Regime II it encoded noise |
| D4 | Benchmark saturation compressed the signal | At mid/high severities, accuracy floors at 12–22% for almost all conditions; ∆B variance concentrated at low severities | Severity grid was too strong for a 91% base model; the accuracy component of the behavior vector saturates |
| D5 | Regime II gave the order pathway zero supervision | Only unordered pairs observed; L_order unusable; associativity loss unusable (no triples) | The two structural mechanisms (order, associativity) had no training signal in the main regime |
| D6 | Too few composition examples | 25 pairs, no triples | Any flexible operator overfits; needs structure or more data |

Non-failures (keep from v1): data integrity (md5-verified CIFAR-10),
leakage boundary, controls (MEMO/PERM), oracle bound, statistics, determinism
(all audited, see `AUDIT.md`).

---

## 2. Research grounding (what the literature says to do differently)

1. **Attention pooling beats linear/mean pooling for compositional
   generalization.** Conklin et al. (sDTM, NeurIPS 2024) show pooling-by-
   multi-headed-attention over compositional structure "drastically" outperforms
   linear pooling across distributional-shift splits
   (https://proceedings.neurips.cc/paper_files/paper/2024/file/ccfa9ba5a84d0e4c620093d27102b7c5-Paper-Conference.pdf).
   → Matches our v1 result (M3 won). v2 makes attention-style aggregation the
   core operator, not a baseline.
2. **Compose in the space where the process is additive.** Compositional Risk
   Minimization (Liu et al., 2024) proves additive *energy* (logit-space)
   classifiers generalize to unseen attribute combinations, with a sharp
   characterization ("discrete affine hull")
   (https://arxiv.org/html/2410.06303v1). CFA (Wang et al., TMLR 2024) likewise
   aligns compositional structure in *feature* space with orthogonal heads
   (https://arxiv.org/abs/2402.02851).
   → v1 composed in an arbitrary learned embedding (M1 additive failed there).
   v2 anchors composition in **behavior/logit space**, where additivity is the
   theoretically right inductive bias.
3. **Make non-commutativity structural, not learned-arbitrary.** Representing
   tokens as matrices and encoding a sequence as the ordered *product*
   ("All You Need Is Non-Commutative Words", 2026,
   https://pith.science/paper/2608.29314) makes order sensitivity a
   mathematical property of the construction (matrix product doesn't swap) —
   the *direction* is learned, the *existence* of order effects is guaranteed.
   Deep-OSG (Chen & Wu, 2023, https://arxiv.org/abs/2302.03358) shows that
   embedding the semigroup (associativity) property into the architecture +
   loss reduces data dependency and error accumulation.
   → v2's operator-product parameterization: each shift = a matrix; composition
   = matrix product. Order falls out structurally; associativity is free.
4. **Ask whether the transformation algebra survives into feature space.**
   "Transformation Laws in Neural Representations" (2026,
   https://arxiv.org/html/2609.18190) studies exactly the homomorphism question
   ρ_τ∘ψ = ψ∘τ for physical transformation families (color groups, heat
   semigroups) and finds the algebra erodes with depth.
   → v2 tests this directly: do learned shift operators compose like the true
   corruptions (associativity error, commutator recovery)?
5. **Set encoders and order.** Vinyals et al., "Order Matters" (ICLR 2016,
   https://3dvar.com/Vinyals2015Order.pdf): order of set inputs matters in
   practice; Read-Process-Write architectures handle unordered sets.
   → v2's set-attention encoder over primitive embeddings, with explicit order
   tokens when order is part of the task.
6. **Severity-calibrated corruption benchmarks.** ImageNet-C (Hendrycks &
   Dietterich) uses 15 corruptions × 5 calibrated severity levels with mean
   corruption error — the standard for exactly our setting
   (https://www.emergentmind.com/topics/imagenet-c).
   → v2 recalibrates severities so single shifts cause ~5–25pp drops (below
   saturation), following the ImageNet-C calibration philosophy.

---

## 3. Revised hypotheses (pre-registered)

| ID | Hypothesis | Decision rule (confirm iff) |
|----|-----------|------------------------------|
| H1′ | **Behavior-anchored composition beats unanchored composition.** A composition operator trained/supervised in behavior space (M-A) or with operator-product structure (M-B) achieves significantly lower held-out BPE than v1's M5 *and* than mean-pooling M0; rank-ρ ≥ 0.8. | Paired Wilcoxon over held-out conditions, p < 0.05 vs both M5 and M0; rank-ρ ≥ 0.8 on V2. |
| H2′ | **Interaction-aware composition beats additive-in-behavior-space** when ground-truth interactions exist (synthetic S2, vision V2). | Lower BPE than behavior-additive baseline on S2 + V2, CI excluding 0. |
| H3′ | **Operator-product order modeling recovers the DIRECTION of non-commutativity.** On held-out reversed pairs, the sign of the predicted behavior difference matches the measured sign (sign-match rate ≥ 80%), commutator-ρ ≥ 0.7, and behavior order error is below commutative baselines. | Sign-match ≥ 80% on S3 + V2 reversed pairs; commutator-ρ ≥ 0.7. |
| H4′ | **Behavior supervision improves the representation** (kept from v1, now with the anchored encoder). | M-A beats its reconstruction-only ablation (p < 0.05). |
| H5′ | **Higher-order supervision helps.** With Regime III (some triples observed), held-out triple BPE is lower than in Regime II (same method). | BPE(Regime III) < BPE(Regime II), p < 0.05, on V2 triples. |
| H6′ | **Synthetic→vision transfer works.** An operator trained on synthetic tiers S2/S3 transfers to V2 without retraining (or with light fine-tuning), beating no-transfer training from scratch. | Transfer BPE ≤ scratch BPE on V2 held-out (CI), rank-ρ ≥ 0.7. |

**Falsification (updated):** the project is unsuccessful if (a) behavior-
anchored composition does not beat M0/M3 on V2, or (b) no method recovers
order *direction* on S3, or (c) held-out BPE shows no consistent benefit over
mean-pooling anywhere. In that case publish the negative result + theory (§8).

---

## 4. Revised method — "Shift Algebra v2"

Three composition mechanisms, all sharing the same shift encoder and behavior
decoder. All are trained with **behavior-primary** supervision.

### 4.1 Shared components

- **Shift encoder** E: input = distribution stats (as v1: feature mean/std,
  pixel mean/std, feature-delta mean/std) **+ the primitive's measured
  behavior vector ∆B_A** (available for primitives — they are observed).
  Output z_A ∈ R^64. (Adding ∆B_A anchors the representation to behavior.)
- **Behavior decoder** G: z_S → ∆B̂_S (5-dim, as v1).
- **Losses:** L = λ_beh·L_beh + λ_repr·L_repr + λ_assoc·L_assoc + λ_reg,
  with L_beh **primary** (λ_beh = 1.0; λ_repr ≤ 0.3). L_assoc is the
  associativity consistency loss — now usable because (i) M-B's operator
  product is associative by construction (so L_assoc regularizes the
  *matrices*), and (ii) Regime III observes triples.

### 4.2 M-A — Behavior-space composition (anchor-first)

Compose directly in behavior space, where quantities are identifiable:

  ∆B̂_{A∘B} = g(∆B_A, ∆B_B, z_A, z_B)

with g = small MLP over concatenation, and a variant g_att = attention pooling
over the set {∆B_A, ∆B_B} (per literature item 1) with a learned interaction
term γ (as in the manuscript's Eq. 19, but in behavior space). The encoder
embeddings enter as auxiliary context. **This is the minimal change that fixes
D2**: the composition target and supervision live in the same identifiable
space as the evaluation metric.

### 4.3 M-B — Operator-product algebra (structure-first)

Each primitive shift i gets a learned matrix T_i ∈ R^{d×d} (d = 32 or 64),
parameterized as **T_i = I + low-rank** (rank r ∈ {4, 8, 16}) so that
identity shifts are easy and the operator stays near-identity (corruptions
are perturbations). A composition S = (A₁, …, A_m) (in application order) gets

  T_S = T_{A_m} · … · T_{A_1}        (ordered matrix product)

and the behavior decoder reads a summary of T_S (flattened upper triangle, or
T_S applied to a learned reference vector). Properties:

- **Non-commutativity is structural** (matrix product); the *direction* is
  learned from data (fixes D3 — literature item 3).
- **Associativity is free** (matrix product is associative) — the manuscript's
  semigroup framing is satisfied by construction (literature item 3,
  Deep-OSG).
- The commutator [T_A, T_B] = T_A T_B − T_B T_A is a principled,
  computable order statistic (manuscript Eq. 69 made literal).

Variants: (a) full matrices, (b) low-rank + identity (default), (c) M-B with
attention-mixed operator summary.

### 4.4 M-C — Attention-set composition (v1 winner, kept)

v1's M3 (cross-attention over primitive embeddings), upgraded: multi-head,
set-attention pooling over {(z_i, ∆B_i)} with **order tokens** when the
composition is ordered (literature items 1, 5). This is the strong baseline
that H1′ must beat.

### 4.5 Training regimes

- **Regime II** (main): primitives + pairs observed (V2 split, §5).
- **Regime III**: additionally observe 6 of 20 triples (tests H5′).
- **Regime I-revised (transfer)**: train the composition operator on
  synthetic tiers S2/S3 (cheap, unlimited compositions), transfer to V2
  (tests H6′).
- **Order supervision**: S3 provides ordered pairs with exact commutators —
  used to pretrain/regularize the order pathway (fixes D5), never the held-out
  V2 pairs.

---

## 5. Revised benchmark

### 5.1 V2 — Vision (CIFAR-10, recalibrated)

- **Same frozen base model** (ResNet-18 w0.25, 91.04%) — reusable.
- **6 primitives** (adds pixelate P and saturation S to v1's A, B, C, D):
  brightness, blur, noise, contrast, pixelate, saturation.
- **Severity recalibration (fixes D4):** pilot experiment first: for each
  primitive, choose 5 severities such that the single-shift accuracy drop is
  ≈ {3, 8, 15, 25, 40}pp (below saturation). Pairs/triples then span a
  graded range without flooring. ImageNet-C-style calibration
  (literature item 6).
- **Behavior vector (fixes D4):** primary = [mean logit-margin drop,
  feature drift, ECE delta, per-class error delta]; **accuracy delta is
  secondary** (reported, not optimized) because it floors.
- **Split (Regime II):** observe all 6 primitives + **all 15 unordered
  pairs** (more composition training data — fixes D6 partially); hold out:
  all 20 triples (C(6,3)), plus 8 ordered pairs and their 8 reverses for the
  order test, plus 3 off-grid conditions.
- **Split (Regime III):** additionally observe 6 triples; hold out 14.
- All v1 controls retained: MEMO, PERM, oracle (direct encoding),
  transductive bound, contamination audit, seed reproducibility.

### 5.2 S — Synthetic tiers (primary mechanism tests, cheap)

- **S1 additive control** (sanity), **S2 interacting** (designed pairwise and
  triple interactions, ground-truth γ known), **S3 non-commutative** (ordered
  pairs with exactly computable commutators, e.g., quantize↔noise,
  nonlinearity↔noise; ground-truth sign and magnitude known). Run FIRST —
  these decide H2′/H3′ with known ground truth and no saturation.
- S3 doubles as the **order-supervision source** for transfer to V2.

### 5.3 N — Natural shift (unchanged role: transfer only, RQ6)

---

## 6. Experiment matrix

| Exp | Question | Methods | Benchmark | Regime |
|-----|----------|---------|-----------|--------|
| E1 | Sanity + interaction test | M-A, M-A-additive, M-B, M-C, M0, M1-beh | S1, S2 | II |
| E2 | Order direction test | M-B, M-C(order), M5(v1), commutative control | S3 | II + order supervision |
| E3 | Main zero-shot test | M-A, M-B, M-C, M0, M1-beh, M5(v1), M1-emb(v1) | V2 | II |
| E4 | Higher-order supervision | M-A, M-B | V2 | II vs III (H5′) |
| E5 | Transfer | M-B, M-C | S→V2 | I-revised (H6′) |
| E6 | Natural transfer | M-B | N | IV |
| E7 | Ablations | M-B ± low-rank, ± identity-residual, ± L_assoc, ± behavior supervision; M-A ± interaction | V2 | II |
| E8 | Controls | MEMO, PERM, oracle, transductive | V2 | II |

Ablation list (pre-registered): remove interaction term; remove order
pathway / use commutative operator; reconstruction-only supervision; no
pairwise training; latent dim ∈ {16, 32, 64, 128}; operator rank ∈ {4, 8, 16};
full vs low-rank matrices; λ_repr ∈ {0, 0.1, 0.3}.

## 7. Metrics and statistics (kept + extended)

Kept from v1: BPE, CGG, RCE, rank-ρ (Spearman on primary behavior
component), IRE in behavior space, paired Wilcoxon + bootstrap CI over
held-out conditions, 5 seeds.

Added:
- **Sign-match rate** for reversed pairs (H3′): fraction of held-out reversed
  pairs where sign(∆B̂_X∘Y − ∆B̂_Y∘X) = sign(measured).
- **Commutator-ρ** against encoder-measured ground truth (as v1).
- **Associativity error** of the operator product on held-out triples
  (zero-shot): ‖T_{A∘B}T_C − T_A T_{B∘C}‖ (H-structure check).
- **Per-severity breakdown**: metrics split by severity band (low/mid/high)
  to expose where each method wins/loses (fixes the saturation blind spot).
- **Calibration**: reliability of predicted degradation as a *decision*
  tool (predicted vs measured, binned).
- **Lipschitz estimate** of the learned composition operator (max Jacobian
  norm over training pairs) + **error compounding curve** vs composition
  order |S| (theory validation, §8).

## 8. Theory track (parallel, publishable regardless)

- Conditional theorem (from the manuscript's Eq. 71 direction): if the true
  shift family admits a representation with an L-Lipschitz composition
  operator, error compounds boundedly; **measure L empirically** for M-B
  (operator product) vs M5 (MLP) — does structure reduce L?
- Test whether operator-product composition satisfies the semigroup property
  approximately (associativity error) and whether that predicts lower
  compounding error (Deep-OSG-style claim, literature item 3).
- Deliverable: a short theory section with the conditional bound + empirical
  L / compounding measurements. This is publishable even if all empirical
  hypotheses fail (honest framing: "we characterize when composition
  generalizes and show current operators do not satisfy the conditions").

## 9. Phases, timeline, compute

| Phase | Content | Est. time (2-core CPU) |
|-------|---------|------------------------|
| P0 | Severity recalibration pilot (choose grids; verify no saturation) | 1–2 h |
| P1 | Synthetic tiers S1–S3 + E1/E2 (decides H2′/H3′ mechanics) | 1–2 days |
| P2 | V2 pools (6 prim × 5 sev + 15 pairs × 5 sev observed; 20 triples + 16 ordered + 3 off-grid held out; behavior on full test pool) | ~1 h |
| P3 | E3 main comparison (7 methods × 5 seeds) + E7 ablations | ~1–2 h |
| P4 | E4 (Regime III), E5 (transfer), E6 (natural) | ~1–2 h |
| P5 | E8 controls + stats + theory measurements + writeup | ~1 h |

Total: ~1 week of wall time, well within the v1 compute budget (base model
reused; no retraining needed). Code reuse: v1 pipeline phases
(build-pools/train-methods/evaluate/audit) are extended, not rewritten;
new modules: behavior-space composition, operator-product algebra, order
tokens, recalibrated config, S-tier generator.

## 10. Fallbacks (decided in advance)

1. **If H1′ confirms (M-A or M-B beats M0/M3, rank-ρ ≥ 0.8):** paper =
   method + benchmark + theory measurements. Strong result.
2. **If only H3′ confirms (order direction recovered, magnitude not):**
   paper = "compositional structure exists but magnitude prediction is
   unsolved" — partial claim, still publishable.
3. **If H1′ fails but H2′/H5′/H6′ give partial signal:** paper =
   benchmark + ablations + honest negative on the main claim.
4. **If everything fails:** publish v1 + v2 negative results + theory
   characterization as a falsification paper (the manuscript's own framing
   explicitly allows this: "if the answer is no, the failure would still
   clarify an important limitation").

## 11. What changes in the manuscript

- Method section: replace the unanchored embedding-space algebra with
  behavior-anchored composition (M-A) + operator-product algebra (M-B);
  attention-set composition (M-C) becomes a first-class variant.
- The antisymmetric MLP pathway is demoted to an ablation (it failed D3).
- Benchmark section: recalibrated severities, 6 primitives, Regime III,
  primary behavior vector without the flooring component.
- Results: v1 negative result reported as "pilot experiment"; v2 as main.
- Theory: conditional bound + empirical Lipschitz/compounding measurements.

*End of plan.*
