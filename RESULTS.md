# Shift Algebra — Experiment Results (Benchmark V, CIFAR-10)

**Status: COMPLETE.** Full pipeline run end-to-end with pre-registered decision
rules. **Headline: the central hypothesis (H1) is NOT supported on this
benchmark.** The proposed Shift Algebra method is significantly *worse* than
simpler baselines at zero-shot prediction of unseen shift compositions. Per the
manuscript's own falsification criteria, the project is unsuccessful. Details
below — this is a negative result and it is reported as such.

## 0. What was run (amendments to the plan, pre-registered in code)

| Item | Plan | As run | Why |
|------|------|--------|-----|
| Base model | ResNet-18, ~93.2% | ResNet-18 **width 0.25** (0.70M params), **91.04%** clean test acc | Amendment 1: sandbox has 2 CPU cores, no GPU; full-width training infeasible. Protocol unchanged. |
| Dataset | CIFAR-10 | Official CIFAR-10 (GitHub mirror; **all 7 file md5s verified** against the torchvision reference fetched from pytorch/vision) | Canonical host blocked; integrity proven by checksum. |
| Split | Regime II revised (App. A.3) | Exactly as specified: 45 observed (4 prim × 5 sev + 5 pairs × 5 sev), 33 held-out (B◦C, C◦B × 5 sev; 4 triples × 5 sev; 3 off-grid) | — |
| Seeds | 5 | 5 (1234–1238); base model fixed seed | — |
| Pool for training labels | train sample (2k) | train sample (2k) | Appendix A.4 leakage boundary |
| Pool for ground truth | full test pool (10k) | full test pool (10k) | — |

Environment: 2 CPU cores, 3.8 GB RAM, PyTorch 2.2.2 CPU. Base training ~1.5 h;
full pipeline (incl. training) ~2.5 h.

## 1. Sanity of the measured behavior (test pool)

- Clean: acc **91.04%**, ECE 0.031, mean conf 0.940.
- Primitive degradations are strong: blur σ=1.5 → 18.7% acc (−72.3pp);
  noise σ=0.15 → 16.5% (−74.5pp).
- **Behavior-level non-commutativity is large and physical:**
  B∘C (blur→noise) = **12.19%** acc vs C∘B (noise→blur) = **18.40%** acc —
  a **6.2pp** order effect in the direction physics predicts (smoothing the
  noise after adding it is less harmful than adding sharp noise to a blurred
  image).
- **Saturation:** at mid/high severities most conditions sit near the
  accuracy floor (12–22%), so ∆B is compressed there; the discriminative
  signal concentrates at low severities. This is a benchmark property to
  remember when reading the metrics (see §6).
- Training labels (train-sample) vs ground truth (test-pool) are on the same
  scale: per-condition correlations 0.999–1.000, mean ratios 0.91–0.99. The
  pool mismatch is negligible.

## 2. Method comparison (33 held-out conditions, mean over 5 seeds)

| method | BPE ↓ | acc MAE ↓ | rank ρ ↑ | CGG ↓ | RCE | OSI-abs ↓ | beh-order-err ↓ | IRE pair ↓ |
|--------|-------|-----------|----------|-------|-----|-----------|-----------------|------------|
| **M3 attention** | **0.236** | 0.080 | **0.881** | **0.236** | **8.4** | 1.155 | 1.073 | **0.460** |
| **M0 no-composition** | 0.381 | **0.076** | 0.821 | 0.381 | 14.1 | 2.032 | 1.330 | 0.580 |
| M2 MLP | 0.481 | 0.150 | 0.793 | 0.481 | 17.2 | 1.533 | 0.873 | 0.852 |
| M6 Shift Algebra (λ=0) | 0.536 | 0.078 | 0.647 | 0.536 | 19.3 | 2.032 | 1.330 | 0.738 |
| **M5 Shift Algebra** | 0.962 | 0.116 | 0.536 | 0.962 | 34.9 | **1.072** | **0.757** | 1.844 |
| M1 additive | 1.828 | 0.342 | 0.708 | 1.828 | 66.3 | 2.032 | 1.330 | 3.313 |
| M8 recon-only | 2.227 | 0.265 | 0.111 | 1.858 | −5.5 | 2.032 | 1.330 | 1.767 |
| MEMO control | 2.240 | 0.265 | n/a (const) | 2.240 | 80.3 | 2.032 | 1.330 | 2.417 |
| PERM control | 3.063 | 0.277 | 0.131 | 2.710 | −8.0 | 1.113 | 1.268 | 4.376 |
| **oracle (not zero-shot)** | **0.028** | — | — | — | 0 | — | — | — |

## 3. Statistical comparisons vs M5 (paired Wilcoxon, 33 conditions;
seed-averaged per-condition squared error; diff = M5_err − other_err,
so **positive = M5 worse**)

| comparison | mean diff | 95% CI | p |
|------------|-----------|--------|---|
| M5 vs **M0 no-composition** | **+0.581** | [+0.221, +1.004] | **0.046** |
| M5 vs **M2 MLP** | **+0.481** | [+0.234, +0.753] | **0.0093** |
| M5 vs **M3 attention** | **+0.726** | [+0.412, +1.082] | **0.00010** |
| M5 vs M6 no-order | +0.426 | [+0.048, +0.868] | 0.19 (n.s.) |
| M5 vs M1 additive | −0.866 | [−1.402, −0.457] | 5.5e-07 |
| M5 vs M8 recon-only | −1.265 | [−1.946, −0.523] | 0.0022 |
| M5 vs MEMO control | −1.278 | [−1.961, −0.535] | 0.0019 |
| M5 vs PERM control | −2.102 | [−2.946, −1.315] | 3.1e-05 |

M5 is **significantly worse** than M0, M2 and M3; significantly better than
M1, M8, MEMO, PERM.

## 4. Decision rules (pre-registered, plan §7)

| rule | result |
|------|--------|
| H1: M5 CGG < M0 CGG | **FAIL** (0.962 vs 0.381; p = 0.046) |
| H2: M5 BPE < M1 additive | PASS (0.962 vs 1.828; p = 5.5e-07) |
| H2: M5 IRE_pair < M1 | PASS (1.84 vs 3.31) |
| H3: M5 OSI-abs < M6 (λ=0) | PASS (1.07 vs 2.03) |
| H3: commutator ρ ≥ 0.7 | **FAIL** (M5 ρ = 0.10; M3 = 0.80) |
| H4: M5 BPE < M8 recon-only | PASS (0.962 vs 2.227; p = 0.0022) |
| H4: rank ρ ≥ 0.8 | **FAIL** (M5 ρ = 0.536; M3 = 0.881) |
| control: MEMO fails | Substantive PASS, threshold FAIL (see §5) |
| control: PERM fails | PASS |
| oracle is an upper bound | PASS (oracle 0.028 ≤ all zero-shot) |

## 5. Controls

- **MEMO (memorization) control:** memorizes the 45 observed conditions
  perfectly in-sample (BPE 3e-06) but predicts a *constant* for all held-out
  conditions: BPE 2.24 (2.3× M5), acc-MAE 0.265 (2.3× M5), rank ρ undefined.
  It substantively fails to generalize. The pre-registered quantitative
  threshold (>3× M5 BPE) was not crossed (observed 2.33×) — a threshold
  calibration miss, not a validity problem: the control's purpose (prove the
  held-out test requires composition) is fulfilled, and M5 beats it
  significantly (p = 0.0019).
- **PERM (permutation) control:** composition targets shuffled during
  training → BPE 3.06, rank ρ 0.131; M5 beats it decisively (p = 3.1e-05).
  Confirms the evaluation signal is compositional structure.
- **Oracle (direct encoding, not zero-shot):** BPE 0.028 — ~8.4× better than
  the best zero-shot method (M3, 0.236). The zero-shot composition problem is
  far from solved; there is a large, well-defined gap to close.

## 6. Findings (honest reading)

1. **H1 is not supported.** Explicit shift composition (M5) is significantly
   *worse* than mean-pooling primitive embeddings (M0) and much worse than
   cross-attention composition (M3) at predicting the behavior of unseen
   compositions. On this benchmark, the degradation of a composed shift is
   largely predictable from the primitive degradations alone, and the learned
   latent composition operator does not add value — it overfits the 25
   observed pairs and misgeneralizes to held-out pairs/triples.
2. **The proposed method overpredicts degradation and even predicts
   below-chance accuracy** on the flagship triple: measured A◦B◦C accuracy
   13.22% (∆acc −77.8pp) vs M5-predicted 5.11% (−85.9pp). Predictions are
   systematically too pessimistic, worst in the saturated regime.
3. **The order pathway captures an order effect but not the true one.**
   Embedding-level order sensitivity is best for M5 (OSI-abs 1.07; commutative
   methods score 2.03 = predict zero order difference), and behavior-level
   order error is also lowest for M5 (0.757 vs 1.330). **But the direction is
   wrong for the flagship pair:** measured ∆acc(B∘C) − ∆acc(C∘B) = **−6.2pp**
   (blur→noise worse), M5 predicts **+21.2pp** (the opposite sign), and the
   embedding-level commutator ranking correlates only ρ = 0.10 with ground
   truth (M3: 0.80). So H3's mechanism works mechanically but does not
   recover the true non-commutativity on this benchmark.
4. **H2's win over additive composition is real but not the point:** M5
   beats M1 (p = 5.5e-07), yet M0 and M3 beat M1 by even more. Additive
   composition in embedding space is actively harmful here (BPE 1.83) —
   embedding-space addition does not correspond to behavior-space addition.
5. **Behavior supervision helps representations (H4, partial):** M5 beats
   reconstruction-only M8 significantly (p = 0.0022). But the absolute
   prediction quality is mediocre (rank ρ 0.536 vs the 0.8 target).
6. **Best zero-shot method is cross-attention (M3):** BPE 0.236, rank ρ 0.881,
   best IRE and commutator recovery. A positive finding for the research
   program: *structured* composition operators generalize to unseen
   compositions better than additive ones — but the specific
   symmetric+antisymmetric MLP parameterization proposed in the manuscript is
   not the winning one on this benchmark.
7. **Benchmark property — saturation:** at mid/high severities the behavior
   saturates near the accuracy floor (12–22%), compressing the compositional
   signal; most metric variance comes from low-severity conditions. A revised
   benchmark should use milder severities (or a behavior vector less prone to
   flooring, e.g. logit/feature-space drift without accuracy).
8. **Verdict per the manuscript's own falsification criteria (XII):**
   *“no consistent benefit exists on unseen compositions”* — **triggered**
   (M5 shows no benefit over M0/M3; it is significantly worse). By the
   pre-registered criteria the project is **unsuccessful**. The negative
   result is still informative and publishable in the manuscript's own
   framing ("if the answer is no, the failure would still clarify an
   important limitation of compositional approaches to robustness").

## 7. What a revised approach would need (for the record)

- Composition operators that generalize from 25 pairs to unseen pairs/triples
  (attention-style set aggregation worked best here; the MLP pair-operator
  overfits).
- Behavior labels/predictions robust to the accuracy floor (predict
  logit/feature drift; use milder severities).
- Order modeling that recovers the *direction* of non-commutativity, not just
  its existence (the antisymmetric pathway currently encodes an arbitrary
  order effect).
- A training signal that ties the latent composition to *behavior-space*
  composition (the current L_repr trains embedding-space reconstruction,
  which is gauge-ambiguous — see the plan's identifiability section).

## 8. Artifacts

- `results/behavior_testpool.csv` — measured behavior for all 78 conditions
  (full 10k test pool).
- `results/metrics_summary.csv`, `results/stats_vs_m5.csv`,
  `results/summary.json` — metrics, statistics, decision rules.
- `results/preds/<method>.npz` — per-seed zero-shot predictions.
- `results/models/M5_seed*.pt` — trained Shift Algebra modules (for the
  oracle / reproducibility checks).
- `results/REPORT.txt` — formatted summary (`python -m src.report`).
- `results/audit_report.json` — automated audit results.
- `AUDIT.md` — post-hoc code review (info-leak and correctness investigation).
- Code: `src/` (config, corruptions, data, models, shift_algebra,
  experiment, report). Data/checkpoints/pools are gitignored (large).
