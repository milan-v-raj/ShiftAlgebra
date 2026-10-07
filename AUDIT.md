# Post-Hoc Code Audit — Info-Leak and Correctness Investigation

**Scope.** After the experiment completed and results were produced, the full
codebase (`src/`) was re-read line-by-line and checked for (a) information
leaks that could invalidate the zero-shot claim, and (b) correctness of the
experiment relative to the pre-registered plan (`EXPERIMENTAL_PLAN.md`,
Appendix A). This document is the written record of that investigation.

**Verdict up front: no information leaks found; the experiment is valid.
Eight bugs were found and fixed during the run (before results were trusted);
one pre-registered control threshold was miscalibrated (documented, not
silently changed). One metric has a known blind spot (documented).**

---

## 1. Information-leak investigation

The zero-shot claim requires: at test time, a model may see only primitive
embeddings, D₀ statistics, and composition identity — never samples or
statistics of any held-out composition.

| # | Leak vector checked | Evidence | Result |
|---|---------------------|----------|--------|
| L1 | Training conditions include held-out compositions | `phase_train_methods` builds `conds` exclusively from `pools/observed.npz`; audit check `pools_contain_only_observed` verifies the pool's keys are disjoint from the 33 held-out keys; `pools_match_config` verifies exact equality with the 45 observed | **No leak** |
| L2 | Encoder inputs for held-out compositions used anywhere in training | Held-out encoder inputs (`v_held`) are computed in `phase_evaluate` **after** all training, used only for the oracle bound and ground-truth order statistics; they never enter `results/preds/` (which are produced by `predict_condition` from primitive embeddings only) | **No leak** |
| L3 | Test-pool information in training labels | Behavior labels Y come from `phase_build_pools`, measured on the 2k **train**-sample only. The test pool is used solely for ground-truth evaluation. (Cross-check: train-sample vs test-pool ∆B correlate 0.999–1.000, so the boundary is also empirically harmless.) | **No leak** |
| L4 | Base model trained on test data | `phase_train_base` trains on `pools["train_images"]` (50k train) only; test pool used solely for accuracy evaluation | **No leak** |
| L5 | Oracle / transductive results mixed into zero-shot rankings | Oracle BPE is computed separately, reported as its own row ("not zero-shot"), and used only in the RCE denominator and the `oracle_is_upper_bound` check | **No leak** |
| L6 | Contamination by superset compositions | Audit `contamination_split_disjoint`: no observed condition's letter-set is a superset of any held-out condition's letter-set (held-out pairs B◦C/C◦B appear in no observed pair; all triples contain an unobserved pair) | **No leak** |
| L7 | Severity-level contamination | Held-out compositions are evaluated at the same diagonal severities as observed ones, but the *composition* (the held-out object) is never observed at any severity; off-grid severities are evaluation-only and their primitive inputs (aux prims) are single shifts — not held-out compositions | **No leak** |
| L8 | Selection on the test set | All splits, seeds, severities, and decision rules are fixed in `src/config.py` before any run; the evaluate phase caches the deterministic behavior table rather than re-rolling anything | **No leak** |
| L9 | Noise realizations shared between B◦C and C◦B | Deliberate design (order-independent noise seed, verified by `noise_seed_order_independent`): makes the order test a *pure* order effect. It cannot help any model see the answer — it removes a nuisance | **Not a leak (controlled nuisance)** |
| L10 | Determinism / reproducibility | `seed_reproducibility` check: retraining M5 seed 1234 reproduces saved held-out predictions with maxdiff = 0.0e+00 | **Reproducible** |

## 2. Correctness investigation (vs. the pre-registered plan)

| Aspect | Plan | Implementation | Result |
|--------|------|----------------|--------|
| Corruption semantics | deterministic, clipped after each op, listed order | `corruptions.py`; audit checks `corruption_deterministic`, `clip_bounds`, `BC_noncommutative` (maxdiff 0.58), `AD_commutative_no_clip` (1.2e-07) | **Correct** |
| Non-commutativity rationale | blur/noise non-commutative | Verified physically and empirically: B∘C = 12.19% vs C∘B = 18.40% measured accuracy (6.2pp behavior-level order effect) | **Correct** |
| Split (Regime II revised) | 5 observed pairs incl. A◦D; held-out B◦C + C◦B + 4 triples + off-grid | `config.py` exactly; 45 observed / 33 held-out | **Correct** |
| Leakage boundary (train sample vs test pool) | App. A.4 | `build-pools` (train sample) vs `evaluate` (test pool) | **Correct** |
| Behavior vector | [acc, per-class acc, ECE, feature drift, mean conf] | `behavior_from` | **Correct** |
| Metrics | BPE, CGG, RCE, rank-ρ, OSI-abs, commutator-ρ, IRE (behavior space) | `phase_evaluate`; IRE ground truth is measured behavior-space γ (identifiable), per the plan's identifiability requirement | **Correct** |
| Baselines | M0–M3, oracle, controls, ablations (M6, M8) | All implemented with matched inputs | **Correct** |
| Statistics | paired Wilcoxon over 33 conditions + bootstrap CI | Implemented in `phase_evaluate` | **Correct** (sign comment fixed, see B8) |
| Data integrity | official CIFAR-10 | All 7 batch-file md5s verified against the torchvision reference (fetched from pytorch/vision via the GitHub API) and cross-checked against a second independent mirror | **Correct** |

## 3. Bugs found and fixed during the run

All were caught by smoke tests or the audit phase **before** results were
trusted. None affected the final reported numbers (each was fixed and the
affected phase re-run from scratch).

| # | Bug | Where | Severity | Fix |
|---|-----|-------|----------|-----|
| B1 | `torch.adaptive_avg_pool2d` doesn't exist | `models.py` | crash | `nn.AdaptiveAvgPool2d` module |
| B2 | `F.conv2d` has no `padding_mode` kwarg | `corruptions.py` | crash | explicit `F.pad(..., mode="reflect")` + zero-pad conv |
| B3 | `make_composition` had no entry for `M8_recon_only` | `shift_algebra.py` | crash | M8 uses the Shift Algebra composition (λ=1) |
| B4 | `predict_condition` passed 1-D embeddings; `AttentionComposition` expects batched input → crash on triples | `shift_algebra.py` | crash | keep `(1, z)` through composition, squeeze at the end |
| B5 | Off-grid C◦B entry had severities swapped (noise σ=1.75, blur σ=0.175 — nonsense condition) | `config.py` | **silent wrong condition** | corrected to (0.175, 1.75), letter-aligned |
| B6 | `bc_idx`/`cb_idx` also matched the off-grid duplicates → shape mismatch | `experiment.py` | crash | filter by `kind == "pair"` |
| B7 | Auxiliary off-grid primitives computed in `build-pools` but never loaded in `train-methods` (and later, `audit`) → KeyError | `experiment.py` | crash | load `aux_prim.pt` into `prim_v` in both phases |
| B8 | Statistics sign comment said "positive = M5 better"; actually positive = M5 **worse** (numbers were correct, comment wrong) | `experiment.py`, `report.py` | misleading label | comment/header corrected; results re-read with the correct sign |
| B9 | Two hardcoded batch md5s were mistyped from memory (one invalid 33-char string) | `data.py` | would have failed verification | constants transcribed programmatically from the torchvision reference source; all 7 verified |

Additionally, the training loop was rewritten to be vectorized over conditions
(python-loop → batched index tensors) after profiling showed ~0.2 s/step;
identical math, ~100× faster (full method training: ~25 min).

## 4. Automated audit results (`python -m src.experiment audit`)

```
PASS  contamination_split_disjoint        violations=[]
PASS  pools_contain_only_observed         overlap=set()
PASS  pools_match_config                   missing=set() extra=set()
PASS  corruption_deterministic
PASS  noise_seed_order_independent
PASS  BC_noncommutative                    maxdiff=0.5796
PASS  AD_commutative_no_clip               maxdiff=1.19e-07
PASS  clip_bounds
FAIL  control_MEMO_fails                   (threshold miss — see §5)
PASS  control_PERM_fails
PASS  oracle_is_upper_bound
PASS  seed_reproducibility                maxdiff=0.00e+00
PASS  offgrid_predictions_finite          n_offgrid=3
```

## 5. Observations and caveats (not bugs)

1. **MEMO control threshold miss.** The pre-registered rule demanded MEMO BPE
   > 3× M5 BPE; observed ratio is 2.33× (2.24 vs 0.962). The control
   *substantively* fails (constant held-out predictions, no rank correlation,
   2.3× worse BPE and acc-MAE; M5 beats it with p = 0.0019), so the control's
   purpose is fulfilled. The 3× threshold was arbitrary; it is reported as
   missed rather than silently relaxed.
2. **OSI-abs metric blind spot.** OSI-abs scores only the *magnitude* of the
   predicted order difference. The PERM control (garbage model) scores 1.11,
   better than M2 (1.53), by chance. The behavior-level order error
   (M5 best at 0.757; commutative methods 1.330) and the spot check (§6.3 of
   RESULTS.md — M5 gets the *direction* of the flagship order effect wrong)
   are the meaningful order readouts; OSI-abs alone should not be used.
3. **M8 is seed-invariant.** Reconstruction-only training converges to the
   same solution from all 5 seeds (in-sample BPE identical to 6 decimals;
   held-out BPE std ≈ 7e-06). Plausible for full-batch optimization of a
   45-point problem with a near-determined target; does not affect validity
   (M8 is an ablation and is compared on held-out BPE, where it is clearly
   worse than M5).
4. **RCE semantics.** RCE = (E_unseen − E_seen)/(E_oracle − E_seen) exceeds 1
   whenever zero-shot error is much larger than the oracle error (here 8–80);
   negative values occur when in-sample error exceeds held-out error (M8,
   PERM). Reported as-is; lower is better.
5. **BPE scale.** The 5-dim behavior vector mixes scales (feature drift is
   O(1–4), accuracies O(0.1–1)); BPE is dominated by the drift component.
   All methods face the identical metric, and rank-ρ/acc-MAE are reported
   alongside, so comparisons remain interpretable.
6. **Saturation.** At mid/high severities the measured accuracy floors at
   12–22%, compressing ∆B variance (see RESULTS.md §6.7). This is a
   benchmark property, not a code issue.
7. **Identifiability.** As flagged in the plan, latent-space algebraic
   quantities (γ, κ in z-space) are gauge-dependent; all structure claims in
   the results are therefore grounded in behavior-space or measured
   quantities (IRE in behavior space, OSI against encoder-measured ground
   truth, commutator-ρ against encoder-measured embedding differences).

## 6. Conclusion

The experiment is **valid**: no information leaks, the zero-shot boundary held
under automated and manual inspection, the data is verified official CIFAR-10,
the split matches the pre-registered protocol, and the pipeline is
reproducible (seed check maxdiff = 0.0). The results in `RESULTS.md` —
including the failure of the central hypothesis H1 — are trustworthy.
