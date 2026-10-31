# Phase 2B Final Report — Adaptive Composition Hazard Mixture

**Protocol:** `V4-P2B-ACHM-CPU-001`  
**Decision:** **ACHM PRIMARY GATE FAILED; useful MAE improvement retained**  
**Scope:** completed CPU-only fresh-confirmation pilot; not the roadmap's final 75+ model Phase 2

## Executive result

ACHM reduced fresh accuracy MAE from **1.719pp** for the preregistered ordered-hazard comparator to **1.593pp**, a **7.3% relative reduction**. The direction was favorable, and ACHM was especially better on length-5 and length-6 words. However, the clustered 95% interval for the 0.126pp improvement was **[-0.025, 0.295]pp**, controlled model-rank Spearman was lower (**0.811 versus 0.822**), and therefore the complete primary gate failed.

The previously frozen Phase-2 metadata-plus-composition regression `g1` remained strongest: **1.241pp MAE**, model-rank Spearman **0.789**, and controlled Spearman **0.827**. ACHM is not a replacement for `g1` on this evidence.

## Why this is a fresh test

The ACHM implementation, fresh-word manifest, gates, active budgets, metrics, and failure policy were committed and pushed before fresh composition outcomes existed:

1. `834ad92` — protocol, fresh manifest, development-only fit, and frozen ACHM;
2. `12f672d` — independent verification script;
3. `d41f334` — 15 fresh CPU training lineages recorded before behavior evaluation;
4. fresh behavior evaluation opened only afterward.

The gate loaded only `calibration` and `method_development` roles from the original Phase-2 development CSV. It did not load the already opened Phase-2 confirmation file. The old confirmation motivated the new hypothesis but provides no confirmation evidence for it.

## Fresh bank and targets

- 15 new training lineages and 30 epoch snapshots;
- five lightweight architecture families and three recipes;
- seeds 42100--42502, disjoint from prior model banks;
- 1,000 balanced CIFAR-10 test images at per-class offsets 200--299;
- image partition disjoint from the Phase-2 development and confirmation behavior partitions;
- 37 calibration conditions;
- 64 new words, 16 at each length 3--6;
- zero overlap with the original 167 words;
- every word paired with its reversal.

Fresh clean accuracy ranged from **30.6% to 52.2%**, with mean **41.0%**. These remain weak two-epoch CPU models.

## Primary and strong-baseline results

| Method | MAE ↓ | Model-rank Spearman ↑ | Controlled Spearman ↑ | Top-1 regret ↓ | Within 1pp ↑ |
|---|---:|---:|---:|---:|---:|
| Prior frozen `g1` | **1.241pp** | **0.789** | **0.827** | **1.316pp** | **35.9%** |
| **ACHM** | **1.593pp** | 0.755 | 0.811 | 1.606pp | 28.1% |
| Symmetric hazard | 1.714pp | 0.763 | 0.822 | 1.606pp | 28.1% |
| Ordered hazard (primary comparator) | 1.719pp | 0.762 | 0.822 | 1.650pp | 28.1% |
| Prior frozen `g0` | 3.043pp | 0.214 | -0.467 | 1.356pp | 29.7% |
| Independent retention | 5.356pp | 0.486 | 0.592 | 4.873pp | 1.6% |
| Dominant shift | 12.328pp | 0.272 | 0.530 | 6.263pp | 7.8% |

ACHM improved MAE and slightly improved top-1 regret over ordered hazard, but lowered raw and controlled rank. Its MAE was also worse than the prior `g1` by 0.352pp; equivalently, `g1` reduced MAE by 22.1% relative to ACHM.

## Preregistered primary gate

| Criterion | Result | Decision |
|---|---:|---|
| ACHM MAE lower than ordered hazard | 1.593pp < 1.719pp | PASS |
| Clustered 95% CI excludes zero | [-0.025, 0.295]pp | **FAIL** |
| Controlled rank higher | 0.811 < 0.822 | **FAIL** |
| Top-1 regret no worse | 1.606pp < 1.650pp | PASS |
| Complete gate | all four required | **FAIL** |

The bootstrap used 10,000 paired draws clustered by fresh reverse-word group and training lineage. Epoch snapshots remained inside their lineage cluster.

## Length behavior

| Length | ACHM MAE | Ordered-hazard MAE | Symmetric-hazard MAE | Prior `g1` MAE |
|---:|---:|---:|---:|---:|
| 3 | 1.128pp | 1.130pp | 1.130pp | 1.453pp |
| 4 | 1.648pp | 1.712pp | 1.709pp | **1.056pp** |
| 5 | 1.671pp | 1.813pp | 1.807pp | **1.145pp** |
| 6 | 1.924pp | 2.220pp | 2.210pp | **1.309pp** |

ACHM's adaptive mixture increasingly helped against a fixed hazard as length grew, but the prior metadata-calibrated `g1` remained much stronger for lengths 4--6. The length breakdown is descriptive and does not rescue the failed primary gate.

## What the gate learned

Mean expert weights were:

| Expert | Overall weight | Length 3 | Length 6 |
|---|---:|---:|---:|
| Dominant shift | 2.9% | 0.4% | 6.6% |
| Independent retention | 7.7% | 1.3% | 16.5% |
| Symmetric compounding | **80.8%** | **93.7%** | **64.9%** |
| Ordered compounding | 8.7% | 4.6% | 12.0% |

Thus the gate mostly learned a regularized symmetric law, with more dominant/independent mass on longer words. This modest interpolation explains why MAE improved without producing a substantially different model ranking.

## Pair-calibration budget

| Unordered pair groups | Directed measurements | ACHM MAE | Controlled rank | Top-1 regret |
|---:|---:|---:|---:|---:|
| 0 | 0 | 5.357pp | 0.592 | 4.873pp |
| 5 | 10 | 3.431pp | 0.693 | 2.673pp |
| 10 | 20 | 1.726pp | 0.798 | 1.811pp |
| 15 | 30 | **1.593pp** | **0.811** | **1.606pp** |

The curve shows that pair calibration is useful and that ten of fifteen bidirectional groups recover much of the full-budget MAE. Because this pilot did not preregister random-selection or optimal-design comparators, it does **not** establish that the greedy selector is optimal or better than random selection.

## Order and uncertainty diagnostics

Order remains unsupported as a headline:

- ACHM reversal MAE: **0.402pp**;
- symmetric reversal MAE: **0.404pp**;
- measurable reversal sign accuracy: **61.2%**.

ACHM barely changed reversal MAE and sign accuracy remained far below the planned 80% gate.

Expert disagreement also failed as an uncertainty score:

- disagreement/error Spearman: **-0.059**;
- selective MAE worsened as coverage decreased;
- 100% coverage MAE was 1.593pp, while 40% lowest-disagreement coverage MAE was 1.887pp.

Therefore disagreement must not be presented as calibrated uncertainty. A future uncertainty method requires explicit calibration, not this heuristic.

## Scientific interpretation

The fresh result supports a narrow statement: **adaptive convex mixing can reduce the pointwise MAE of a fixed hazard law, particularly on longer words.** It does not support the stronger claim that ACHM is the best forecasting method or that its improvement is statistically established across independent clusters. The earlier `g1` result transferred better and remains the strongest CPU evidence.

The appropriate next method question is not to tune ACHM on these outcomes. This entire Phase-2B confirmation is now opened. Any revision requires another fresh test. Under the current CPU constraint, the responsible stopping point is to retain this negative gate, publish the budget curve, and avoid a rescue run.
