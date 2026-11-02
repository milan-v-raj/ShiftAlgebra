# Phase 3A Final Report — Realistic-Corruption Composition Confirmation

**Protocol:** `V4-P3A-CIFAR100-REALISTIC-CORRUPTIONS-001`  
**Decision:** **REALISTIC COMPOSITION GATE FAILED (6/7 criteria passed)**  
**Interpretation:** strong favorable forecasting evidence was retained, but the preregistered all-required gate failed because accuracy-selection regret was slightly worse.

## Scope

Phase 3A tested the primary order-blind symmetric composition forecaster on six deterministic clean-room common-corruption categories: brightness, Gaussian noise, JPEG-style block-DCT quantization, diagonal motion blur, pixelation, and a shot-noise approximation. These are **ImageNet-C-style categories**, not official ImageNet-C implementations or official severity levels.

The development bank comprised 36 reconstructed Phase-2H snapshots evaluated on CIFAR-100 class offsets 80--89 and 60 development words. The frozen method was then tested once on 18 fresh lineages/36 snapshots, class offsets 90--99, and 60 disjoint confirmation words. Each target set contained 30 exact reversal pairs. Calibration comprised identity, six singles, and all 30 directed distinct-primitive pairs.

## Prospective chronology

1. Protocol, word sets, corruptions, model roster, gates, and code were committed at `d32633d` before Phase-3A behavior was opened.
2. Development behavior was opened; metric-specific exponents and ridge coefficients were frozen and committed at `98d4a13`.
3. Fresh confirmation checkpoints were trained, hashed, and committed with behavior still unopened at `baec63e`.
4. Confirmation behavior was then evaluated once. No method, coefficient, exponent, scale, corruption, split, word, or gate was retuned.

## Primary errors

All MAEs are in their native units. Macro error divides each coordinate by its development-only target standard deviation and averages the six coordinates.

| Coordinate | Metadata/clean `g0` MAE | Fitted symmetric MAE | Relative reduction | Hierarchical normalized improvement 95% CI |
|---|---:|---:|---:|---:|
| Margin | 0.007467 | **0.006688** | 10.4% | [-0.00784, 0.10135] |
| Feature drift | 0.679988 | **0.550282** | 19.1% | [0.03543, 0.23368] |
| ECE | 0.018555 | **0.015983** | 13.9% | [0.01142, 0.19839] |
| Balanced accuracy | 0.012737 | **0.009312** | 26.9% | [0.07285, 0.20257] |
| Accuracy | 0.012737 | **0.009312** | 26.9% | [0.07285, 0.20257] |
| Mean confidence | 0.012636 | **0.011019** | 12.8% | [0.00489, 0.10063] |

The fitted symmetric method improved all six point MAEs. Its macro normalized MAE fell from **0.576501 to 0.471656**, an **18.2% relative reduction**. The family/lineage/reversal hierarchical improvement was **0.104845**, 95% CI **[0.050407, 0.154644]**. Five coordinate intervals excluded zero; margin's coordinate-specific interval did not.

Balanced and overall accuracy are numerically redundant on this class-balanced partition, but both were retained exactly as preregistered.

## Comparator results

`g0` uses clean-reference behavior plus model and word metadata but no single- or pair-corruption behavior; “metadata” in the inherited V4 naming includes that clean-reference scalar.

| Method | Macro normalized MAE ↓ | Macro controlled model-rank Spearman ↑ | Accuracy top-1 regret ↓ |
|---|---:|---:|---:|
| Metadata/clean ridge `g0` | 0.576501 | 0.191236 | **0.1083 pp** |
| **Fitted symmetric composition** | **0.471656** | **0.613512** | 0.1367 pp |
| Independent-retention formula | 1.430770 | 0.503191 | **0.0633 pp** |
| Envelope formula | 0.586328 | 0.547733 | 0.8767 pp |
| Unfitted symmetric-pair formula | 0.542695 | 0.557679 | 0.1750 pp |

The fitted method beat every comparator on macro normalized MAE and substantially improved controlled rank over `g0`. The independent formula illustrates that low selection regret does not imply accurate full-vector forecasts: it had the best regret but by far the worst macro error.

## Preregistered gate

| Criterion | Result |
|---|---|
| Macro normalized MAE below `g0` | PASS |
| Hierarchical 95% improvement interval excludes zero | PASS |
| At least four of six coordinate MAEs improve | PASS (6/6) |
| Better macro error than independent and envelope formulas | PASS |
| Macro controlled rank improves over `g0` | PASS |
| Accuracy top-1 selection regret is no worse than `g0` | **FAIL** (0.1367 vs 0.1083 pp) |
| At least five of six family improvements are positive | PASS (6/6) |

Because the gate was conjunctive, the final decision is **FAILED**. The unfavorable selection result is retained without tuning. The absolute regret difference was 0.0283 percentage points, but no tolerance was preregistered, so strict numerical worsening fails the criterion.

## Family, reversal, and robustness behavior

All six architecture-family mean normalized improvements were positive:

- ConvNeXt holdout: 0.143683;
- depthwise holdout: 0.141688;
- dilated holdout: 0.108879;
- residual holdout: 0.098784;
- shuffle holdout: 0.074436;
- gated holdout: 0.061601.

The symmetric model was exactly invariant under reversal. Real outcomes were not: mean absolute reversal gaps were 0.009336 margin, 0.804229 feature drift, 0.022105 ECE, 0.012468 balanced accuracy, 0.012468 accuracy, and 0.016477 mean confidence. This is an explicit limitation of the primary method, retained after Phase 2H showed that order awareness was not a broadly reliable improvement.

The labeled no-retuning sensitivity that removes redundant balanced accuracy retained a positive improvement of **0.098480**, hierarchical 95% CI **[0.040333, 0.147263]**. Improvements were also positive for both epochs, all word lengths, all six families, and all three recipes. These checks do not replace the primary failed gate.

## Correctness and leakage audit

The post-outcome verifier passed **35/35** checks. It independently recalculated all per-metric and macro MAEs, reconstructed the seven gate decisions, checked exact reversal invariance, validated all checkpoint/behavior/forecast/development hashes, matched forecast targets to measured behavior, and confirmed disjoint model, image, and target-word partitions.

The detailed audit is in [`INFORMATION_AUDIT.md`](INFORMATION_AUDIT.md). No confirmation-outcome path into fitting was found. One procedural limitation remains: calibration and target outcomes are stored by the same evaluator before the frozen forecaster writes predictions, so sealing is enforced by committed chronology, method hashes, and audited code flow rather than by a cryptographically isolated prediction service.

## Conclusion

Phase 3A provides favorable evidence that symmetric composition-aware forecasting transfers to this new common-corruption vocabulary on CIFAR-100: all six errors improve, the hierarchical aggregate interval is positive, controlled ranking improves sharply, every family sign is favorable, and the fitted method beats all four requested comparators on macro error. It nevertheless does **not** pass its complete prospective confirmation gate because top-1 accuracy-selection regret worsens slightly.

The justified headline remains narrow: **composition-aware forecasting transfers favorably in error and ranking, while robust selection improvement and order sensitivity are not established.** This phase does not establish official ImageNet-C, ImageNet, natural-shift, independent-dataset, modern-checkpoint, or universal algebra performance.
