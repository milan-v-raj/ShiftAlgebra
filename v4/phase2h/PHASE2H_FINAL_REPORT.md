# Phase 2H Final Report — Prospective Order-Awareness Ablation

**Protocol:** `V4-P2H-CIFAR100-ORDER-ABLATION-001`

**Primary decision:** **ORDER-AWARENESS GATE FAILED (5/6 criteria passed)**

## Plain-language result

A composition-aware model that ignores order already captures almost all of the forecasting benefit. Adding one directed order feature reduced macro normalized MAE only from **0.3524 to 0.3494**—a **0.84% relative improvement**—and improved reversal-contrast prediction, but slightly reduced controlled model-ranking performance from **0.7320 to 0.7285**. Because rank was preregistered as an all-required criterion, the complete order-awareness gate failed.

The correct conclusion is not that order has no signal. The order-aware model had a small positive hierarchical MAE improvement interval **[0.00031, 0.00722]**, improved four of six coordinate MAEs, improved all six family-level mean errors, and reduced reversal-contrast error by 13.7%. However, those gains were not broad enough to establish order awareness as a reliable headline contribution.

## Prospective comparison

Both methods were fitted on Phase-2F development outcomes and frozen before Phase-2H training or behavior evaluation. They used the same clean condition, six singles, thirty directed pair measurements, ridge penalty, metadata, target scales, and symmetric/independent/envelope features.

- **Symmetric composition:** directions are averaged before feature construction.
- **Order-aware composition:** identical model plus exactly one `ordered - symmetric` feature.

Confirmation used 18 fresh lineages, 36 snapshots, CIFAR-100 offsets 60--79, and sixty new words arranged into thirty reversal pairs.

## Complete six-coordinate result

| Coordinate MAE | Symmetric composition | Order-aware | Direction |
|---|---:|---:|---|
| Margin | 0.006351 | **0.006253** | order improves |
| Feature drift | 0.322257 | **0.317417** | order improves |
| ECE | 0.022524 | **0.022517** | order improves slightly |
| Balanced accuracy | **0.006399** | 0.006437 | symmetric better |
| Accuracy | **0.006399** | 0.006437 | symmetric better |
| Mean confidence | 0.011242 | **0.010810** | order improves |

| Aggregate endpoint | Symmetric composition | Order-aware |
|---|---:|---:|
| Macro normalized MAE ↓ | 0.352402 | **0.349429** |
| Macro controlled model-rank Spearman ↑ | **0.732009** | 0.728517 |
| Accuracy top-1 regret ↓ | **0.650pp** | **0.650pp** |

For context, the metadata-only `g0` macro normalized MAE was 0.960842. Thus symmetric composition alone reduced error by 63.3%; order awareness added less than one additional relative percentage point.

## Primary gate

| Preregistered criterion | Result | Decision |
|---|---:|---|
| Order-aware macro normalized MAE lower | 0.3494 < 0.3524 | PASS |
| Hierarchical improvement CI excludes zero | [0.00031, 0.00722] | PASS |
| At least four coordinate MAEs improve | 4/6 | PASS |
| Reversal-contrast MAE improves | 0.1629 < 0.1886 | PASS |
| Controlled model rank no worse | 0.7285 < 0.7320 | **FAIL** |
| Accuracy regret no worse | 0.650pp = 0.650pp | PASS |
| Complete all-required gate | 5/6 | **FAIL** |

All six architecture-family mean MAE differences favored order awareness, but several effects were very small. The complete gate is retained as failed without tuning.

## Reversal-specific behavior

The symmetric model is exactly reversal invariant: its maximum predicted difference between a word and its reverse was below `4e-15`. True behavior was not reversal invariant.

Order awareness reduced macro normalized reversal-contrast MAE from **0.1886 to 0.1629**, with hierarchical improvement interval **[0.0090, 0.0466]**. Contrast Spearman was strongest for feature drift (0.791), confidence (0.702), and margin (0.623), but weak for accuracy (0.049) and negative for ECE (-0.208). This shows detectable order information for some behavior channels, not reliable universal order benefit.

## Unique-coordinate sensitivity

Balanced and overall accuracy are redundant on the balanced partition. Removing balanced accuracy preserves the decision:

- macro normalized MAE: 0.36287 symmetric versus 0.35894 order-aware;
- controlled rank: **0.76264 symmetric versus 0.76023 order-aware**;
- four of five remaining metrics improve;
- hierarchical MAE improvement CI: **[0.00039, 0.00937]**;
- reversal-contrast MAE: 0.2063 versus 0.1745.

The order-awareness gate therefore still fails because controlled rank declines.

## Scientific interpretation

The strongest supported framing is now clearer:

> Composition-aware forecasting is robust and transferable; explicit order awareness is a small, channel-dependent refinement that is not yet reliable enough for the headline method.

Order features may still be useful when feature drift, confidence, or margin differences under reversal matter. They should be presented as a secondary ablation or optional expert—not as proof of a broadly non-commutative Shift Algebra.

This remains a CIFAR-100 synthetic-transformation result using tiny two-epoch CPU models and pair calibration. It does not establish order value for natural shifts, another dataset, ImageNet, or modern public checkpoints.
