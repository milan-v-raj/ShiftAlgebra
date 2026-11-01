# Retrospective Architecture Synthesis

Phases 2D and 2E are combined descriptively without changing either failed gate.

| Family | g0 error minus g1 error |
|---|---:|
| phase2d:dense_holdout | +0.453pp |
| phase2d:vgg_holdout | +1.904pp |
| phase2e:attention_holdout | -0.642pp |
| phase2e:inception_holdout | +1.327pp |
| phase2e:mixer_holdout | +6.669pp |
| phase2e:squeeze_holdout | +2.285pp |

Macro-family improvement: **1.999pp**; hierarchical sensitivity interval **[0.244, 4.086]pp**.
Families favoring `g1`: **5/6**; independence-assuming one-sided sign p=0.1094.

This is not a new confirmation: all families use CIFAR-10, the same transformations and recipes, and lightweight custom models. Phase 2D and Phase 2E both remain failed.
