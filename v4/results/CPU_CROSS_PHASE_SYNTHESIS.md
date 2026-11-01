# CPU Cross-Phase Synthesis

Four independently frozen confirmation banks are summarized without refitting any method. This retrospective aggregation is not a new confirmation and does not revise any stage gate.

| Bank | Lineages | Models | Words | g0 MAE | g1 MAE | Controlled rank g0 → g1 |
|---|---:|---:|---:|---:|---:|---:|
| phase2 | 15 | 30 | 90 | 3.304pp | 1.737pp | -0.235 → 0.657 |
| phase2b | 15 | 30 | 64 | 3.043pp | 1.241pp | -0.467 → 0.827 |
| phase2c | 15 | 30 | 64 | 2.694pp | 1.410pp | 0.007 → 0.695 |
| phase2d | 6 | 12 | 64 | 2.978pp | 1.800pp | 0.270 → 0.621 |

Macro-bank MAE: **3.005pp (`g0`) → 1.547pp (`g1`)**.
Hierarchical macro-bank improvement: **1.448pp**, bootstrap sensitivity interval **[0.954, 1.901]pp**.
All 4/4 banks favor `g1`; an independence-assuming one-sided sign test gives p=0.0625.

**Inference warning:** the four banks share CIFAR-10, transformation semantics, and some architecture classes. The interval is descriptive retrospective stability analysis, not independent-dataset confirmation.

Total: 51 lineages, 102 snapshots, 282 bank-word environments, and 7308 prediction rows.

Gate record: Phase 2 PASS; Phase 2B FAIL; Phase 2C PASS; Phase 2D FAIL. These outcomes are not pooled into a single all-positive claim.
