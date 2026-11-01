# CPU Cross-Phase Synthesis

Five independently frozen confirmation banks are summarized without refitting any method. This retrospective aggregation is not a new confirmation and does not revise any stage gate.

| Bank | Lineages | Models | Words | g0 MAE | g1 MAE | Controlled rank g0 → g1 |
|---|---:|---:|---:|---:|---:|---:|
| phase2 | 15 | 30 | 90 | 3.304pp | 1.737pp | -0.235 → 0.657 |
| phase2b | 15 | 30 | 64 | 3.043pp | 1.241pp | -0.467 → 0.827 |
| phase2c | 15 | 30 | 64 | 2.694pp | 1.410pp | 0.007 → 0.695 |
| phase2d | 6 | 12 | 64 | 2.978pp | 1.800pp | 0.270 → 0.621 |
| phase2e | 12 | 24 | 60 | 4.443pp | 2.034pp | 0.159 → 0.755 |

Macro-bank MAE: **3.292pp (`g0`) → 1.644pp (`g1`)**.
Hierarchical macro-bank improvement: **1.641pp**, bootstrap sensitivity interval **[1.072, 2.301]pp**.
All 5/5 banks favor `g1`; an independence-assuming one-sided sign test gives p=0.0312.

**Inference warning:** the five banks share CIFAR-10, transformation semantics, and some architecture classes. The interval is descriptive retrospective stability analysis, not independent-dataset confirmation.

Total: 63 lineages, 126 snapshots, 342 bank-word environments, and 8748 prediction rows.

Gate record: Phase 2 PASS; Phase 2B FAIL; Phase 2C PASS; Phase 2D FAIL; Phase 2E FAIL. These outcomes are not pooled into a single all-positive claim.
