# Phase 2D — Architecture-Family Holdout (CPU)

**Protocol:** `V4-P2D-ARCH-HOLDOUT-CPU-001`
**Status:** complete; **primary architecture-holdout gate failed (3/4 criteria passed)**

This one-shot test applies unchanged `g0`, `g1`, ACHM, active/random five-pair designs, and four alternative selectors to TinyVGG and TinyDenseNet families absent from method development. It uses six new lineages, twelve snapshots, a disjoint 1,000-image partition, and 64 words absent from all earlier target sets.

Frozen `g1` reduced MAE from **2.978pp to 1.800pp**, improved controlled rank from **0.270 to 0.621**, and lowered regret. The 1.178pp clustered improvement interval was **[-0.042, 2.293]pp**, so the complete gate failed. Full ACHM achieved 2.312pp MAE but better ranking/regret. The earlier active selector did not replicate its complete gate; magnitude-only selection was the strongest tested five-pair heuristic at 4.517pp.

See `PHASE2D_FINAL_REPORT.md` and `INFORMATION_AUDIT.md`.
