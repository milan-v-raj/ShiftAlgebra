# Phase 2C — Active Pair Calibration (CPU Pilot)

**Protocol:** `V4-P2C-ACTIVE-PAIR-CPU-001`
**Status:** fresh confirmation complete; **active-pair primary gate passed**

This phase tests a distinct practical question after retaining the failed Phase-2B ACHM gate: can five development-selected bidirectional pair groups outperform the expected result of selecting five groups uniformly at random?

The active design and twenty random controls use the same gate family, pair budget, source development roles, and fixed regularization. New lineages, image partition, and words supply a one-shot fresh confirmation. Full-budget ACHM and the prior frozen `g1` are references.

## Fresh result

The active five-group design achieved **3.176pp MAE**, compared with **3.636pp mean** and **3.818pp median** across twenty random five-group designs. The expected-random improvement was **0.459pp**, with clustered 95% CI **[0.119, 0.889]pp**. Controlled rank and top-1 regret also beat random medians, so all preregistered criteria passed.

The active design beat 17/20 random draws, but the best random subset reached 2.592pp. Full fifteen-group ACHM reached 1.603pp and prior `g1` reached 1.410pp using three times as many directed pair measurements. See `PHASE2C_FINAL_REPORT.md` and `INFORMATION_AUDIT.md`.

This is not a rescue of ACHM and not the final 75+ model benchmark.
