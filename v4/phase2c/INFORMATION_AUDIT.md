# Phase 2C Information and Correctness Audit

**Protocol:** `V4-P2C-ACTIVE-PAIR-CPU-001`
**Audit outcome:** access boundary and numerical checks pass; positive gate retained without expansion

## Chronology

- Commit `fa6f5ed` froze all method, design, model, word, metric, and analysis choices before fresh behavior outcomes.
- Commit `b201678` recorded 15 newly trained lineages before composition evaluation.
- Fresh behavior evaluation, forecasts, and analysis followed those commits.
- No method or gate was altered after outcomes were opened.

## Method access

`freeze_designs.py` loads only `v4/phase2/results/development-behavior.csv` with roles `{calibration, method_development}`. Its frozen artifact explicitly forbids:

- `v4/phase2/results/confirmation-behavior.csv`;
- `v4/phase2b/results/fresh-behavior.csv`.

The active five-group design was inherited unchanged from the pre-Phase-2B freeze. Twenty random subsets were generated from seed 22403 before Phase-2C outcomes. All gates used the same fixed L2 coefficient and development source.

## Fresh separation

- Model seeds and IDs are disjoint from Phase 2 and Phase 2B.
- Behavior images use per-class offsets 300--399; earlier stages used 0--49, 100--199, and 200--299.
- All 64 words are absent from the original 167 words and the 64 Phase-2B words.
- Every word and reversal share one cluster.
- Training data are shared by design, but no fresh composed-target behavior enters gate fitting.

## Fair pair-budget comparison

Every active or random design measures five unordered groups and both directions, for ten directed conditions. Unmeasured residuals are set to zero. All designs use identical model metadata, primitive measurements, gate capacity, regularization, and development rows.

The random primary estimand averages absolute error over twenty frozen random designs. It estimates expected loss over random calibration-set selection; it is not represented as a deployable ensemble prediction. Full-budget ACHM and `g1` are clearly separated because they use thirty directed pair measurements.

## Cluster and metric accounting

- 30 snapshots correspond to 15 inferential lineages.
- 64 words correspond to 32 reversal groups.
- Epoch snapshots remain nested within lineage.
- The bootstrap jointly resamples lineages and reversal groups.
- Controlled rank residualizes clean accuracy, parameter count, epoch, architecture family, and recipe.

## Automated verification

`verify_phase2c.py` checks:

1. source-development hash and fresh-word split;
2. one active and twenty unique equal-budget random designs;
3. 3,030 fresh behavior rows and behavior-file hash;
4. 1,920 fresh forecast rows;
5. finite bounded predictions for active, full, prior `g1`, and all twenty random designs;
6. independent recomputation of primary-method MAEs.

All checks pass, as do all Phase-2C unit tests.

## Claim boundaries

- The active selector beats expected random selection, not every random draw.
- Three of twenty random subsets have lower MAE.
- The experiment compares one heuristic with random selection, not D-optimal or uncertainty-based design.
- The same lightweight families/recipes occur in development and confirmation.
- CIFAR-10 and weak two-epoch models are the only setting.
- This is not evidence for order, calibrated uncertainty, a universal composition law, or official ODP superiority.
- The positive active-pair gate does not change the failed Phase-2B ACHM method gate.
