# Phase 2B Information and Correctness Audit

**Protocol:** `V4-P2B-ACHM-CPU-001`  
**Audit outcome:** no target leak detected; primary gate correctly failed

## Chronology

- Commit `834ad92` froze the protocol, model/word manifest, gate code, development-only fit, pair budgets, primary comparator, metrics, and gate before any fresh model behavior was measured.
- Commit `12f672d` added independent artifact verification before fresh behavior evaluation.
- Commit `d41f334` recorded training of all fresh lineages before composition evaluation.
- Fresh evaluation, forecast, and analysis occurred after those commits and were not used for tuning or rerunning.

## Development access

`freeze_method.py` points only to `v4/phase2/results/development-behavior.csv`. `load_behavior(..., {"calibration", "method_development"})` filters rows while reading. Automated tests confirm that the loaded role set is exactly `{calibration, method_development}`. The opened original Phase-2 confirmation CSV is never passed to the ACHM fitter.

The source development CSV SHA-256 is frozen in `PHASE2B_MANIFEST.json` and independently rechecked by `verify_phase2b.py`.

## Fresh target separation

- Fresh seeds and lineage IDs do not overlap Phase 2.
- The new behavior partition uses 100 images per class at offsets 200--299.
- Phase-2 development used offsets 0--49; Phase-2 confirmation used offsets 100--199.
- All 64 target words are absent from the previous 167-word manifest.
- Reverse words are kept together, preventing a reversal from silently entering a different role.
- The training set is shared, as intended for new CIFAR training lineages; target behavior images and labels are not used while fitting ACHM.

## Allowed calibration information

The R2 forecaster is allowed labeled clean, single, and selected ordered-pair measurements for each fresh model. It receives no image, label, confidence, or measured accuracy from any fresh length-3--6 word when producing that word's prediction.

`g0_prior` and `g1_prior` use coefficients frozen in the earlier Phase-2 development stage. They are applied without refitting.

## Model and cluster accounting

- 30 snapshots are present.
- They come from 15 independent training lineages.
- Epoch-1 and epoch-2 snapshots from one lineage are never treated as independent bootstrap lineages.
- 64 fresh words form 32 reversal groups.
- The primary bootstrap resamples reversal groups and lineages, retaining nested rows.

## Numerical checks

`verify_phase2b.py` independently verifies:

1. manifest and source-development hashes;
2. 64 unique fresh words and the frozen method status;
3. 3,030 behavior rows = 30 models × 101 conditions;
4. 1,920 fresh forecast rows = 30 models × 64 words;
5. bounded finite ACHM, hazard, `g0`, and `g1` predictions;
6. independent recomputation of every method MAE in `PHASE2B_ANALYSIS.json`.

All checks pass.

## Negative-result handling

The complete primary gate requires all four preregistered criteria. ACHM passed MAE direction and regret but failed the clustered-CI and controlled-rank criteria. The analysis status is therefore `ACHM_PRIMARY_GATE_FAILED`. No criterion was removed, no seed or word was replaced, and no confirmation-guided hyperparameter change was made.

The per-length table in the final report is descriptive. It does not alter the gate. Likewise, pair-budget, order, and uncertainty analyses remain secondary.

## Remaining limitations

- Five architecture names refer to deliberately small CPU models, not modern production-scale families.
- Architecture and recipe categories are represented in both development and confirmation; this is lineage/word transfer, not family holdout.
- CIFAR-10 is the only dataset.
- Models are weak two-epoch classifiers.
- The active selector lacks a random-selection comparator.
- Expert disagreement is not calibrated uncertainty and failed its diagnostic.
- Official ODP target-available baselines were not rerun in Phase 2B.
- This experiment does not complete the roadmap's 75+ model Phase 2.
