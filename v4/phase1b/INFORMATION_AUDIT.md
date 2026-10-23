# Phase 1B Information-Leak and Correctness Audit

**Protocol:** `V4-P1B-CALIBRATED-001`
**Audit timing:** after the one-shot confirmation, before final result commit
**Conclusion:** no detected target-to-fit leak or result-changing implementation error; declared design limitations remain

## 1. Freeze and provenance checks

- Stage-A implementation was committed as `5ff297827e46fd0fde9b5d8298df13409d36fc74` before calibration.
- The preregistration record was pushed before calibration.
- Calibration candidate tables and `FROZEN_STRENGTHS.json` were committed as `44f55c1b0081d16339a6dd22649dc4abf7a60a46` before pair/target evaluation.
- The strength-freeze record was pushed before confirmation.
- Checkpoint hashes exactly match the completed Phase-1 freeze:
  - CIFAR: `9f5815dd3ab43edd5d09636e8d2688fa9ab8aaae346143ec6d5184bd99944848`;
  - EMNIST: `05c49cd3678ee537accac6380d33002e4d8f512e5eb93093b18f0701f815ca53`.
- The manifest hash remains `d2be1ca5c899659482204630d1f2b755587b71ee994372a7b716e5c5ad591264`.

## 2. Severity-calibration access audit

`v4.phase1b.calibration`:

- does not accept or load a word manifest;
- iterates only over the six primitive names and the frozen scalar grid;
- applies exactly one primitive per candidate;
- uses only class-stable Phase-1 development indices (CIFAR offset 0, 100/class; EMNIST offset 0, 40/class);
- records `ordered_pairs=false` and `length_3_to_6_targets=false` in both calibration artifacts;
- selects by the committed nearest-0.08 normalized-drop rule with smaller-alpha tie break.

The expected candidate count is exactly `6 primitives × 26 strengths = 156` per dataset; both artifacts contain 156 rows.

No confirmation image or composed outcome enters severity selection.

## 3. Forecast-fitting access audit

`v4.phase1.forecast.run_forecast_evaluation` loads the aggregate behavior CSV, then constructs three disjoint objects:

- `primitive_behavior` from six length-one rows;
- `pair_targets` from 30 length-two rows;
- `target_words` from 130 length-three-to-six word identities.

All fitting calls occur before target truth is stacked for scoring:

- `fit_lp(primitive_behavior, pair_words, pair_targets)`;
- `fit_potential_ensemble(primitive_behavior, pair_words, pair_targets, frozen_p)`;
- pair residual lookup from pair targets only;
- pair-feature ridge from pair targets only.

Target rows are used only after all models are fitted, inside the prediction/scoring loop. The optional per-target records added for Phase 1B are emitted after scoring and do not feed back into a fit.

The CSV physically contains target aggregates by the time forecasting runs, so the seal is enforced by committed code and call structure rather than cryptographic process isolation. This is a limitation of the runner, not evidence of a leak. Unit tests verify 30 calibration pairs, 130 scored targets, and `target_access_during_fit=false`.

## 4. Transformation correctness

Tests establish that:

- attenuation strength 0 returns the input bit-for-bit;
- attenuation strength 1 returns the exact Phase-1 endpoint operator bit-for-bit;
- all outputs are finite and bounded;
- words are applied chronologically;
- impulse-noise fields are deterministic and batching independent;
- strength maps must contain exactly the six declared primitives and values in `[0,1]`.

`Identity` bypasses transformation. For non-identity primitive `T`, the code applies `torch.lerp(x, T(x), alpha)` at each chronological step, matching the preregistered equation.

## 5. Partition and sample-count checks

- CIFAR development and confirmation indices are disjoint by class-local offsets `[0,100)` and `[100,1000)`.
- EMNIST development and confirmation indices are disjoint by offsets `[0,40)` and `[40,400)`.
- Confirmation row counts are 9,000 CIFAR and 16,920 EMNIST for every condition.
- Each behavior artifact contains exactly 167 unique conditions with length counts `1/6/30/40/30/30/30` for identity/lengths 1--6.
- Each fingerprint artifact contains exactly six vectors of dimension 301.
- No exclusions are present.

The confirmation base images were already used by Phase 1 at different endpoint severities. The report therefore does not claim an independent image sample. The Phase-1B transformed targets themselves remained unopened until after Stage B.

## 6. Metric and inference checks

- Accuracy differences are reported in percentage points.
- Primary macro MAE gives CIFAR and EMNIST equal weight.
- The best baseline is selected only from the six declared non-potential methods.
- Bootstrap seed, draw count, hierarchy, comparator, and interval rule match the preregistration.
- Bootstrap resampling is paired: comparator and potential errors are differenced within each target before family/dataset aggregation.
- Dynamic-range thresholds match the frozen formulas.
- Accuracy clipping is applied before scoring and its fraction is reported; all Phase-1B methods have zero clipping.
- Direct recomputation from per-target rows confirms reported aggregate MAEs and reveals positive signed bias for every method/target.

The bootstrap has only two datasets and the comparator is selected on the scored outcomes. The protocol declared both facts. The interval should be read as an internal clustered diagnostic, not a population-wide confidence statement. Comparator selection is adverse to the proposed method here, but its uncertainty is not separately modeled.

## 7. Post-result code review findings

No result-changing defect was found. Two structural scientific limitations are important:

1. Pair residual and potential interaction terms are normalized by pair count. This prevents explosive long-word forecasts but assumes an intensive average interaction correction. Phase 1B shows extensive compounding damage that this form cannot represent.
2. The confirmation runner computes all condition aggregates before fitting, rather than persisting predictions before target evaluation. The fitting function still does not consume target values, but a future benchmark should enforce this contract with separate calibration and target processes/files.

Neither limitation justifies changing or rerunning Phase 1B.

## 8. Verification executed

- Phase-1B tests: 5/5 pass.
- Phase-1 tests: 8/8 pass.
- Phase-0 tests: 38/38 pass.
- Python compile checks pass.
- `git diff --check` passes after documentation finalization.
- Phase-1B verifier reports `COMPLETE_METHOD_GATE_FAILED`, with both 167-condition runs valid.

The failed result is therefore retained as the protocol outcome.
