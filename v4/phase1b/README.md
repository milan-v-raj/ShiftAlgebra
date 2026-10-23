# Phase 1B — Severity-Calibrated Mason Extension

**Protocol:** `V4-P1B-CALIBRATED-001`
**Current status:** Stage-A implementation; target conditions remain unopened

Phase 1B is a separately preregistered test of the saturation diagnosis from Phase 1. It keeps the classifiers, 167-word manifest, R2 access regime, forecasting methods, and confirmation partitions fixed, while replacing the severe primitive endpoints with a common attenuation homotopy.

The completed Phase-1 failure is not modified. Phase 1B is also not guaranteed to favor Shift Algebra.

## Leakage barrier

Severity strengths are selected using only clean and single-primitive accuracy on the old development partitions. The calibration executable cannot evaluate pairs or composed words. The selected strengths must be committed in a second freeze before either confirmation pipeline is run.

## Layout

- `corruptions.py` — endpoint-preserving attenuation homotopy;
- `calibration.py` — clean/single-only deterministic strength selection;
- `freeze_strengths.py` — creates the Stage-B freeze artifact;
- `pipeline.py` — one-shot 167-condition confirmation;
- `analyze.py` — frozen dynamic-range checks, metrics, bootstrap, and gates;
- `verify_phase1b.py` — stage-aware verifier;
- `results/` — compact calibration and eventual confirmation artifacts.

See [`../protocol/phase1b_preregistration.md`](../protocol/phase1b_preregistration.md) for the exact design and gates.
