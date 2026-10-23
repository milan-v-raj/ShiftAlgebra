# Phase 1B — Severity-Calibrated Mason Extension

**Protocol:** `V4-P1B-CALIBRATED-001`
**Status:** one-shot confirmation complete; dynamic-range and method gates failed

Phase 1B was a separately preregistered test of the saturation diagnosis from Phase 1. It kept the classifiers, 167-word manifest, R2 access regime, forecasting methods, and confirmation partitions fixed, while replacing the severe primitive endpoints with a common attenuation homotopy.

The negative Phase-1 result remains unchanged. Phase 1B is an additional negative result, not a rescue.

## Result

Dataset-macro accuracy MAE:

- pair residual: **18.837pp**;
- potential Shift Algebra: **20.575pp** (**9.22% worse**);
- clustered comparator-minus-potential 95% CI: **[-3.277, -0.380]pp**.

The maximum envelope no longer won, confirming that its Phase-1 advantage was severity dependent. However, mild primitive effects still compounded much faster than all frozen pair-calibrated methods predicted. The dynamic-range guard also failed at CIFAR length six and EMNIST lengths four through six.

The secondary order channel passed: potential algebra improved reversal-difference MAE over signed `L_p` (2.208pp versus 3.109pp), but that did not offset magnitude error.

See [`PHASE1B_FINAL_REPORT.md`](PHASE1B_FINAL_REPORT.md) for the complete result and [`INFORMATION_AUDIT.md`](INFORMATION_AUDIT.md) for the post-run leak/correctness audit.

## Leakage barrier used

Severity strengths were selected using only clean and single-primitive accuracy on the old development partitions. The calibration executable cannot evaluate pairs or composed words. Complete candidate tables and selected strengths were committed in a second freeze before either confirmation pipeline ran.

## Layout

- `corruptions.py` — endpoint-preserving attenuation homotopy;
- `calibration.py` — clean/single-only deterministic strength selection;
- `FROZEN_STRENGTHS.json` — Stage-B selected strengths and hashes;
- `pipeline.py` — one-shot 167-condition confirmation;
- `analyze.py` — frozen dynamic-range checks, metrics, bootstrap, and gates;
- `verify_phase1b.py` — stage-aware verifier;
- `results/` — compact calibration and confirmation artifacts.

## Verify

```bash
python v4/phase1b/verify_phase1b.py
```

See [`../protocol/phase1b_preregistration.md`](../protocol/phase1b_preregistration.md) for the exact frozen design and gates.
