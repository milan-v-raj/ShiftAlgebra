# Five-Pair Selector Transfer Synthesis

This retrospective table compares equal-budget designs that were fixed using original Phase-2 development roles. It does not choose a new selector or create a confirmation gate.

| Bank | Active | Magnitude / variance | Coverage | D-optimal | Random mean |
|---|---:|---:|---:|---:|---:|
| Phase 2C | **3.176pp** | not prospectively recorded | not prospectively recorded | not prospectively recorded | 3.636pp |
| Phase 2D | 6.071pp | **4.517pp** | 6.519pp | 8.004pp | 6.452pp |
| Phase 2E | 3.664pp | **3.055pp** | 4.861pp | 5.080pp | 4.699pp |

## Interpretation

- The active design passed its Phase-2C gate but failed its complete Phase-2D replication gate.
- Active point MAE remains below expected random MAE in both architecture extensions, but no Phase-2E replication inference was preregistered.
- Magnitude and variance choose the same group set and are strongest on both architecture banks.
- Coverage and graph D-optimal objectives do not improve point MAE here.
- Full fifteen-group calibration and prior `g1` remain materially stronger than budgeted selectors.

The safest prospective choice for another equal-budget study would be to freeze magnitude selection as the primary design before collecting a genuinely new dataset. The current architecture outcomes must not be used as that confirmation.
