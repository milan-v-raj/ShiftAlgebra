# Five-Pair Selector Transfer Synthesis

This retrospective table compares equal-budget designs that were fixed using original Phase-2 development roles. It does not choose a new selector or create a confirmation gate.

| Bank | Active | Magnitude / variance | Coverage | D-optimal | Random mean |
|---|---:|---:|---:|---:|---:|
| Phase 2C | **3.176pp** | not prospectively recorded | not prospectively recorded | not prospectively recorded | 3.636pp |
| Phase 2D | 6.071pp | **4.517pp** | 6.519pp | 8.004pp | 6.452pp |
| Phase 2E | 3.664pp | **3.055pp** | 4.861pp | 5.080pp | 4.699pp |
| Phase 2F (six-metric normalized) | not tested | **0.271** | not primary | not primary | 0.338 mean |

## Interpretation

- The active design passed its Phase-2C gate but failed its complete Phase-2D replication gate.
- Active point MAE remains below expected random MAE in both architecture extensions, but no Phase-2E replication inference was preregistered.
- Magnitude and variance choose the same group set and are strongest on both CIFAR-10 architecture banks.
- On fresh CIFAR-100 full-vector confirmation, multi-metric magnitude selection passed all four preregistered criteria against expected random calibration.
- Coverage and graph D-optimal objectives do not improve point MAE here.
- Full fifteen-group calibration and prior `g1` remain materially stronger than budgeted selectors.

Phase 2F provides the requested prospective new-dataset confirmation for magnitude selection. Broader claims still require natural shifts or another independently frozen dataset; Phase-2F outcomes must not be used to retune that future design.
