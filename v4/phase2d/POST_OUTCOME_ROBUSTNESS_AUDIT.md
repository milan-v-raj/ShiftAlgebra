# Phase 2D Post-Outcome Robustness Audit

This audit was performed only after the frozen Phase-2D result was retained. It is a sensitivity analysis, not a replacement test, and it does not alter the failed gate.

## Provenance and leak review

- Freeze commit: `e71b360`.
- Training-record commit: `9cbdc6a`.
- Outcome/report commit: `637b420`.
- All twelve local checkpoint hashes match the pre-evaluation training record.
- The committed behavior hash matches `evaluation.json`.
- Frozen prior models, ACHM gates, selector sets, words, image offsets, architecture definitions, and decision rules precede outcome evaluation.
- The only post-outcome analysis-code correction converted NumPy booleans to JSON-serializable Python booleans. It changed no value or decision.
- No route from Phase-2D target accuracy to method fitting or pair selection was found.

## Heterogeneity of the `g0` minus `g1` MAE improvement

Positive values favor `g1`.

| Stratum | Improvement |
|---|---:|
| TinyVGG | +1.904pp |
| TinyDenseNet | +0.453pp |
| ERM/flip | +2.490pp |
| Noise augmentation | +1.106pp |
| Label smoothing | -0.061pp |
| Length 3 | +2.921pp |
| Length 4 | +0.805pp |
| Length 5 | +0.255pp |
| Length 6 | +0.732pp |

Both architecture-family averages and all 32 reversal-group averages favor `g1`. Five of six lineage averages favor `g1`; the TinyDenseNet label-smoothing lineage is negative at -1.420pp.

An exact sign-flip sensitivity test over the six lineage means gives one-sided `p=0.0625` and two-sided `p=0.125`. This agrees with the preregistered bootstrap decision: the direction is consistent but six independent lineages are insufficient for a conventional confirmatory claim.

## Statistical caveats discovered by audit

1. Architecture generalization has only two family-level units. The frozen bootstrap clusters lineages and reversal groups, as preregistered, but does not establish population-level inference over architecture families.
2. Controlled rank uses twelve correlated snapshots with many metadata controls and should be treated as a descriptive endpoint.
3. Epoch snapshots share lineages. The primary bootstrap handles this dependence, but point ranking and regret metrics still summarize snapshots.
4. The cross-phase bootstrap contains only four banks that share CIFAR-10, corruption semantics, and some architecture classes. Its interval is a descriptive stability calculation, not an independent-dataset confidence statement.

## Conclusion

No information leak, checkpoint mismatch, or arithmetic error was found. The audit strengthens confidence that the recorded numbers are genuine while reinforcing the conservative conclusion: Phase 2D remains **failed**, with encouraging but heterogeneous point transfer that needs more independent architecture families and datasets.

Machine-readable details and artifact hashes are in `results/POST_OUTCOME_ROBUSTNESS.json`.
