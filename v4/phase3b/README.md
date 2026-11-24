# Phase 3B — CIFAR-10 Realistic-Corruption Replication

Phase 3B prospectively replicated Phase 3A on CIFAR-10 with the same six clean-room corruption operators, symmetric primary method, four comparators, six-coordinate behavior vector, and seven all-required gates.

**Final status: `CROSS_DATASET_REPLICATION_GATE_FAILED` (6/7).** All six coordinate MAEs, the aggregate hierarchical interval, controlled rank, and all six family signs favored the fitted symmetric method, but accuracy top-1 regret was slightly worse than `g0` (0.7275 vs 0.7125 percentage points). The unfavorable result is retained without tuning.

See the [`final report`](PHASE3B_FINAL_REPORT.md), [`information audit`](INFORMATION_AUDIT.md), and [`prospective protocol`](../protocol/phase3b_cifar10_replication_preregistration.md).

## Frozen chronology

| Boundary | Pushed commit |
|---|---|
| Prospective protocol before development behavior | `603e2b5` |
| Development behavior and fitted method before confirmation training | `18282c5` |
| Confirmation checkpoint ledger before confirmation behavior | `c7d5b57` |

## Reproduce committed analyses

```bash
.venv/bin/python -m v4.phase3b.analyze
.venv/bin/python -m v4.phase3b.robustness
.venv/bin/python -m v4.phase3b.verify
```

The verifier passed 40/40 checks. These commands use compact committed artifacts and do not retrain models or reopen confirmation behavior.

## Full prospective execution order

```bash
# 1. Freeze and push the protocol before behavior.
.venv/bin/python -m v4.phase3b.manifest
.venv/bin/python -m unittest v4.phase3b.test_phase3b -v

# 2. Train/evaluate development and freeze/push the method.
.venv/bin/python -m v4.phase3b.train_bank --split development
.venv/bin/python -m v4.phase3b.evaluate_bank --split development
.venv/bin/python -m v4.phase3b.freeze_methods

# 3. Train/hash confirmation checkpoints and push the sealed ledger.
.venv/bin/python -m v4.phase3b.train_bank --split confirmation

# 4. Open confirmation once and retain the result.
.venv/bin/python -m v4.phase3b.evaluate_bank --split confirmation
.venv/bin/python -m v4.phase3b.forecast
.venv/bin/python -m v4.phase3b.analyze
.venv/bin/python -m v4.phase3b.robustness
.venv/bin/python -m v4.phase3b.verify
```

Dataset caches and checkpoints remain ignored. Compact manifests, hashes, behavior tables, forecasts, analyses, audits, and reports are committed.
