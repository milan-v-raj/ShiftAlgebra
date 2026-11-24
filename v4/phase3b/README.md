# Phase 3B — CIFAR-10 Realistic-Corruption Replication

Phase 3B prospectively replicates Phase 3A on a different dataset. It retains the same six clean-room corruption operators, symmetric primary method, four comparators, six-coordinate behavior vector, and seven all-required gates.

The only method-structure change required by the dataset is the accuracy chance floor: 0.10 for ten CIFAR-10 classes. Coefficients, exponents, and metric scales are fitted only on fresh Phase-3B development behavior.

See [`../protocol/phase3b_cifar10_replication_preregistration.md`](../protocol/phase3b_cifar10_replication_preregistration.md).

## Prospective execution order

```bash
# 1. Freeze and commit protocol before behavior.
.venv/bin/python -m v4.phase3b.manifest
.venv/bin/python -m unittest v4.phase3b.test_phase3b -v

# 2. Train/evaluate development and freeze the method; commit before confirmation training.
.venv/bin/python -m v4.phase3b.train_bank --split development
.venv/bin/python -m v4.phase3b.evaluate_bank --split development
.venv/bin/python -m v4.phase3b.freeze_methods

# 3. Train/hash confirmation checkpoints; commit the ledger before behavior.
.venv/bin/python -m v4.phase3b.train_bank --split confirmation

# 4. Open confirmation once and retain the result.
.venv/bin/python -m v4.phase3b.evaluate_bank --split confirmation
.venv/bin/python -m v4.phase3b.forecast
.venv/bin/python -m v4.phase3b.analyze
.venv/bin/python -m v4.phase3b.robustness
.venv/bin/python -m v4.phase3b.verify
```

Dataset caches and checkpoints remain ignored. Compact manifests, hashes, behavior tables, forecasts, analyses, audits, and reports are committed.
