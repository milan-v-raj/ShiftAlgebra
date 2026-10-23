#!/usr/bin/env python3
"""One-command verification for completed Shift Algebra V4 Phase-0 artifacts."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v4.benchmark.schema import BenchmarkManifest

REQUIRED = [
    "v4/README.md",
    "v4/KNOWN_LIMITATIONS.md",
    "v4/src/algebra.py",
    "v4/src/models.py",
    "v4/src/fingerprints.py",
    "v4/src/order_models.py",
    "v4/analysis/metrics.py",
    "v4/baselines/source_family.py",
    "v4/benchmark/schema.py",
    "v4/benchmark/manifests/mason_exact.json",
    "v4/literature/CLAIM_MATRIX.csv",
    "v4/protocol/phase1_preregistration.md",
    "v4/protocol/LICENSE_PROVENANCE.md",
]

CLAIM_COLUMNS = {
    "paper",
    "venue",
    "year",
    "task",
    "target_samples_available",
    "ordered_compositions",
    "primitive_only_forecast",
    "pair_calibration",
    "models",
    "datasets",
    "metrics",
    "code_url",
    "exact_overlap",
    "notes",
}


def main() -> None:
    missing = [path for path in REQUIRED if not (ROOT / path).is_file()]
    if missing:
        raise SystemExit(f"missing Phase-0 artifacts: {missing}")

    test_result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "v4/tests", "-v"],
        cwd=ROOT,
        check=False,
    )
    if test_result.returncode:
        raise SystemExit(test_result.returncode)

    with (ROOT / "v4/literature/CLAIM_MATRIX.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if set(reader.fieldnames or []) != CLAIM_COLUMNS:
            raise SystemExit("literature claim matrix columns do not match contract")
        if len(rows) < 15 or any(not all(row.values()) for row in rows):
            raise SystemExit("literature claim matrix is incomplete")

    manifest = BenchmarkManifest.read(ROOT / "v4/benchmark/manifests/mason_exact.json")
    dataset_counts = Counter(condition.dataset for condition in manifest.conditions)
    expected = {"EMNIST": 167, "CIFAR-10": 167, "FaceScrub": 167}
    if dataset_counts != expected:
        raise SystemExit(f"unexpected Mason counts: {dataset_counts}")

    summary = {
        "phase": 0,
        "status": "IN_PROGRESS",
        "tests": "PASS",
        "required_artifacts": len(REQUIRED),
        "literature_rows": len(rows),
        "mason_conditions": len(manifest.conditions),
        "mason_conditions_per_dataset": expected,
        "mason_manifest_sha256": manifest.sha256(),
        "next_blockers": [
            "external multi-index literature refresh",
            "Phase-1 checkpoint/license freeze",
            "five-model CIFAR compute pilot",
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
