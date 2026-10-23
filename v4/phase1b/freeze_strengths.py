#!/usr/bin/env python3
"""Freeze deterministic Phase-1B calibration outputs before target evaluation."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from v4.phase1.corruptions import PRIMITIVES
from v4.phase1b import PROTOCOL_ID

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "v4/phase1b/results"
OUTPUT = ROOT / "v4/phase1b/FROZEN_STRENGTHS.json"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    datasets = {}
    for key, expected_name in (("cifar10", "CIFAR-10"), ("emnist", "EMNIST")):
        path = RESULTS / f"{key}-calibration.json"
        payload = json.loads(path.read_text())
        if payload["protocol"] != PROTOCOL_ID or payload["dataset"] != expected_name:
            raise ValueError(f"wrong calibration payload in {path}")
        access = payload["information_access"]
        if access["ordered_pairs"] or access["length_3_to_6_targets"]:
            raise ValueError(f"forbidden target access in {path}")
        if set(payload["selected"]) != set(PRIMITIVES):
            raise ValueError(f"incomplete primitive selection in {path}")
        datasets[key] = {
            "dataset": expected_name,
            "model_id": payload["model_id"],
            "checkpoint_sha256": payload["classifier_checkpoint_sha256"],
            "calibration_artifact": str(path.relative_to(ROOT)),
            "calibration_artifact_sha256": file_sha256(path),
            "all_selected_within_band": payload["all_selected_within_band"],
            "strengths": {
                primitive: payload["selected"][primitive]["alpha"]
                for primitive in PRIMITIVES
            },
            "selected_normalized_drops": {
                primitive: payload["selected"][primitive]["normalized_drop"]
                for primitive in PRIMITIVES
            },
        }
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    frozen = {
        "protocol": PROTOCOL_ID,
        "status": "STRENGTHS_FROZEN_TARGETS_UNOPENED",
        "preregistration_code_commit": commit,
        "selection_rule": (
            "nearest normalized clean-to-chance accuracy drop to 0.08 on the "
            "single-only development grid; smaller-alpha tie break"
        ),
        "datasets": datasets,
    }
    OUTPUT.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
