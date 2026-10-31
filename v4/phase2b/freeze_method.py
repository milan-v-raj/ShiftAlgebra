#!/usr/bin/env python3
"""Fit and freeze ACHM strictly from Phase-2 development roles."""

from pathlib import Path

from v4.phase2b.method import freeze_method

ROOT = Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    freeze_method(
        ROOT / "v4/phase2/results/development-behavior.csv",
        ROOT / "v4/phase2b/FROZEN_ACHM.json",
        ROOT / "v4/phase2b/results/development-forecast.csv",
    )
    print(ROOT / "v4/phase2b/FROZEN_ACHM.json")
