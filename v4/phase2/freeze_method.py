#!/usr/bin/env python3
"""Freeze Phase-2 extensive-hazard scaling and meta-regressions."""

from __future__ import annotations

import json

from v4.phase2.forecast import freeze_development


if __name__ == "__main__":
    print(json.dumps(freeze_development(), indent=2, sort_keys=True))
