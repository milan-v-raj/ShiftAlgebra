#!/usr/bin/env python3
"""Apply the frozen Phase-2 method to the confirmation model bank."""

from v4.phase2.forecast import forecast_confirmation


if __name__ == "__main__":
    rows = forecast_confirmation()
    print(f"wrote {len(rows)} confirmation prediction rows")
