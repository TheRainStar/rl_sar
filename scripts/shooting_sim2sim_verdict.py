#!/usr/bin/env python3
# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
"""Summarize MuJoCo shooting_telemetry.csv after a sim2sim policy run.

Example:
  python3 scripts/shooting_sim2sim_verdict.py policy/q25/shooting_telemetry.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def _f(row: dict, key: str, default: float = float("nan")) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def main() -> int:
    parser = argparse.ArgumentParser(description="Judge sim2sim policy telemetry")
    parser.add_argument("csv", type=Path, help="policy/q25/shooting_telemetry.csv")
    parser.add_argument("--min-aim-s", type=float, default=2.0, help="ignore first N seconds after aim_on")
    args = parser.parse_args()

    if not args.csv.is_file():
        print(f"missing csv: {args.csv}", file=sys.stderr)
        return 2

    with args.csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("empty telemetry csv", file=sys.stderr)
        return 2

    aim_rows = [r for r in rows if _f(r, "aim_on") >= 0.5]
    if not aim_rows:
        print("no aim_on=1 samples; press [1] and wait for telemetry")
        return 1

    t0 = _f(aim_rows[0], "t")
    eval_rows = [r for r in aim_rows if _f(r, "t") >= t0 + args.min_aim_s]
    if not eval_rows:
        eval_rows = aim_rows

    n = len(eval_rows)
    stand_n = sum(1 for r in eval_rows if _f(r, "stand_ok") >= 0.5)
    hit_n = sum(1 for r in eval_rows if _f(r, "hit") >= 0.5)
    recoil_n = sum(1 for r in eval_rows if _f(r, "recoil_on") >= 0.5)
    mode = int(_f(eval_rows[-1], "ctrl_mode", 1.0))
    z = [_f(r, "base_z") for r in eval_rows]
    pitch = [_f(r, "pitch_deg") for r in eval_rows]
    lat = [_f(r, "lateral_miss_m") for r in eval_rows]
    vert = [_f(r, "vertical_miss_m") for r in eval_rows]

    def mean(xs: list[float]) -> float:
        return sum(xs) / max(len(xs), 1)

    stand_rate = stand_n / n
    hit_rate = hit_n / n
    passed = stand_rate >= 0.80 and hit_rate >= 0.50

    print(f"file: {args.csv}")
    print(f"mode: {'SIM2SIM' if mode == 1 else 'DEPLOY'}  samples={n}  recoil_samples={recoil_n}")
    print(
        f"stand_ok={stand_rate:.0%}  hit={hit_rate:.0%}  "
        f"z_mean={mean(z):.3f}  pitch_mean={mean(pitch):.2f}  "
        f"lat_mean={mean(lat):.3f}  vert_mean={mean(vert):.3f}"
    )
    print(
        "PASS: policy held stance and hit often enough"
        if passed
        else "FAIL: policy did not transfer (kneel/miss). Isaac play ≠ MuJoCo sim2sim."
    )
    if mode != 1:
        print("NOTE: this CSV is DEPLOY mode; a PASS does not mean the policy works.")
    return 0 if passed and mode == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
