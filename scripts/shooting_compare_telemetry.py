#!/usr/bin/env python3
# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
"""Compare Isaac (robot_lab) vs MuJoCo (rl_sar) shooting telemetry CSVs.

Examples:
  python3 scripts/shooting_compare_telemetry.py \\
      --isaac /path/to/isaac_shooting_telemetry.csv \\
      --mujoco policy/q25/shooting_telemetry.csv \\
      --out /tmp/shooting_compare
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


def _load_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"empty csv: {path}")
        cols = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name in reader.fieldnames:
                try:
                    cols[name].append(float(row[name]))
                except (TypeError, ValueError):
                    cols[name].append(float("nan"))
    return {k: np.asarray(v, dtype=np.float64) for k, v in cols.items()}


def _resample_to(src_t: np.ndarray, src_y: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
    if src_t.size == 0:
        return np.full_like(dst_t, np.nan, dtype=np.float64)
    order = np.argsort(src_t)
    t = src_t[order]
    y = src_y[order]
    # drop duplicate timestamps
    uniq_t, idx = np.unique(t, return_index=True)
    uniq_y = y[idx]
    return np.interp(dst_t, uniq_t, uniq_y, left=np.nan, right=np.nan)


def _stats(name: str, a: np.ndarray, b: np.ndarray) -> str:
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return f"{name}: no overlapping samples"
    da = a[mask]
    db = b[mask]
    err = da - db
    return (
        f"{name}: isaac_mean={da.mean():.4f} mujoco_mean={db.mean():.4f} "
        f"|diff|_mean={np.abs(err).mean():.4f} max={np.abs(err).max():.4f} "
        f"rmse={math.sqrt(float(np.mean(err * err))):.4f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Isaac vs MuJoCo shooting telemetry")
    parser.add_argument("--isaac", type=Path, required=True, help="Isaac play CSV")
    parser.add_argument("--mujoco", type=Path, required=True, help="MuJoCo shooting_telemetry.csv")
    parser.add_argument("--out", type=Path, default=Path("/tmp/shooting_compare"), help="output dir")
    parser.add_argument("--max-t", type=float, default=None, help="optional time crop (seconds)")
    args = parser.parse_args()

    isaac = _load_csv(args.isaac)
    mujoco = _load_csv(args.mujoco)
    args.out.mkdir(parents=True, exist_ok=True)

    t_i = isaac["t"]
    t_m = mujoco["t"]
    if args.max_t is not None:
        t_i_mask = t_i <= args.max_t
        t_m_mask = t_m <= args.max_t
        isaac = {k: v[t_i_mask] for k, v in isaac.items()}
        mujoco = {k: v[t_m_mask] for k, v in mujoco.items()}
        t_i = isaac["t"]
        t_m = mujoco["t"]

    # Common timeline: denser of the two, clipped to overlap.
    t0 = max(float(t_i.min(initial=0.0)), float(t_m.min(initial=0.0)))
    t1 = min(float(t_i.max(initial=0.0)), float(t_m.max(initial=0.0)))
    if t1 <= t0:
        print("[ERROR] no overlapping time range")
        return 1
    dt = min(np.median(np.diff(t_i)) if t_i.size > 1 else 0.02, np.median(np.diff(t_m)) if t_m.size > 1 else 0.02)
    dt = float(max(dt, 1e-3))
    t_common = np.arange(t0, t1 + 0.5 * dt, dt)

    compare_keys = [
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "base_z",
        "lateral_miss_m",
        "vertical_miss_m",
        "aim_error",
        "delta_y",
        "delta_z",
    ]
    # include a few joint torques if present
    for name in ("FR_thigh_joint", "FL_thigh_joint", "RR_calf_joint", "RL_calf_joint"):
        for prefix in ("tau_est_", "q_"):
            key = prefix + name
            if key in isaac and key in mujoco:
                compare_keys.append(key)

    lines = [
        f"isaac={args.isaac}",
        f"mujoco={args.mujoco}",
        f"overlap_t=[{t0:.3f}, {t1:.3f}] s, samples={t_common.size}",
        "",
    ]
    series = {}
    for key in compare_keys:
        if key not in isaac or key not in mujoco:
            lines.append(f"{key}: missing in one CSV")
            continue
        yi = _resample_to(t_i, isaac[key], t_common)
        ym = _resample_to(t_m, mujoco[key], t_common)
        series[key] = (yi, ym)
        lines.append(_stats(key, yi, ym))

    report = "\n".join(lines) + "\n"
    report_path = args.out / "compare_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"[OK] wrote {report_path}")

    if plt is None:
        print("[WARN] matplotlib not installed; skip plots")
        return 0

    plot_groups = [
        ("attitude", ["roll_deg", "pitch_deg", "yaw_deg"]),
        ("gyro", ["gyro_x", "gyro_y", "gyro_z"]),
        ("aim", ["lateral_miss_m", "vertical_miss_m", "aim_error"]),
        ("delta", ["delta_y", "delta_z"]),
    ]
    for group_name, keys in plot_groups:
        keys = [k for k in keys if k in series]
        if not keys:
            continue
        fig, axes = plt.subplots(len(keys), 1, figsize=(10, 2.4 * len(keys)), sharex=True)
        if len(keys) == 1:
            axes = [axes]
        for ax, key in zip(axes, keys):
            yi, ym = series[key]
            ax.plot(t_common, yi, label="isaac", linewidth=1.2)
            ax.plot(t_common, ym, label="mujoco", linewidth=1.2)
            ax.set_ylabel(key)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")
        axes[-1].set_xlabel("t (s)")
        fig.suptitle(f"Isaac vs MuJoCo: {group_name}")
        fig.tight_layout()
        out = args.out / f"compare_{group_name}.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"[OK] wrote {out}")

    # torque / joint subset
    torque_keys = [k for k in series if k.startswith("tau_est_") or k.startswith("q_")]
    if torque_keys:
        fig, axes = plt.subplots(len(torque_keys), 1, figsize=(10, 2.2 * len(torque_keys)), sharex=True)
        if len(torque_keys) == 1:
            axes = [axes]
        for ax, key in zip(axes, torque_keys):
            yi, ym = series[key]
            ax.plot(t_common, yi, label="isaac", linewidth=1.0)
            ax.plot(t_common, ym, label="mujoco", linewidth=1.0)
            ax.set_ylabel(key.replace("_joint", ""))
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")
        axes[-1].set_xlabel("t (s)")
        fig.suptitle("Isaac vs MuJoCo: joints")
        fig.tight_layout()
        out = args.out / "compare_joints.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"[OK] wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
