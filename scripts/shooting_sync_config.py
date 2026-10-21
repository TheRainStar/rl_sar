#!/usr/bin/env python3
# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
"""Sync / validate rl_sar Q25 shooting deploy config against robot_lab training.

Examples:
  python3 scripts/shooting_sync_config.py --check
  python3 scripts/shooting_sync_config.py --apply
  python3 scripts/shooting_sync_config.py --apply --copy-policy /path/to/exported/policy.pt
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML required: pip install pyyaml") from exc

RL_SAR_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROBOT_LAB = Path.home() / "robot_lab"
CONFIG_PATH = RL_SAR_ROOT / "policy" / "q25" / "robot_lab" / "config.yaml"
CONFIG_KEY = "q25/robot_lab"

# Isaac Lab Q25 runtime order (robot.joint_names / actions).
Q25_JOINT_ORDER = [
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
]


def _read_text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def _extract_float_assign(text: str, name: str) -> float | None:
    m = re.search(rf"^{re.escape(name)}\s*=\s*([-+eE0-9.]+)", text, flags=re.M)
    return float(m.group(1)) if m else None


def _extract_tuple2(text: str, name: str) -> tuple[float, float] | None:
    m = re.search(rf"^{re.escape(name)}\s*=\s*\(\s*([-+eE0-9.]+)\s*,\s*([-+eE0-9.]+)\s*\)", text, flags=re.M)
    return (float(m.group(1)), float(m.group(2))) if m else None


def load_expected_from_robot_lab(robot_lab_root: Path) -> dict:
    """Parse robot_lab sources (no Isaac runtime required)."""
    constants = _read_text(
        robot_lab_root
        / "source/robot_lab/robot_lab/tasks/manager_based/locomotion/shooting_stability/constants.py"
    )
    velocity_pretrain = _read_text(
        robot_lab_root
        / "source/robot_lab/robot_lab/tasks/manager_based/locomotion/shooting_stability/velocity_pretrain.py"
    )
    env_cfg = _read_text(
        robot_lab_root
        / "source/robot_lab/robot_lab/tasks/manager_based/locomotion/shooting_stability/shooting_stability_env_cfg.py"
    )
    assets = _read_text(robot_lab_root / "source/robot_lab/robot_lab/assets/deeprobotics.py")

    target_distance = _extract_float_assign(constants, "TARGET_DISTANCE_M") or 100.0
    target_height = _extract_float_assign(constants, "TARGET_HEIGHT_M") or 1.3
    hit_radius = _extract_float_assign(constants, "TARGET_HIT_RADIUS_M") or 0.25
    recoil_interval = (_extract_tuple2(constants, "RECOIL_INTERVAL_S") or (0.08, 0.08))[0]
    recoil_range = _extract_tuple2(constants, "RECOIL_FORCE_RANGE_N") or (32.0, 40.0)
    # Deploy uses signed -X force; magnitude matches training mid-range (or fixed stage force).
    recoil_force = -0.5 * (abs(recoil_range[0]) + abs(recoil_range[1]))

    ang_vel_scale = 0.25
    dof_vel_scale = 0.05
    dof_pos_scale = 1.0
    if "base_ang_vel.scale = 0.25" in velocity_pretrain:
        ang_vel_scale = 0.25
    if "joint_vel.scale = 0.05" in velocity_pretrain:
        dof_vel_scale = 0.05
    if "joint_pos.scale = 1.0" in velocity_pretrain:
        dof_pos_scale = 1.0

    hip_scale = 0.125
    leg_scale = 0.25
    m_hip = re.search(r'"\.\*_hip_joint"\s*:\s*([0-9.]+)', env_cfg)
    m_leg = re.search(r'"\^\(\?!.*_hip_joint\)\.\*"\s*:\s*([0-9.]+)', env_cfg)
    if m_hip:
        hip_scale = float(m_hip.group(1))
    if m_leg:
        leg_scale = float(m_leg.group(1))

    # PD / effort from DEEPROBOTICS_Q25_CFG actuators (skip URDF joint_drive stiffness=0).
    kp = 80.0
    kd = 2.0
    effort_hip = 85.0
    effort_calf = 180.0
    if "DEEPROBOTICS_Q25_CFG" in assets:
        q25_block = assets.split("DEEPROBOTICS_Q25_CFG", 1)[1].split("\n)\n", 1)[0]
        hip_m = re.search(
            r'HipThigh"\s*:\s*DCMotorCfg\((.*?)\)\s*,\s*\n\s*"Calf"',
            q25_block,
            flags=re.S,
        )
        calf_m = re.search(r'Calf"\s*:\s*DCMotorCfg\((.*?)\)\s*,?\s*\n\s*\}', q25_block, flags=re.S)
        if hip_m:
            stiff = re.search(r"stiffness=([0-9.]+)", hip_m.group(1))
            damp = re.search(r"damping=([0-9.]+)", hip_m.group(1))
            effort = re.search(r"effort_limit=([0-9.]+)", hip_m.group(1))
            if stiff:
                kp = float(stiff.group(1))
            if damp:
                kd = float(damp.group(1))
            if effort:
                effort_hip = float(effort.group(1))
        if calf_m:
            effort = re.search(r"effort_limit=([0-9.]+)", calf_m.group(1))
            if effort:
                effort_calf = float(effort.group(1))

    action_scale = []
    torque_limits = []
    rl_kp = []
    rl_kd = []
    for name in Q25_JOINT_ORDER:
        if "hip" in name:
            action_scale.append(hip_scale)
            torque_limits.append(effort_hip)
        elif "calf" in name:
            action_scale.append(leg_scale)
            torque_limits.append(effort_calf)
        else:
            action_scale.append(leg_scale)
            torque_limits.append(effort_hip)
        rl_kp.append(kp)
        rl_kd.append(kd)

    # Isaac runtime order: 4 hips, 4 thighs, 4 calves
    default_dof_pos = [0.0] * 4 + [-0.67] * 4 + [1.3] * 4
    m_thigh = re.search(r'"\.\*_thigh_joint"\s*:\s*([-+0-9.]+)', assets)
    m_calf = re.search(r'"\.\*_calf_joint"\s*:\s*([-+0-9.]+)', assets)
    if m_thigh and m_calf:
        default_dof_pos = [0.0] * 4 + [float(m_thigh.group(1))] * 4 + [float(m_calf.group(1))] * 4

    return {
        "action_scale": action_scale,
        "ang_vel_scale": ang_vel_scale,
        "dof_vel_scale": dof_vel_scale,
        "dof_pos_scale": dof_pos_scale,
        "lin_vel_scale": 1.0,
        "rl_kp": rl_kp,
        "rl_kd": rl_kd,
        "fixed_kp": list(rl_kp),
        "fixed_kd": list(rl_kd),
        "torque_limits": torque_limits,
        "default_dof_pos": default_dof_pos,
        "shooting_target_distance": target_distance,
        "shooting_target_height": target_height,
        "shooting_hit_radius": hit_radius,
        "shooting_target_azimuth": 0.0,
        # +2.5 deg = robot_lab TARGET_AZIMUTH_RANGE_DEG edge (non-facing board).
        "shooting_offset_target_azimuth": 0.0436332313,
        "shooting_recoil_interval": recoil_interval,
        "shooting_recoil_duration": 0.01,
        "shooting_recoil_force": recoil_force,
        "num_of_dofs": 12,
        "num_observations": 52,
    }


def _fmt_num(v: float) -> str:
    if float(v).is_integer():
        return f"{float(v):.1f}"
    s = f"{float(v):.6f}".rstrip("0").rstrip(".")
    return s


def _fmt_list(values: list[float], per_row: int = 3) -> str:
    chunks = []
    for i in range(0, len(values), per_row):
        row = ", ".join(_fmt_num(v) for v in values[i : i + per_row])
        chunks.append(("                 " + row) if i else row)
    return ",\n".join(chunks)


def _almost_equal(a, b, tol: float = 1e-6) -> bool:
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b))
    return abs(float(a) - float(b)) <= tol


def check_config(current: dict, expected: dict) -> list[str]:
    diffs = []
    for key, exp in expected.items():
        if key not in current:
            diffs.append(f"MISSING {key}: expected {exp}")
            continue
        cur = current[key]
        if not _almost_equal(cur, exp):
            diffs.append(f"DIFF {key}: deploy={cur} | train={exp}")
    return diffs


def apply_config(config_path: Path, expected: dict, dry_run: bool = False) -> None:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if CONFIG_KEY not in data:
        raise KeyError(f"{CONFIG_KEY} not found in {config_path}")
    node = data[CONFIG_KEY]
    for key, value in expected.items():
        node[key] = value

    # Preserve comments poorly with yaml dump — rewrite key blocks via text patch for critical fields.
    text = config_path.read_text(encoding="utf-8")

    def replace_scalar(key: str, value: float) -> None:
        nonlocal text
        text = re.sub(
            rf"(^[ ]*{re.escape(key)}:\s*)([-+eE0-9.]+)",
            rf"\g<1>{value}",
            text,
            count=1,
            flags=re.M,
        )

    def replace_list(key: str, values: list[float], per_row: int = 3) -> None:
        nonlocal text
        formatted = _fmt_list(values, per_row=per_row)
        # Match YAML list that may span multiple lines until next key at same indent.
        pattern = rf"(^[ ]*{re.escape(key)}:\s*)\[[^\]]*\]"
        # multiline lists use newlines inside []
        pattern = rf"(^[ ]*{re.escape(key)}:\s*)\[(.*?)\]"
        repl = rf"\g<1>[{formatted}]"
        new_text, n = re.subn(pattern, repl, text, count=1, flags=re.M | re.S)
        if n == 0:
            print(f"[WARN] could not patch list key {key}; using yaml dump fallback later")
        else:
            text = new_text

    for key in (
        "ang_vel_scale",
        "dof_vel_scale",
        "dof_pos_scale",
        "lin_vel_scale",
        "shooting_target_distance",
        "shooting_target_height",
        "shooting_hit_radius",
        "shooting_target_azimuth",
        "shooting_offset_target_azimuth",
        "shooting_recoil_interval",
        "shooting_recoil_duration",
        "shooting_recoil_force",
        "num_of_dofs",
        "num_observations",
    ):
        if key in expected:
            replace_scalar(key, float(expected[key]))

    for key in (
        "action_scale",
        "rl_kp",
        "rl_kd",
        "fixed_kp",
        "fixed_kd",
        "torque_limits",
        "default_dof_pos",
    ):
        if key in expected:
            replace_list(key, list(expected[key]), per_row=3)

    if dry_run:
        print(text)
        return
    config_path.write_text(text, encoding="utf-8")
    print(f"[OK] updated {config_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync rl_sar Q25 shooting config with robot_lab")
    parser.add_argument("--robot-lab", type=Path, default=DEFAULT_ROBOT_LAB, help="robot_lab root")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="rl_sar config.yaml")
    parser.add_argument("--check", action="store_true", help="only check mismatches")
    parser.add_argument("--apply", action="store_true", help="write expected values into config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="print patched config without writing")
    parser.add_argument("--copy-policy", type=Path, default=None, help="copy policy.pt into policy/q25/robot_lab/")
    args = parser.parse_args()

    if not args.check and not args.apply and args.copy_policy is None:
        args.check = True

    expected = load_expected_from_robot_lab(args.robot_lab)
    print(f"[INFO] robot_lab: {args.robot_lab}")
    print(
        "[INFO] expected scales: "
        f"action hip={expected['action_scale'][0]}, thigh/calf={expected['action_scale'][1]}; "
        f"ang_vel={expected['ang_vel_scale']}, dof_vel={expected['dof_vel_scale']}; "
        f"recoil_force={expected['shooting_recoil_force']}"
    )

    data = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    current = data.get(CONFIG_KEY, {})
    diffs = check_config(current, expected)
    if diffs:
        print(f"[WARN] {len(diffs)} mismatch(es):")
        for d in diffs:
            print(f"  - {d}")
    else:
        print("[OK] deploy config matches robot_lab expected values")

    if args.apply or args.dry_run:
        apply_config(args.config, expected, dry_run=args.dry_run)

    if args.copy_policy is not None:
        src = args.copy_policy
        dst = RL_SAR_ROOT / "policy" / "q25" / "robot_lab" / "policy.pt"
        if not src.is_file():
            print(f"[ERROR] policy not found: {src}", file=sys.stderr)
            return 1
        shutil.copy2(src, dst)
        print(f"[OK] copied {src} -> {dst}")

    return 1 if diffs and args.check and not args.apply else 0


if __name__ == "__main__":
    raise SystemExit(main())
