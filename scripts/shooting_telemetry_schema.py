# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
"""Shared shooting telemetry CSV schema for Isaac (robot_lab) <-> MuJoCo (rl_sar)."""

from __future__ import annotations

# Isaac Lab Q25 runtime order (robot.joint_names / actions).
Q25_JOINT_NAMES = [
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

BASE_COLUMNS = [
    "t",
    "roll_deg",
    "pitch_deg",
    "yaw_deg",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "base_x",
    "base_y",
    "base_z",
    "target_x",
    "target_y",
    "target_z",
    "impact_x",
    "impact_y",
    "impact_z",
    "delta_x",
    "delta_y",
    "delta_z",
    "lateral_miss_m",
    "vertical_miss_m",
    "aim_error",
    "hit",
    "aim_on",
    "recoil_on",
]


def shooting_telemetry_header(joint_names: list[str] | None = None) -> list[str]:
    names = joint_names or Q25_JOINT_NAMES
    cols = list(BASE_COLUMNS)
    for name in names:
        cols.extend([f"q_{name}", f"tau_est_{name}", f"tau_cmd_{name}"])
    return cols


def shooting_telemetry_header_csv(joint_names: list[str] | None = None) -> str:
    return ",".join(shooting_telemetry_header(joint_names)) + "\n"
