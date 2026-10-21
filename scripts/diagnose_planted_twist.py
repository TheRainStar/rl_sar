#!/usr/bin/env python3
"""Measure planted hip-twist: body yaw vs foot slip (no free-joint qpos writes)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[1]
MJCF = ROOT / "src/rl_sar_zoo/q25_description/mjcf/shooting_stability.xml"
STAND = np.array(
    [0.0, 0.0, 0.0, 0.0, -0.65, -0.65, -0.90, -0.90, 1.236, 1.236, 1.164, 1.164],
    dtype=np.float64,
)
JOINT_NAMES = [
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
FOOT_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]


def rpy(q):
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sinp = 2 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def joint_q(model, data):
    return np.array(
        [data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]] for n in JOINT_NAMES]
    )


def foot_xy(model, data):
    out = np.zeros((4, 2))
    for i, name in enumerate(FOOT_NAMES):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        out[i] = data.xpos[bid][:2]
    return out


def main():
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data = mujoco.MjData(model)
    model.opt.timestep = 0.005
    for name, q in zip(JOINT_NAMES, STAND):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[jid]] = q
        data.qvel[model.jnt_dofadr[jid]] = 0.0
    data.qpos[0:3] = [0.096, 0.001, 0.578]
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    kp = np.array([200, 200, 200, 200, 280, 280, 300, 300, 320, 320, 360, 360], dtype=np.float64)
    kd = np.array([6, 6, 6, 6, 8, 8, 9, 9, 9, 9, 10, 10], dtype=np.float64)
    act = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n.replace("_joint", "")) for n in JOINT_NAMES]

    def pd_step(q_des):
        q = joint_q(model, data)
        dq = np.array(
            [data.qvel[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]] for n in JOINT_NAMES]
        )
        tau = np.clip(kp * (q_des - q) - kd * dq, -85, 180)
        for a, t in zip(act, tau):
            data.ctrl[a] = t
        mujoco.mj_step(model, data)

    for _ in range(300):
        pd_step(STAND)
    feet0 = foot_xy(model, data)
    yaw0 = rpy(data.qpos[3:7])[2]
    z0 = float(data.qpos[2])
    print(f"hold z={z0:.3f} yaw={math.degrees(yaw0):.3f}deg feet=\n{np.round(feet0, 4)}")

    az = math.radians(2.5)
    for gain in (3.2, 6.0, 10.0):
        # reset to hold by continuing from current is messy; re-hold from stand
        for name, q in zip(JOINT_NAMES, STAND):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            data.qpos[model.jnt_qposadr[jid]] = q
        data.qpos[0:3] = [0.096, 0.001, 0.578]
        data.qpos[3:7] = [1, 0, 0, 0]
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        for _ in range(300):
            pd_step(STAND)
        feet0 = foot_xy(model, data)
        yaw0 = rpy(data.qpos[3:7])[2]
        hip = gain * az
        q_des = STAND.copy()
        q_des[0] += hip
        q_des[1] -= hip
        q_des[2] += hip
        q_des[3] -= hip
        # ramp over 1.0 s
        for i in range(400):
            a = min(1.0, (i + 1) / 200.0)
            pd_step(STAND + a * (q_des - STAND))
        feet1 = foot_xy(model, data)
        yaw1 = rpy(data.qpos[3:7])[2]
        slip = np.linalg.norm(feet1 - feet0, axis=1)
        print(
            f"gain={gain:4.1f} hip={hip:.3f}  yaw {math.degrees(yaw0):+.3f}->{math.degrees(yaw1):+.3f} "
            f"(d={math.degrees(yaw1-yaw0):+.3f} deg, want +2.5)  "
            f"foot_slip_mm={np.round(slip*1000, 1)}  z={data.qpos[2]:.3f}"
        )


def planted_yaw(model, data, dyaw, feet):
    cx, cy = feet[:, :2].mean(axis=0)
    c, s = math.cos(dyaw), math.sin(dyaw)
    dx, dy = data.qpos[0] - cx, data.qpos[1] - cy
    data.qpos[0] = cx + c * dx - s * dy
    data.qpos[1] = cy + s * dx + c * dy
    roll, pitch, yaw = rpy(data.qpos[3:7])
    yaw += dyaw
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    data.qpos[3] = cr * cp * cy + sr * sp * sy
    data.qpos[4] = sr * cp * cy - cr * sp * sy
    data.qpos[5] = cr * sp * cy + sr * cp * sy
    data.qpos[6] = cr * cp * sy - sr * sp * cy


def restore_ik(model, data, feet, iters=4, clamp=0.03):
    nv = model.nv
    jacp = np.zeros((3, nv))
    legs = [
        ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"],
        ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"],
        ["RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"],
        ["RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"],
    ]
    for _ in range(iters):
        mujoco.mj_forward(model, data)
        for f, joints in enumerate(legs):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, FOOT_NAMES[f])
            err = np.append(feet[f], data.xpos[bid][2]) - data.xpos[bid]
            err = np.array([feet[f][0] - data.xpos[bid][0], feet[f][1] - data.xpos[bid][1], 0.0])
            # keep XY planted; leave Z to contact
            if float(np.dot(err, err)) < 1e-12:
                continue
            mujoco.mj_jacBody(model, data, jacp, None, bid)
            dofs = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in joints]
            J = jacp[:, dofs]
            A = J.T @ J + 1e-3 * np.eye(3)
            try:
                dq = np.linalg.solve(A, J.T @ err)
            except np.linalg.LinAlgError:
                continue
            for j, name in enumerate(joints):
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                qadr = model.jnt_qposadr[jid]
                q = data.qpos[qadr] + float(np.clip(dq[j], -clamp, clamp))
                lo, hi = model.jnt_range[jid]
                data.qpos[qadr] = float(np.clip(q, lo, hi))


def main_slow_ik():
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data = mujoco.MjData(model)
    model.opt.timestep = 0.005
    for name, q in zip(JOINT_NAMES, STAND):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[jid]] = q
    data.qpos[0:3] = [0.096, 0.001, 0.578]
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    kp = np.array([200, 200, 200, 200, 280, 280, 300, 300, 320, 320, 360, 360], dtype=np.float64)
    kd = np.array([6, 6, 6, 6, 8, 8, 9, 9, 9, 9, 10, 10], dtype=np.float64)
    act = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n.replace("_joint", "")) for n in JOINT_NAMES]

    def pd_step(q_des):
        q = joint_q(model, data)
        dq = np.array(
            [data.qvel[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]] for n in JOINT_NAMES]
        )
        tau = np.clip(kp * (q_des - q) - kd * dq, -85, 180)
        for a, t in zip(act, tau):
            data.ctrl[a] = t
        mujoco.mj_step(model, data)

    for _ in range(300):
        pd_step(STAND)
    feet0 = foot_xy(model, data)
    # lock full 3d feet for IK xy
    feet_lock = np.zeros((4, 2))
    feet_lock[:] = feet0
    z0 = float(data.qpos[2])
    remain = math.radians(2.5)
    hold = STAND.copy()
    print("\n--- slow planted yaw + XY-only IK, PD tracks hold ---")
    for i in range(500):
        step = float(np.clip(remain, -0.0008, 0.0008))
        if abs(step) > 1e-6:
            planted_yaw(model, data, step, np.column_stack([feet_lock, np.zeros(4)]))
            restore_ik(model, data, feet_lock)
            hold = joint_q(model, data)
            remain -= step
        pd_step(hold)
        if i in (0, 49, 99, 199, 499):
            slip = np.linalg.norm(foot_xy(model, data) - feet0, axis=1)
            yaw = math.degrees(rpy(data.qpos[3:7])[2])
            print(
                f"  t={i*0.005:.2f}s z={data.qpos[2]:.3f} yaw={yaw:+.2f} remain={math.degrees(remain):+.2f} "
                f"slip_mm={np.round(slip*1000,1)}"
            )
    slip = np.linalg.norm(foot_xy(model, data) - feet0, axis=1)
    print(
        f"SLOW-IK RESULT z={data.qpos[2]:.3f} (hold {z0:.3f}) yaw={math.degrees(rpy(data.qpos[3:7])[2]):+.2f} "
        f"slip_mm={np.round(slip*1000,1)} "
        f"{'COLLAPSE' if data.qpos[2] < 0.40 else 'up'}"
    )


def main_command_ik_pose():
    """Solve planted-yaw joints on a copy, command them with PD on the live sim."""
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data = mujoco.MjData(model)
    model.opt.timestep = 0.005
    for name, q in zip(JOINT_NAMES, STAND):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[jid]] = q
    data.qpos[0:3] = [0.096, 0.001, 0.578]
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    kp = np.array([200, 200, 200, 200, 280, 280, 300, 300, 320, 320, 360, 360], dtype=np.float64)
    kd = np.array([6, 6, 6, 6, 8, 8, 9, 9, 9, 9, 10, 10], dtype=np.float64)
    act = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n.replace("_joint", "")) for n in JOINT_NAMES]

    def pd_step(q_des):
        q = joint_q(model, data)
        dq = np.array(
            [data.qvel[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]] for n in JOINT_NAMES]
        )
        tau = np.clip(kp * (q_des - q) - kd * dq, -85, 180)
        for a, t in zip(act, tau):
            data.ctrl[a] = t
        mujoco.mj_step(model, data)

    for _ in range(300):
        pd_step(STAND)
    feet0 = foot_xy(model, data)
    z0 = float(data.qpos[2])

    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    scratch.qvel[:] = 0
    mujoco.mj_forward(model, scratch)
    feet = foot_xy(model, scratch)
    planted_yaw(model, scratch, math.radians(2.5), np.column_stack([feet, np.zeros(4)]))
    restore_ik(model, scratch, feet, iters=12, clamp=0.15)
    mujoco.mj_forward(model, scratch)
    q_twist = joint_q(model, scratch)
    print("\n--- command IK pose via PD only (no live qpos write) ---")
    print(f"dq joints={np.round(q_twist - STAND, 3)}")
    print(f"scratch yaw={math.degrees(rpy(scratch.qpos[3:7])[2]):+.2f} scratch z={scratch.qpos[2]:.3f}")

    for i in range(400):
        a = min(1.0, (i + 1) / 200.0)
        pd_step((1 - a) * STAND + a * q_twist)
        if i in (0, 99, 199, 399):
            slip = np.linalg.norm(foot_xy(model, data) - feet0, axis=1)
            yaw = math.degrees(rpy(data.qpos[3:7])[2])
            print(
                f"  t={i*0.005:.2f}s z={data.qpos[2]:.3f} yaw={yaw:+.2f} slip_mm={np.round(slip*1000,1)}"
            )
    slip = np.linalg.norm(foot_xy(model, data) - feet0, axis=1)
    print(
        f"CMD-IK RESULT z={data.qpos[2]:.3f} (hold {z0:.3f}) yaw={math.degrees(rpy(data.qpos[3:7])[2]):+.2f} "
        f"slip_mm={np.round(slip*1000,1)} {'COLLAPSE' if data.qpos[2] < 0.40 else 'up'}"
    )


if __name__ == "__main__":
    main()
    main_slow_ik()
    main_command_ik_pose()
