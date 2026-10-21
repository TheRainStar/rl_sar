#!/usr/bin/env python3
"""Reproduce the C++ planted-yaw + 3-DoF foot IK that runs after [1]."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

try:
    import mujoco
except ImportError as exc:
    raise SystemExit("Need mujoco in this env") from exc

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
LEG_JOINTS = [
    ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"],
    ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"],
    ["RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"],
    ["RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"],
]


def solve_dls3(J: np.ndarray, err: np.ndarray, lam: float = 1e-3) -> np.ndarray | None:
    A = J.T @ J + lam * np.eye(3)
    b = J.T @ err
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None


def set_stand(model, data):
    for name, q in zip(JOINT_NAMES, STAND):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[jid]] = q
        data.qvel[model.jnt_dofadr[jid]] = 0.0
    data.qpos[0:3] = [0.096, 0.001, 0.578]
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)


def foot_pos(model, data):
    out = np.zeros((4, 3))
    for i, name in enumerate(FOOT_NAMES):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        out[i] = data.xpos[bid]
    return out


def rpy(q):
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sinp = 2 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


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


def restore_ik(model, data, feet, iters=8):
    nv = model.nv
    jacp = np.zeros((3, nv))
    max_abs_dq = 0.0
    for _ in range(iters):
        mujoco.mj_forward(model, data)
        for f, joints in enumerate(LEG_JOINTS):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, FOOT_NAMES[f])
            err = feet[f] - data.xpos[bid]
            if float(np.dot(err, err)) < 1e-12:
                continue
            mujoco.mj_jacBody(model, data, jacp, None, bid)
            dofs = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in joints]
            J = jacp[:, dofs]
            dq = solve_dls3(J, err)
            if dq is None:
                continue
            max_abs_dq = max(max_abs_dq, float(np.max(np.abs(dq))))
            for j, name in enumerate(joints):
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                qadr = model.jnt_qposadr[jid]
                q = data.qpos[qadr] + float(np.clip(dq[j], -0.15, 0.15))
                lo, hi = model.jnt_range[jid]
                data.qpos[qadr] = float(np.clip(q, lo, hi))
                data.qvel[model.jnt_dofadr[jid]] = 0.0
    mujoco.mj_forward(model, data)
    return max_abs_dq


def joint_q(model, data):
    return np.array(
        [data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]] for n in JOINT_NAMES]
    )


def main():
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data = mujoco.MjData(model)
    set_stand(model, data)
    q0 = joint_q(model, data)
    z0 = float(data.qpos[2])
    print(f"stand z={z0:.3f} q={np.round(q0, 3)}")

    # One typical [1] tick: residual yaw ~0.001 rad, face_gain=0.15, plus a few cm lateral.
    dyaw = 0.15 * 0.001 + 1.0 * (0.05 / 100.0)
    dyaw = float(np.clip(dyaw, -0.03, 0.03))
    print(f"\n--- single planted IK step dyaw={dyaw:.5f} rad ---")
    feet = foot_pos(model, data)
    planted_yaw(model, data, dyaw, feet)
    max_dq = restore_ik(model, data, feet)
    q1 = joint_q(model, data)
    print(f"max|raw dq|={max_dq:.3f} z={data.qpos[2]:.3f}")
    print(f"dq joints={np.round(q1 - q0, 3)}")
    print(f"q after ={np.round(q1, 3)}")

    print("\n--- 40 kinematic ticks ---")
    set_stand(model, data)
    for i in range(40):
        feet = foot_pos(model, data)
        w, x, y, z = data.qpos[3:7]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        dy = float(np.clip(0.15 * (0.0 - yaw) + 0.0005, -0.03, 0.03))
        planted_yaw(model, data, dy, feet)
        restore_ik(model, data, feet)
    qn = joint_q(model, data)
    print(f"z={data.qpos[2]:.3f} q={np.round(qn, 3)}")

    print("\n--- physics: PD hold 1s, then planted IK+qpos inject every 5ms for 1s ---")
    set_stand(model, data)
    model.opt.timestep = 0.005
    kp = np.array([200, 200, 200, 200, 280, 280, 300, 300, 320, 320, 360, 360], dtype=np.float64)
    kd = np.array([6, 6, 6, 6, 8, 8, 9, 9, 9, 9, 10, 10], dtype=np.float64)
    act = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n.replace("_joint", "")) for n in JOINT_NAMES]

    def pd_step(q_des):
        q = joint_q(model, data)
        dq = np.array([data.qvel[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]] for n in JOINT_NAMES])
        tau = np.clip(kp * (q_des - q) - kd * dq, -85, 180)
        for a, t in zip(act, tau):
            data.ctrl[a] = t
        mujoco.mj_step(model, data)

    for _ in range(200):
        pd_step(STAND)
    print(f"after hold z={data.qpos[2]:.3f} pitch={rpy(data.qpos[3:7])[1]*180/math.pi:.2f}deg")
    z_hold = float(data.qpos[2])
    q_hold = joint_q(model, data)
    for i in range(200):
        feet = foot_pos(model, data)
        w, x, y, z = data.qpos[3:7]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        dy = float(np.clip(1.0 * ((0.0 - 0.0) ) + 0.15 * (0.0 - yaw), -0.03, 0.03))
        if abs(dy) > 1e-6 or True:
            planted_yaw(model, data, max(dy, 0.002), feet)
            restore_ik(model, data, feet)
        pd_step(joint_q(model, data))  # like SyncCommandToCurrentJoints
        if i in (0, 19, 49, 99, 199):
            print(f"  t={i*0.005:.2f}s z={data.qpos[2]:.3f} thighs={np.round(joint_q(model, data)[4:8], 3)}")
    qn = joint_q(model, data)
    print(f"final z={data.qpos[2]:.3f} (hold {z_hold:.3f}) q={np.round(qn, 3)}")
    print(f"dq from hold={np.round(qn - q_hold, 3)}")
    fallen = data.qpos[2] < 0.40 or abs(rpy(data.qpos[3:7])[1]) > 0.4
    print("RESULT:", "COLLAPSE" if fallen else "stable")

    print("\n--- physics: PD hold then FREE-JOINT yaw only (fixed red-target path) ---")
    set_stand(model, data)
    for _ in range(200):
        pd_step(STAND)
    z_hold = float(data.qpos[2])
    for i in range(200):
        w, x, y, z = data.qpos[3:7]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        dy = float(np.clip(0.15 * (0.0 - yaw), -0.03, 0.03))
        if abs(dy) > 1e-4:
            roll, pitch, _ = rpy(data.qpos[3:7])
            yaw += dy
            cr, sr = math.cos(roll / 2), math.sin(roll / 2)
            cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
            cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
            data.qpos[3] = cr * cp * cy + sr * sp * sy
            data.qpos[4] = sr * cp * cy - cr * sp * sy
            data.qpos[5] = cr * sp * cy + sr * cp * sy
            data.qpos[6] = cr * cp * sy - sr * sp * cy
        pd_step(STAND)
        if i in (0, 99, 199):
            print(f"  t={i*0.005:.2f}s z={data.qpos[2]:.3f}")
    print("FREE-YAW RESULT:", "COLLAPSE" if data.qpos[2] < 0.40 else "stable", f"z={data.qpos[2]:.3f} (hold {z_hold:.3f})")

    print("\n--- physics: centroid yaw 2.5deg, no IK (offset-target path) ---")
    set_stand(model, data)
    for _ in range(200):
        pd_step(STAND)
    z_hold = float(data.qpos[2])
    remaining = math.radians(2.5)
    for i in range(200):
        step = float(np.clip(remaining, -0.03, 0.03))
        if abs(step) > 1e-4:
            planted_yaw(model, data, step, foot_pos(model, data))
            remaining -= step
        pd_step(STAND)
        if i in (0, 4, 99, 199):
            print(f"  t={i*0.005:.2f}s z={data.qpos[2]:.3f} remain_deg={remaining*180/math.pi:.2f}")
    print("CENTROID-YAW RESULT:", "COLLAPSE" if data.qpos[2] < 0.40 else "stable", f"z={data.qpos[2]:.3f}")

    print("\n--- physics: press-[1] red-target servo (anchor + pitch/yaw, free-joint yaw, no IK) ---")
    set_stand(model, data)
    for _ in range(200):
        pd_step(STAND)
    pitch_trim = 0.0
    yaw_trim = 0.0
    mz_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "muzzle_pos")
    mq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "muzzle_quat")
    mz_adr = int(model.sensor_adr[mz_id])
    mq_adr = int(model.sensor_adr[mq_id])
    target = np.array([100.0, 0.0, 1.3])
    for i in range(400):
        mujoco.mj_forward(model, data)
        mpos = data.sensordata[mz_adr : mz_adr + 3]
        mquat = data.sensordata[mq_adr : mq_adr + 4]
        # Ray vs x=100 plane
        fwd = np.array([
            1 - 2 * (mquat[2] ** 2 + mquat[3] ** 2),
            2 * (mquat[1] * mquat[2] + mquat[0] * mquat[3]),
            2 * (mquat[1] * mquat[3] - mquat[0] * mquat[2]),
        ])
        if abs(fwd[0]) < 1e-6:
            impact = target.copy()
        else:
            t = (100.0 - mpos[0]) / fwd[0]
            impact = mpos + t * fwd
        vert_err = target[2] - impact[2]
        lat_err = -(impact[1] - target[1])
        dt = 0.005
        pitch_trim = float(np.clip(pitch_trim + 0.006 * vert_err * dt, -0.40, 0.40))
        yaw_trim = float(np.clip(yaw_trim + 0.008 * lat_err * dt, -0.30, 0.30))
        pitch_cmd = float(np.clip(0.015 * vert_err + pitch_trim, -0.40, 0.40))
        yaw_cmd = float(np.clip(0.020 * lat_err + yaw_trim, -0.30, 0.30))
        q_des = STAND.copy()
        q_des[4] += pitch_cmd
        q_des[5] += pitch_cmd
        q_des[6] -= pitch_cmd
        q_des[7] -= pitch_cmd
        q_des[0] += yaw_cmd
        q_des[1] -= yaw_cmd
        q_des[2] += yaw_cmd
        q_des[3] -= yaw_cmd
        w, x, y, z = data.qpos[3:7]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        dyaw = float(np.clip(1.0 * (-(impact[1] - target[1]) / 100.0) + 0.15 * (0.0 - yaw), -0.03, 0.03))
        if abs(dyaw) > 1e-4:
            roll, pitch, _ = rpy(data.qpos[3:7])
            yaw += dyaw
            cr, sr = math.cos(roll / 2), math.sin(roll / 2)
            cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
            cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
            data.qpos[3] = cr * cp * cy + sr * sp * sy
            data.qpos[4] = sr * cp * cy - cr * sp * sy
            data.qpos[5] = cr * sp * cy + sr * cp * sy
            data.qpos[6] = cr * cp * sy - sr * sp * cy
        pd_step(q_des)
        if i in (0, 99, 199, 399):
            print(
                f"  t={i*0.005:.2f}s z={data.qpos[2]:.3f} lat={impact[1]-target[1]:+.3f} "
                f"vert={impact[2]-target[2]:+.3f} pitch={rpy(data.qpos[3:7])[1]*180/math.pi:.2f}"
            )
    print("PRESS-1 RESULT:", "COLLAPSE" if data.qpos[2] < 0.40 else "stable", f"z={data.qpos[2]:.3f}")


if __name__ == "__main__":
    main()
