#!/usr/bin/env python3
"""Headless MuJoCo shooting rollout for sim2sim debug (no GUI).

Joint order matches Isaac Lab Q25 runtime robot.joint_names:
  [FL,FR,RL,RR]_hip, [FL,FR,RL,RR]_thigh, [FL,FR,RL,RR]_calf
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
import yaml

try:
    import mujoco
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Need mujoco: conda activate rl_sar_env && pip install mujoco") from exc

ROOT = Path(__file__).resolve().parents[1]
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


def quat_rotate_inverse_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    a = v * (2.0 * w * w - 1.0)
    cross = np.array([y * v[2] - z * v[1], z * v[0] - x * v[2], x * v[1] - y * v[0]])
    b = cross * w * 2.0
    c = np.array([x, y, z]) * np.dot(np.array([x, y, z]), v) * 2.0
    return a - b + c


def quat_rotate_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return quat_rotate_inverse_wxyz(np.array([w, -x, -y, -z]), v)


def quat_to_rpy_deg(q: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = [float(v) for v in q]
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sinp = 2 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    r2d = 180.0 / math.pi
    return roll * r2d, pitch * r2d, yaw * r2d


def load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["q25/robot_lab"]


def sensor_adr(model: mujoco.MjModel, name: str) -> int:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sid < 0:
        raise KeyError(name)
    return int(model.sensor_adr[sid])


class Q25ShootingRollout:
    def __init__(
        self,
        mjcf: Path,
        policy_path: Path,
        cfg: dict,
        *,
        clip_actions: float | None = None,
        blend_duration: float = 1.5,
        use_body_lin_vel: bool = True,
    ):
        self.model = mujoco.MjModel.from_xml_path(str(mjcf))
        self.data = mujoco.MjData(self.model)
        self.cfg = cfg
        self.dt = float(cfg.get("dt", 0.005))
        self.decimation = int(cfg.get("decimation", 4))
        self.default = np.asarray(cfg["default_dof_pos"], dtype=np.float32)
        self.action_scale = np.asarray(cfg["action_scale"], dtype=np.float32)
        self.kp = np.asarray(cfg["rl_kp"], dtype=np.float32)
        self.kd = np.asarray(cfg["rl_kd"], dtype=np.float32)
        self.torque_limits = np.asarray(cfg["torque_limits"], dtype=np.float32)
        self.ang_vel_scale = float(cfg.get("ang_vel_scale", 0.25))
        self.dof_vel_scale = float(cfg.get("dof_vel_scale", 0.05))
        self.lin_vel_scale = float(cfg.get("lin_vel_scale", 1.0))
        self.target_distance = float(cfg.get("shooting_target_distance", 100.0))
        self.target_height = float(cfg.get("shooting_target_height", 1.3))
        self.hit_radius = float(cfg.get("shooting_hit_radius", 0.25))
        self.target_azimuth = float(cfg.get("shooting_target_azimuth", 0.0))
        self.use_body_lin_vel = use_body_lin_vel
        lo, hi = cfg.get("clip_actions_lower"), cfg.get("clip_actions_upper")
        if clip_actions is not None:
            self.clip_lo = -abs(clip_actions) * np.ones(12, dtype=np.float32)
            self.clip_hi = abs(clip_actions) * np.ones(12, dtype=np.float32)
        elif lo and hi:
            self.clip_lo = np.asarray(lo, dtype=np.float32)
            self.clip_hi = np.asarray(hi, dtype=np.float32)
        else:
            self.clip_lo = -100 * np.ones(12, dtype=np.float32)
            self.clip_hi = 100 * np.ones(12, dtype=np.float32)
        self.blend_duration = float(blend_duration)
        self.policy = torch.jit.load(str(policy_path), map_location="cpu")
        self.policy.eval()

        self.act_ids, self.q_adr, self.dq_adr = [], [], []
        for n in JOINT_NAMES:
            short = n.replace("_joint", "")
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, short)
            if aid < 0:
                raise KeyError(short)
            self.act_ids.append(aid)
            self.q_adr.append(sensor_adr(self.model, f"{short}_pos"))
            self.dq_adr.append(sensor_adr(self.model, f"{short}_vel"))
        self.imu_quat_adr = sensor_adr(self.model, "imu_quat")
        self.imu_gyro_adr = sensor_adr(self.model, "imu_gyro")
        self.frame_vel_adr = sensor_adr(self.model, "frame_vel")
        self.frame_pos_adr = sensor_adr(self.model, "frame_pos")
        self.muzzle_pos_adr = sensor_adr(self.model, "muzzle_pos")
        self.muzzle_quat_adr = sensor_adr(self.model, "muzzle_quat")

        self.last_action = np.zeros(12, dtype=np.float32)
        self.blend_start = self.default.copy()
        self.blend_t = 1.0
        self.episode_step = 0

    def reset_crouch(self):
        key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "crouch")
        if key >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.last_action[:] = 0
        self.episode_step = 0
        self.blend_t = 1.0

    def set_stand_pose(self, stand_q: np.ndarray):
        for name, q in zip(JOINT_NAMES, stand_q):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.data.qpos[self.model.jnt_qposadr[jid]] = q
            self.data.qvel[self.model.jnt_dofadr[jid]] = 0.0
        self.data.qpos[2] = float(self.cfg.get("spawn_z", 0.56))
        self.data.qpos[3:7] = np.array([1, 0, 0, 0], dtype=np.float64)
        mujoco.mj_forward(self.model, self.data)

    def read_state(self):
        q = np.array([self.data.sensordata[a] for a in self.q_adr], dtype=np.float32)
        dq = np.array([self.data.sensordata[a] for a in self.dq_adr], dtype=np.float32)
        quat = np.array(self.data.sensordata[self.imu_quat_adr : self.imu_quat_adr + 4], dtype=np.float32)
        gyro = np.array(self.data.sensordata[self.imu_gyro_adr : self.imu_gyro_adr + 3], dtype=np.float32)
        lin_w = np.array(self.data.sensordata[self.frame_vel_adr : self.frame_vel_adr + 3], dtype=np.float32)
        lin = quat_rotate_inverse_wxyz(quat, lin_w) if self.use_body_lin_vel else lin_w
        base_pos = np.array(self.data.sensordata[self.frame_pos_adr : self.frame_pos_adr + 3], dtype=np.float32)
        muzzle_pos = np.array(self.data.sensordata[self.muzzle_pos_adr : self.muzzle_pos_adr + 3], dtype=np.float32)
        muzzle_quat = np.array(self.data.sensordata[self.muzzle_quat_adr : self.muzzle_quat_adr + 4], dtype=np.float32)
        return q, dq, quat, gyro, lin, base_pos, muzzle_pos, muzzle_quat

    def aim_error(self, muzzle_pos, muzzle_quat) -> float:
        az = self.target_azimuth
        forward = quat_rotate_wxyz(muzzle_quat, np.array([1, 0, 0], dtype=np.float32))
        forward /= max(np.linalg.norm(forward), 1e-6)
        # Board-frame lateral miss (Isaac training intent). Do not fold pitch/sag
        # into aim_error or the twist actor saturates the hips.
        nx, ny = -math.cos(az), -math.sin(az)
        target = np.array(
            [self.target_distance * math.cos(az), self.target_distance * math.sin(az), self.target_height],
            dtype=np.float32,
        )
        denom = nx * forward[0] + ny * forward[1]
        if abs(denom) < 1e-6:
            return 1.0
        t = (nx * target[0] + ny * target[1] - (nx * muzzle_pos[0] + ny * muzzle_pos[1])) / denom
        if t <= 0:
            return 1.0
        impact = muzzle_pos + t * forward
        lat = abs((impact[1] - target[1]) * math.cos(az) - (impact[0] - target[0]) * math.sin(az))
        return float(np.clip(lat / (2.0 * self.hit_radius), 0.0, 1.0))

    def countdown(self) -> float:
        interval = float(self.cfg.get("shooting_recoil_interval", 0.08))
        policy_dt = self.dt * self.decimation
        cycle = max(int(interval / max(policy_dt, 1e-6)), 1)
        phase = (self.episode_step % cycle) / float(cycle)
        return 2.0 * phase - 1.0

    def build_obs(self, q, dq, quat, gyro, lin, muzzle_pos, muzzle_quat) -> np.ndarray:
        grav = quat_rotate_inverse_wxyz(quat, np.array([0, 0, -1], dtype=np.float32))
        az = self.target_azimuth
        obs = np.concatenate(
            [
                lin * self.lin_vel_scale,
                gyro * self.ang_vel_scale,
                grav,
                np.zeros(3, dtype=np.float32),
                q - self.default,
                dq * self.dof_vel_scale,
                self.last_action,
                np.array([self.aim_error(muzzle_pos, muzzle_quat)], dtype=np.float32),
                np.array([self.countdown()], dtype=np.float32),
                np.array([math.sin(az), math.cos(az)], dtype=np.float32),
            ]
        ).astype(np.float32)
        assert obs.shape[0] == 52
        return obs

    def pd_torque(self, q, dq, q_des) -> np.ndarray:
        return np.clip(self.kp * (q_des - q) - self.kd * dq, -self.torque_limits, self.torque_limits)

    def apply_torque(self, tau):
        for aid, t in zip(self.act_ids, tau):
            self.data.ctrl[aid] = t

    def hold_pose(self, q_des: np.ndarray, seconds: float, kp=None, kd=None):
        old_kp, old_kd = self.kp, self.kd
        if kp is not None:
            self.kp = np.asarray(kp, dtype=np.float32)
        if kd is not None:
            self.kd = np.asarray(kd, dtype=np.float32)
        for _ in range(int(seconds / self.dt)):
            q, dq, *_ = self.read_state()
            self.apply_torque(self.pd_torque(q, dq, q_des))
            mujoco.mj_step(self.model, self.data)
        self.kp, self.kd = old_kp, old_kd

    def seed_from_current(self):
        q, *_ = self.read_state()
        self.blend_start = q.copy()
        self.blend_t = 0.0
        self.last_action = (q - self.default) / np.maximum(self.action_scale, 1e-6)

    def policy_step(self):
        q, dq, quat, gyro, lin, base_pos, muzzle_pos, muzzle_quat = self.read_state()
        obs = self.build_obs(q, dq, quat, gyro, lin, muzzle_pos, muzzle_quat)
        with torch.no_grad():
            action = self.policy(torch.from_numpy(obs).view(1, -1)).numpy()[0].astype(np.float32)
        action = np.clip(action, self.clip_lo, self.clip_hi)
        self.last_action = action
        q_des = action * self.action_scale + self.default
        if self.blend_duration > 1e-6 and self.blend_t < 1.0:
            self.blend_t = min(1.0, self.blend_t + (self.dt * self.decimation) / self.blend_duration)
            a = self.blend_t
            q_des = (1 - a) * self.blend_start + a * q_des
        for _ in range(self.decimation):
            q, dq, *_ = self.read_state()
            self.apply_torque(self.pd_torque(q, dq, q_des))
            mujoco.mj_step(self.model, self.data)
        self.episode_step += 1
        q, dq, quat, gyro, lin, base_pos, muzzle_pos, muzzle_quat = self.read_state()
        roll, pitch, yaw = quat_to_rpy_deg(quat)
        return {
            "t": self.episode_step * self.dt * self.decimation,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "base_z": float(base_pos[2]),
            "aim_error": self.aim_error(muzzle_pos, muzzle_quat),
            "action_abs_max": float(np.max(np.abs(action))),
            "q_des_thigh_fr": float(q_des[5]),
            "q_thigh_fr": float(q[5]),
            "q_hip_fr": float(q[1]),
            "q_calf_fr": float(q[9]),
            "fallen": abs(roll) > 45 or abs(pitch) > 45 or base_pos[2] < 0.25,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mjcf", type=Path, default=ROOT / "src/rl_sar_zoo/q25_description/mjcf/shooting_stability.xml")
    parser.add_argument("--policy", type=Path, default=ROOT / "policy/q25/robot_lab/policy.pt")
    parser.add_argument("--config", type=Path, default=ROOT / "policy/q25/robot_lab/config.yaml")
    parser.add_argument("--base-yaml", type=Path, default=ROOT / "policy/q25/base.yaml")
    parser.add_argument("--hold-s", type=float, default=2.0)
    parser.add_argument("--policy-s", type=float, default=8.0)
    parser.add_argument("--clip", type=float, default=None)
    parser.add_argument("--blend", type=float, default=None)
    parser.add_argument("--no-body-lin-vel", action="store_true")
    parser.add_argument("--stand-default", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("/tmp/mujoco_rollout.csv"))
    parser.add_argument("--tag", type=str, default="run")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    base = yaml.safe_load(args.base_yaml.read_text(encoding="utf-8"))["q25"]
    cfg.setdefault("dt", base.get("dt", 0.005))
    cfg.setdefault("decimation", base.get("decimation", 4))
    cfg.setdefault("spawn_z", base.get("spawn_z", 0.56))
    stand = np.asarray(base.get("shooting_stand_dof_pos", cfg["default_dof_pos"]), dtype=np.float32)
    if args.stand_default:
        stand = np.asarray(cfg["default_dof_pos"], dtype=np.float32)
    blend = args.blend if args.blend is not None else float(cfg.get("shooting_policy_blend_duration", 1.5))

    env = Q25ShootingRollout(
        args.mjcf, args.policy, cfg, clip_actions=args.clip, blend_duration=blend, use_body_lin_vel=not args.no_body_lin_vel
    )
    env.reset_crouch()
    env.set_stand_pose(stand)
    env.hold_pose(
        stand,
        args.hold_s,
        kp=np.asarray(base.get("stand_hold_kp", cfg["rl_kp"]), dtype=np.float32),
        kd=np.asarray(base.get("stand_hold_kd", cfg["rl_kd"]), dtype=np.float32),
    )
    env.seed_from_current()

    rows, fallen_at = [], None
    for _ in range(int(args.policy_s / (env.dt * env.decimation))):
        info = env.policy_step()
        rows.append(info)
        if info["fallen"]:
            fallen_at = info["t"]
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["t", "roll", "pitch", "yaw", "base_z", "aim_error", "action_abs_max", "q_des_thigh_fr", "q_thigh_fr", "q_hip_fr", "q_calf_fr", "fallen"]
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})

    if not rows:
        print(f"[{args.tag}] FAIL no steps")
        return 2
    pitches = [r["pitch"] for r in rows]
    zs = [r["base_z"] for r in rows]
    ok = fallen_at is None and abs(pitches[-1]) < 20 and zs[-1] > 0.30
    print(
        f"[{args.tag}] steps={len(rows)} fallen_at={fallen_at} "
        f"pitch_end={pitches[-1]:.2f} pitch_max={max(abs(p) for p in pitches):.2f} "
        f"z_end={zs[-1]:.3f} z_min={min(zs):.3f} aim_end={rows[-1]['aim_error']:.3f} "
        f"hipFR={rows[-1]['q_hip_fr']:.3f} thFR={rows[-1]['q_thigh_fr']:.3f} "
        f"roll_end={rows[-1]['roll']:.2f} => {'PASS' if ok else 'FAIL'}"
    )
    print(f"[{args.tag}] csv={args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
