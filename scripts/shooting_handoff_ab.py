#!/usr/bin/env python3
"""Compare MuJoCo policy handoff recipes (no GUI)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path("/home/liuzhen/workspace/rl_sar")
sys.path.insert(0, str(ROOT / "scripts"))
from shooting_mujoco_rollout import Q25ShootingRollout  # noqa: E402


def run(tag: str, *, policy: Path, isaac_reset: bool, last_action_zero: bool, clip: float, kp: float, kd: float, z: float, hold_s: float, blend: float, stand_default: bool) -> None:
    cfg = yaml.safe_load((ROOT / "policy/q25/robot_lab/config.yaml").read_text())["q25/robot_lab"]
    base = yaml.safe_load((ROOT / "policy/q25/base.yaml").read_text())["q25"]
    cfg.setdefault("dt", 0.005)
    cfg.setdefault("decimation", 4)
    cfg["rl_kp"] = [kp] * 12
    cfg["rl_kd"] = [kd] * 12
    env = Q25ShootingRollout(
        ROOT / "src/rl_sar_zoo/q25_description/mjcf/shooting_stability.xml",
        policy,
        cfg,
        clip_actions=clip,
        blend_duration=blend,
    )
    env.reset_crouch()
    stand = np.asarray(cfg["default_dof_pos"] if stand_default or isaac_reset else base["shooting_stand_dof_pos"], dtype=np.float32)
    env.set_stand_pose(stand)
    env.data.qpos[2] = z
    env.data.qpos[0:2] = 0.0
    env.data.qpos[3:7] = np.array([1, 0, 0, 0], dtype=np.float64)
    env.data.qvel[:] = 0
    import mujoco
    mujoco.mj_forward(env.model, env.data)
    if hold_s > 0:
        env.hold_pose(stand, hold_s, kp=np.asarray(base["stand_hold_kp"], dtype=np.float32), kd=np.asarray(base["stand_hold_kd"], dtype=np.float32))
    q, dq, quat, gyro, lin, base_pos, muzzle_pos, muzzle_quat = env.read_state()
    env.blend_start = q.copy()
    env.blend_t = 0.0 if blend > 0 else 1.0
    env.last_action[:] = 0.0 if last_action_zero else (q - env.default) / np.maximum(env.action_scale, 1e-6)
    env.episode_step = 0
    print(
        f"\n[{tag}] start z={base_pos[2]:.3f} aim={env.aim_error(muzzle_pos, muzzle_quat):.3f} "
        f"q_th={q[4:8].round(3)} last_a0={env.last_action[0]:.3f} policy={policy.name}"
    )
    fallen = None
    last = None
    for i in range(int(4.0 / (env.dt * env.decimation))):
        last = env.policy_step()
        if i in (0, 9, 49):
            print(f"  t={last['t']:.2f} z={last['base_z']:.3f} pitch={last['pitch']:.1f} aim={last['aim_error']:.2f} |a|={last['action_abs_max']:.2f}")
        if last["fallen"]:
            fallen = last["t"]
            break
    print(
        f"  => {'FAIL' if fallen else 'held'} fallen_at={fallen} "
        f"z={last['base_z']:.3f} pitch={last['pitch']:.1f}" if last else "  => no steps"
    )


def main() -> None:
    twist = ROOT / "policy/q25/robot_lab/policy.pt"
    stab = ROOT / "policy/q25/robot_lab/policy_stab0614.pt"
    cases = [
        ("current-handoff twist", dict(policy=twist, isaac_reset=False, last_action_zero=True, clip=1.0, kp=80, kd=2, z=0.58, hold_s=1.5, blend=1.5, stand_default=False)),
        ("isaac-reset twist", dict(policy=twist, isaac_reset=True, last_action_zero=True, clip=1.0, kp=80, kd=2, z=0.45, hold_s=0.0, blend=0.0, stand_default=True)),
        ("isaac-reset stab0614", dict(policy=stab, isaac_reset=True, last_action_zero=True, clip=1.0, kp=80, kd=2, z=0.45, hold_s=0.0, blend=0.0, stand_default=True)),
        ("isaac-reset twist z0.52", dict(policy=twist, isaac_reset=True, last_action_zero=True, clip=1.0, kp=80, kd=2, z=0.52, hold_s=0.0, blend=0.0, stand_default=True)),
    ]
    for tag, kwargs in cases:
        run(tag, **kwargs)


if __name__ == "__main__":
    main()
