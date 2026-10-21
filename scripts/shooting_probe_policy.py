#!/usr/bin/env python3
"""Compare twist policy.pt on Isaac-reset vs MuJoCo seed observations."""

from __future__ import annotations

import csv
from pathlib import Path

import torch

POLICY = Path("/home/liuzhen/workspace/rl_sar/policy/q25/robot_lab/policy.pt")
OBS_DUMP = Path("/home/liuzhen/workspace/rl_sar/policy/q25/shooting_obs_dump.csv")
DEFAULT = torch.tensor([0, 0, 0, 0, -0.67, -0.67, -0.67, -0.67, 1.3, 1.3, 1.3, 1.3])
SCALE = torch.tensor([0.125, 0.125, 0.125, 0.125, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
NAMES = [
    "FL_hip", "FR_hip", "RL_hip", "RR_hip",
    "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
    "FL_calf", "FR_calf", "RL_calf", "RR_calf",
]


def load_latest_seed(path: Path) -> torch.Tensor | None:
    if not path.is_file():
        return None
    last = None
    with path.open() as f:
        for row in csv.DictReader(f):
            if row.get("tag") == "seed":
                last = row
    if last is None:
        return None
    return torch.tensor([float(last[f"o{i}"]) for i in range(52)], dtype=torch.float32)


def isaac_reset_obs(
    *,
    aim_error: float = 0.0,
    last_action: torch.Tensor | None = None,
    countdown: float = -1.0,
    azimuth: tuple[float, float] = (0.0, 1.0),
    gravity: tuple[float, float, float] = (0.0, 0.0, -1.0),
    lin_vel: tuple[float, float, float] = (0.0, 0.0, 0.0),
    joint_pos: torch.Tensor | None = None,
) -> torch.Tensor:
    obs = torch.zeros(52)
    obs[0:3] = torch.tensor(lin_vel)
    obs[3:6] = 0.0  # ang_vel * 0.25
    obs[6:9] = torch.tensor(gravity)
    obs[9:12] = 0.0
    if joint_pos is not None:
        obs[12:24] = joint_pos
    obs[24:36] = 0.0
    if last_action is not None:
        obs[36:48] = last_action
    obs[48] = aim_error
    obs[49] = countdown
    obs[50] = azimuth[0]
    obs[51] = azimuth[1]
    return obs


def run(model, obs: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        out = model(obs.unsqueeze(0)).squeeze(0).cpu()
    return out


def report(title: str, actions: torch.Tensor) -> None:
    q = DEFAULT + actions * SCALE
    print(f"\n=== {title} ===")
    print("actions:", " ".join(f"{a:+.3f}" for a in actions.tolist()))
    print("q_des  :", " ".join(f"{v:+.3f}" for v in q.tolist()))
    print("sat±1  :", int((actions.abs() >= 0.99).sum()), "/12")
    front_thigh = float(q[4:6].mean())
    rear_thigh = float(q[6:8].mean())
    rear_calf = float(q[10:12].mean())
    print(
        f"front_thigh={front_thigh:+.3f} rear_thigh={rear_thigh:+.3f} "
        f"rear_calf={rear_calf:+.3f}  (kneel if rear_calf>1.65 or front_thigh<-1.2)"
    )


def probe_file(path: Path) -> None:
    print(f"\n########## {path} ##########")
    model = torch.jit.load(str(path), map_location="cpu")
    model.eval()
    seed = load_latest_seed(OBS_DUMP)
    report("Isaac reset: zeros + aim=0 + last_action=0", run(model, isaac_reset_obs(aim_error=0.0)))
    report("Isaac reset: zeros + aim=1 + last_action=0", run(model, isaac_reset_obs(aim_error=1.0)))
    if seed is not None:
        report("MuJoCo seed obs (as dumped)", run(model, seed))


def main() -> None:
    candidates = [
        POLICY,
        Path("/home/liuzhen/workspace/rl_sar/policy/q25/robot_lab/policy_stab0614.pt"),
        Path("/home/liuzhen/robot_lab/logs/rsl_rl/deeprobotics_q25_shooting_twist/2026-07-22_00-26-38/exported/policy.pt"),
    ]
    for path in candidates:
        if path.is_file():
            probe_file(path)
    return
    model = torch.jit.load(str(POLICY), map_location="cpu")
    model.eval()

    seed = load_latest_seed(OBS_DUMP)
    if seed is not None:
        report("MuJoCo seed obs (as dumped)", run(model, seed))
        patched = seed.clone()
        patched[36:48] = 0
        report("MuJoCo seed but last_action=0 (Isaac reset)", run(model, patched))
        patched2 = seed.clone()
        patched2[36:48] = 0
        patched2[48] = 0.0
        report("MuJoCo seed, last_action=0, aim_error=0", run(model, patched2))
        patched3 = seed.clone()
        patched3[0:3] = 0
        patched3[36:48] = 0
        patched3[48] = 0.0
        report("MuJoCo seed, lin_vel=0, last_action=0, aim_error=0", run(model, patched3))

    report("Isaac reset: zeros + aim=0 + last_action=0", run(model, isaac_reset_obs(aim_error=0.0)))
    report("Isaac reset: zeros + aim=1 + last_action=0", run(model, isaac_reset_obs(aim_error=1.0)))
    report("Isaac reset: zeros + aim=0.5 + last_action=0", run(model, isaac_reset_obs(aim_error=0.5)))

    if seed is not None:
        # Reorder MuJoCo grouped joint_pos/vel into URDF interleaved (FL h/t/c, FR h/t/c, ...)
        g2u = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
        urdf = seed.clone()
        jp = seed[12:24]
        jv = seed[24:36]
        urdf[12:24] = jp[g2u]
        urdf[24:36] = jv[g2u]
        report("MuJoCo seed with joint_pos/vel in URDF interleaved order", run(model, urdf))
        urdf[36:48] = 0
        urdf[48] = 0
        report("URDF-order joints + last_action=0 + aim=0", run(model, urdf))

        # Velocity-Q25 order: FR h/t/c, FL h/t/c, RR h/t/c, RL h/t/c
        # grouped FL/FR/RL/RR hip,thigh,calf -> velocity
        # grouped idx: 0 FL_h 1 FR_h 2 RL_h 3 RR_h 4 FL_t 5 FR_t 6 RL_t 7 RR_t 8 FL_c 9 FR_c 10 RL_c 11 RR_c
        g2v = [1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10]
        vel = seed.clone()
        vel[12:24] = jp[g2v]
        vel[24:36] = jv[g2v]
        report("MuJoCo seed with joint_pos/vel in velocity FR/FL/RR/RL order", run(model, vel))


if __name__ == "__main__":
    main()
