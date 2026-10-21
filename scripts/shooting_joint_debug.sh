#!/usr/bin/env bash
# Joint debug helper: sync config, remind Isaac/MuJoCo commands, compare CSVs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBOT_LAB="${ROBOT_LAB:-$HOME/robot_lab}"
ISAAC_CSV="${ISAAC_CSV:-/tmp/isaac_shooting_telemetry.csv}"
MUJOCO_CSV="${MUJOCO_CSV:-$ROOT/policy/q25/shooting_telemetry.csv}"
COMPARE_OUT="${COMPARE_OUT:-/tmp/shooting_compare}"
POLICY_SRC="${POLICY_SRC:-}"

cd "$ROOT"

echo "==> [1/3] Sync / check deploy config against robot_lab"
python3 "$ROOT/scripts/shooting_sync_config.py" --robot-lab "$ROBOT_LAB" --check || true
if [[ "${APPLY_CONFIG:-0}" == "1" ]]; then
  python3 "$ROOT/scripts/shooting_sync_config.py" --robot-lab "$ROBOT_LAB" --apply
fi
if [[ -n "$POLICY_SRC" ]]; then
  python3 "$ROOT/scripts/shooting_sync_config.py" --copy-policy "$POLICY_SRC"
fi

echo
echo "==> [2/3] Capture telemetry"
echo "Isaac play (in robot_lab env), example:"
echo "  cd \"$ROBOT_LAB\""
echo "  python scripts/reinforcement_learning/rsl_rl/play.py \\"
echo "    --task RobotLab-Isaac-ShootingStability-Deeprobotics-Q25-v0 \\"
echo "    --num_envs 1 --play_straight_ahead --play_no_co_rotate \\"
echo "    --play_telemetry_csv $ISAAC_CSV --play_telemetry_interval 0.5 \\"
echo "    --play_telemetry_max_time 20"
echo
echo "MuJoCo (in rl_sar):"
echo "  ./cmake_build/bin/rl_sim_mujoco q25 shooting_stability"
echo "  # press 0 -> 1 (-> 2), CSV writes to:"
echo "  #   $MUJOCO_CSV"

if [[ "${COMPARE:-0}" == "1" ]]; then
  echo
  echo "==> [3/3] Compare CSVs"
  python3 "$ROOT/scripts/shooting_compare_telemetry.py" \
    --isaac "$ISAAC_CSV" \
    --mujoco "$MUJOCO_CSV" \
    --out "$COMPARE_OUT"
else
  echo
  echo "==> [3/3] Compare later with:"
  echo "  COMPARE=1 ISAAC_CSV=$ISAAC_CSV MUJOCO_CSV=$MUJOCO_CSV $0"
  echo "  # or:"
  echo "  python3 scripts/shooting_compare_telemetry.py --isaac $ISAAC_CSV --mujoco $MUJOCO_CSV --out $COMPARE_OUT"
fi
