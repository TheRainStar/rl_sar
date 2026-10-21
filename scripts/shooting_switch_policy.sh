#!/usr/bin/env bash
# Optional helper: copy a named export onto policy.pt (fallback name).
# Runtime FSM loads shooting_frontal_model / shooting_offset_model from config.yaml
# (policy_stab0614.pt vs policy_twist.pt) and does not require this script.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIR="$ROOT/policy/q25/robot_lab"
MODE="${1:-}"

usage() {
  echo "Usage: $0 [twist|stab0614|show]"
  echo "  twist     : copy twist export onto policy.pt (fallback only)"
  echo "  stab0614  : copy June hold export onto policy.pt (fallback only)"
  echo "  show      : print current policy md5"
  echo
  echo "Preferred: set shooting_frontal_model / shooting_offset_model in config.yaml."
}

case "$MODE" in
  show|"")
    md5sum "$DIR/policy.pt" "$DIR/policy_stab0614.pt" "$DIR/policy_twist.pt" 2>/dev/null || true
    ;;
  twist)
    if [[ ! -f "$DIR/policy_twist.pt" ]]; then
      cp -f "$DIR/policy.pt" "$DIR/policy_twist.pt"
    fi
    cp -f "$DIR/policy_twist.pt" "$DIR/policy.pt"
    echo "[OK] policy.pt <- policy_twist.pt"
    md5sum "$DIR/policy.pt"
    ;;
  stab0614)
    if [[ ! -f "$DIR/policy_twist.pt" ]]; then
      cp -f "$DIR/policy.pt" "$DIR/policy_twist.pt"
    fi
    cp -f "$DIR/policy_stab0614.pt" "$DIR/policy.pt"
    echo "[OK] policy.pt <- policy_stab0614.pt"
    md5sum "$DIR/policy.pt"
    ;;
  *)
    usage; exit 1
    ;;
esac
