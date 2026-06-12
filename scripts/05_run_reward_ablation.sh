#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
python "${ROOT_DIR}/src/training/rerun_reward_phase.py" --config "${ROOT_DIR}/configs/default.yaml"
