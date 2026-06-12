#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

bash "${ROOT_DIR}/scripts/01_train_yolov8.sh"
bash "${ROOT_DIR}/scripts/02_run_closed_loop_ablation.sh"
bash "${ROOT_DIR}/scripts/03_run_algorithmic_baselines.sh"
bash "${ROOT_DIR}/scripts/04_run_noise_robustness.sh"
bash "${ROOT_DIR}/scripts/05_run_reward_ablation.sh"
bash "${ROOT_DIR}/scripts/06_run_latency_benchmark.sh"
