#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
python - <<PY
from pathlib import Path
from training.final_pipeline import load_config, ensure_dirs, prepare_detection
from evaluation.make_figures import make_all_figures

root = Path("${ROOT_DIR}")
cfg = load_config(root / "configs" / "default.yaml")
dirs = ensure_dirs(cfg["results_dir"])
prepare_detection(cfg, dirs)
make_all_figures(cfg, dirs)
print("Fig. 5 detection artifacts regenerated.")
PY
