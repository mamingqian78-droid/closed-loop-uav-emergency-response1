#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
python - <<PY
from pathlib import Path
from training.final_pipeline import load_config, ensure_dirs
from evaluation.make_figures import fig9, fig10, set_style

root = Path("${ROOT_DIR}")
cfg = load_config(root / "configs" / "default.yaml")
dirs = ensure_dirs(cfg["results_dir"])
set_style(cfg["figures"]["font"])
fig9(dirs, int(cfg["figures"]["dpi"]))
fig10(dirs, int(cfg["figures"]["dpi"]))
print("Fig. 9 and Fig. 10 regenerated from robustness records.")
PY
