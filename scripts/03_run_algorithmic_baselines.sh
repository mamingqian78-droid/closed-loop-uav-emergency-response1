#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
python - <<PY
from pathlib import Path
from training.final_pipeline import load_config, ensure_dirs, make_tables

root = Path("${ROOT_DIR}")
cfg = load_config(root / "configs" / "default.yaml")
dirs = ensure_dirs(cfg["results_dir"])
make_tables(dirs)
print("Table 4 regenerated from phase3 baseline records.")
PY
