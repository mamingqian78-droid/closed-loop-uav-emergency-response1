#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
python - <<PY
from pathlib import Path
from sb3_contrib import MaskablePPO
from training.final_pipeline import load_config, ensure_dirs, measure_efficiency

root = Path("${ROOT_DIR}")
cfg = load_config(root / "configs" / "default.yaml")
dirs = ensure_dirs(cfg["results_dir"])
model_path = dirs["models"] / f"C4_ppo_seed{cfg['train_seeds'][0]}.zip"
model = MaskablePPO.load(str(model_path), device="cpu")
measure_efficiency(cfg, dirs, model)
print(f"Table 6 regenerated at {dirs['tables']}")
PY
