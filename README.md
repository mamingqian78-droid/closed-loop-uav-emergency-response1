# Code Package for This Manuscript

This folder contains the code package prepared for journal submission.
It reproduces the YOLOv8 detection results, closed-loop UAV dispatch
simulation, baseline comparisons, robustness analysis, reward ablation,
latency benchmarking, and final figures/tables.

## Quick Start

```bash
cd code
pip install -r requirements.txt
bash scripts/reproduce_all.sh
```

The main fully executable pipeline is:

```bash
python src/training/final_pipeline.py --config configs/default.yaml
```

The plotting script can regenerate Fig. 5-Fig. 10 from existing CSV/NPY
records:

```bash
python src/evaluation/make_figures.py
```

## Directory Map

- `configs/`: YAML files for each experiment condition.
- `src/envs/`: Gym-style UAV emergency response environment and helpers.
- `src/perception/`: YOLOv8, empirical confusion matrix, and noise models.
- `src/dssm/`: disaster-state matrix and local patch extraction.
- `src/policies/`: PPO/DQN/A2C/random/greedy policy definitions.
- `src/training/`: training loops and the complete final pipeline.
- `src/evaluation/`: metrics, statistical tests, latency benchmark, and plotting.
- `scripts/`: shell entry points for reproducing the manuscript figures/tables.

## Data

The expected AIDERv2 directory layout is:

```text
data/aiderv2/
  train/{earthquake,flood,normal,wildfire}
  val/{earthquake,flood,normal,wildfire}
  test/{earthquake,flood,normal,wildfire}
```

If the dataset is not already present, set `AIDERV2_URL` and run:

```bash
bash src/perception/aiderv2_download.sh
```

## Reproducibility Notes

Stable-Baselines3 may log up to 140288 steps instead of the nominal 140000 because PPO rollouts are aligned to `n_steps=512`. All seed assignments, hyperparameters, and training steps are specified in the paper (Table 1) and in `configs/*.yaml`.
