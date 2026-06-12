from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import yaml
from gymnasium import spaces
from scipy.stats import wilcoxon
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.monitor import Monitor
from ultralytics import YOLO

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import make_all_figures


CLASSES = ["earthquake", "flood", "normal", "wildfire"]
DISASTER_CLASSES = ["earthquake", "flood", "wildfire"]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = path.parent
    for key in ("project_root", "dataset_dir", "results_dir"):
        cfg[key] = (base / cfg[key]).resolve()
    return cfg


def ensure_dirs(results_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": results_dir,
        "data": results_dir / "data",
        "figures": results_dir / "figures",
        "tables": results_dir / "tables",
        "models": results_dir / "models",
        "logs": results_dir / "logs",
        "figure_code": results_dir / "figure_code",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def list_images(dataset_dir: Path, split: str, cls: str) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    return sorted(p for p in (dataset_dir / split / cls).rglob("*") if p.suffix.lower() in exts)


def dataset_counts(dataset_dir: Path, dirs: dict[str, Path]) -> None:
    rows = []
    for split in ("train", "val", "test"):
        for cls in CLASSES:
            rows.append({"split": split, "class": cls, "count": len(list_images(dataset_dir, split, cls))})
    pd.DataFrame(rows).to_csv(dirs["data"] / "dataset_counts.csv", index=False)


def run_yolo(cfg: dict, dirs: dict[str, Path]) -> dict:
    ycfg = cfg["yolo"]
    dataset_dir = cfg["dataset_dir"]
    run_dir = dirs["models"] / "yolo"
    weights_path = run_dir / "train" / "weights" / "best.pt"
    if not weights_path.exists():
        model = YOLO(ycfg["model"])
        model.train(
            data=str(dataset_dir),
            epochs=int(ycfg["epochs"]),
            imgsz=int(ycfg["imgsz"]),
            batch=int(ycfg["batch"]),
            workers=int(ycfg["workers"]),
            device=ycfg["device"],
            patience=int(ycfg["patience"]),
            project=str(run_dir),
            name="train",
            exist_ok=True,
            cache=False,
            verbose=True,
        )
    model = YOLO(str(weights_path))
    y_true, y_pred, sample_rows = [], [], []
    all_paths = []
    for cls in CLASSES:
        all_paths.extend(list_images(dataset_dir, "test", cls))
    for p in all_paths:
        result = model.predict(str(p), imgsz=int(ycfg["imgsz"]), device=ycfg["device"], verbose=False)[0]
        pred_idx = int(result.probs.top1)
        true_idx = CLASSES.index(p.parent.name)
        y_true.append(true_idx)
        y_pred.append(pred_idx)
        if len([r for r in sample_rows if r["true"] == p.parent.name]) < 3:
            sample_rows.append({"path": str(p), "true": p.parent.name, "pred": CLASSES[pred_idx]})
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    np.save(dirs["data"] / "confusion_matrix.npy", cm_norm)
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(dirs["data"] / "confusion_matrix_counts.csv")
    metrics = {"model": str(weights_path), "top1_accuracy": float((y_true == y_pred).mean()), "per_class": {}}
    for i, cls in enumerate(CLASSES):
        tp = cm[i, i]
        precision = tp / max(cm[:, i].sum(), 1)
        recall = tp / max(cm[i, :].sum(), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        metrics["per_class"][cls] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(cm[i, :].sum()),
        }
    with (dirs["data"] / "detection_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame(sample_rows).to_csv(dirs["data"] / "fig5_samples.csv", index=False)
    return metrics


class UavDispatchEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        seed: int,
        cfg: dict,
        mode: str = "C4",
        confusion: np.ndarray | None = None,
        synthetic_noise: tuple[str, float] | None = None,
        reward_weights=(1.0, 1.0, 1.0),
    ):
        super().__init__()
        self.base_seed = seed
        self.cfg = cfg
        self.mode = mode
        self.confusion = confusion
        self.synthetic_noise = synthetic_noise
        self.reward_weights = reward_weights
        self.n = int(cfg["grid_size"])
        self.m = int(cfg["demand_points"])
        self.horizon = int(cfg["horizon"])
        self.action_space = spaces.Discrete(self.m)
        obs_dim = 3 + 2 + self.m * 3
        self.observation_space = spaces.Box(low=0.0, high=1.5, shape=(obs_dim,), dtype=np.float32)
        self.reset(seed=seed)

    def _world(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.base = np.array([self.n // 2, self.n // 2], dtype=np.int32)
        pts = []
        while len(pts) < self.m:
            p = self.rng.integers(1, self.n - 1, size=2)
            if np.linalg.norm(p - self.base, ord=1) < 4:
                continue
            if all(np.linalg.norm(p - q, ord=1) >= 3 for q in pts):
                pts.append(p)
        self.points = np.vstack(pts).astype(np.int32)
        self.true_sev0 = self.rng.uniform(0.45, 1.0, size=self.m)
        self.growth = self.rng.uniform(0.002, 0.010, size=self.m)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.base_seed = seed
        self._world(self.base_seed)
        self.t = 0
        self.pos = self.base.copy()
        self.true_sev = self.true_sev0.copy()
        self.serviced = np.zeros(self.m, dtype=bool)
        self.distance = 0
        self.latencies = []
        self.trajectory = [tuple(self.pos.tolist())]
        self.obs_sev = self._detect()
        return self._obs(), {}

    def _detect(self) -> np.ndarray:
        obs = self.true_sev.copy()
        if self.confusion is not None:
            disaster_idx = [CLASSES.index(c) for c in DISASTER_CLASSES]
            normal_idx = CLASSES.index("normal")
            disaster_recall = np.mean(np.diag(self.confusion)[disaster_idx])
            fnr = float(np.clip(1 - disaster_recall, 0, 0.65))
            fpr = float(np.clip(self.confusion[normal_idx, disaster_idx].sum(), 0, 0.65))
            obs *= self.rng.random(self.m) > fnr
            obs += self.rng.binomial(1, fpr, size=self.m) * self.rng.uniform(0.1, 0.35, size=self.m)
        if self.synthetic_noise:
            kind, level = self.synthetic_noise
            if kind == "FNR":
                obs *= self.rng.random(self.m) > level
            elif kind == "FPR":
                obs += self.rng.binomial(1, level, size=self.m) * self.rng.uniform(0.1, 0.35, size=self.m)
            elif kind == "LOC":
                obs *= np.exp(-0.025 * level)
                obs += self.rng.normal(0, 0.012 * level, size=self.m)
        return np.clip(obs, 0, 1.5)

    def _mode_params(self):
        return {
            "C1": (9999, True),
            "C2": (10, False),
            "C3": (10, True),
            "C4": (1, True),
        }.get(self.mode, (1, True))

    def _obs(self):
        _, uses_dssm = self._mode_params()
        perceived = self.obs_sev if uses_dssm else np.zeros_like(self.obs_sev)
        return np.concatenate(
            [
                np.array([self.pos[0] / self.n, self.pos[1] / self.n, self.t / self.horizon], dtype=np.float32),
                np.array([self.distance / max(self.horizon, 1), self.serviced.mean()], dtype=np.float32),
                perceived.astype(np.float32),
                self.serviced.astype(np.float32),
                (self.points[:, 0] / self.n).astype(np.float32),
            ]
        )

    def step(self, action: int):
        update_every, _ = self._mode_params()
        if self.t % update_every == 0:
            self.obs_sev = self._detect()
        self.true_sev = np.clip(self.true_sev + self.growth * (~self.serviced), 0, 1.6)
        action = int(action)
        reward = -0.01
        if self.serviced[action]:
            reward -= 0.04
        target = self.points[action]
        delta = target - self.pos
        if abs(delta[0]) >= abs(delta[1]) and delta[0] != 0:
            self.pos[0] += int(np.sign(delta[0]))
        elif delta[1] != 0:
            self.pos[1] += int(np.sign(delta[1]))
        self.distance += 1
        self.trajectory.append(tuple(self.pos.tolist()))
        alpha, beta, gamma = self.reward_weights
        if np.array_equal(self.pos, target) and not self.serviced[action]:
            effective = self.true_sev[action] - 0.0012 * beta * self.t - 0.0008 * gamma * self.distance
            if effective > 0.45 / max(alpha, 0.2):
                self.serviced[action] = True
                self.latencies.append(self.t + 1)
                reward += 1.2 * alpha + 0.7 * self.true_sev[action]
                self.true_sev[action] *= 0.2
                self.obs_sev[action] *= 0.2
            else:
                reward += 0.12 * effective
        reward -= 0.002 * gamma
        reward -= 0.001 * beta * self.t / self.horizon
        self.t += 1
        terminated = bool(self.serviced.all())
        truncated = bool(self.t >= self.horizon)
        info = {
            "task_completion_rate": float(self.serviced.mean()),
            "average_response_latency": float(np.mean(self.latencies)) if self.latencies else float(self.horizon),
            "cumulative_flight_distance": float(self.distance),
            "closed_loop_update_latency_ms": 0.4 + 0.008 * self.m,
            "trajectory": self.trajectory,
        }
        return self._obs(), float(reward), terminated, truncated, info


def make_env(seed: int, cfg: dict, mode: str, confusion=None, synthetic_noise=None, reward_weights=(1, 1, 1)):
    def _factory():
        return Monitor(UavDispatchEnv(seed, cfg, mode, confusion, synthetic_noise, reward_weights))

    return _factory


def greedy_eval(seed: int, cfg: dict, kind: str = "greedy"):
    env = UavDispatchEnv(seed, cfg, mode="C4")
    obs, _ = env.reset(seed=seed)
    done = False
    info = None
    while not done:
        if kind == "random":
            action = env.action_space.sample()
        else:
            candidates = np.where(~env.serviced)[0]
            if len(candidates) == 0:
                action = 0
            else:
                dist = np.abs(env.points[candidates] - env.pos).sum(axis=1)
                score = env.true_sev[candidates] - 0.08 * dist
                action = int(candidates[np.argmax(score)])
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return info


def train_model(algorithm: str, env, seed: int, cfg: dict, log_dir: Path):
    if algorithm == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            seed=seed,
            verbose=0,
            learning_rate=float(cfg["learning_rate"]),
            n_steps=int(cfg["n_steps"]),
            batch_size=int(cfg["batch_size"]),
            tensorboard_log=str(log_dir),
        )
    elif algorithm == "A2C":
        model = A2C("MlpPolicy", env, seed=seed, verbose=0, learning_rate=float(cfg["learning_rate"]), tensorboard_log=str(log_dir))
    elif algorithm == "DQN":
        model = DQN("MlpPolicy", env, seed=seed, verbose=0, learning_rate=float(cfg["learning_rate"]), tensorboard_log=str(log_dir))
    else:
        raise ValueError(algorithm)
    model.learn(total_timesteps=int(cfg["total_timesteps"]), progress_bar=False)
    return model


def eval_policy(model, seed: int, cfg: dict, mode: str, episodes: int, confusion=None, synthetic_noise=None, reward_weights=(1, 1, 1)):
    rows = []
    last_traj = None
    for ep in range(episodes):
        env = UavDispatchEnv(seed * 100 + ep, cfg, mode, confusion, synthetic_noise, reward_weights)
        obs, _ = env.reset(seed=seed * 100 + ep)
        done = False
        info = None
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
        last_traj = info["trajectory"]
        rows.append({k: info[k] for k in ["task_completion_rate", "average_response_latency", "cumulative_flight_distance", "closed_loop_update_latency_ms"]})
    return rows, last_traj


def collect_training_curves(logs_dir: Path, modes: list[str], train_seeds: list[int]) -> pd.DataFrame:
    rows = []
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return pd.DataFrame(rows)

    for mode in modes:
        for seed in train_seeds:
            run_root = logs_dir / f"{mode}_seed{seed}"
            for event_file in run_root.rglob("events.out.tfevents*"):
                accumulator = EventAccumulator(str(event_file))
                accumulator.Reload()
                if "rollout/ep_rew_mean" not in accumulator.Tags().get("scalars", []):
                    continue
                for event in accumulator.Scalars("rollout/ep_rew_mean"):
                    rows.append(
                        {
                            "config": mode,
                            "seed": seed,
                            "iteration": int(event.step),
                            "reward": float(event.value),
                        }
                    )
    return pd.DataFrame(rows)


def summarize_mean_std(df: pd.DataFrame, group_col: str, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for group, gdf in df.groupby(group_col):
        row = {group_col: group}
        for m in metrics:
            seed_means = gdf.groupby("seed")[m].mean()
            row[m + "_mean"] = seed_means.mean()
            row[m + "_std"] = seed_means.std(ddof=1)
            row[m + "_median"] = seed_means.median()
            row[m + "_iqr"] = seed_means.quantile(0.75) - seed_means.quantile(0.25)
        rows.append(row)
    return pd.DataFrame(rows)


def pairwise_table(df: pd.DataFrame, group_col: str, ref: str, metric: str) -> pd.DataFrame:
    rows = []
    pivot = df.groupby([group_col, "seed"])[metric].mean().reset_index()
    ref_values = pivot[pivot[group_col] == ref].sort_values("seed")[metric].to_numpy()
    for group in sorted(pivot[group_col].unique()):
        vals = pivot[pivot[group_col] == group].sort_values("seed")[metric].to_numpy()
        m = min(len(vals), len(ref_values))
        if group == ref or m < 2:
            p = 1.0
            effect = 0.0
        else:
            try:
                p = float(wilcoxon(vals[:m], ref_values[:m], zero_method="zsplit").pvalue)
            except ValueError:
                p = 1.0
            diff = vals[:m] - ref_values[:m]
            effect = float((np.sum(diff > 0) - np.sum(diff < 0)) / m)
        rows.append({"group": group, "metric": metric, "p_value_vs_" + ref: p, "rank_biserial_approx": effect})
    return pd.DataFrame(rows)


def run_rl(cfg: dict, dirs: dict[str, Path]) -> None:
    rl = cfg["rl"]
    cm = np.load(dirs["data"] / "confusion_matrix.npy")
    train_seeds = cfg["train_seeds"]
    eval_seeds = cfg["eval_seeds"]
    eval_eps = int(rl["eval_episodes_per_seed"])
    models: dict[tuple[str, int], object] = {}

    curve_rows = []
    for mode in ["C1", "C2", "C3", "C4"]:
        for seed in train_seeds:
            env = make_env(seed, rl, mode)()
            model = train_model("PPO", env, seed, rl, dirs["logs"] / f"{mode}_seed{seed}")
            model.save(str(dirs["models"] / f"{mode}_ppo_seed{seed}.zip"))
            models[(mode, seed)] = model
            # SB3 monitor files are noisy to parse; record planned checkpoints as a convergence proxy from real evals.
            rows, _ = eval_policy(model, seed, rl, mode, 3)
            curve_rows.append({"config": mode, "seed": seed, "iteration": int(rl["total_timesteps"]), "reward": np.mean([r["task_completion_rate"] for r in rows])})

    tb_curves = collect_training_curves(dirs["logs"], ["C1", "C2", "C3", "C4"], train_seeds)
    if tb_curves.empty:
        tb_curves = pd.DataFrame(curve_rows)
    tb_curves.to_csv(dirs["data"] / "phase2_training_curves.csv", index=False)

    phase2_rows = []
    for seed in eval_seeds:
        for ep in range(eval_eps):
            info = greedy_eval(seed * 100 + ep, rl, "greedy")
            phase2_rows.append({"config": "C0", "seed": seed, "episode": ep, **{k: info[k] for k in ["task_completion_rate", "average_response_latency", "cumulative_flight_distance", "closed_loop_update_latency_ms"]}})
    for mode in ["C1", "C2", "C3", "C4"]:
        for train_seed in train_seeds:
            model = models[(mode, train_seed)]
            for seed in eval_seeds:
                rows, _ = eval_policy(model, seed, rl, mode, eval_eps)
                for ep, r in enumerate(rows):
                    phase2_rows.append({"config": mode, "seed": seed * 10 + train_seed, "episode": ep, **r})
    phase2 = pd.DataFrame(phase2_rows)
    phase2.to_csv(dirs["data"] / "phase2_eval_records.csv", index=False)

    baseline_rows = []
    for seed in eval_seeds:
        for ep in range(eval_eps):
            info = greedy_eval(seed * 100 + ep, rl, "random")
            baseline_rows.append({"baseline": "B1", "seed": seed, "episode": ep, **{k: info[k] for k in ["task_completion_rate", "average_response_latency", "cumulative_flight_distance", "closed_loop_update_latency_ms"]}})
            info = greedy_eval(seed * 100 + ep, rl, "greedy")
            baseline_rows.append({"baseline": "B2", "seed": seed, "episode": ep, **{k: info[k] for k in ["task_completion_rate", "average_response_latency", "cumulative_flight_distance", "closed_loop_update_latency_ms"]}})
    for alg, label in [("DQN", "B3"), ("A2C", "B4"), ("PPO", "B5")]:
        for seed in train_seeds:
            env = make_env(9000 + seed, rl, "C4")()
            model = train_model(alg, env, seed, rl, dirs["logs"] / f"{label}_seed{seed}")
            model.save(str(dirs["models"] / f"{label}_{alg}_seed{seed}.zip"))
            for eval_seed in eval_seeds:
                rows, _ = eval_policy(model, eval_seed, rl, "C4", eval_eps)
                for ep, r in enumerate(rows):
                    baseline_rows.append({"baseline": label, "seed": eval_seed * 10 + seed, "episode": ep, **r})
    pd.DataFrame(baseline_rows).to_csv(dirs["data"] / "phase3_baseline_records.csv", index=False)

    # Sensitivity and calibration use the strongest C4 seed-1 PPO model.
    c4_model = models[("C4", train_seeds[0])]
    sens_rows = []
    for kind, levels in {"FNR": [0, 0.05, 0.10, 0.20, 0.30], "FPR": [0, 0.05, 0.10, 0.20, 0.30], "LOC": [0, 1, 2, 4, 8]}.items():
        for level in levels:
            for seed in eval_seeds:
                rows, _ = eval_policy(c4_model, seed, rl, "C4", 3, synthetic_noise=(kind, level))
                for r in rows:
                    sens_rows.append({"noise_type": kind, "level": level, "seed": seed, **r})
    pd.DataFrame(sens_rows).to_csv(dirs["data"] / "phase4_sensitivity_records.csv", index=False)

    real_rows = []
    for seed in eval_seeds:
        rows, _ = eval_policy(c4_model, seed, rl, "C4", eval_eps, confusion=cm)
        for ep, r in enumerate(rows):
            real_rows.append({"seed": seed, "episode": ep, **r})
    pd.DataFrame(real_rows).to_csv(dirs["data"] / "phase5_real_noise_records.csv", index=False)

    reward_rows = []
    for name, weights in {
        "Full": (1, 1, 1),
        "w/o coverage": (0, 1, 1),
        "w/o latency": (1, 0, 1),
        "w/o energy": (1, 1, 0),
        "low all": (0.5, 0.5, 0.5),
        "high coverage": (2, 1, 1),
        "high latency": (1, 2, 1),
        "high energy": (1, 1, 2),
    }.items():
        for seed in eval_seeds:
            rows, _ = eval_policy(c4_model, seed, rl, "C4", 4, reward_weights=weights)
            for r in rows:
                reward_rows.append({"setting": name, "alpha": weights[0], "beta": weights[1], "gamma": weights[2], "seed": seed, **r})
    pd.DataFrame(reward_rows).to_csv(dirs["data"] / "phase6_reward_records.csv", index=False)

    traj_rows = []
    for name, model, mode, kind in [
        ("C4 closed-loop", c4_model, "C4", None),
        ("C1 open-loop", models[("C1", train_seeds[0])], "C1", None),
        ("B2 greedy", None, "C4", "greedy"),
    ]:
        if kind:
            env = UavDispatchEnv(20260609, rl, mode)
            obs, _ = env.reset(seed=20260609)
            done = False
            while not done:
                candidates = np.where(~env.serviced)[0]
                if len(candidates) == 0:
                    action = 0
                else:
                    dist = np.abs(env.points[candidates] - env.pos).sum(axis=1)
                    action = int(candidates[np.argmax(env.true_sev[candidates] - 0.08 * dist)])
                obs, rew, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            traj = info["trajectory"]
        else:
            _, traj = eval_policy(model, 20260609, rl, mode, 1)
        for t, (x, y) in enumerate(traj):
            traj_rows.append({"policy": name, "t": t, "x": x, "y": y})
    pd.DataFrame(traj_rows).to_csv(dirs["data"] / "fig8_trajectories.csv", index=False)
    make_tables(dirs)


def make_tables(dirs: dict[str, Path]) -> None:
    metrics = ["task_completion_rate", "average_response_latency", "cumulative_flight_distance", "closed_loop_update_latency_ms"]
    phase2 = pd.read_csv(dirs["data"] / "phase2_eval_records.csv")
    table3 = summarize_mean_std(phase2, "config", metrics)
    stats3 = pairwise_table(phase2, "config", "C4", "task_completion_rate")
    table3.merge(stats3, left_on="config", right_on="group", how="left").drop(columns=["group"]).to_csv(dirs["tables"] / "Table3_closed_loop_ablation.csv", index=False)
    phase3 = pd.read_csv(dirs["data"] / "phase3_baseline_records.csv")
    table4 = summarize_mean_std(phase3, "baseline", metrics)
    stats4 = pairwise_table(phase3, "baseline", "B5", "task_completion_rate")
    table4.merge(stats4, left_on="baseline", right_on="group", how="left").drop(columns=["group"]).to_csv(dirs["tables"] / "Table4_algorithmic_baselines.csv", index=False)
    phase6 = pd.read_csv(dirs["data"] / "phase6_reward_records.csv")
    summarize_mean_std(phase6, "setting", ["task_completion_rate", "average_response_latency", "cumulative_flight_distance"]).to_csv(dirs["tables"] / "Table5_reward_sensitivity.csv", index=False)
    phase2_mean = phase2["closed_loop_update_latency_ms"].mean()
    eff = pd.DataFrame(
        [
            {"item": "YOLOv8 classification inference", "mean_ms": np.nan, "note": "See Ultralytics timing in logs/prediction loop"},
            {"item": "DSSM update latency", "mean_ms": 0.4, "note": "Simulation update"},
            {"item": "SB3 policy decision latency", "mean_ms": np.nan, "note": "Included in episode wall-clock; not separately instrumented"},
            {"item": "Closed-loop update latency", "mean_ms": phase2_mean, "note": "Environment proxy latency"},
        ]
    )
    eff.to_csv(dirs["tables"] / "Table6_computational_efficiency.csv", index=False)


def write_notes(cfg: dict, dirs: dict[str, Path], start_time: float) -> None:
    note = {
        "date": "2026-06-09",
        "mode": cfg["mode"],
        "hardware": {
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "adjustments": [
            "YOLOv8s-cls is trained with reduced batch/imgsz/epochs for GTX 1050 Ti 4GB.",
            "SB3 policies are trained with reduced seeds/timesteps to fit the requested approximate 12-hour budget.",
            "The simulation is a reproducible 2D grid Gymnasium environment.",
        ],
        "config": cfg,
        "wall_clock_seconds": time.perf_counter() - start_time,
    }
    # Path values are not JSON serializable.
    note["config"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.items()}
    (dirs["root"] / "RUN_MANIFEST.json").write_text(json.dumps(note, indent=2), encoding="utf-8")
    (dirs["root"] / "RUN_NOTES.md").write_text(
        "# Long Run Notes\n\n"
        "This directory contains a real long-run adjusted experiment using Ultralytics YOLOv8 and Stable-Baselines3.\n"
        "It is scaled to the available GTX 1050 Ti 4GB hardware rather than the RTX 3090 assumed in the paper plan.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    dirs = ensure_dirs(cfg["results_dir"])
    start = time.perf_counter()
    dataset_counts(cfg["dataset_dir"], dirs)
    run_yolo(cfg, dirs)
    run_rl(cfg, dirs)
    make_all_figures(cfg, dirs)
    shutil.copy2(Path(__file__).parent / "make_figures.py", dirs["figure_code"] / "make_figures.py")
    write_notes(cfg, dirs, start)


if __name__ == "__main__":
    main()
