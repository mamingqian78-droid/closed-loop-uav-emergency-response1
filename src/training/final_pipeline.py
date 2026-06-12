from __future__ import annotations

import argparse
import json
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
from sb3_contrib import MaskablePPO
from ultralytics import YOLO

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import make_all_figures
from run_long_experiment import dataset_counts, list_images


CLASSES = ["earthquake", "flood", "normal", "wildfire"]
DISASTER_CLASSES = ["earthquake", "flood", "wildfire"]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = path.parent
    for key in ("project_root", "dataset_dir", "results_dir", "source_detection_dir"):
        if key in cfg and cfg[key] is not None:
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


def run_yolo(cfg: dict, dirs: dict[str, Path]) -> dict:
    ycfg = cfg["yolo"]
    dataset_dir = cfg["dataset_dir"]
    run_dir = dirs["models"] / ycfg.get("run_name", "yolo")
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
    normal_idx = CLASSES.index("normal")
    threshold = float(ycfg.get("confidence_threshold", 0.0))
    all_paths: list[Path] = []
    for cls in CLASSES:
        all_paths.extend(list_images(dataset_dir, "test", cls))

    y_true, y_pred, rows = [], [], []
    results = model.predict(
        [str(p) for p in all_paths],
        imgsz=int(ycfg["imgsz"]),
        batch=int(ycfg["batch"]),
        device=ycfg["device"],
        verbose=False,
        stream=True,
    )
    for p, result in zip(all_paths, results):
        probs = result.probs.data.detach().cpu().numpy()
        pred_idx = int(probs.argmax())
        conf = float(probs[pred_idx])
        if pred_idx != normal_idx and conf < threshold:
            pred_idx = normal_idx
        true_idx = CLASSES.index(p.parent.name)
        y_true.append(true_idx)
        y_pred.append(pred_idx)
        rows.append({"path": str(p), "true": p.parent.name, "pred": CLASSES[pred_idx], "confidence": conf})

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for t, p in zip(y_true_arr, y_pred_arr):
        cm[t, p] += 1
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    np.save(dirs["data"] / "confusion_matrix.npy", cm_norm)
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(dirs["data"] / "confusion_matrix_counts.csv")

    metrics = {
        "model": str(weights_path),
        "imgsz": int(ycfg["imgsz"]),
        "confidence_threshold": threshold,
        "top1_accuracy": float((y_true_arr == y_pred_arr).mean()),
        "per_class": {},
    }
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
    disaster_idx = [CLASSES.index(c) for c in DISASTER_CLASSES]
    metrics["empirical_fnr"] = float(1 - np.diag(cm_norm)[disaster_idx].mean())
    metrics["empirical_fpr"] = float(cm_norm[normal_idx, disaster_idx].sum())
    (dirs["data"] / "detection_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    sample_df = pd.DataFrame(rows)
    samples = pd.concat(
        [
            sample_df[sample_df["true"] != sample_df["pred"]].head(4),
            sample_df[sample_df["true"] == sample_df["pred"]].groupby("true", group_keys=False).head(1),
        ],
        ignore_index=True,
    ).head(8)
    samples.to_csv(dirs["data"] / "fig5_samples.csv", index=False)
    return metrics


def prepare_detection(cfg: dict, dirs: dict[str, Path]) -> None:
    source = cfg.get("source_detection_dir")
    needed = [
        "confusion_matrix.npy",
        "confusion_matrix_counts.csv",
        "detection_metrics.json",
        "fig5_samples.csv",
        "dataset_counts.csv",
    ]
    if source and all((source / "data" / name).exists() for name in needed):
        for name in needed:
            shutil.copy2(source / "data" / name, dirs["data"] / name)
        src_weights = source / "models" / "yolo"
        dst_weights = dirs["models"] / "yolo"
        if src_weights.exists() and not dst_weights.exists():
            shutil.copytree(src_weights, dst_weights)
        return
    dataset_counts(cfg["dataset_dir"], dirs)
    run_yolo(cfg, dirs)


class FinalDispatchEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        seed: int,
        cfg: dict,
        mode: str = "C4",
        confusion: np.ndarray | None = None,
        synthetic_noise: tuple[str, float] | None = None,
        reward_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        super().__init__()
        self.base_seed = seed
        self.cfg = cfg
        self.mode = mode
        self.confusion = confusion
        self.synthetic_noise = synthetic_noise
        self.reward_weights = reward_weights
        self.n = int(cfg["grid_size"])
        self.real_n = int(cfg["real_targets"])
        self.decoy_n = int(cfg["decoy_targets"])
        self.slots = self.real_n + self.decoy_n
        self.horizon = int(cfg["horizon"])
        self.update_period = int(cfg["update_period"])
        self.action_space = spaces.Discrete(self.slots)
        self.features_per_slot = 7
        obs_dim = 6 + self.slots * self.features_per_slot
        self.observation_space = spaces.Box(low=-1.0, high=2.0, shape=(obs_dim,), dtype=np.float32)
        self._last_update_ms = 0.0
        self.reset(seed=seed)

    def _world(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.base = np.array([self.n // 2, self.n // 2], dtype=np.int32)
        pts: list[np.ndarray] = []
        while len(pts) < self.slots:
            p = self.rng.integers(1, self.n - 1, size=2)
            if np.linalg.norm(p - self.base, ord=1) < 4:
                continue
            if all(np.linalg.norm(p - q, ord=1) >= 3 for q in pts):
                pts.append(p)
        self.true_pos = np.vstack(pts[: self.real_n]).astype(np.int32)
        self.decoy_pos = np.vstack(pts[self.real_n :]).astype(np.int32)
        self.all_pos = np.vstack([self.true_pos, self.decoy_pos]).astype(np.int32)
        self.true_sev0 = self.rng.uniform(0.38, 0.82, size=self.real_n)
        self.growth = self.rng.uniform(0.0008, 0.0028, size=self.real_n)
        release = np.zeros(self.real_n, dtype=np.int32)
        late = self.rng.choice(self.real_n, size=max(5, self.real_n // 2), replace=False)
        release[late] = self.rng.integers(70, 300, size=len(late))
        self.release_time = release

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.base_seed = seed
        self._world(self.base_seed)
        self.t = 0.0
        self.pos = self.base.copy()
        self.distance = 0.0
        self.energy = float(self.horizon * 0.88)
        self.true_sev = self.true_sev0.copy()
        self.serviced_real = np.zeros(self.real_n, dtype=bool)
        self.visited_decoy = np.zeros(self.decoy_n, dtype=bool)
        self.completion_times: list[float] = []
        self.trajectory = [tuple(self.pos.tolist())]
        self.update_latencies_ms: list[float] = []
        self._refresh_observation(force=True)
        return self._obs(), {}

    def _mode_params(self) -> tuple[int, bool]:
        return {
            "C1": (10**9, True),
            "C2": (self.update_period, False),
            "C3": (self.update_period, True),
            "C4": (1, True),
        }.get(self.mode, (1, True))

    def _noise_levels(self) -> tuple[float, float, float]:
        fnr = 0.05
        fpr = 0.10
        loc = 0.25
        if self.confusion is not None:
            disaster_idx = [CLASSES.index(c) for c in DISASTER_CLASSES]
            normal_idx = CLASSES.index("normal")
            fnr = float(np.clip(1 - np.diag(self.confusion)[disaster_idx].mean(), 0, 0.45))
            fpr = float(np.clip(self.confusion[normal_idx, disaster_idx].sum(), 0, 0.35))
        if self.synthetic_noise is not None:
            kind, level = self.synthetic_noise
            if kind == "FNR":
                fnr = float(level)
            elif kind == "FPR":
                fpr = float(level)
            elif kind == "LOC":
                loc = float(level)
        return fnr, fpr, loc

    def _refresh_observation(self, force: bool = False) -> None:
        start = time.perf_counter()
        fnr, fpr, loc = self._noise_levels()
        real_visible = (self.t >= self.release_time) & (~self.serviced_real)
        real_detected = real_visible & (self.rng.random(self.real_n) > fnr)
        obs_sev = np.zeros(self.slots, dtype=np.float32)
        active = np.zeros(self.slots, dtype=np.float32)
        conf = np.zeros(self.slots, dtype=np.float32)
        age = np.zeros(self.slots, dtype=np.float32)
        obs_pos = self.all_pos.astype(np.float32).copy()

        obs_sev[: self.real_n] = np.where(real_detected, self.true_sev + self.rng.normal(0, 0.025, self.real_n), 0)
        active[: self.real_n] = real_detected.astype(np.float32)
        conf[: self.real_n] = np.where(real_detected, self.rng.uniform(0.76, 0.98, self.real_n), 0)
        age[: self.real_n] = np.clip((self.t - self.release_time) / self.horizon, 0, 1)

        synthetic_fpr = self.synthetic_noise is not None and self.synthetic_noise[0] == "FPR"
        if synthetic_fpr:
            decoy_prob = min(0.05 + 2.85 * fpr, 0.94)
            decoy_sev_low = 0.72 + 0.45 * fpr
            decoy_sev_high = 1.05 + 0.65 * fpr
            decoy_conf_low = 0.70 + 0.18 * fpr
            decoy_conf_high = 0.98
        else:
            decoy_prob = min(0.15 + 2.50 * fpr, 0.82)
            decoy_sev_low = 0.55
            decoy_sev_high = 1.10
            decoy_conf_low = 0.45
            decoy_conf_high = 0.82
        decoy_active = (self.rng.random(self.decoy_n) < decoy_prob) & (~self.visited_decoy)
        obs_sev[self.real_n :] = np.where(decoy_active, self.rng.uniform(decoy_sev_low, decoy_sev_high, self.decoy_n), 0)
        active[self.real_n :] = decoy_active.astype(np.float32)
        conf[self.real_n :] = np.where(decoy_active, self.rng.uniform(decoy_conf_low, decoy_conf_high, self.decoy_n), 0)
        age[self.real_n :] = np.where(decoy_active, self.rng.uniform(0.0, 0.35, self.decoy_n), 0)

        if loc > 0:
            sigma = loc if self.synthetic_noise and self.synthetic_noise[0] == "LOC" else 0.25
            obs_pos += self.rng.normal(0, sigma, size=obs_pos.shape)
            obs_pos = np.clip(obs_pos, 0, self.n - 1)

        self.obs_pos = obs_pos
        self.obs_sev = np.clip(obs_sev, 0, 1.5)
        self.obs_active = active
        self.obs_conf = conf
        self.obs_age = age
        elapsed = (time.perf_counter() - start) * 1000
        self._last_update_ms = elapsed
        self.update_latencies_ms.append(elapsed)

    def _obs(self) -> np.ndarray:
        _, uses_dssm = self._mode_params()
        if uses_dssm:
            sev = self.obs_sev
            active = self.obs_active
            conf = self.obs_conf
            age = self.obs_age
            pos = self.obs_pos
        else:
            sev = np.zeros_like(self.obs_sev)
            active = np.zeros_like(self.obs_active)
            conf = np.zeros_like(self.obs_conf)
            age = np.zeros_like(self.obs_age)
            pos = self.obs_pos
        served = np.concatenate([self.serviced_real.astype(np.float32), self.visited_decoy.astype(np.float32)])
        slots = np.column_stack(
            [
                sev,
                pos[:, 0] / self.n,
                pos[:, 1] / self.n,
                active,
                served,
                conf,
                age,
            ]
        )
        head = np.array(
            [
                self.pos[0] / self.n,
                self.pos[1] / self.n,
                self.t / self.horizon,
                self.energy / max(self.horizon, 1),
                self.serviced_real.mean(),
                active.mean(),
            ],
            dtype=np.float32,
        )
        return np.concatenate([head, slots.ravel().astype(np.float32)]).astype(np.float32)

    def action_masks(self) -> np.ndarray:
        _, uses_dssm = self._mode_params()
        served = np.concatenate([self.serviced_real.astype(bool), self.visited_decoy.astype(bool)])
        if uses_dssm:
            mask = (self.obs_active > 0.5) & (~served)
        else:
            mask = ~served
        if not np.any(mask):
            mask = ~served
        if not np.any(mask):
            mask = np.ones(self.slots, dtype=bool)
        return mask.astype(bool)

    def _advance(self, dt: float) -> None:
        dt = float(dt)
        self.t += dt
        self.true_sev = np.clip(self.true_sev + self.growth * dt * (~self.serviced_real), 0, 1.35)

    def step(self, action: int):
        update_every, _ = self._mode_params()
        if self.t > 0 and int(self.t) % max(update_every, 1) == 0:
            self._refresh_observation()
        action = int(action)
        alpha, beta, gamma = self.reward_weights
        reward = -0.03
        done_penalty = False
        active = self.obs_active[action] > 0.5
        already_done = action < self.real_n and self.serviced_real[action]
        already_done = already_done or (action >= self.real_n and self.visited_decoy[action - self.real_n])
        if (not active) or already_done:
            self._advance(4)
            self.energy -= 1.8
            reward -= 0.28
            done_penalty = True
        else:
            target_obs = self.obs_pos[action]
            travel = float(np.abs(target_obs - self.pos).sum())
            target_true = self.all_pos[action]
            search = float(np.abs(target_true - target_obs).sum())
            dt = travel + search + (13 if action < self.real_n else 16)
            self.distance += travel + search
            self.energy -= travel * (0.85 + 0.15 * gamma) + search * 0.55 + 2.0
            self.pos = np.rint(target_true).astype(np.int32)
            self.trajectory.append(tuple(self.pos.tolist()))
            self._advance(dt)
            if action < self.real_n:
                false_alarm = False
                if self.synthetic_noise is not None and self.synthetic_noise[0] == "FPR":
                    false_alarm = self.rng.random() < min(0.82 * float(self.synthetic_noise[1]), 0.55)
                if false_alarm:
                    reward -= 1.15 + 0.30 * beta + 0.20 * gamma
                    return self._obs(), float(reward), False, bool(self.t >= self.horizon or self.energy <= 0), self._info()
                if self.synthetic_noise is not None and self.synthetic_noise[0] == "FNR":
                    miss_probability = min(1.70 * float(self.synthetic_noise[1]), 0.68)
                elif self.confusion is not None:
                    miss_probability = min(1.70 * self._noise_levels()[0], 0.68)
                else:
                    miss_probability = 0.0
                if self.rng.random() < miss_probability:
                    reward -= 0.95 + 0.25 * beta + 0.15 * gamma
                    return self._obs(), float(reward), False, bool(self.t >= self.horizon or self.energy <= 0), self._info()
                if (self.t >= self.release_time[action]) and (not self.serviced_real[action]):
                    sev = float(self.true_sev[action])
                    latency_cost = self.t / self.horizon
                    energy_cost = self.distance / max(self.horizon, 1)
                    self.serviced_real[action] = True
                    self.completion_times.append(self.t)
                    reward += alpha * (2.6 * sev + 1.02) + 0.08
                    reward -= 0.55 * beta * latency_cost + 0.35 * gamma * energy_cost
                else:
                    reward -= 0.35
            else:
                self.visited_decoy[action - self.real_n] = True
                reward -= 1.1 + 0.25 * beta + 0.18 * gamma
        reward -= 0.0015 * beta * self.t
        reward -= 0.0020 * gamma * max(self.distance, 0)
        if done_penalty and self.t > self.horizon * 0.9:
            reward -= 0.2
        terminated = bool(self.serviced_real.all())
        truncated = bool(self.t >= self.horizon or self.energy <= 0)
        if terminated:
            reward += 4.0 * alpha
        return self._obs(), float(reward), terminated, truncated, self._info()

    def _info(self) -> dict:
        missing = self.real_n - int(self.serviced_real.sum())
        latencies = list(self.completion_times) + [self.horizon] * missing
        return {
            "task_completion_rate": float(self.serviced_real.mean()),
            "average_response_latency": float(np.mean(latencies)),
            "cumulative_flight_distance": float(self.distance),
            "closed_loop_update_latency_ms": float(np.mean(self.update_latencies_ms[-10:])) if self.update_latencies_ms else self._last_update_ms,
            "trajectory": self.trajectory,
            "false_dispatches": int(self.visited_decoy.sum()),
        }


def make_env(seed: int, cfg: dict, mode: str, confusion=None, synthetic_noise=None, reward_weights=(1, 1, 1)):
    def _factory():
        return Monitor(FinalDispatchEnv(seed, cfg, mode, confusion, synthetic_noise, tuple(reward_weights)))

    return _factory


def choose_greedy(env: FinalDispatchEnv, static_obs: dict | None = None, random: bool = False) -> int:
    if random:
        return int(env.action_space.sample())
    sev = env.obs_sev if static_obs is None else static_obs["sev"]
    active = env.obs_active if static_obs is None else static_obs["active"]
    conf = env.obs_conf if static_obs is None else static_obs["conf"]
    pos = env.obs_pos if static_obs is None else static_obs["pos"]
    served = np.concatenate([env.serviced_real.astype(bool), env.visited_decoy.astype(bool)])
    candidates = np.where((active > 0.5) & (~served))[0]
    if len(candidates) == 0:
        candidates = np.where(~served)[0]
    if len(candidates) == 0:
        return 0
    dist = np.abs(pos[candidates] - env.pos).sum(axis=1)
    score = sev[candidates] - 0.018 * dist
    return int(candidates[np.argmax(score)])


def static_snapshot(env: FinalDispatchEnv) -> dict:
    sev = np.zeros(env.slots, dtype=np.float32)
    active = np.zeros(env.slots, dtype=np.float32)
    conf = np.zeros(env.slots, dtype=np.float32)
    pos = env.all_pos.astype(np.float32).copy()
    visible = env.rng.random(env.real_n) < 0.90
    active[: env.real_n] = visible.astype(np.float32)
    sev[: env.real_n] = np.where(visible, env.true_sev0 + env.rng.normal(0, 0.12, env.real_n), 0)
    conf[: env.real_n] = np.where(visible, env.rng.uniform(0.62, 0.88, env.real_n), 0)
    decoy_active = env.rng.random(env.decoy_n) < 0.14
    active[env.real_n :] = decoy_active.astype(np.float32)
    sev[env.real_n :] = np.where(decoy_active, env.rng.uniform(0.45, 0.9, env.decoy_n), 0)
    conf[env.real_n :] = np.where(decoy_active, env.rng.uniform(0.45, 0.76, env.decoy_n), 0)
    pos += env.rng.normal(0, 1.1, size=pos.shape)
    pos = np.clip(pos, 0, env.n - 1)
    return {"sev": np.clip(sev, 0, 1.4), "active": active, "conf": conf, "pos": pos}


def heuristic_eval(seed: int, cfg: dict, kind: str, episodes: int = 1, mode: str = "C4", synthetic_noise=None, confusion=None):
    rows = []
    last_traj = None
    for ep in range(episodes):
        env = FinalDispatchEnv(seed * 100 + ep, cfg, mode=mode, synthetic_noise=synthetic_noise, confusion=confusion)
        obs, _ = env.reset(seed=seed * 100 + ep)
        static = None
        if kind == "static":
            static = static_snapshot(env)
            env.obs_sev = static["sev"].copy()
            env.obs_active = static["active"].copy()
            env.obs_conf = static["conf"].copy()
            env.obs_pos = static["pos"].copy()
        done = False
        info = env._info()
        while not done:
            action = choose_greedy(env, static_obs=static, random=(kind == "random"))
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        last_traj = info["trajectory"]
        rows.append({k: info[k] for k in METRICS})
    return rows, last_traj


def train_model(algorithm: str, env, seed: int, cfg: dict, log_dir: Path):
    if algorithm == "PPO":
        model = MaskablePPO(
            "MlpPolicy",
            env,
            seed=seed,
            verbose=0,
            learning_rate=float(cfg["learning_rate"]),
            n_steps=int(cfg["n_steps"]),
            batch_size=int(cfg["batch_size"]),
            gamma=float(cfg["gamma"]),
            tensorboard_log=str(log_dir),
        )
    elif algorithm == "A2C":
        model = A2C(
            "MlpPolicy",
            env,
            seed=seed,
            verbose=0,
            learning_rate=float(cfg["learning_rate"]),
            tensorboard_log=str(log_dir),
            gamma=float(cfg["gamma"]),
        )
    elif algorithm == "DQN":
        model = DQN(
            "MlpPolicy",
            env,
            seed=seed,
            verbose=0,
            learning_rate=float(cfg["learning_rate"]),
            tensorboard_log=str(log_dir),
            gamma=float(cfg["gamma"]),
            buffer_size=50000,
            learning_starts=2000,
            batch_size=int(cfg["batch_size"]),
        )
    else:
        raise ValueError(algorithm)
    model.learn(total_timesteps=int(cfg["total_timesteps"]), progress_bar=False)
    return model


METRICS = ["task_completion_rate", "average_response_latency", "cumulative_flight_distance", "closed_loop_update_latency_ms"]


def eval_policy(model, seed: int, cfg: dict, mode: str, episodes: int, confusion=None, synthetic_noise=None, reward_weights=(1, 1, 1)):
    rows = []
    last_traj = None
    for ep in range(episodes):
        env = FinalDispatchEnv(seed * 100 + ep, cfg, mode, confusion, synthetic_noise, tuple(reward_weights))
        obs, _ = env.reset(seed=seed * 100 + ep)
        done = False
        info = env._info()
        while not done:
            try:
                action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
            except TypeError:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
        last_traj = info["trajectory"]
        rows.append({k: info[k] for k in METRICS})
    return rows, last_traj


def collect_training_curves(logs_dir: Path, modes: list[str], train_seeds: list[int]) -> pd.DataFrame:
    rows = []
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return pd.DataFrame(rows)
    for mode in modes:
        for seed in train_seeds:
            for event_file in (logs_dir / f"{mode}_seed{seed}").rglob("events.out.tfevents*"):
                acc = EventAccumulator(str(event_file))
                acc.Reload()
                if "rollout/ep_rew_mean" not in acc.Tags().get("scalars", []):
                    continue
                for e in acc.Scalars("rollout/ep_rew_mean"):
                    rows.append({"config": mode, "seed": seed, "iteration": int(e.step), "reward": float(e.value)})
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


def make_tables(dirs: dict[str, Path]) -> None:
    phase2 = pd.read_csv(dirs["data"] / "phase2_eval_records.csv")
    table3 = summarize_mean_std(phase2, "config", METRICS)
    stats3 = pairwise_table(phase2, "config", "C4", "task_completion_rate")
    table3.merge(stats3, left_on="config", right_on="group", how="left").drop(columns=["group"]).to_csv(dirs["tables"] / "Table3_closed_loop_ablation.csv", index=False)

    phase3 = pd.read_csv(dirs["data"] / "phase3_baseline_records.csv")
    table4 = summarize_mean_std(phase3, "baseline", METRICS)
    stats4 = pairwise_table(phase3, "baseline", "B5", "task_completion_rate")
    table4.merge(stats4, left_on="baseline", right_on="group", how="left").drop(columns=["group"]).to_csv(dirs["tables"] / "Table4_algorithmic_baselines.csv", index=False)

    phase6 = pd.read_csv(dirs["data"] / "phase6_reward_records.csv")
    summarize_mean_std(phase6, "setting", ["task_completion_rate", "average_response_latency", "cumulative_flight_distance"]).to_csv(dirs["tables"] / "Table5_reward_sensitivity.csv", index=False)


def measure_efficiency(cfg: dict, dirs: dict[str, Path], model) -> None:
    rl = cfg["rl"]
    env = FinalDispatchEnv(777, rl, "C4")
    obs, _ = env.reset(seed=777)
    n = 1000
    t0 = time.perf_counter()
    for _ in range(n):
        env._refresh_observation()
    dssm_ms = (time.perf_counter() - t0) * 1000 / n
    t0 = time.perf_counter()
    for _ in range(n):
        try:
            model.predict(obs, deterministic=True, action_masks=env.action_masks())
        except TypeError:
            model.predict(obs, deterministic=True)
    policy_ms = (time.perf_counter() - t0) * 1000 / n
    t0 = time.perf_counter()
    for _ in range(n):
        env._refresh_observation()
        try:
            action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
        except TypeError:
            action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        if terminated or truncated:
            obs, _ = env.reset()
    closed_ms = (time.perf_counter() - t0) * 1000 / n
    yolo_ms = np.nan
    ycfg = cfg.get("yolo", {})
    weights = dirs["models"] / ycfg.get("run_name", "yolo") / "train" / "weights" / "best.pt"
    sample_paths = list_images(cfg["dataset_dir"], "test", CLASSES[0])
    if weights.exists() and sample_paths:
        detector = YOLO(str(weights))
        sample = str(sample_paths[0])
        for _ in range(10):
            detector.predict(sample, imgsz=int(ycfg.get("imgsz", 96)), device=ycfg.get("device", "cpu"), verbose=False)
        t0 = time.perf_counter()
        for _ in range(n):
            detector.predict(sample, imgsz=int(ycfg.get("imgsz", 96)), device=ycfg.get("device", "cpu"), verbose=False)
        yolo_ms = (time.perf_counter() - t0) * 1000 / n
    pd.DataFrame(
        [
            {
                "item": "YOLOv8 classification inference",
                "mean_ms": yolo_ms,
                "note": f"1000 batch=1 predictions, imgsz={ycfg.get('imgsz', 96)}, threshold={ycfg.get('confidence_threshold', 0.0)}",
            },
            {"item": "DSSM update latency", "mean_ms": dssm_ms, "note": "1000 simulated updates"},
            {"item": "SB3 policy decision latency", "mean_ms": policy_ms, "note": "1000 deterministic policy calls"},
            {"item": "Closed-loop update latency", "mean_ms": closed_ms, "note": "DSSM update + policy decision + environment step"},
            {"item": "End-to-end perception+control latency", "mean_ms": yolo_ms + closed_ms, "note": "YOLOv8 inference + closed-loop update latency"},
        ]
    ).to_csv(dirs["tables"] / "Table6_computational_efficiency.csv", index=False)


def run_rl(cfg: dict, dirs: dict[str, Path]) -> None:
    rl = cfg["rl"]
    cm = np.load(dirs["data"] / "confusion_matrix.npy")
    train_seeds = cfg["train_seeds"]
    eval_seeds = cfg["eval_seeds"]
    eval_eps = int(rl["eval_episodes_per_seed"])
    sens_eps = int(rl["sensitivity_episodes_per_seed"])
    models: dict[tuple[str, int], object] = {}

    for mode in ["C1", "C2", "C3", "C4"]:
        for seed in train_seeds:
            env = make_env(seed, rl, mode)()
            model = train_model("PPO", env, seed, rl, dirs["logs"] / f"{mode}_seed{seed}")
            model.save(str(dirs["models"] / f"{mode}_ppo_seed{seed}.zip"))
            models[(mode, seed)] = model

    curves = collect_training_curves(dirs["logs"], ["C1", "C2", "C3", "C4"], train_seeds)
    curves.to_csv(dirs["data"] / "phase2_training_curves.csv", index=False)

    phase2_rows = []
    for seed in eval_seeds:
        rows, _ = heuristic_eval(seed, rl, "static", eval_eps, mode="C1")
        for ep, r in enumerate(rows):
            phase2_rows.append({"config": "C0", "seed": seed, "episode": ep, **r})
    for mode in ["C1", "C2", "C3", "C4"]:
        for train_seed in train_seeds:
            model = models[(mode, train_seed)]
            for seed in eval_seeds:
                rows, _ = eval_policy(model, seed, rl, mode, eval_eps)
                for ep, r in enumerate(rows):
                    phase2_rows.append({"config": mode, "seed": seed * 10 + train_seed, "episode": ep, **r})
    pd.DataFrame(phase2_rows).to_csv(dirs["data"] / "phase2_eval_records.csv", index=False)

    baseline_rows = []
    for seed in eval_seeds:
        for label, kind in [("B1", "random"), ("B2", "greedy")]:
            rows, _ = heuristic_eval(seed, rl, kind, eval_eps, mode="C4")
            for ep, r in enumerate(rows):
                baseline_rows.append({"baseline": label, "seed": seed, "episode": ep, **r})
    baseline_models = {}
    for alg, label in [("DQN", "B3"), ("A2C", "B4"), ("PPO", "B5")]:
        for seed in train_seeds:
            env = make_env(9000 + seed, rl, "C4")()
            model = train_model(alg, env, seed, rl, dirs["logs"] / f"{label}_seed{seed}")
            model.save(str(dirs["models"] / f"{label}_{alg}_seed{seed}.zip"))
            baseline_models[(label, seed)] = model
            for eval_seed in eval_seeds:
                rows, _ = eval_policy(model, eval_seed, rl, "C4", eval_eps)
                for ep, r in enumerate(rows):
                    baseline_rows.append({"baseline": label, "seed": eval_seed * 10 + seed, "episode": ep, **r})
    pd.DataFrame(baseline_rows).to_csv(dirs["data"] / "phase3_baseline_records.csv", index=False)

    c4_model = models[("C4", train_seeds[0])]
    sens_rows = []
    for kind, levels in {"FNR": [0, 0.05, 0.10, 0.20, 0.30], "FPR": [0, 0.05, 0.10, 0.20, 0.30], "LOC": [0, 1, 2, 4, 8]}.items():
        for level in levels:
            for seed in eval_seeds:
                rows, _ = eval_policy(c4_model, seed, rl, "C4", sens_eps, synthetic_noise=(kind, level))
                for ep, r in enumerate(rows):
                    sens_rows.append({"noise_type": kind, "level": level, "seed": seed, "episode": ep, **r})
    pd.DataFrame(sens_rows).to_csv(dirs["data"] / "phase4_sensitivity_records.csv", index=False)

    real_rows = []
    for seed in eval_seeds:
        rows, _ = eval_policy(c4_model, seed, rl, "C4", eval_eps, confusion=cm)
        for ep, r in enumerate(rows):
            real_rows.append({"seed": seed, "episode": ep, **r})
    pd.DataFrame(real_rows).to_csv(dirs["data"] / "phase5_real_noise_records.csv", index=False)

    reward_rows = []
    for setting, weights in rl["reward_settings"].items():
        for seed in train_seeds:
            env = make_env(12000 + seed, rl, "C4", reward_weights=weights)()
            model = train_model("PPO", env, seed, rl, dirs["logs"] / f"reward_{setting.replace(' ', '_').replace('/', 'no')}_seed{seed}")
            model.save(str(dirs["models"] / f"reward_{setting.replace(' ', '_').replace('/', 'no')}_seed{seed}.zip"))
            for eval_seed in eval_seeds:
                rows, _ = eval_policy(model, eval_seed, rl, "C4", max(4, eval_eps // 2), reward_weights=weights)
                for ep, r in enumerate(rows):
                    reward_rows.append({"setting": setting, "alpha": weights[0], "beta": weights[1], "gamma": weights[2], "seed": eval_seed * 10 + seed, "episode": ep, **r})
    pd.DataFrame(reward_rows).to_csv(dirs["data"] / "phase6_reward_records.csv", index=False)

    traj_rows = []
    for name, runner in [
        ("C4 closed-loop", ("model", c4_model, "C4")),
        ("C1 open-loop", ("model", models[("C1", train_seeds[0])], "C1")),
        ("B2 greedy", ("heuristic", None, "C4")),
    ]:
        if runner[0] == "model":
            _, traj = eval_policy(runner[1], 20260610, rl, runner[2], 1)
        else:
            _, traj = heuristic_eval(20260610, rl, "greedy", 1, mode=runner[2])
        for t, (x, y) in enumerate(traj):
            traj_rows.append({"policy": name, "t": t, "x": x, "y": y})
    pd.DataFrame(traj_rows).to_csv(dirs["data"] / "fig8_trajectories.csv", index=False)

    make_tables(dirs)
    measure_efficiency(cfg, dirs, c4_model)


def write_notes(cfg: dict, dirs: dict[str, Path], start_time: float) -> None:
    note = {
        "date": "2026-06-10",
        "mode": cfg["mode"],
        "hardware": {
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "adjustments": [
            "YOLO detection artifacts are reused from the completed real YOLOv8 training run when available.",
            "The UAV environment is revised after diagnostic audit: false positives create decoy targets, localization noise perturbs target coordinates, and greedy baselines no longer read true severity.",
            "Reward ablations retrain PPO policies instead of only changing evaluation-time scoring.",
        ],
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.items()},
        "wall_clock_seconds": time.perf_counter() - start_time,
    }
    (dirs["root"] / "RUN_MANIFEST.json").write_text(json.dumps(note, indent=2), encoding="utf-8")
    (dirs["root"] / "RUN_NOTES.md").write_text(
        "# Final Run Notes\n\n"
        "This run replaces the pilot dispatch simulator with a revised environment that implements decoy false positives, localization error, service time, energy cost, fair baseline observations, and measured computational efficiency.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    dirs = ensure_dirs(cfg["results_dir"])
    start = time.perf_counter()
    prepare_detection(cfg, dirs)
    run_rl(cfg, dirs)
    make_all_figures(cfg, dirs)
    shutil.copy2(Path(__file__).parent / "make_figures.py", dirs["figure_code"] / "make_figures.py")
    shutil.copy2(Path(__file__), dirs["root"] / "run_final_experiment.py")
    shutil.copy2(args.config, dirs["root"] / "final_config.yaml")
    write_notes(cfg, dirs, start)


if __name__ == "__main__":
    main()
