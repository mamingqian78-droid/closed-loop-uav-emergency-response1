from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import pandas as pd

from make_figures import make_all_figures
from final_pipeline import ensure_dirs, load_config, make_env, make_tables, train_model, eval_policy


def safe_name(setting: str) -> str:
    return setting.replace(" ", "_").replace("/", "no")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("experiment_code/final_config.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    dirs = ensure_dirs(cfg["results_dir"])
    rl = cfg["rl"]
    eval_eps = int(rl["eval_episodes_per_seed"])
    started = time.perf_counter()
    rows = []
    status_path = dirs["logs"] / "reward_phase_status.csv"

    for setting, weights in rl["reward_settings"].items():
        for seed in cfg["train_seeds"]:
            step_started = time.perf_counter()
            status = {
                "setting": setting,
                "train_seed": seed,
                "status": "training",
                "elapsed_seconds": time.perf_counter() - started,
            }
            pd.DataFrame([status]).to_csv(status_path, mode="a", header=not status_path.exists(), index=False)
            env = make_env(12000 + seed, rl, "C4", reward_weights=weights)()
            model = train_model("PPO", env, seed, rl, dirs["logs"] / f"reward_{safe_name(setting)}_seed{seed}")
            model.save(str(dirs["models"] / f"reward_{safe_name(setting)}_seed{seed}.zip"))
            for eval_seed in cfg["eval_seeds"]:
                eval_rows, _ = eval_policy(model, eval_seed, rl, "C4", max(4, eval_eps // 2), reward_weights=weights)
                for ep, record in enumerate(eval_rows):
                    rows.append(
                        {
                            "setting": setting,
                            "alpha": weights[0],
                            "beta": weights[1],
                            "gamma": weights[2],
                            "seed": eval_seed * 10 + seed,
                            "episode": ep,
                            **record,
                        }
                    )
            pd.DataFrame(rows).to_csv(dirs["data"] / "phase6_reward_records.csv", index=False)
            status = {
                "setting": setting,
                "train_seed": seed,
                "status": "done",
                "elapsed_seconds": time.perf_counter() - started,
                "step_seconds": time.perf_counter() - step_started,
            }
            pd.DataFrame([status]).to_csv(status_path, mode="a", header=False, index=False)

    make_tables(dirs)
    make_all_figures(cfg, dirs)
    shutil.copy2(Path("experiment_code/make_figures.py"), dirs["figure_code"] / "make_figures.py")
    shutil.copy2(Path("experiment_code/run_final_experiment.py"), dirs["root"] / "run_final_experiment.py")
    shutil.copy2(args.config, dirs["root"] / "final_config.yaml")
    print("reward phase complete")


if __name__ == "__main__":
    main()
