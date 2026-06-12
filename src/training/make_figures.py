from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import LineCollection
from PIL import Image, ImageOps


CLASSES = ["earthquake", "flood", "normal", "wildfire"]
DISASTER_CLASSES = ["earthquake", "flood", "wildfire"]


def set_style(font: str) -> None:
    plt.rcParams.update(
        {
            "font.family": font,
            "font.size": 8,
            "text.color": "black",
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "axes.grid": False,
            "xtick.labelsize": 7,
            "xtick.color": "black",
            "ytick.labelsize": 7,
            "ytick.color": "black",
            "legend.fontsize": 7,
            "figure.titlesize": 10,
            "axes.linewidth": 0.8,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )
    sns.set_style("ticks", {"xtick.bottom": True, "ytick.left": True, "axes.grid": False})


def style_axis(ax, *, ticks: bool = True) -> None:
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)
    ax.tick_params(
        axis="both",
        which="both",
        bottom=True,
        left=True,
        top=False,
        right=False,
        colors="black",
        labelcolor="black",
        direction="in",
        length=4 if ticks else 0,
        width=0.9,
    )
    ax.tick_params(axis="both", which="minor", bottom=True, left=True, top=False, right=False, direction="in", length=2 if ticks else 0, width=0.7, colors="black")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    ax.title.set_color("black")


def style_legend(ax) -> None:
    legend = ax.get_legend()
    if legend is None:
        return
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(0.6)
    for text in legend.get_texts():
        text.set_color("black")


def save(fig, path: Path, dpi: int) -> None:
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def fig5(dirs: dict[str, Path], dpi: int) -> None:
    cm = np.load(dirs["data"] / "confusion_matrix.npy")
    samples = pd.read_csv(dirs["data"] / "fig5_samples.csv")
    fig = plt.figure(figsize=(9.2, 3.5))
    fig.text(0.215, 0.93, "(a) Confusion matrix", ha="center", va="center", color="black", fontsize=9)
    fig.text(0.705, 0.93, "(b) Representative disaster-scene classifications", ha="center", va="center", color="black", fontsize=9)

    ax = fig.add_axes([0.060, 0.170, 0.300, 0.640])
    cax = fig.add_axes([0.375, 0.170, 0.017, 0.640])
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", cbar=True, cbar_ax=cax, square=True, ax=ax, vmin=0, vmax=1, linewidths=0)
    ax.set_xticklabels(CLASSES, rotation=35, ha="right")
    ax.set_yticklabels(CLASSES, rotation=0)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    style_axis(ax)
    if ax.collections and ax.collections[0].colorbar is not None:
        cbar = ax.collections[0].colorbar
        cbar.outline.set_edgecolor("black")
        cbar.ax.tick_params(color="black", labelcolor="black", direction="in", length=4, width=0.8)
    shown = samples.head(8).reset_index(drop=True)
    x0, y_top, y_bottom = 0.430, 0.550, 0.175
    col_w, row_h = 0.125, 0.285
    x_gap = 0.020
    for i, row in shown.iterrows():
        col = i % 4
        row_i = i // 4
        x = x0 + col * (col_w + x_gap)
        y = y_top if row_i == 0 else y_bottom
        ax_i = fig.add_axes([x, y, col_w, row_h])
        img = Image.open(row["path"]).convert("RGB")
        img = ImageOps.fit(img, (320, 230), method=Image.Resampling.LANCZOS)
        ax_i.imshow(img, interpolation="nearest", aspect="auto")
        ax_i.set_xticks([])
        ax_i.set_yticks([])
        ax_i.text(
            0.02,
            0.98,
            f"T: {row['true']}\nP: {row['pred']}",
            transform=ax_i.transAxes,
            ha="left",
            va="top",
            fontsize=5.5,
            color="black",
            bbox={"facecolor": "white", "edgecolor": "black", "linewidth": 0.3, "alpha": 0.82, "pad": 1.6},
        )
        for spine in ax_i.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.6)
    save(fig, dirs["figures"] / "Fig5_detection_performance.png", dpi)


def fig6(dirs: dict[str, Path], dpi: int) -> None:
    df = pd.read_csv(dirs["data"] / "phase2_training_curves.csv")
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    colors = sns.color_palette("colorblind", n_colors=df["config"].nunique())
    for color, (cfg, g) in zip(colors, df.groupby("config")):
        agg = g.groupby("iteration")["reward"].agg(["mean", "std", "count"]).reset_index()
        ci = 1.96 * agg["std"].fillna(0) / np.sqrt(agg["count"])
        x = agg["iteration"] / 1000.0
        ax.plot(x, agg["mean"], label=cfg, lw=1.6, color=color, marker="o", markevery=max(len(agg) // 8, 1), ms=3)
        ax.fill_between(x, agg["mean"] - ci, agg["mean"] + ci, color=color, alpha=0.16, linewidth=0)
    ax.set_xlabel("Training environment steps (x10^3)")
    ax.set_ylabel("Mean episodic reward")
    ymin = float(df["reward"].min())
    ymax = float(df["reward"].max())
    pad = max((ymax - ymin) * 0.08, 0.5)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.legend(ncol=3, frameon=True)
    style_axis(ax)
    style_legend(ax)
    save(fig, dirs["figures"] / "Fig6_training_convergence.png", dpi)


def fig7(dirs: dict[str, Path], dpi: int) -> None:
    df = pd.read_csv(dirs["data"] / "phase2_eval_records.csv")
    metrics = [
        ("task_completion_rate", "Task completion rate"),
        ("average_response_latency", "Average response latency"),
        ("cumulative_flight_distance", "Cumulative flight distance"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=True)
    for idx, (ax, (metric, label)) in enumerate(zip(axes.ravel()[:3], metrics)):
        sns.boxplot(
            data=df,
            x="config",
            y=metric,
            ax=ax,
            color="#9ecae1",
            fliersize=1.8,
            linewidth=0.9,
            showmeans=True,
            meanprops={"marker": "D", "markerfacecolor": "#1f1f1f", "markeredgecolor": "#1f1f1f", "markersize": 2.8},
        )
        ax.set_xlabel("")
        ax.set_ylabel(label)
        ax.set_title(f"({chr(ord('a') + idx)}) {label}")
        ymin = float(df[metric].min())
        ymax = float(df[metric].max())
        if np.isclose(ymin, ymax):
            pad = max(abs(ymax) * 0.04, 0.02)
            ax.set_ylim(ymin - pad, ymax + pad)
        elif metric == "cumulative_flight_distance":
            ax.set_ylim(max(0, ymin - 5), ymax + 5)
        style_axis(ax)

    ax = axes.ravel()[3]
    table6_path = dirs["tables"] / "Table6_computational_efficiency.csv"
    if table6_path.exists():
        latency = pd.read_csv(table6_path)
        labels = {
            "YOLOv8 classification inference": "YOLOv8",
            "DSSM update latency": "DSSM",
            "SB3 policy decision latency": "Policy",
            "Closed-loop update latency": "Closed loop",
            "End-to-end perception+control latency": "End-to-end",
        }
        latency = latency[latency["item"].isin(labels)].copy()
        latency["label"] = latency["item"].map(labels)
        order = ["DSSM", "Policy", "Closed loop", "YOLOv8", "End-to-end"]
        latency["label"] = pd.Categorical(latency["label"], categories=order, ordered=True)
        latency = latency.sort_values("label")
        colors = ["#8dd3c7", "#80b1d3", "#bebada", "#fb8072", "#b3de69"]
        ax.barh(latency["label"].astype(str), latency["mean_ms"], color=colors[: len(latency)], edgecolor="#4d4d4d", linewidth=0.6)
        xmin = max(float(latency["mean_ms"].min()) * 0.45, 0.01)
        xmax = float(latency["mean_ms"].max()) * 2.1
        ax.set_xscale("log")
        ax.set_xlim(xmin, xmax)
        for y, value in enumerate(latency["mean_ms"]):
            text = f"{value:.3f}" if value < 0.1 else f"{value:.2f}"
            ax.text(float(value) * 1.10, y, text, va="center", ha="left", fontsize=6.5)
        ax.set_xlabel("Latency (ms, log scale)")
        ax.set_ylabel("")
        ax.set_title("(d) Computational latency components")
        style_axis(ax)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Table 6 latency data unavailable", ha="center", va="center")
    save(fig, dirs["figures"] / "Fig7_closed_loop_boxplots.png", dpi)


def fig8(dirs: dict[str, Path], dpi: int) -> None:
    df = pd.read_csv(dirs["data"] / "fig8_trajectories.csv")
    points_path = dirs["data"] / "fig8_scene_points.csv"
    points = pd.read_csv(points_path) if points_path.exists() else pd.DataFrame()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), constrained_layout=True)
    for idx, (ax, (name, g)) in enumerate(zip(axes, df.groupby("policy"))):
        if not points.empty:
            pg_bg = points[(points["policy"] == name) & (points["kind"] == "real")]
            if not pg_bg.empty:
                limit_bg = max(float(df["x"].max()), float(df["y"].max()), 32)
                xs = np.linspace(0, limit_bg, 80)
                ys = np.linspace(0, limit_bg, 80)
                xx, yy = np.meshgrid(xs, ys)
                heat = np.zeros_like(xx)
                for _, row in pg_bg.iterrows():
                    spread = 2.1 if row["completed"] == 1 else 2.8
                    weight = 0.75 if row["completed"] == 1 else 1.10
                    heat += weight * np.exp(-((xx - row["x"]) ** 2 + (yy - row["y"]) ** 2) / (2 * spread**2))
                if heat.max() > 0:
                    ax.contourf(xx, yy, heat / heat.max(), levels=np.linspace(0.18, 1.0, 6), cmap="Reds", alpha=0.18)
        if len(g) > 1:
            xy = g[["x", "y"]].to_numpy()
            segments = np.stack([xy[:-1], xy[1:]], axis=1)
            lc = LineCollection(segments, cmap="viridis", linewidth=1.5)
            lc.set_array(g["t"].to_numpy()[:-1])
            ax.add_collection(lc)
        else:
            ax.plot(g["x"], g["y"], lw=1.4)
        if not points.empty:
            pg = points[points["policy"] == name]
            real = pg[pg["kind"] == "real"]
            decoy = pg[pg["kind"] == "decoy"]
            done = real[real["completed"] == 1]
            missed = real[real["completed"] == 0]
            if not decoy.empty:
                ax.scatter(decoy["x"], decoy["y"], s=16, marker="^", color="#f28e2b", alpha=0.45, label="false alert")
            if not missed.empty:
                ax.scatter(missed["x"], missed["y"], s=22, marker="x", color="#d62728", linewidths=0.9, label="missed")
            if not done.empty:
                ax.scatter(done["x"], done["y"], s=22, marker="+", color="#2ca02c", linewidths=1.0, label="served")
        ax.scatter(g["x"].iloc[0], g["y"].iloc[0], s=28, marker="s", label="start")
        ax.scatter(g["x"].iloc[-1], g["y"].iloc[-1], s=28, marker="o", label="end")
        ax.set_title(f"({chr(ord('a') + idx)}) {name}")
        limit = max(float(df["x"].max()), float(df["y"].max()), 32)
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_aspect("equal")
        ax.set_xticks(np.linspace(0, limit, 5))
        ax.set_yticks(np.linspace(0, limit, 5))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        style_axis(ax)
    axes[0].legend(loc="upper right", frameon=True)
    style_legend(axes[0])
    save(fig, dirs["figures"] / "Fig8_dispatch_trajectories.png", dpi)


def fig9(dirs: dict[str, Path], dpi: int) -> None:
    df = pd.read_csv(dirs["data"] / "phase4_sensitivity_records.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.7), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.03, wspace=0.16, hspace=0.02)
    titles = {"FNR": "False negative rate", "FPR": "False positive rate", "LOC": "Localization error"}
    for idx, (ax, kind) in enumerate(zip(axes, ["FNR", "FPR", "LOC"])):
        g = df[df["noise_type"] == kind]
        agg = g.groupby("level")["task_completion_rate"].agg(["mean", "std"]).reset_index()
        ax.errorbar(agg["level"], agg["mean"], yerr=agg["std"], marker="o", lw=1.4, capsize=3)
        ax.set_title(f"({chr(ord('a') + idx)}) {titles[kind]}")
        ax.set_xlabel("Noise level" if kind != "LOC" else "Sigma (cells)")
        ax.set_ylabel("Task completion rate")
        ax.set_ylim(0, 1.02)
        style_axis(ax)
    save(fig, dirs["figures"] / "Fig9_noise_sensitivity.png", dpi)


def fig10(dirs: dict[str, Path], dpi: int) -> None:
    cm = np.load(dirs["data"] / "confusion_matrix.npy")
    sens = pd.read_csv(dirs["data"] / "phase4_sensitivity_records.csv")
    real = pd.read_csv(dirs["data"] / "phase5_real_noise_records.csv")
    real_perf = real["task_completion_rate"].mean()
    disaster_idx = [CLASSES.index(c) for c in DISASTER_CLASSES]
    normal_idx = CLASSES.index("normal")
    fnr = 1 - np.diag(cm)[disaster_idx].mean()
    fpr = cm[normal_idx, disaster_idx].sum()
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.2), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.03, wspace=0.14, hspace=0.02)
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", cbar=True, square=True, ax=axes[0], vmin=0, vmax=1, linewidths=0)
    axes[0].set_xticklabels(CLASSES, rotation=35, ha="right")
    axes[0].set_yticklabels(CLASSES, rotation=0)
    axes[0].set_title("(a) Empirical confusion matrix")
    axes[0].set_xlabel("Predicted class")
    axes[0].set_ylabel("True class")
    style_axis(axes[0])
    if axes[0].collections and axes[0].collections[0].colorbar is not None:
        cbar = axes[0].collections[0].colorbar
        cbar.outline.set_edgecolor("black")
        cbar.ax.tick_params(color="black", labelcolor="black", length=3, width=0.8)
    curves = {}
    for kind, label, color in [
        ("FNR", "Synthetic FNR curve", "#1f77b4"),
        ("FPR", "Synthetic FPR curve", "#ff7f0e"),
    ]:
        curve = sens[sens["noise_type"] == kind].groupby("level")["task_completion_rate"].mean().reset_index()
        curves[kind] = curve
        axes[1].plot(curve["level"], curve["task_completion_rate"], marker="o", lw=1.3, label=label, color=color)
    fnr_curve = curves["FNR"].sort_values("level")
    order = np.argsort(fnr_curve["task_completion_rate"].to_numpy())
    equiv_fnr = float(np.interp(real_perf, fnr_curve["task_completion_rate"].to_numpy()[order], fnr_curve["level"].to_numpy()[order]))
    axes[1].axhline(real_perf, color="#d62728", ls="--", lw=1.2, label=f"Real matrix={real_perf:.2f}")
    axes[1].axvline(fnr, color="#9467bd", ls="-.", lw=1.1, label=f"Empirical FNR={fnr:.2f}")
    axes[1].axvline(fpr, color="#8c564b", ls=":", lw=1.1, label=f"Empirical FPR={fpr:.2f}")
    axes[1].scatter([equiv_fnr], [real_perf], marker="*", s=75, color="#d62728", zorder=5, label=f"Equivalent FNR={equiv_fnr:.2f}")
    axes[1].set_xlabel("Equivalent synthetic noise level")
    axes[1].set_ylabel("Task completion rate")
    axes[1].set_xlim(-0.01, 0.31)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("(b) Real-noise calibration")
    axes[1].legend(frameon=True)
    style_axis(axes[1])
    style_legend(axes[1])
    save(fig, dirs["figures"] / "Fig10_real_noise_calibration.png", dpi)


def make_all_figures(cfg: dict, dirs: dict[str, Path]) -> None:
    set_style(cfg["figures"]["font"])
    dpi = int(cfg["figures"]["dpi"])
    fig5(dirs, dpi)
    fig6(dirs, dpi)
    fig7(dirs, dpi)
    fig8(dirs, dpi)
    fig9(dirs, dpi)
    fig10(dirs, dpi)


def find_project_root(start: Path | None = None) -> Path:
    """Locate the project root so this file can be run directly from PyCharm."""
    current = (start or Path(__file__).resolve()).resolve()
    search_dirs = [current if current.is_dir() else current.parent, *current.parents]
    for candidate in search_dirs:
        if (candidate / "paper_results_final" / "data").exists():
            return candidate
    return Path.cwd().resolve()


def default_config_and_dirs(project_root: Path | None = None) -> tuple[dict, dict[str, Path]]:
    root = project_root or find_project_root()
    results_dir = root / "paper_results_final"
    dirs = {
        "root": results_dir,
        "data": results_dir / "data",
        "figures": results_dir / "figures",
        "tables": results_dir / "tables",
        "models": results_dir / "models",
        "logs": results_dir / "logs",
        "figure_code": results_dir / "figure_code",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    cfg = {
        "figures": {
            "dpi": 600,
            "font": "Arial",
        }
    }
    return cfg, dirs


if __name__ == "__main__":
    config, output_dirs = default_config_and_dirs()
    make_all_figures(config, output_dirs)
    print(f"Figures regenerated in: {output_dirs['figures']}")
