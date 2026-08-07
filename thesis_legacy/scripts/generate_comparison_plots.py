"""
Generate thesis-quality comparison plots from benchmark results.

Reads results/summary_{task}.csv produced by benchmark.py and creates:
  - Box plots of CTE distribution per controller
  - CDF of trailer CTE across all test paths
  - Bar charts of completion rate and jackknife rate with error bars
  - Inference time comparison

Usage
-----
    python thesis/scripts/generate_comparison_plots.py \\
        --results results/ \\
        --out thesis/figures/experiment_results/

One .pdf (and .png) is saved per figure into the output directory.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------

CONTROLLER_LABELS = {
    "td3": "TD3 (RL)",
    "fpp": "Pure Pursuit",
    "rpp": "Reverse PP",
    "mpc": "MPC",
}

CONTROLLER_COLORS = {
    "td3": "#2196F3",
    "fpp": "#FF9800",
    "rpp": "#FF9800",
    "mpc": "#4CAF50",
}

TASK_TITLES = {
    "forward":      "Forward Path Following",
    "reverse":      "Reverse Path Following",
    "obstacle_fwd": "Forward Obstacle Avoidance",
    "obstacle_rev": "Reverse Obstacle Avoidance",
}

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "pdf.fonttype": 42,  # editable text in PDFs
})


def _save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}.pdf/png")


def _label(ctrl: str) -> str:
    return CONTROLLER_LABELS.get(ctrl, ctrl)


def _color(ctrl: str) -> str:
    return CONTROLLER_COLORS.get(ctrl, "#9E9E9E")


# -----------------------------------------------------------------------
# Individual plot functions
# -----------------------------------------------------------------------

def plot_cte_boxplot(df: pd.DataFrame, task: str, out_dir: Path):
    """Box plot: trailer CTE distribution per controller."""
    controllers = df["controller"].unique().tolist()
    data = [df[df["controller"] == c]["mean_abs_cte_trailer"].dropna().values for c in controllers]
    labels = [_label(c) for c in controllers]
    colors = [_color(c) for c in controllers]

    fig, ax = plt.subplots(figsize=(max(4, len(controllers) * 1.8), 4.5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False,
                    medianprops={"linewidth": 2, "color": "black"})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_ylabel("Mean |CTE| — trailer (m)")
    ax.set_title(f"Cross-Track Error Distribution\n{TASK_TITLES.get(task, task)}")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_dir, f"{task}_cte_boxplot")


def plot_cte_cdf(df: pd.DataFrame, task: str, out_dir: Path):
    """CDF of mean trailer CTE across all test scenarios."""
    controllers = df["controller"].unique().tolist()
    fig, ax = plt.subplots(figsize=(5.5, 4))

    for ctrl in controllers:
        vals = np.sort(df[df["controller"] == ctrl]["mean_abs_cte_trailer"].dropna().values)
        if len(vals) == 0:
            continue
        cdf = np.arange(1, len(vals) + 1) / len(vals)
        ax.step(vals, cdf, label=_label(ctrl), color=_color(ctrl), linewidth=1.8)

    ax.set_xlabel("Mean |CTE| — trailer (m)")
    ax.set_ylabel("Cumulative Fraction")
    ax.set_title(f"CTE CDF\n{TASK_TITLES.get(task, task)}")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, out_dir, f"{task}_cte_cdf")


def plot_completion_rates(df: pd.DataFrame, task: str, out_dir: Path):
    """Grouped bar chart: completion rate, jackknife rate, collision rate."""
    controllers = df["controller"].unique().tolist()
    metrics = [("completed", "Completion Rate"), ("jackknifed", "Jackknife Rate"), ("collided", "Collision Rate")]

    x = np.arange(len(metrics))
    width = 0.8 / max(len(controllers), 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, ctrl in enumerate(controllers):
        sub = df[df["controller"] == ctrl]
        rates = [sub[m].mean() * 100 if m in sub.columns else 0.0 for m, _ in metrics]
        offset = (i - len(controllers) / 2 + 0.5) * width
        ax.bar(x + offset, rates, width * 0.9, label=_label(ctrl),
               color=_color(ctrl), alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in metrics])
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"Episode Outcome Rates\n{TASK_TITLES.get(task, task)}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_dir, f"{task}_outcome_rates")


def plot_inference_time(df: pd.DataFrame, task: str, out_dir: Path):
    """Bar chart of mean and P95 inference time per controller."""
    controllers = df["controller"].unique().tolist()
    means = []
    p95s = []
    labels = []
    colors = []
    for ctrl in controllers:
        sub = df[df["controller"] == ctrl]
        if "mean_inference_ms" not in sub.columns:
            continue
        means.append(sub["mean_inference_ms"].mean())
        p95s.append(sub["p95_inference_ms"].mean() if "p95_inference_ms" in sub.columns else float("nan"))
        labels.append(_label(ctrl))
        colors.append(_color(ctrl))

    if not means:
        return

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(4, len(labels) * 1.8), 4))
    bars = ax.bar(x, means, color=colors, alpha=0.8, label="Mean")
    ax.scatter(x, p95s, color="black", zorder=5, marker="^", s=60, label="P95")
    ax.axhline(y=100, color="red", linestyle="--", linewidth=1.2, label="dt = 100 ms")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Inference Time (ms)")
    ax.set_title(f"Controller Computational Cost\n{TASK_TITLES.get(task, task)}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_dir, f"{task}_inference_time")


def plot_hitch_angle(df: pd.DataFrame, task: str, out_dir: Path):
    """Box plot of mean absolute hitch angle per controller."""
    controllers = df["controller"].unique().tolist()
    data = [np.degrees(df[df["controller"] == c]["mean_abs_hitch_angle"].dropna().values)
            for c in controllers]
    labels = [_label(c) for c in controllers]
    colors = [_color(c) for c in controllers]

    fig, ax = plt.subplots(figsize=(max(4, len(controllers) * 1.8), 4.5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    medianprops={"linewidth": 2, "color": "black"})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_ylabel("Mean |Hitch Angle| (°)")
    ax.set_title(f"Hitch Angle Distribution\n{TASK_TITLES.get(task, task)}")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_dir, f"{task}_hitch_angle")


def plot_cte_by_difficulty(df: pd.DataFrame, task: str, out_dir: Path):
    """Bar chart: mean trailer CTE broken down by path difficulty."""
    if "difficulty" not in df.columns:
        return
    difficulties = ["straight", "mild", "aggressive", "scurve"]
    difficulties = [d for d in difficulties if d in df["difficulty"].values]
    controllers = df["controller"].unique().tolist()

    x = np.arange(len(difficulties))
    width = 0.8 / max(len(controllers), 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, ctrl in enumerate(controllers):
        sub = df[df["controller"] == ctrl]
        means = [sub[sub["difficulty"] == d]["mean_abs_cte_trailer"].mean() for d in difficulties]
        stds = [sub[sub["difficulty"] == d]["mean_abs_cte_trailer"].std() for d in difficulties]
        offset = (i - len(controllers) / 2 + 0.5) * width
        ax.bar(x + offset, means, width * 0.9, yerr=stds, capsize=3,
               label=_label(ctrl), color=_color(ctrl), alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in difficulties])
    ax.set_xlabel("Path Difficulty")
    ax.set_ylabel("Mean |CTE| — trailer (m)")
    ax.set_title(f"CTE by Path Difficulty\n{TASK_TITLES.get(task, task)}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_dir, f"{task}_cte_by_difficulty")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def process_task(csv_path: Path, out_dir: Path):
    task = csv_path.stem.replace("summary_", "")
    print(f"\nProcessing {csv_path.name}  (task={task})")

    df = pd.read_csv(csv_path)

    # Cast boolean columns
    for col in ["completed", "jackknifed", "collided", "truncated"]:
        if col in df.columns:
            df[col] = df[col].map(lambda v: str(v).lower() in ("true", "1", "yes"))

    task_out = out_dir / task
    plot_cte_boxplot(df, task, task_out)
    plot_cte_cdf(df, task, task_out)
    plot_completion_rates(df, task, task_out)
    plot_inference_time(df, task, task_out)
    plot_hitch_angle(df, task, task_out)
    plot_cte_by_difficulty(df, task, task_out)


def main():
    parser = argparse.ArgumentParser(description="Generate thesis comparison plots")
    parser.add_argument(
        "--results", type=Path, default=Path("results"),
        help="Directory containing summary_*.csv files (default: results/)"
    )
    parser.add_argument(
        "--out", type=Path, default=Path("thesis/figures/experiment_results"),
        help="Output directory for figures (default: thesis/figures/experiment_results/)"
    )
    parser.add_argument(
        "--task", default=None,
        help="Process only this task (default: all summary_*.csv files found)"
    )
    args = parser.parse_args()

    if args.task:
        csv_files = [args.results / f"summary_{args.task}.csv"]
    else:
        csv_files = sorted(args.results.glob("summary_*.csv"))

    if not csv_files:
        print(f"No summary_*.csv files found in {args.results}")
        sys.exit(1)

    for csv_path in csv_files:
        if csv_path.exists():
            process_task(csv_path, args.out)
        else:
            print(f"  Skipping missing file: {csv_path}")

    print("\nAll figures saved.")


if __name__ == "__main__":
    main()
