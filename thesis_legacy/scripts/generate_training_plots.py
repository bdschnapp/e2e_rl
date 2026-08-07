"""Generate training curve plots from SB3 log data."""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_monitor_csv(path: Path):
    """Load a Stable Baselines3 Monitor CSV log."""
    data = np.genfromtxt(path, delimiter=",", skip_header=2, names=True)
    return data


def plot_training_curve(log_path: Path, output_dir: Path):
    data = load_monitor_csv(log_path)
    timesteps = np.cumsum(data["l"])
    rewards = data["r"]

    # Smooth with a rolling window
    window = 50
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(timesteps[window - 1 :], smoothed, linewidth=1.5)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode Return")
    ax.set_title("Training Curve")
    ax.grid(True, alpha=0.3)

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "training_curve.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path, help="Path to monitor.csv")
    parser.add_argument("--out", type=Path, default=Path("figures/experiment_results"))
    args = parser.parse_args()
    plot_training_curve(args.log, args.out)
