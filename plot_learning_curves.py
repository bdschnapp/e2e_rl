"""
Plot evaluation learning curves from SB3 EvalCallback output.

Each training run produces  models/<scenario>/<obs_tag>/<reward>/logs/evaluations.npz
with arrays:
    timesteps  — shape (n_evals,)
    results    — shape (n_evals, n_eval_episodes)  episode returns at each checkpoint
    ep_lengths — shape (n_evals, n_eval_episodes)

Usage
-----
    # Overlay three runs on one figure:
    python plot_learning_curves.py \\
        --runs "State=models/forward/state/dense/logs/evaluations.npz" \\
               "Lidar=models/forward/lidar/dense/logs/evaluations.npz" \\
               "BEV-CNN=models/forward/bev_scratch/dense/logs/evaluations.npz" \\
        --title "Forward Obs Ablation" \\
        --out thesis/figures/experiment_results/forward_obs_ablation.pdf

    # Quick check on a single run:
    python plot_learning_curves.py \\
        --runs "Reverse/State=models/reverse/state/dense/logs/evaluations.npz" \\
        --out /tmp/quick_check.png
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Apply a 1-D rolling mean with window w. Edges use available samples."""
    if w <= 1:
        return x
    out = np.empty_like(x)
    for i in range(len(x)):
        lo = max(0, i - w + 1)
        out[i] = x[lo : i + 1].mean()
    return out


def load_run(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load evaluations.npz and return (timesteps, mean_return, std_return)."""
    data = np.load(path)
    timesteps = data["timesteps"]               # (n_evals,)
    results   = data["results"]                 # (n_evals, n_eval_episodes)
    mean_ret  = results.mean(axis=1)
    std_ret   = results.std(axis=1)
    return timesteps, mean_ret, std_ret


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    runs: list[tuple[str, str]],
    title: str,
    out: Path,
    smooth: int,
):
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for label, path in runs:
        if not Path(path).exists():
            print(f"  [WARN] not found, skipping: {path}")
            continue

        timesteps, mean_ret, std_ret = load_run(path)
        mean_sm = _rolling_mean(mean_ret, smooth)
        std_sm  = _rolling_mean(std_ret,  smooth)

        line, = ax.plot(timesteps, mean_sm, label=label, linewidth=1.8)
        ax.fill_between(
            timesteps,
            mean_sm - std_sm,
            mean_sm + std_sm,
            alpha=0.18,
            color=line.get_color(),
        )

        # Terminal summary
        tail = mean_ret[-10:]
        print(f"  {label:<30s}  final-10-eval mean: {tail.mean():.3f}  ± {tail.std():.3f}")

    ax.set_xlabel("Environment steps", fontsize=11)
    ax.set_ylabel("Mean episode return", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Overlay evaluation learning curves from evaluations.npz files."
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        metavar="LABEL=PATH",
        required=True,
        help=(
            'One or more label=path pairs, e.g. '
            '"State=models/forward/state/dense/logs/evaluations.npz"'
        ),
    )
    parser.add_argument(
        "--title",
        default="Learning Curves",
        help="Figure title (default: 'Learning Curves').",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("learning_curve.pdf"),
        help="Output file path (PDF or PNG). Default: learning_curve.pdf.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Rolling-mean window size for smoothing (default: 1 = no smoothing).",
    )

    args = parser.parse_args()

    # Parse "LABEL=PATH" pairs
    parsed: list[tuple[str, str]] = []
    for item in args.runs:
        if "=" not in item:
            parser.error(f"--runs entries must be 'LABEL=PATH', got: {item!r}")
        label, path = item.split("=", 1)
        parsed.append((label.strip(), path.strip()))

    main(runs=parsed, title=args.title, out=args.out, smooth=args.smooth)
