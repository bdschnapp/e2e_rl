"""Generate thesis figures from the saved sweep CSVs — no re-training needed.

Reads <outdir>/summary.csv + <outdir>/curves.csv (written by run_stage1_sweep.py) and
writes figures to <outdir>/figures/:
  final_<metric>.png     grouped bar (x=algo, grouped by reward), faceted by direction,
                         with 95% CI error bars   [from summary.csv]
  curve_<metric>.png     metric vs timesteps, one line per algo (mean over seeds+rewards),
                         faceted by direction      [from curves.csv]

Pure csv + matplotlib (no pandas) so it's dependency-light.

    python scripts/plot_stage1.py --outdir results_stage1
"""

import os
import csv
import argparse
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BAR_METRICS = ["completion", "trailer_cte", "max_hitch", "jackknife"]
CURVE_METRICS = ["completion", "trailer_cte"]


def _read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def plot_final_bars(summary, outdir):
    directions = sorted({r["direction"] for r in summary})
    algos = sorted({r["algo"] for r in summary})
    rewards = sorted({r["reward"] for r in summary})
    for metric in BAR_METRICS:
        fig, axes = plt.subplots(1, len(directions), figsize=(6 * len(directions), 4), squeeze=False)
        for ax, direction in zip(axes[0], directions):
            w = 0.8 / max(len(rewards), 1)
            for j, reward in enumerate(rewards):
                means, cis, xs = [], [], []
                for i, algo in enumerate(algos):
                    row = next((r for r in summary if r["algo"] == algo and r["direction"] == direction
                                and r["reward"] == reward), None)
                    means.append(_f(row[f"{metric}_mean"]) if row else float("nan"))
                    cis.append(_f(row[f"{metric}_ci95"]) if row else float("nan"))
                    xs.append(i + j * w)
                ax.bar(xs, means, w, yerr=cis, capsize=2, label=reward)
            ax.set_xticks([i + 0.4 - w / 2 for i in range(len(algos))])
            ax.set_xticklabels(algos, rotation=20)
            ax.set_title(f"{direction}"); ax.set_ylabel(metric)
            ax.legend(fontsize=7, title="reward")
        fig.suptitle(f"Final {metric} (mean ± 95% CI)")
        fig.tight_layout()
        p = os.path.join(outdir, "figures", f"final_{metric}.png")
        fig.savefig(p, dpi=130); plt.close(fig)
        print(f"  wrote {p}")


def plot_curves(curves, outdir):
    directions = sorted({r["direction"] for r in curves})
    algos = sorted({r["algo"] for r in curves})
    for metric in CURVE_METRICS:
        fig, axes = plt.subplots(1, len(directions), figsize=(6 * len(directions), 4), squeeze=False)
        for ax, direction in zip(axes[0], directions):
            for algo in algos:
                pts = defaultdict(list)  # timestep -> [values over seeds/rewards]
                for r in curves:
                    if r["algo"] == algo and r["direction"] == direction:
                        pts[int(_f(r["timesteps"]))].append(_f(r[metric]))
                if not pts:
                    continue
                ts = sorted(pts)
                mean = [np.nanmean(pts[t]) for t in ts]
                ax.plot(ts, mean, marker="o", ms=3, label=algo)
            ax.set_title(direction); ax.set_xlabel("timesteps"); ax.set_ylabel(metric)
            ax.legend(fontsize=7)
        fig.suptitle(f"{metric} vs training (mean over seeds+rewards)")
        fig.tight_layout()
        p = os.path.join(outdir, "figures", f"curve_{metric}.png")
        fig.savefig(p, dpi=130); plt.close(fig)
        print(f"  wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results_stage1")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.outdir, "figures"), exist_ok=True)
    summary = _read(os.path.join(args.outdir, "summary.csv"))
    curves = _read(os.path.join(args.outdir, "curves.csv"))
    plot_final_bars(summary, args.outdir)
    plot_curves(curves, args.outdir)
    print(f"# figures in {os.path.join(args.outdir, 'figures')}")


if __name__ == "__main__":
    main()
