"""Faithful native rollout + visualization of a reverse tractor-only model.

Runs the model in its EXACT training env (tractor_trailer_rl, reusing
train_baseline.build_cfg) on GENTLE (steering-required) paths, so a "never steer"
policy would fail. The smoke test that the actor deliberately steers and tracks
before we trust the ROS deploy.

  # static multi-episode summary PNG (headless):
  /home/ben/Ben/Thesis/e2e_rl/venv/bin/python scripts/viz_reverse_rollout.py --episodes 6

  # WATCH it drive live (animated window; needs a display):
  /home/ben/Ben/Thesis/e2e_rl/venv/bin/python scripts/viz_reverse_rollout.py --live --episodes 3
"""
import os, sys, argparse
os.environ.setdefault("SDL_VIDEODRIVER", "dummy"); os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")

LIVE = "--live" in sys.argv
import matplotlib
matplotlib.use("TkAgg" if LIVE else "Agg")   # e2e_rl venv has tkinter, not Qt
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from dataclasses import replace

sys.path.insert(0, str(Path(__file__).resolve().parent))   # for train_baseline
from train_baseline import build_cfg
from tractor_trailer_rl import Direction, VehicleKind
from tractor_trailer_rl.envs.base import LaneFollowingEnv

ALL_KINDS = ["straight", "gentle", "sharp", "winding", "lab_seam", "lab_corner"]

def make_cfg(kind="gentle"):
    cfg = build_cfg(Direction.REVERSE, VehicleKind.TRACTOR_ONLY)
    kp = {k: (1.0 if k == kind else 0.0) for k in ALL_KINDS}
    return replace(cfg, path=replace(cfg.path, mild_only=False, kind_probs=kp))

def load(args):
    cfg = make_cfg(args.kind)
    env = LaneFollowingEnv(cfg)
    from stable_baselines3 import TD3
    model = TD3.load(args.model, env=None, device="cpu")
    print(f"loaded {args.model}")
    return cfg, env, model

def run_live(args):
    cfg, env, model = load(args)
    half = cfg.world.lane_centerline_half_width_m
    plt.ion()
    fig, ax = plt.subplots(figsize=(14, 5))
    for ep in range(args.episodes):
        obs, info = env.reset(seed=100 + ep)
        xs, ys = env.xx, env.yy
        ax.clear()
        ax.plot(xs, ys, "b-", lw=1.2)
        ax.fill_between(xs, ys - half, ys + half, color="0.85", zorder=0)
        ax.set_xlim(xs.min() - 2, xs.max() + 2)
        ax.set_ylim(ys.min() - half - 1, ys.max() + half + 1)
        (trail,) = ax.plot([], [], "-", color="orange", lw=1.5)
        (truck,) = ax.plot([], [], "ko", ms=9)
        head = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                           arrowprops=dict(arrowstyle="->", color="red", lw=2))
        tx, ty = [], []
        done = False; step = 0
        while not done:
            a, _ = model.predict(obs, deterministic=True); a = np.asarray(a).flatten()
            obs, r, term, trunc, info = env.step(a)
            px, py = info["tractor_pos"]; yaw = info["tractor_yaw"]
            tx.append(px); ty.append(py); step += 1
            trail.set_data(tx, ty); truck.set_data([px], [py])
            head.set_position((px, py)); head.xy = (px + 1.2*np.cos(yaw), py + 1.2*np.sin(yaw))
            ax.set_title(f"ep{ep}  step{step}  steer_rate={a[0]:+.2f}  stop={a[1]:+.2f}")
            done = term or trunc
            if step % args.skip == 0:
                plt.pause(args.dt)
        plt.pause(0.4)
    plt.ioff(); print("done"); plt.show()

def run_static(args):
    cfg, env, model = load(args)
    half = cfg.world.lane_centerline_half_width_m
    n = args.episodes
    fig, axs = plt.subplots(2, n, figsize=(4.2 * n, 8))
    if n == 1: axs = axs.reshape(2, 1)
    for ep in range(n):
        obs, info = env.reset(seed=100 + ep)
        xs_path, ys_path = env.xx.copy(), env.yy.copy()
        traj, steers, stops = [], [], []
        done = False; R = 0.0
        while not done:
            obs_in = obs.copy()
            if args.mirror:
                # Replicate the EXACT bridge transform proven by compare_bridge_vs_native:
                # state[:5] negated, lidar beams reversed. Then un-mirror the steer rate.
                obs_in[:5] = -obs_in[:5]
                obs_in[5:] = obs_in[5:][::-1]
            a, _ = model.predict(obs_in, deterministic=True); a = np.asarray(a).flatten()
            if args.mirror:
                a = a.copy(); a[0] = -a[0]   # node's -action[0] un-mirror
            obs, r, term, trunc, info = env.step(a)
            traj.append(info["tractor_pos"]); steers.append(float(a[0]))
            stops.append(float(a[1]) if a.size > 1 else 0.0); R += float(r); done = term or trunc
        traj = np.array(traj)
        ok = "OK" if (trunc and not term) else "EARLY-TERM"
        order = np.argsort(xs_path)
        e_y = traj[:, 1] - np.interp(traj[:, 0], xs_path[order], ys_path[order])
        rms = float(np.sqrt(np.mean(e_y ** 2)))
        ax = axs[0, ep]
        ax.plot(xs_path, ys_path, "b-", lw=1.2)
        ax.fill_between(xs_path, ys_path - half, ys_path + half, color="0.85", zorder=0)
        ax.scatter(traj[:, 0], traj[:, 1], c=np.arange(len(traj)), cmap="plasma", s=5)
        ax.plot(traj[0, 0], traj[0, 1], "go", ms=8); ax.plot(traj[-1, 0], traj[-1, 1], "rx", ms=9)
        ax.set_title(f"ep{ep} {ok} len{len(traj)} R{R:.0f} | RMS e_y={rms:.2f}m")
        ax.set_ylim(ys_path.min() - half - 0.6, ys_path.max() + half + 0.6)
        ax2 = axs[1, ep]
        ax2.plot(e_y, "k-", lw=1, label="cross-track e_y (m)")
        ax2.axhspan(-half, half, color="0.88", zorder=0); ax2.axhline(0, color="0.6", lw=0.6)
        ax2.plot(np.array(steers) * half, "b-", lw=0.7, alpha=0.6, label="steer_rate (scaled)")
        ax2.set_title("tracking + steering"); ax2.set_xlabel("step"); ax2.set_ylim(-2*half, 2*half)
        if ep == 0: ax2.legend(fontsize=7)
    fig.suptitle("Reverse tractor-only — native ttrl rollout on GENTLE paths (steering required)")
    fig.tight_layout(); fig.savefig(args.out, dpi=95, bbox_inches="tight")
    print(f"wrote {args.out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser(
        "~/Ben/Thesis/Electrans_project/lab_models_ttrl_baseline/reverse_tractor_only/best_model.zip"))
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--kind", default="gentle", choices=ALL_KINDS)
    ap.add_argument("--mirror", action="store_true",
                    help="apply the EXACT bridge Y-mirror (negate state, reverse lidar, un-mirror "
                         "steer) in closed loop -- tests whether the bridge transform itself breaks it")
    ap.add_argument("--live", action="store_true", help="animated window instead of static PNG")
    ap.add_argument("--dt", type=float, default=0.01, help="live: pause per frame (s)")
    ap.add_argument("--skip", type=int, default=2, help="live: redraw every Nth step (speed)")
    ap.add_argument("--out", default=os.path.expanduser(
        "~/Ben/Thesis/Electrans_project/lab_map_capture/reverse_smoke.png"))
    args = ap.parse_args()
    (run_live if args.live else run_static)(args)

if __name__ == "__main__":
    main()
