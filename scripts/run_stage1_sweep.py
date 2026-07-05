"""Stage-1 ablation sweep with plot-ready, re-run-free output.

Loops the algorithm 2x2 (+ optional SAC) x reward x direction at fixed obs, x seeds,
trains each via SB3 on the batched GPU env, and saves EVERYTHING needed to generate
thesis figures/tables later WITHOUT re-training:

  <outdir>/curves.csv     one row per (run, eval-checkpoint): metadata + full metric
                          set vs timesteps  -> learning-curve plots
  <outdir>/summary.csv    one row per config: final metrics mean +/- 95% CI over
                          seeds + steps-to-threshold  -> bar charts / result tables
  <outdir>/manifest.jsonl one line per run: full config, vehicle params, lr, wall-clock,
                          final metrics  -> reproducibility + param provenance

Every figure/table then derives from these CSVs; re-run only to change the experiment.

    TTRL_BACKEND=cupy python scripts/run_stage1_sweep.py --preset shunt \
        --algos ppo td3 ppo_disc dqn --seeds 0 1 2 --steps 800000 --outdir results_stage1
"""

import os
import sys
import json
import time
import csv
import math
import argparse
import warnings
from dataclasses import replace, asdict

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np  # noqa: E402
from tractor_trailer_rl.batched import backend  # noqa: E402
from tractor_trailer_rl.config import ActionMode  # noqa: E402
from run_ablations_gpu import cell_to_cfg  # noqa: E402
from train_sb3 import build_model, evaluate, EvalCurveCallback  # noqa: E402

# algorithm token -> (sb3 algo, discrete?)  — the 2x2 + SAC
ALGO_MAP = {"ppo": ("ppo", False), "td3": ("td3", False), "sac": ("sac", False),
            "dqn": ("dqn", True), "ppo_disc": ("ppo", True)}
FWD_REWARDS = ["dense", "tractor_focus", "multiplicative", "guided"]
REV_REWARDS = ["dense", "no_hitch", "multiplicative", "guided"]
METRICS = ["completion", "trailer_cte", "max_hitch", "jackknife", "ep_return"]


def train_one(algo, direction, obs, reward, seed, args):
    from tractor_trailer_rl.batched.sb3_adapter import SB3BatchedVecEnv
    sb3_algo, discrete = ALGO_MAP[algo]
    cfg = cell_to_cfg(f"{direction}:{obs}:{reward}", preset=args.preset,
                      fixed_speed=args.fixed_speed, mild_paths=args.mild_paths)
    if discrete:
        cfg = replace(cfg, action=replace(cfg.action, mode=ActionMode.DISCRETE))
    lr = 1e-3 if sb3_algo == "ppo" else 2e-3
    steps = args.ppo_steps if sb3_algo == "ppo" else args.steps  # per-algo budget

    env = SB3BatchedVecEnv(cfg, args.n_envs, path_pool_size=1024)
    eval_env = SB3BatchedVecEnv(cfg, min(args.n_envs, 256), path_pool_size=512)
    model = build_model(sb3_algo, env, lr, args.batch_size, seed)
    cb = EvalCurveCallback(eval_env, args.eval_every or max(steps // 8, 1))
    t0 = time.perf_counter()
    model.learn(total_timesteps=steps, progress_bar=False, callback=cb)
    wall = time.perf_counter() - t0
    final = evaluate(model, eval_env, steps=800)
    best = cb.best or final
    v = cfg.vehicle
    meta = dict(algo=algo, sb3_algo=sb3_algo, discrete=discrete, direction=direction,
                obs=obs, reward=reward, seed=seed, preset=args.preset, lr=lr,
                batch_size=args.batch_size, n_envs=args.n_envs, steps=steps,
                wall_s=round(wall, 1), steps_to_threshold=cb.steps_to_threshold,
                veh_m=v.m, veh_wheelbase=round(v.lf + v.lr, 3),
                veh_trailer_L=v.trailer_length_m, veh_max_steer=v.max_steer_angle_rad,
                bend_scale=cfg.path.bend_scale, best_timesteps=best.get("timesteps"))
    return cb.curve, best, final, meta


def ci95(xs):
    xs = [x for x in xs if x == x]
    return 1.96 * np.std(xs, ddof=1) / math.sqrt(len(xs)) if len(xs) >= 2 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", nargs="+", default=["ppo", "td3", "ppo_disc", "dqn"])
    ap.add_argument("--rewards", nargs="+", default=None, help="default: all valid per direction")
    ap.add_argument("--directions", nargs="+", default=["forward", "reverse"])
    ap.add_argument("--obs", nargs="+", default=["lidar24"],
                    help="observation(s) to cross: state, lidar4/8/16/24/32")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--preset", choices=["lab", "truck", "shunt"], default="shunt")
    ap.add_argument("--fixed_speed", action="store_true", default=True)
    ap.add_argument("--mild_paths", action="store_true", default=True)
    ap.add_argument("--n_envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=500000, help="budget for off-policy (TD3/DQN/SAC)")
    ap.add_argument("--ppo_steps", type=int, default=2000000, help="budget for PPO/PPO-disc (on-policy)")
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--eval_every", type=int, default=None)
    ap.add_argument("--outdir", default="results_stage1")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    curves_path = os.path.join(args.outdir, "curves.csv")
    summary_path = os.path.join(args.outdir, "summary.csv")
    manifest_path = os.path.join(args.outdir, "manifest.jsonl")
    curve_cols = (["run_id", "algo", "direction", "obs", "reward", "seed", "timesteps"] + METRICS
                  + ["n_episodes"])

    # RESUME: skip runs already completed (present in manifest without an error). Each
    # run is flushed on completion, so a crash/kill only loses the in-flight run; a
    # re-invocation continues where it left off. Adding seeds/obs later just appends.
    done = set()
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if "error" not in rec and "run_id" in rec:
                    done.add(rec["run_id"])

    cf = open(curves_path, "a", newline="")
    cw = csv.DictWriter(cf, fieldnames=curve_cols)
    if os.path.getsize(curves_path) == 0:
        cw.writeheader()
    mf = open(manifest_path, "a")

    # build the run list (algo x obs x reward x direction x seed)
    runs = []
    for direction in args.directions:
        rewards = args.rewards or (FWD_REWARDS if direction == "forward" else REV_REWARDS)
        for algo in args.algos:
            for obs in args.obs:
                for reward in rewards:
                    for seed in args.seeds:
                        runs.append((algo, direction, obs, reward, seed))
    todo = [r for r in runs
            if f"{r[0]}__{r[1]}__{r[2]}__{r[3]}__s{r[4]}" not in done]
    print(f"# Stage-1 sweep: {len(runs)} runs ({len(done)} done, {len(todo)} to run) -> "
          f"{args.outdir}  (preset={args.preset} steps={args.steps} n_envs={args.n_envs})")

    for i, (algo, direction, obs, reward, seed) in enumerate(todo):
        run_id = f"{algo}__{direction}__{obs}__{reward}__s{seed}"
        try:
            curve, best, final, meta = train_one(algo, direction, obs, reward, seed, args)
        except Exception as e:  # keep the sweep alive; record the failure and continue
            print(f"  [{i+1}/{len(todo)}] {run_id}  FAILED: {type(e).__name__}: {e}")
            mf.write(json.dumps({"run_id": run_id, "error": f"{type(e).__name__}: {e}"}) + "\n"); mf.flush()
            continue
        for pt in curve:
            row = dict(run_id=run_id, algo=algo, direction=direction, obs=obs, reward=reward,
                       seed=seed, timesteps=pt["timesteps"], n_episodes=pt.get("n_episodes"))
            row.update({m: pt.get(m) for m in METRICS})
            cw.writerow(row)
        cf.flush()
        meta_out = dict(run_id=run_id, **meta,
                        **{f"best_{k}": best[k] for k in METRICS},
                        **{f"final_{k}": final[k] for k in METRICS})
        mf.write(json.dumps(meta_out) + "\n"); mf.flush()
        print(f"  [{i+1}/{len(todo)}] {run_id}  BEST compl={best['completion']:.2f} "
              f"cte={best['trailer_cte']:.3f} hitch={best['max_hitch']:.3f} @{best.get('timesteps')} "
              f"| final={final['completion']:.2f} ({meta['wall_s']:.0f}s)")

    cf.close(); mf.close()

    # summary recomputed from the FULL manifest (all completed runs) so it's correct
    # after any resume: mean +/- 95% CI over seeds per (algo, direction, obs, reward).
    from collections import defaultdict
    groups = defaultdict(list)
    with open(manifest_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if "error" in rec or "best_completion" not in rec:
                continue
            groups[(rec["algo"], rec["direction"], rec["obs"], rec["reward"])].append(rec)
    # summary uses the BEST checkpoint per run (not final) — the ablation compares
    # each run's best policy.
    sfields = ["algo", "direction", "obs", "reward", "n_seeds"]
    for m in METRICS:
        sfields += [f"{m}_mean", f"{m}_ci95"]
    sfields += ["steps_to_threshold_mean"]
    with open(summary_path, "w", newline="") as sf:
        sw = csv.DictWriter(sf, fieldnames=sfields); sw.writeheader()
        for key, recs in sorted(groups.items()):
            algo, direction, obs, reward = key
            row = dict(algo=algo, direction=direction, obs=obs, reward=reward, n_seeds=len(recs))
            for m in METRICS:
                vals = [r[f"best_{m}"] for r in recs]
                row[f"{m}_mean"] = float(np.mean(vals)); row[f"{m}_ci95"] = ci95(vals)
            stts = [r["steps_to_threshold"] for r in recs if r.get("steps_to_threshold") is not None]
            row["steps_to_threshold_mean"] = float(np.mean(stts)) if stts else float("nan")
            sw.writerow(row)
    print(f"# wrote {curves_path}, {summary_path}, {manifest_path}")


if __name__ == "__main__":
    main()
