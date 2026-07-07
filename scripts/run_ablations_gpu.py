"""Multi-seed ablation runner on the batched GPU env (C7 infrastructure).

Trains each (direction x observation x reward) cell for several seeds using the
on-GPU TD3, and writes a CSV of per-seed final evaluation metrics plus mean +/-
95% CI across seeds. This is what turns the thesis's single-seed ablation tables
into seeded results with confidence intervals (Ch.5 "Statistical Rigor").

This is the *harness*; the exact cell list, step budgets, and TD3 hyperparameters
should be dialed in on the target GPU. Kept small and explicit on purpose.

    TTRL_BACKEND=cupy python scripts/run_ablations_gpu.py \
        --cells forward:lidar24:multiplicative reverse:lidar24:multiplicative \
        --seeds 0 1 2 --steps 300000 --n_envs 1024 --out results_gpu.csv
"""

import os
import sys
import csv
import math
import time
import argparse
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tractor_trailer_rl.batched import backend  # noqa: E402
from tractor_trailer_rl.batched.backend import to_numpy  # noqa: E402
from tractor_trailer_rl.config import (lab_config, truck_config, shunt_truck_config,  # noqa: E402
                                       Direction, VehicleKind, ObsConfig)
import numpy as np  # noqa: E402


def cell_to_cfg(cell, preset="lab", fixed_speed=False, mild_paths=False, pool_file=None,
                speed_range=None, lidar_step=None, guide_transition=None,
                stop_penalty=None, stop_threshold=None):
    direction_s, obs_s, reward_s = cell.split(":")
    direction = Direction.REVERSE if direction_s == "reverse" else Direction.FORWARD
    preset_fn = {"truck": truck_config, "lab": lab_config,
                 "shunt": shunt_truck_config}[preset]
    base = preset_fn(direction=direction, vehicle_kind=VehicleKind.TRAILER)
    # BEV perception cell: bevN => 32x32 image obs (bevN sets S=N; bare "bev" => 32),
    # lidar off. Otherwise state (beams=0) or lidarN.
    bev_size = 0
    if obs_s == "state":
        beams = 0
    elif obs_s.startswith("bev"):
        beams = 0
        bev_size = int(obs_s[3:]) if obs_s[3:].isdigit() else 32
    elif obs_s.startswith("lidar") and obs_s[5:].isdigit():
        beams = int(obs_s[5:])   # lidar4 / lidar8 / lidar16 / lidar24 / lidar32
    else:
        beams = 24
    cfg = replace(base, obs=replace(base.obs, lidar_beams=beams, bev_size=bev_size),
                  reward=replace(base.reward, mode=reward_s))
    if guide_transition is not None:
        # guided reward alpha-decay horizon (env-steps); <=0 => no decay (pure PP imitation)
        cfg = replace(cfg, reward=replace(cfg.reward, guide_transition_steps=int(guide_transition)))
    if stop_penalty is not None or stop_threshold is not None:
        # lane-following stop-gate stability: dominate the stop (penalty >= crash) so the
        # agent never stops on a feasible path, and/or raise the trigger threshold.
        ov = {}
        if stop_penalty is not None: ov["stop_penalty"] = float(stop_penalty)
        if stop_threshold is not None: ov["stop_threshold"] = float(stop_threshold)
        cfg = replace(cfg, action=replace(cfg.action, **ov))
    if lidar_step is not None:
        # override lidar march step (e.g. 0.15 to reproduce v4's pre-fix effective step)
        cfg = replace(cfg, obs=replace(cfg.obs, lidar_step_m=float(lidar_step)))
    if mild_paths:
        # Stage-1 task: straight + gentle only (solvable from scratch, clean
        # attribution). Drops sharp/winding/lab_* kinds.
        cfg = replace(cfg, path=replace(cfg.path, mild_only=True))
    if speed_range is not None:
        # Variable speed as REGULARISATION (agilex/tractor_trailer_rl finding: a fixed
        # speed trains brittle policies). explicit_min/max override all path kinds.
        smin, smax = speed_range
        cfg = replace(cfg, speed_random=replace(cfg.speed_random, enabled=True,
                                                explicit_min=float(smin), explicit_max=float(smax)))
    elif fixed_speed:
        # No-obstacle ablation protocol: hold speed fixed (as the SB3 tables do) so
        # completion isn't capped by low-speed episodes timing out before the goal.
        cfg = replace(cfg, speed_random=replace(cfg.speed_random, enabled=False))
    if pool_file:
        # Load the PP-validated "safe spawn" pool (scripts/build_safe_pool.py) so no
        # run can fail from an infeasible path geometry. Speeds come from the file.
        cfg = replace(cfg, path=replace(cfg.path, pool_file=pool_file))
    return cfg


def evaluate(agent, env, episodes_steps=1500):
    """Deterministic rollout; report mean episode return + completion proxy."""
    import torch
    obs = backend.xp  # noqa
    o = env.reset(seed=12345)
    from tractor_trailer_rl.batched.td3 import _to_torch, _to_backend
    dev = agent.device
    ot = _to_torch(o, dev).float()
    ep_ret = None
    rets, comps = [], []
    N = env.num_envs
    import numpy as _np
    ep_ret = _np.zeros(N)
    for _ in range(episodes_steps):
        a = agent._act(ot, explore=False)
        o, r, term, trunc, info = env.step(_to_backend(a))
        ot = _to_torch(o, dev).float()
        r_np = to_numpy(r); done = to_numpy(term) | to_numpy(trunc)
        ep_ret += r_np
        if done.any():
            rets.extend(ep_ret[done].tolist())
            # completion = reached the goal, read from info BEFORE auto-reset
            succ = info["success"]
            comps.extend(succ[done].tolist())
            ep_ret[done] = 0.0
    return (float(np.mean(rets)) if rets else float("nan"),
            float(np.mean(comps)) if comps else float("nan"))


def ci95(xs):
    xs = [x for x in xs if x == x]
    if len(xs) < 2:
        return float("nan")
    return 1.96 * (np.std(xs, ddof=1) / math.sqrt(len(xs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", required=True,
                    help="direction:obs:reward, e.g. forward:lidar24:multiplicative")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--n_envs", type=int, default=32,
                    help="moderate N for off-policy TD3 (huge N tanks sample-efficiency; "
                         "use PPO for N>>100)")
    ap.add_argument("--preset", choices=["lab", "truck", "shunt"], default="shunt")
    ap.add_argument("--updates_per_step", type=int, default=32,
                    help="gradient updates per env tick; keep ~= n_envs for UTD~1")
    ap.add_argument("--batch_size", type=int, default=256,
                    help="gradient minibatch (raise to 2048-8192 to use the GPU on big-N runs)")
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="actor+critic learning rate (raise with batch_size)")
    ap.add_argument("--fixed_speed", action="store_true",
                    help="hold speed fixed (no-obstacle ablation protocol, matches SB3 tables)")
    ap.add_argument("--mild_paths", action="store_true",
                    help="Stage-1 task: straight+gentle paths only (solvable from scratch)")
    ap.add_argument("--out", default="results_gpu.csv")
    args = ap.parse_args()

    from tractor_trailer_rl.batched.env import BatchedLaneFollowingEnv
    from tractor_trailer_rl.batched.td3 import TD3

    rows = []
    for cell in args.cells:
        cfg = cell_to_cfg(cell, preset=args.preset, fixed_speed=args.fixed_speed,
                          mild_paths=args.mild_paths)
        per_seed = []
        for seed in args.seeds:
            t0 = time.perf_counter()
            env = BatchedLaneFollowingEnv(cfg, args.n_envs)
            agent = TD3(env, seed=seed, batch_size=args.batch_size,
                        actor_lr=args.lr, critic_lr=args.lr)
            agent.train(args.steps, start_steps=min(20_000, args.steps // 10),
                        updates_per_step=args.updates_per_step, log_every=50)
            ret, comp = evaluate(agent, env)
            dt = time.perf_counter() - t0
            print(f"[{cell} seed={seed}] eval_return={ret:.2f} completion={comp:.2f} ({dt:.0f}s)")
            per_seed.append((seed, ret, comp))
            rows.append(dict(cell=cell, seed=seed, eval_return=ret, completion=comp, secs=round(dt)))
        rets = [r for _, r, _ in per_seed]; comps = [c for _, _, c in per_seed]
        print(f"== {cell}: return {np.nanmean(rets):.2f} +/- {ci95(rets):.2f}  "
              f"completion {np.nanmean(comps):.2f} +/- {ci95(comps):.2f} (n={len(rets)})")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cell", "seed", "eval_return", "completion", "secs"])
        w.writeheader(); w.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
