"""Train TD3 on the batched GPU env (the CuPy-port trainer).

Example (forward lab lane-following, state+lidar obs):
    TTRL_BACKEND=cupy python scripts/train_batched_td3.py --steps 300000 --n_envs 1024

This is the on-GPU replacement for the SB3 SubprocVecEnv training path; the same
script parameterised over seeds is what the multi-seed ablation reruns (C7) call.
"""

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tractor_trailer_rl.batched import backend  # noqa: E402
from tractor_trailer_rl.config import lab_config, truck_config, Direction, VehicleKind  # noqa: E402


def build_cfg(args):
    preset = lab_config if args.preset == "lab" else truck_config
    direction = Direction.REVERSE if args.direction == "reverse" else Direction.FORWARD
    kind = VehicleKind.TRACTOR_ONLY if args.tractor_only else VehicleKind.TRAILER
    return preset(direction=direction, vehicle_kind=kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=["lab", "truck"], default="lab")
    ap.add_argument("--direction", choices=["forward", "reverse"], default="forward")
    ap.add_argument("--tractor_only", action="store_true")
    ap.add_argument("--n_envs", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--start_steps", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from tractor_trailer_rl.batched.env import BatchedLaneFollowingEnv
    from tractor_trailer_rl.batched.td3 import TD3

    cfg = build_cfg(args)
    print(f"# backend={backend.get_backend()} preset={args.preset} dir={args.direction} "
          f"trailer={not args.tractor_only} n_envs={args.n_envs} obs_dim="
          f"{cfg.obs.lidar_beams}+state seed={args.seed}")
    env = BatchedLaneFollowingEnv(cfg, args.n_envs)
    agent = TD3(env, seed=args.seed)
    t0 = time.perf_counter()
    returns = agent.train(args.steps, start_steps=args.start_steps)
    dt = time.perf_counter() - t0
    print(f"# done: {args.steps:,} env steps in {dt:.1f}s "
          f"({args.steps / dt:,.0f} steps/s); episodes={len(returns)}")


if __name__ == "__main__":
    main()
