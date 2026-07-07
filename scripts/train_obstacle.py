"""Curriculum trainer for the obstacle stop-gate env (Track D, D3).

Stage A: lane-following in the STOP_SIGNAL action space (obstacles OFF) -> the base
         tracking skill with the correct 2-D action head. Warm-startable from the
         corrected Stage-1 ablation protocol (same obs/action).
Stage B: warm-start from Stage A, obstacles ON -> learn to steer around obstacles
         (via the hidden local-planner reward) and to stop on impassable layouts.

Flags:
  --scratch        skip Stage A (from-scratch baseline; answers "did the base skill
                   transfer help?")
  --warm_start P   reuse an existing Stage-A .zip instead of training one
  --algo {td3,ppo,sac,dqn}

Discrete algos (dqn) use the flat Discrete(n*2)=steer x {go,stop} in BOTH stages;
continuous algos use STOP_SIGNAL [steer, stop]. Obs/action spaces match across
stages so the warm start is a plain SB3 load (no weight surgery).

    TTRL_BACKEND=cupy python scripts/train_obstacle.py --algo td3 --direction forward
"""
import os, sys, argparse, warnings, time
from dataclasses import replace
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from tractor_trailer_rl.batched import backend  # noqa: E402
from tractor_trailer_rl.config import (ActionMode, Direction,  # noqa: E402
                                       shunt_truck_obstacle_config)
from run_ablations_gpu import cell_to_cfg  # noqa: E402
from train_sb3 import build_model, evaluate  # noqa: E402

DISCRETE_ALGOS = {"dqn"}


def _cls(algo):
    from stable_baselines3 import TD3, PPO, SAC, DQN
    return {"td3": TD3, "ppo": PPO, "sac": SAC, "dqn": DQN}[algo]


def _set_action(cfg, discrete):
    mode = ActionMode.DISCRETE if discrete else ActionMode.STOP_SIGNAL
    return replace(cfg, action=replace(cfg.action, mode=mode))


def stageA_cfg(direction, reward, discrete):
    # lane-following, obstacles off, but in the stop-capable action space.
    # Match the proven Stage-1 sweep protocol: dominate the stop on feasible paths
    # (stop_penalty >= crash 500) and disable guided-alpha decay, so the lane-follower
    # never learns to stop on a clear road. (The old 2e-3 / no-stop-penalty recipe
    # diverged "with the stop" -> Stage A scored 0% completion.)
    cfg = cell_to_cfg(f"{direction}:lidar24:{reward}", preset="shunt",
                      fixed_speed=True, mild_paths=True,
                      guide_transition=-1, stop_penalty=500.0)
    return _set_action(cfg, discrete)


def stageB_cfg(direction, discrete):
    cfg = shunt_truck_obstacle_config(
        direction=Direction.REVERSE if direction == "reverse" else Direction.FORWARD)
    return _set_action(cfg, discrete)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="td3", choices=["td3", "ppo", "sac", "dqn"])
    ap.add_argument("--direction", default="forward", choices=["forward", "reverse"])
    ap.add_argument("--reward", default="multiplicative", help="Stage-A lane-following reward")
    ap.add_argument("--scratch", action="store_true", help="skip Stage A (from-scratch baseline)")
    ap.add_argument("--warm_start", default=None, help="existing Stage-A .zip to reuse")
    ap.add_argument("--stageA_steps", type=int, default=300000)
    ap.add_argument("--stageB_steps", type=int, default=None, help="default 2M ppo / 700k else")
    ap.add_argument("--n_envs", type=int, default=32,
                    help="off-policy env count (sweep 'gentle/stable' regime); 256 was undertrained")
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results_obstacle")
    ap.add_argument("--eval_pool_file", default="path_pools/shunt_mild_safe.npz",
                    help="PP-validated feasible pool for HONEST Stage-A completion eval")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    from tractor_trailer_rl.batched.sb3_adapter import SB3BatchedVecEnv
    discrete = args.algo in DISCRETE_ALGOS
    lr = 1e-3  # proven-stable for all algos (2e-3 off-policy diverged with the stop)
    stageB_steps = args.stageB_steps or (2_000_000 if args.algo == "ppo" else 700_000)
    tag = f"{args.algo}_{args.direction}_s{args.seed}" + ("_scratch" if args.scratch else "")

    envB = SB3BatchedVecEnv(stageB_cfg(args.direction, discrete), args.n_envs, path_pool_size=1024)

    if args.scratch:
        print(f"[{tag}] from scratch on obstacles ({stageB_steps} steps)")
        model = build_model(args.algo, envB, lr, args.batch_size, args.seed)
    else:
        pathA = args.warm_start
        if pathA is None:
            print(f"[{tag}] Stage A: lane-following STOP_SIGNAL ({args.stageA_steps} steps)")
            cfgA = stageA_cfg(args.direction, args.reward, discrete)
            envA = SB3BatchedVecEnv(cfgA, args.n_envs, path_pool_size=1024)
            mA = build_model(args.algo, envA, lr, args.batch_size, args.seed)
            t0 = time.perf_counter(); mA.learn(total_timesteps=args.stageA_steps, progress_bar=False)
            pathA = os.path.join(args.out, f"stageA_{args.algo}_{args.direction}_s{args.seed}.zip")
            mA.save(pathA)
            # HONEST Stage-A eval (matches the sweep): PP-validated feasible pool + stop
            # disabled, so completion measures tracking, not stop-misfires.
            evalA_cfg = replace(cfgA, path=replace(cfgA.path, pool_file=args.eval_pool_file))
            evalA_env = SB3BatchedVecEnv(evalA_cfg, min(args.n_envs, 256), path_pool_size=512)
            evalA_env.env._eval_no_stop = True
            evA = evaluate(mA, evalA_env, steps=800)
            print(f"[{tag}] Stage A done ({time.perf_counter()-t0:.0f}s) "
                  f"compl={evA['completion']:.2f} cte={evA['trailer_cte']:.3f} -> {pathA}")
        print(f"[{tag}] Stage B: warm-start {os.path.basename(pathA)}, obstacles ON ({stageB_steps} steps)")
        model = _cls(args.algo).load(pathA, env=envB, device="auto")

    t0 = time.perf_counter()
    model.learn(total_timesteps=stageB_steps, progress_bar=False)
    pathB = os.path.join(args.out, f"stageB_{tag}.zip")
    model.save(pathB)
    ev = evaluate(model, envB, steps=1000)
    print(f"[{tag}] Stage B done ({time.perf_counter()-t0:.0f}s) compl={ev['completion']:.2f} "
          f"cte={ev['trailer_cte']:.3f} -> {pathB}")
    print(f"[{tag}] eval table:  python scripts/eval_obstacle.py --model {pathB} --direction {args.direction}")


if __name__ == "__main__":
    main()
