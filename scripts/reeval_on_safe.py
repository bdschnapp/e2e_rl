"""Re-evaluate an already-trained lidar24-reverse model on the safe (feasible) pool
vs a random pool, at the SAME lidar step it was trained with. Answers: is the sub-100%
completion a 'true' miss-rate on feasible paths, or an artifact of infeasible eval paths?
No retraining — loads the saved best checkpoint."""
import os, sys, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from dataclasses import replace
from run_ablations_gpu import cell_to_cfg
from tractor_trailer_rl.config import ActionMode
from tractor_trailer_rl.batched.sb3_adapter import SB3BatchedVecEnv
from train_sb3 import evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--lidar_step", type=float, default=0.15)
ap.add_argument("--safe_pool", default="path_pools/shunt_mild_safe.npz")
ap.add_argument("--steps", type=int, default=1200)
args = ap.parse_args()

from stable_baselines3 import TD3
model = TD3.load(args.model, device="cuda")

def make_env(pool_file):
    cfg = cell_to_cfg("reverse:lidar24:multiplicative", preset="shunt",
                      fixed_speed=True, mild_paths=True, lidar_step=args.lidar_step,
                      pool_file=pool_file)
    cfg = replace(cfg, action=replace(cfg.action, mode=ActionMode.STOP_SIGNAL))
    return SB3BatchedVecEnv(cfg, 256, path_pool_size=512)

for label, pf in [("random pool (v4-style eval)", None),
                  ("SAFE pool (feasible eval)", args.safe_pool)]:
    env = make_env(pf)
    m = evaluate(model, env, steps=args.steps)
    print(f"  {label:32s}: completion={m['completion']:.3f}  cte={m['trailer_cte']:.3f}  "
          f"hitch={m['max_hitch']:.3f}  n={m['n_episodes']}")
