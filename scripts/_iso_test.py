import sys, os, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"src"); sys.path.insert(0,"scripts")
import numpy as np
from dataclasses import replace
from tractor_trailer_rl.batched import backend
from tractor_trailer_rl.config import ActionMode
from run_ablations_gpu import cell_to_cfg
from tractor_trailer_rl.batched.sb3_adapter import SB3BatchedVecEnv
from train_sb3 import build_model, evaluate, EvalCurveCallback
mode = sys.argv[1]  # fixed | stop
cfg = cell_to_cfg("forward:lidar24:multiplicative", preset="shunt", fixed_speed=True, mild_paths=True)
cfg = replace(cfg, action=replace(cfg.action, mode=ActionMode.FIXED_SPEED if mode=="fixed" else ActionMode.STOP_SIGNAL))
env = SB3BatchedVecEnv(cfg, 256, path_pool_size=1024)
ev = SB3BatchedVecEnv(cfg, 256, path_pool_size=512)
m = build_model("td3", env, 2e-3, 4096, 0)
cb = EvalCurveCallback(ev, 100_000)
m.learn(total_timesteps=400_000, progress_bar=False, callback=cb)
b = cb.best or evaluate(m, ev, 400)
print(f"ISO[{mode}] action_dim={env.action_space.shape} BEST completion={b['completion']:.2f} cte={b['trailer_cte']:.3f}")
