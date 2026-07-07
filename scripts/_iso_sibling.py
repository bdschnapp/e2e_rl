import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"src"); sys.path.insert(0,"scripts")
import numpy as np
from dataclasses import replace
from tractor_trailer_rl.batched import backend
from tractor_trailer_rl.config import ActionMode
from run_ablations_gpu import cell_to_cfg
from tractor_trailer_rl.batched.sb3_adapter import SB3BatchedVecEnv
from train_sb3 import evaluate, EvalCurveCallback
from tractor_trailer_rl.actions import modes as action_modes
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
cfg = cell_to_cfg("forward:lidar24:multiplicative", preset="shunt", fixed_speed=True, mild_paths=True)
cfg = replace(cfg, action=replace(cfg.action, mode=ActionMode.STOP_SIGNAL))
N = 16                                   # gentle, sibling-like data regime
env = SB3BatchedVecEnv(cfg, N, path_pool_size=1024)
ev  = SB3BatchedVecEnv(cfg, 256, path_pool_size=512)
sigma = action_modes.noise_sigma(cfg, 2)   # [steer=0.05, stop=0.05]
noise = NormalActionNoise(mean=np.zeros(2), sigma=sigma)
# SIBLING hyperparams: SB3 TD3 defaults + lr 1e-3 + action noise (batch 256, buffer 1M,
# train_freq episode, gradient_steps -1, net [400,300])
m = TD3("MlpPolicy", env, action_noise=noise, learning_rate=1e-3, learning_starts=10_000,
        device="cuda", verbose=0)
cb = EvalCurveCallback(ev, 40_000)
m.learn(total_timesteps=300_000, progress_bar=False, callback=cb)
b = cb.best or evaluate(m, ev, 400)
print(f"ISO[sibling-hparams,2D-STOP] BEST completion={b['completion']:.2f} cte={b['trailer_cte']:.3f} maxhitch={b['max_hitch']:.3f}")
for c in cb.curve: print(f"  t={c['timesteps']} compl={c['completion']:.2f}")
