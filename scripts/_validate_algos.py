import sys, time, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"src"); sys.path.insert(0,"scripts")
from dataclasses import replace
from tractor_trailer_rl.batched import backend
from tractor_trailer_rl.config import ActionMode
from run_ablations_gpu import cell_to_cfg
from tractor_trailer_rl.batched.sb3_adapter import SB3BatchedVecEnv
from train_sb3 import build_model, evaluate, EvalCurveCallback

# (algo, sb3_algo, discrete, direction, n_envs, steps)
JOBS = [
    ("td3","td3",False,"forward",32,300_000),
    ("dqn","dqn",True,"forward",32,300_000),
    ("ppo","ppo",False,"forward",256,1_000_000),
    ("ppo_disc","ppo",True,"forward",256,1_000_000),
    ("td3","td3",False,"reverse",32,300_000),   # the hard case
]
for algo,sb3_algo,discrete,direction,n_envs,steps in JOBS:
    cfg = cell_to_cfg(f"{direction}:lidar24:multiplicative", preset="shunt", fixed_speed=True, mild_paths=True)
    mode = ActionMode.DISCRETE if discrete else ActionMode.STOP_SIGNAL
    cfg = replace(cfg, action=replace(cfg.action, mode=mode))
    env = SB3BatchedVecEnv(cfg, n_envs, path_pool_size=1024)
    ev  = SB3BatchedVecEnv(cfg, 256, path_pool_size=512)
    m = build_model(sb3_algo, env, 1e-3, 4096, 0)
    cb = EvalCurveCallback(ev, max(steps//6,1))
    t0=time.perf_counter(); m.learn(total_timesteps=steps, progress_bar=False, callback=cb)
    wall=time.perf_counter()-t0
    b = cb.best or evaluate(m, ev, 400)
    curve=" ".join(f"{c['timesteps']//1000}k:{c['completion']:.2f}" for c in cb.curve)
    print(f"RESULT {algo:9} {direction:8} n_envs={n_envs:3} steps={steps} "
          f"BEST_compl={b['completion']:.2f} cte={b['trailer_cte']:.3f} wall={wall:.0f}s | {curve}", flush=True)
print("VALIDATION DONE", flush=True)
