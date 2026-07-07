import sys, time, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"src"); sys.path.insert(0,"scripts")
from dataclasses import replace
from tractor_trailer_rl.batched import backend
from tractor_trailer_rl.config import ActionMode
from run_ablations_gpu import cell_to_cfg
from tractor_trailer_rl.batched.sb3_adapter import SB3BatchedVecEnv
from train_sb3 import build_model, evaluate
reward = sys.argv[1] if len(sys.argv)>1 else "multiplicative"
n_envs = int(sys.argv[2]) if len(sys.argv)>2 else 32
cfg = cell_to_cfg(f"reverse:lidar24:{reward}", preset="shunt", fixed_speed=True, mild_paths=True)
cfg = replace(cfg, action=replace(cfg.action, mode=ActionMode.STOP_SIGNAL))
env = SB3BatchedVecEnv(cfg, n_envs, path_pool_size=1024)
ev  = SB3BatchedVecEnv(cfg, 256, path_pool_size=512)
m = build_model("td3", env, 1e-3, 4096, 0)
print(f"REV TEST reward={reward} n_envs={n_envs} (live eval every 100k)", flush=True)
t0=time.perf_counter(); best=0.0
for k in range(15):
    m.learn(100_000, progress_bar=False, reset_num_timesteps=(k==0))
    r = evaluate(m, ev, 400)
    best=max(best, r['completion'])
    print(f"  t={(k+1)*100}k compl={r['completion']:.2f} cte={r['trailer_cte']:.3f} maxhitch={r['max_hitch']:.3f} best={best:.2f} ({time.perf_counter()-t0:.0f}s)", flush=True)
print(f"REV DONE reward={reward} best_compl={best:.2f}", flush=True)
