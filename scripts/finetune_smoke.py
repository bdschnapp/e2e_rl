import os
os.environ.setdefault("SDL_VIDEODRIVER","dummy")
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ.setdefault(v,"1")
from dataclasses import replace
import numpy as np
from tractor_trailer_rl import lab_config, Direction, VehicleKind, ActionMode
from tractor_trailer_rl.train.runner import finetune

def main():
    cfg = lab_config(Direction.FORWARD, VehicleKind.TRAILER)
    cfg = replace(cfg,
        action=replace(cfg.action, mode=ActionMode.STOP_SIGNAL, stop_threshold=0.8,
                       stop_noise_sigma=0.05, stop_penalty=500.0),
        reward=replace(cfg.reward, mode="multiplicative"),
        speed_random=replace(cfg.speed_random, explicit_min=0.5, explicit_max=0.8),
        path=replace(cfg.path, mild_only=False))
    init="/home/ben/Ben/Thesis/Electrans_project/lab_models_ttrl_baseline/forward_trailer/best_model.zip"
    out="/tmp/claude-1000/-home-ben-Ben-Thesis-Electrans-project/226fc11f-b81b-4403-9b53-3bc9252c0984/scratchpad/ttrl_ft_smoke"
    finetune(init, cfg, timesteps=80000, n_envs=8, out_dir=out, eval_freq=10000,
             n_eval_episodes=10, device="auto", seed=0)
    d=np.load(os.path.join(out,"logs","evaluations.npz"))
    for t,r,l in zip(d["timesteps"], d["results"].mean(1), d["ep_lengths"].mean(1)):
        print(f"  t={t:6d} reward={r:8.1f} len={l:6.1f}")
    print("FT SMOKE OK")

if __name__ == "__main__":
    main()
