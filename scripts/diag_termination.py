"""Diagnose WHY episodes end for a trained reverse policy: success / stop / jackknife
/ oob / timeout. Tests the hypothesis that the late-training collapse and the residual
~5% miss are driven by spurious STOP actions, not jackknifes."""
import os, sys, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from dataclasses import replace
from run_ablations_gpu import cell_to_cfg
from tractor_trailer_rl.config import ActionMode
from tractor_trailer_rl.batched.sb3_adapter import SB3BatchedVecEnv

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--obs", default="state")
ap.add_argument("--reward", default="multiplicative")
ap.add_argument("--pool", default="path_pools/shunt_mild_safe.npz")
ap.add_argument("--steps", type=int, default=1500)
args = ap.parse_args()

from stable_baselines3 import TD3
model = TD3.load(args.model, device="cuda")

cfg = cell_to_cfg(f"reverse:{args.obs}:{args.reward}", preset="shunt",
                  fixed_speed=True, mild_paths=True, pool_file=args.pool)
cfg = replace(cfg, action=replace(cfg.action, mode=ActionMode.STOP_SIGNAL))
env = SB3BatchedVecEnv(cfg, 256, path_pool_size=512)
env.env._diag_causes = True   # turn on termination-cause breakdown

# priority order for attributing a terminated episode to a single cause
CAUSES = ["cause_success", "cause_stop", "cause_jackknife", "cause_oob", "cause_timeout"]
tally = {c: 0 for c in CAUSES}
tally["other"] = 0
stop_fires = 0          # how many env-steps the deterministic policy commands a STOP
total_steps = 0

o = env.reset()
for _ in range(args.steps):
    a, _ = model.predict(o, deterministic=True)
    stop_fires += int((np.asarray(a)[:, 1] > cfg.action.stop_threshold).sum())
    total_steps += env.num_envs
    o, r, dones, infos = env.step(a)
    for i in np.nonzero(dones)[0]:
        for c in CAUSES:
            if infos[i].get(c, False):
                tally[c] += 1; break
        else:
            tally["other"] += 1

n = sum(tally.values())
print(f"model: {os.path.basename(args.model)}  ({n} episodes)")
print(f"  stop-action command rate: {stop_fires/total_steps:.4f} of all steps")
for c in CAUSES + ["other"]:
    print(f"  {c:16s}: {tally[c]:5d}  ({tally[c]/max(n,1)*100:5.1f}%)")
miss = n - tally["cause_success"]
if miss:
    print(f"  -> of the {miss} NON-completions: "
          f"stop={tally['cause_stop']/miss*100:.0f}%  jackknife={tally['cause_jackknife']/miss*100:.0f}%  "
          f"oob={tally['cause_oob']/miss*100:.0f}%  timeout={tally['cause_timeout']/miss*100:.0f}%")
