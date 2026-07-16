"""Phase-2B geometry-encoder RL, GPU/cupy-native (NEW file; the stable replacement for
geom_rl.py). The frozen encoder runs IN the env via cupy (no torch in the env loop), so
this is the same cupy-env + torch-policy pattern that 2A/bc_rescue ran stably.

Freeze the geometry-pretrained encoder; observation = [true delta,gamma + predicted
exteroceptive geometry]; train TD3 (MlpPolicy) FROM SCRATCH. Completion & CTE measured
from the TRUE state (stashed by the env), never the prediction.

    TTRL_BACKEND=cupy python3 -u scripts/geom_obs_rl.py --encoder geom_encoder_fwd_coord.pt \
        --coord --steps 500000 --eval_every 50000 --seed 0
"""

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np  # noqa: E402
from dataclasses import replace  # noqa: E402
from tractor_trailer_rl.batched import backend  # noqa: E402
from tractor_trailer_rl.batched.backend import to_numpy  # noqa: E402
from tractor_trailer_rl.config import ActionMode  # noqa: E402
from run_ablations_gpu import cell_to_cfg  # noqa: E402
from train_sb3 import _td3_action_noise  # noqa: E402
from tractor_trailer_rl.batched.sb3_adapter import SB3BatchedVecEnv  # noqa: E402
from tractor_trailer_rl.batched.bev_sdf_env import SdfObsCfg  # noqa: E402
from tractor_trailer_rl.batched.geom_obs_env import GeomObsEnv  # noqa: E402

POOL = "path_pools/shunt_mild_safe.npz"
JACK = 1.4


class GeomObsVecEnv(SB3BatchedVecEnv):
    """SB3 VecEnv over GeomObsEnv (mirrors sb3_stability.SdfVecEnv)."""
    def __init__(self, cfg, num_envs, encoder_ckpt, sdf_cfg=SdfObsCfg(True), proprio=(0, 1), path_pool_size=4096):
        from stable_baselines3.common.vec_env.base_vec_env import VecEnv
        self.env = GeomObsEnv(cfg, num_envs, encoder_ckpt, sdf_cfg=sdf_cfg,
                              proprio=proprio, path_pool_size=path_pool_size)
        VecEnv.__init__(self, num_envs, self.env.single_observation_space, self.env.single_action_space)
        self._actions = None
        self._seed = None


def make_cfgs(direction):
    cfg = cell_to_cfg(f"{direction}:bev:multiplicative", preset="shunt", fixed_speed=True, mild_paths=True)
    cfg = replace(cfg, action=replace(cfg.action, mode=ActionMode.STOP_SIGNAL))
    eval_cfg = replace(cfg, path=replace(cfg.path, pool_file=POOL)) if os.path.exists(POOL) else cfg
    return cfg, eval_cfg


def geom_eval(model, ev, steps=800):
    """Completion from info['is_success']; CTE/hitch from the env's stashed TRUE state."""
    o = ev.reset(); N = ev.num_envs
    ep_cte = np.zeros(N); ep_len = np.zeros(N); ep_maxh = np.zeros(N)
    comps, ctes, hitches, jacks = [], [], [], []
    for _ in range(steps):
        a, _ = model.predict(o, deterministic=True)
        o, r, dones, infos = ev.step(a)
        tv = to_numpy(ev.env._last_true_vec)           # (N, 8) TRUE state
        ep_cte += np.abs(tv[:, 4]); ep_maxh = np.maximum(ep_maxh, np.abs(tv[:, 1])); ep_len += 1
        for i in range(N):
            if dones[i]:
                comps.append(float(infos[i].get("is_success", 0.0)))
                ctes.append(ep_cte[i] / max(ep_len[i], 1)); hitches.append(ep_maxh[i])
                jacks.append(1.0 if ep_maxh[i] > JACK else 0.0)
                ep_cte[i] = ep_len[i] = ep_maxh[i] = 0.0
    m = lambda xs: float(np.mean(xs)) if xs else float("nan")
    return dict(completion=m(comps), trailer_cte=m(ctes), max_hitch=m(hitches), jackknife=m(jacks), n=len(comps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--direction", choices=["forward", "reverse"], default="forward")
    ap.add_argument("--coord", action="store_true")
    ap.add_argument("--n_envs", type=int, default=32)
    ap.add_argument("--eval_envs", type=int, default=64, help="eval env count (small => smaller eval-time memory burst)")
    ap.add_argument("--steps", type=int, default=500_000)
    ap.add_argument("--eval_every", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch as th
    from stable_baselines3 import TD3
    from stable_baselines3.common.callbacks import BaseCallback

    ckpt = th.load(args.encoder, weights_only=False)
    sdf_cfg = SdfObsCfg(args.coord)
    cfg, eval_cfg = make_cfgs(args.direction)
    print(f"# geom-obs-RL (cupy encoder) backend={backend.get_backend()} enc={args.encoder} "
          f"dir={args.direction} steps={args.steps} seed={args.seed}", flush=True)

    train = GeomObsVecEnv(cfg, args.n_envs, ckpt, sdf_cfg=sdf_cfg, path_pool_size=1024)
    # smaller eval env (64, was 256) to shrink the eval-time encoder memory burst that
    # correlated with the segfault (see PHASE2_GEOMETRY_ENCODER.md §6)
    ev = GeomObsVecEnv(eval_cfg, args.eval_envs, ckpt, sdf_cfg=sdf_cfg, path_pool_size=256)
    ev.env._eval_no_stop = True

    model = TD3("MlpPolicy", train, learning_rate=1e-3, learning_starts=10_000,
                action_noise=_td3_action_noise(train), seed=args.seed, device="cuda", verbose=0)

    curve = []
    class _CB(BaseCallback):
        def __init__(self): super().__init__(); self._last = 0
        def _on_step(self):
            if self.num_timesteps - self._last >= args.eval_every:
                self._last = self.num_timesteps
                m = geom_eval(self.model, ev, 500); m["t"] = int(self.num_timesteps); curve.append(m)
                print(f"  t={m['t']} compl={m['completion']:.3f} cte={m['trailer_cte']:.3f} hitch={m['max_hitch']:.3f}", flush=True)
            return True

    t0 = time.perf_counter()
    model.learn(total_timesteps=args.steps, progress_bar=False, callback=_CB())
    fin = geom_eval(model, ev, 800)
    _key = lambda c: (c["completion"], -(c["trailer_cte"] if c["trailer_cte"] == c["trailer_cte"] else 1e9))
    best = max(curve, key=_key) if curve else fin
    print(f"# GEOM-OBS-RL DONE ({time.perf_counter()-t0:.0f}s): BEST compl={best['completion']:.3f} "
          f"cte={best['trailer_cte']:.3f} @{best.get('t','NA')} | final compl={fin['completion']:.3f} cte={fin['trailer_cte']:.3f}", flush=True)
    print("# curve (t, compl, cte):", [(c["t"], round(c["completion"], 3), round(c["trailer_cte"], 3)) for c in curve], flush=True)


if __name__ == "__main__":
    main()
