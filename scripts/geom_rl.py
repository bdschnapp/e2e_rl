"""Phase-2B geometry-encoder RL (NEW file, additive; remote-run).

Freeze the geometry-pretrained encoder and use it as a FIXED perception module: the RL
observation is [true proprio (delta, gamma) + encoder-predicted exteroceptive geometry
(e_y, e_psi, e_y_t, e_psi_t, k1, k2)]. Then train TD3 FROM SCRATCH on that low-dim
vector (plain MlpPolicy) — no BC, no critic warm-up, no unfreeze.

Hypothesis: a frozen good encoder gives stable, task-relevant features from step 0, so
RL-from-scratch behaves like the 2A `state` run (=1.0, no collapse) — with the key
difference that the geometry now comes from the image. Completion & CTE are measured
from the TRUE state (stashed in info), never the prediction.

    TTRL_BACKEND=cupy python3 -u scripts/geom_rl.py --encoder geom_encoder_fwd_coord.pt \
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
import gymnasium as gym  # noqa: E402
from tractor_trailer_rl.batched import backend  # noqa: E402
from tractor_trailer_rl.config import ActionMode  # noqa: E402
from run_ablations_gpu import cell_to_cfg  # noqa: E402
from train_sb3 import _td3_action_noise  # noqa: E402
from sb3_stability import SdfVecEnv  # noqa: E402
from tractor_trailer_rl.batched.bev_sdf_env import SdfObsCfg  # noqa: E402
from geom_encoder import GeomEncoder  # noqa: E402

POOL = "path_pools/shunt_mild_safe.npz"
JACK = 1.4


def make_cfgs(direction):
    cfg = cell_to_cfg(f"{direction}:bev:multiplicative", preset="shunt", fixed_speed=True, mild_paths=True)
    cfg = replace(cfg, action=replace(cfg.action, mode=ActionMode.STOP_SIGNAL))
    eval_cfg = replace(cfg, path=replace(cfg.path, pool_file=POOL)) if os.path.exists(POOL) else cfg
    return cfg, eval_cfg


def geom_state_vecenv(venv, ckpt, proprio=(0, 1), enc_device="cpu"):
    """VecEnvWrapper: replace the Dict obs with [true proprio + predicted geometry].
    True state is stashed in info['true_vector'] so metrics use ground truth."""
    import torch as th
    from stable_baselines3.common.vec_env import VecEnvWrapper

    class _GeomWrap(VecEnvWrapper):
        def __init__(self):
            enc = GeomEncoder(ckpt["in_ch"], ckpt["out_dim"], ckpt["img_size"]).to(enc_device)
            enc.load_state_dict(ckpt["state_dict"]); enc.eval()
            self.enc = enc; self.dev = enc_device
            self.mean = th.as_tensor(ckpt["label_mean"], dtype=th.float32, device=enc_device)
            self.std = th.as_tensor(ckpt["label_std"], dtype=th.float32, device=enc_device)
            self.proprio = proprio
            super().__init__(venv, observation_space=venv.observation_space["vector"])

        def _tf(self, obs):
            with th.no_grad():
                img = th.as_tensor(np.asarray(obs["image"], dtype=np.float32), device=self.dev)
                pred = (self.enc(img) * self.std + self.mean).cpu().numpy()
            true = np.asarray(obs["vector"], dtype=np.float32)
            out = pred.astype(np.float32)
            for d in self.proprio:
                out[:, d] = true[:, d]          # override proprio dims with the true (measurable) values
            return out, true

        def reset(self):
            o, _ = self._tf(self.venv.reset()); return o

        def step_wait(self):
            obs, r, d, infos = self.venv.step_wait()
            o, true = self._tf(obs)
            for i in range(len(infos)):
                infos[i]["true_vector"] = true[i]
                tobs = infos[i].get("terminal_observation")
                if isinstance(tobs, dict):   # transform the Dict terminal obs -> geometry vector for the buffer
                    tv, _ = self._tf({"image": np.asarray(tobs["image"], np.float32)[None],
                                      "vector": np.asarray(tobs["vector"], np.float32)[None]})
                    infos[i]["terminal_observation"] = tv[0]
            return o, r, d, infos

    return _GeomWrap()


def geom_eval(model, env, steps=800):
    """Deterministic rollout; completion from info['is_success'], CTE/hitch from the TRUE
    state in info['true_vector'] (index 4 = trailer CTE, 1 = hitch)."""
    o = env.reset(); N = env.num_envs
    ep_cte = np.zeros(N); ep_len = np.zeros(N); ep_maxh = np.zeros(N)
    comps, ctes, hitches, jacks = [], [], [], []
    for _ in range(steps):
        a, _ = model.predict(o, deterministic=True)
        o, r, dones, infos = env.step(a)
        for i in range(N):
            tv = infos[i].get("true_vector")
            if tv is not None:
                ep_cte[i] += abs(float(tv[4])); ep_maxh[i] = max(ep_maxh[i], abs(float(tv[1]))); ep_len[i] += 1
            if dones[i]:
                comps.append(float(infos[i].get("is_success", 0.0)))
                ctes.append(ep_cte[i] / max(ep_len[i], 1)); hitches.append(ep_maxh[i])
                jacks.append(1.0 if ep_maxh[i] > JACK else 0.0)
                ep_cte[i] = ep_len[i] = ep_maxh[i] = 0.0
    m = lambda xs: float(np.mean(xs)) if xs else float("nan")
    return dict(completion=m(comps), trailer_cte=m(ctes), max_hitch=m(hitches),
                jackknife=m(jacks), n=len(comps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--direction", choices=["forward", "reverse"], default="forward")
    ap.add_argument("--coord", action="store_true")
    ap.add_argument("--n_envs", type=int, default=32)
    ap.add_argument("--steps", type=int, default=500_000)
    ap.add_argument("--eval_every", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--enc_device", default="cpu", help="device for the frozen encoder's per-step inference "
                    "(cpu avoids torch-GPU/cupy-GPU interleaving in the obs loop — fixes the segfault)")
    args = ap.parse_args()

    import torch as th
    th.set_num_threads(1)   # avoid CPU thread over-subscription vs cupy (native-crash guard on these boxes)
    from stable_baselines3 import TD3
    from stable_baselines3.common.callbacks import BaseCallback

    ckpt = th.load(args.encoder, weights_only=False)
    sdf_cfg = SdfObsCfg(args.coord)
    cfg, eval_cfg = make_cfgs(args.direction)
    print(f"# geom-RL backend={backend.get_backend()} enc={args.encoder} dir={args.direction} "
          f"steps={args.steps} seed={args.seed}", flush=True)

    train = geom_state_vecenv(SdfVecEnv(cfg, args.n_envs, sdf_cfg=sdf_cfg, path_pool_size=1024), ckpt, enc_device=args.enc_device)
    ev_inner = SdfVecEnv(eval_cfg, 256, sdf_cfg=sdf_cfg, path_pool_size=512)
    ev_inner.env._eval_no_stop = True
    ev = geom_state_vecenv(ev_inner, ckpt, enc_device=args.enc_device)

    model = TD3("MlpPolicy", train, learning_rate=1e-3, learning_starts=10_000,
                action_noise=_td3_action_noise(train), seed=args.seed, device="cuda", verbose=0)

    curve = [];
    class _CB(BaseCallback):
        def __init__(self): super().__init__(); self._last = 0; self.best = None; self._bk = None
        def _on_step(self):
            if self.num_timesteps - self._last >= args.eval_every:
                self._last = self.num_timesteps
                m = geom_eval(self.model, ev, 500); m["t"] = int(self.num_timesteps)
                curve.append(m)
                cte = m["trailer_cte"] if m["trailer_cte"] == m["trailer_cte"] else 1e9
                k = (m["completion"], -cte)
                if self._bk is None or k > self._bk: self._bk = k; self.best = dict(m)
                print(f"  t={m['t']} compl={m['completion']:.3f} cte={m['trailer_cte']:.3f} "
                      f"hitch={m['max_hitch']:.3f}", flush=True)
            return True

    t0 = time.perf_counter()
    model.learn(total_timesteps=args.steps, progress_bar=False, callback=_CB())
    fin = geom_eval(model, ev, 800)
    _key = lambda c: (c["completion"], -(c["trailer_cte"] if c["trailer_cte"] == c["trailer_cte"] else 1e9))
    best = max(curve, key=_key) if curve else fin
    print(f"# GEOM-RL DONE ({time.perf_counter()-t0:.0f}s): "
          f"BEST compl={best['completion']:.3f} cte={best['trailer_cte']:.3f} @{best.get('t','NA')} "
          f"| final compl={fin['completion']:.3f} cte={fin['trailer_cte']:.3f}", flush=True)
    print("# curve (t, compl, cte):", [(c["t"], round(c["completion"], 3), round(c["trailer_cte"], 3)) for c in curve], flush=True)


if __name__ == "__main__":
    main()
