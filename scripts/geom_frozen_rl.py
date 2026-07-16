"""Phase-2B/3 geometry RL, STABLE form (NEW file). The frozen geometry-distillation
encoder runs as the SB3 policy's features_extractor (NOT in the env loop) — the exact
frozen-CNN-in-policy pattern that bc_rescue ran without a single crash. Avoids the
cupy/torch encoder-in-env-loop segfault entirely.

Obs = SDF-BEV Dict{vector(true state), image}. The frozen encoder maps the image to the
predicted geometry; proprio dims (delta,gamma) are overridden with the true (measurable)
values; the policy MLP acts on that vector, trained FROM SCRATCH. Because obs['vector']
is the TRUE state, eval (train_sb3.evaluate) reports true completion/CTE.

    TTRL_BACKEND=cupy python3 -u scripts/geom_frozen_rl.py --encoder geom_encoder_fwd_coord.pt \
        --coord --steps 500000 --eval_every 50000 --seed 0
"""

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from dataclasses import replace  # noqa: E402
from tractor_trailer_rl.batched import backend  # noqa: E402
from tractor_trailer_rl.config import ActionMode  # noqa: E402
from run_ablations_gpu import cell_to_cfg  # noqa: E402
from train_sb3 import evaluate, EvalCurveCallback, _td3_action_noise  # noqa: E402
from sb3_stability import SdfVecEnv  # noqa: E402
from tractor_trailer_rl.batched.bev_sdf_env import SdfObsCfg  # noqa: E402

POOL = "path_pools/shunt_mild_safe.npz"


def frozen_geom_extractor_class(ckpt_path, proprio=(0, 1)):
    """SB3 features_extractor: frozen encoder(image) -> predicted geometry, proprio dims
    overridden with the true obs['vector'] values. No trainable params (fully frozen)."""
    import torch as th
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from geom_encoder import GeomEncoder

    class FrozenGeomExtractor(BaseFeaturesExtractor):
        def __init__(self, observation_space):
            ck = th.load(ckpt_path, weights_only=False)
            out_dim = int(ck["out_dim"])
            super().__init__(observation_space, features_dim=out_dim)
            enc = GeomEncoder(ck["in_ch"], out_dim, ck["img_size"])
            enc.load_state_dict(ck["state_dict"]); enc.eval()
            for p in enc.parameters():
                p.requires_grad_(False)
            self.enc = enc
            self.register_buffer("mean", th.as_tensor(ck["label_mean"], dtype=th.float32))
            self.register_buffer("std", th.as_tensor(ck["label_std"], dtype=th.float32))
            self.proprio = set(proprio)

        def forward(self, obs):
            with th.no_grad():
                pred = self.enc(obs["image"]) * self.std + self.mean   # (N, out_dim) de-standardized
            vec = obs["vector"]
            cols = [(vec[:, i:i + 1] if i in self.proprio else pred[:, i:i + 1])
                    for i in range(pred.shape[1])]
            return th.cat(cols, dim=1)

    return FrozenGeomExtractor


def make_cfgs(direction):
    cfg = cell_to_cfg(f"{direction}:bev:multiplicative", preset="shunt", fixed_speed=True, mild_paths=True)
    cfg = replace(cfg, action=replace(cfg.action, mode=ActionMode.STOP_SIGNAL))
    eval_cfg = replace(cfg, path=replace(cfg.path, pool_file=POOL)) if os.path.exists(POOL) else cfg
    return cfg, eval_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--direction", choices=["forward", "reverse"], default="forward")
    ap.add_argument("--coord", action="store_true")
    ap.add_argument("--n_envs", type=int, default=32)
    ap.add_argument("--steps", type=int, default=500_000)
    ap.add_argument("--eval_every", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from stable_baselines3 import TD3
    sdf_cfg = SdfObsCfg(args.coord)
    cfg, eval_cfg = make_cfgs(args.direction)
    print(f"# geom-frozen-RL backend={backend.get_backend()} enc={args.encoder} dir={args.direction} "
          f"steps={args.steps} seed={args.seed}", flush=True)

    train = SdfVecEnv(cfg, args.n_envs, sdf_cfg=sdf_cfg, path_pool_size=1024)
    ev = SdfVecEnv(eval_cfg, 256, sdf_cfg=sdf_cfg, path_pool_size=512)
    ev.env._eval_no_stop = True

    pk = {"features_extractor_class": frozen_geom_extractor_class(args.encoder, (0, 1))}
    model = TD3("MultiInputPolicy", train, learning_rate=1e-3, learning_starts=10_000,
                action_noise=_td3_action_noise(train), policy_kwargs=pk,
                seed=args.seed, device="cuda", verbose=0)

    cb = EvalCurveCallback(ev, args.eval_every)
    t0 = time.perf_counter()
    model.learn(total_timesteps=args.steps, progress_bar=False, callback=cb)
    fin = evaluate(model, ev, steps=800)
    best = cb.best or fin
    print(f"# GEOM-FROZEN-RL DONE ({time.perf_counter()-t0:.0f}s): "
          f"BEST completion={best['completion']:.3f} cte={best['trailer_cte']:.3f} @{best.get('timesteps')} "
          f"| final completion={fin['completion']:.3f} cte={fin['trailer_cte']:.3f}", flush=True)
    print("# curve (t, compl, cte):",
          [(c['timesteps'], round(c['completion'], 3), round(c['trailer_cte'], 3)) for c in cb.curve], flush=True)


if __name__ == "__main__":
    main()
