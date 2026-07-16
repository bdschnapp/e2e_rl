"""Phase-2B screening: which lever rescues the BEV image toward state parity?

Short, single-seed forward-lane screens (shunt, mild, multiplicative, TD3) comparing
occupancy vs SDF vs SDF+coord, scratch vs stability extractor. Reuses the frozen
train_sb3 helpers WITHOUT modifying them. NEW file; touches nothing run_stage1_sweep
depends on.

    TTRL_BACKEND=cupy python3 scripts/screen_bev_rescue.py --steps 200000
"""

import os
import sys
import time
import argparse
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from dataclasses import replace  # noqa: E402
from tractor_trailer_rl.batched import backend  # noqa: E402
from tractor_trailer_rl.config import ActionMode  # noqa: E402
from tractor_trailer_rl.batched.sb3_adapter import SB3BatchedVecEnv  # noqa: E402
from tractor_trailer_rl.batched.bev_sdf_env import SdfObsCfg  # noqa: E402
from run_ablations_gpu import cell_to_cfg  # noqa: E402
from train_sb3 import build_model, EvalCurveCallback  # noqa: E402
from sb3_stability import SdfVecEnv, build_td3, stability_extractor_class  # noqa: E402

POOL = "path_pools/shunt_mild_safe.npz"


def make_cfgs(obs):
    """(train_cfg, eval_cfg) for a forward multiplicative cell: train on random paths,
    eval on the PP-safe pool (mirrors run_stage1_sweep). STOP_SIGNAL action (TD3)."""
    cfg = cell_to_cfg(f"forward:{obs}:multiplicative", preset="shunt",
                      fixed_speed=True, mild_paths=True)
    cfg = replace(cfg, action=replace(cfg.action, mode=ActionMode.STOP_SIGNAL))
    eval_cfg = replace(cfg, path=replace(cfg.path, pool_file=POOL)) if os.path.exists(POOL) else cfg
    return cfg, eval_cfg


def make_envs(kind, obs, n_envs, sdf_cfg=None):
    cfg, eval_cfg = make_cfgs(obs)
    if kind == "sdf":
        train = SdfVecEnv(cfg, n_envs, sdf_cfg=sdf_cfg, path_pool_size=1024)
        ev = SdfVecEnv(eval_cfg, 256, sdf_cfg=sdf_cfg, path_pool_size=512)
    else:
        train = SB3BatchedVecEnv(cfg, n_envs, path_pool_size=1024)
        ev = SB3BatchedVecEnv(eval_cfg, 256, path_pool_size=512)
    ev.env._eval_no_stop = True   # fair completion (matches the sweep eval protocol)
    return train, ev


# (name, env_kind, obs_token, sdf_cfg, model_builder)
def variants():
    stab = lambda: stability_extractor_class()
    return [
        ("state (ref)",           "base", "state",   None,                 "scratch"),
        ("lidar24 (ref)",         "base", "lidar24", None,                 "scratch"),
        ("bev occ scratch",       "base", "bev",     None,                 "scratch"),
        ("bev sdf scratch",       "sdf",  "bev",     SdfObsCfg(False),     "scratch"),
        ("bev sdf+coord scratch", "sdf",  "bev",     SdfObsCfg(True),      "scratch"),
        ("bev occ  +stability",   "base", "bev",     None,                 "stability"),
        ("bev sdf+coord+stability","sdf", "bev",     SdfObsCfg(True),      "stability"),
    ]


def run_variant(name, kind, obs, sdf_cfg, builder, steps, eval_every, seed):
    train, ev = make_envs(kind, obs, n_envs=32, sdf_cfg=sdf_cfg)
    if builder == "stability":            # LayerNorm + separate actor/critic encoders
        model = build_td3(train, seed, extractor_class=stability_extractor_class(),
                          share_features_extractor=False)
    elif builder == "stability_enclr":    # + CNN encoder on 0.1x LR (slow the noisy branch)
        model = build_td3(train, seed, extractor_class=stability_extractor_class(),
                          share_features_extractor=False, encoder_lr_scale=0.1)
    elif builder == "stability_sg":       # image encoder detached in the policy forward
        # (actor + critic both rely on the state vector; the LayerNormed image branch is an
        # inert, non-destabilising feature) -> tests whether removing the noisy image gradient
        # recovers ~state parity. Shared extractor so the detach applies to the whole policy.
        model = build_td3(train, seed,
                          extractor_class=stability_extractor_class(stop_grad_image=True),
                          share_features_extractor=True)
    else:
        model = build_model("td3", train, 1e-3, 4096, seed)
    cb = EvalCurveCallback(ev, eval_every)
    t0 = time.perf_counter()
    model.learn(total_timesteps=steps, progress_bar=False, callback=cb)
    wall = time.perf_counter() - t0
    b = cb.best or {}
    return dict(name=name, best_compl=b.get("completion", float("nan")),
                best_cte=b.get("trailer_cte", float("nan")),
                at=b.get("timesteps"), wall=round(wall))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--eval_every", type=int, default=25_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=None, help="substring filter on variant names")
    args = ap.parse_args()

    print(f"# 2B screen backend={backend.get_backend()} steps={args.steps} seed={args.seed}")
    rows = []
    for name, kind, obs, sdf_cfg, builder in variants():
        if args.only and not any(s in name for s in args.only):
            continue
        try:
            r = run_variant(name, kind, obs, sdf_cfg, builder, args.steps, args.eval_every, args.seed)
        except Exception as e:
            r = dict(name=name, best_compl=float("nan"), best_cte=float("nan"),
                     at=None, wall=0, err=f"{type(e).__name__}: {e}")
        rows.append(r)
        tag = r.get("err", "")
        print(f"  {r['name']:28s} compl={r['best_compl']:.3f} cte={r['best_cte']:.3f} "
              f"@{r['at']} ({r['wall']}s) {tag}")

    print("\n=== 2B SCREEN SUMMARY (best completion @%d steps) ===" % args.steps)
    print(f"{'variant':30s} {'completion':>10s} {'trailer_cte':>12s}")
    for r in rows:
        print(f"{r['name']:30s} {r['best_compl']:>10.3f} {r['best_cte']:>12.3f}")


if __name__ == "__main__":
    main()
