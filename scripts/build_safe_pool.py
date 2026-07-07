"""Build a PP-VALIDATED "safe spawn" path pool.

Motivation
----------
Some randomly generated paths bend hard right at the spawn point. Forward driving
(minimum-phase) tolerates this, but reverse (non-minimum-phase: the trailer must be
pre-positioned) cannot articulate in time and crashes regardless of the policy. So a
fraction of the old pool was *infeasible from spawn* -- a run could collapse on an
unlucky path draw, not a bad policy. That contaminates the ablation.

Fix: keep only paths a tuned pure-pursuit controller completes BOTH forward AND
reverse from the (clean) spawn. PP is the geometric feasibility oracle -- if PP can
do it, the geometry is trackable, so any RL failure is attributable to the policy,
not the path. By construction PP completion on the saved pool is 100%.

Output: an .npz with keys xs (P,), ys (POOL,P), lo (POOL,), hi (POOL,) that the env
loads via cfg.path.pool_file (see BatchedLaneFollowingEnv._load_pool).

    TTRL_BACKEND=cupy python scripts/build_safe_pool.py \
        --preset shunt --mild --fixed_speed --pool_size 1024 \
        --out path_pools/shunt_mild_safe.npz
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from run_ablations_gpu import cell_to_cfg                                  # noqa: E402
from tractor_trailer_rl.paths import generator as pathgen                 # noqa: E402
from tractor_trailer_rl.batched import backend                           # noqa: E402
from tractor_trailer_rl.batched.backend import xp                         # noqa: E402
from tractor_trailer_rl.batched.sb3_adapter import SB3BatchedVecEnv       # noqa: E402
from tractor_trailer_rl.batched.pure_pursuit import BatchedPurePursuit    # noqa: E402


def gen_candidates(rng, cfg, n):
    """n candidate paths from the cfg's path distribution. Returns xs(P,), ys(n,P)."""
    xs = None
    ys = []
    for _ in range(n):
        xx, yy, _, _ = pathgen.generate_path(rng, cfg.world, cfg.path)
        if xs is None:
            xs = np.asarray(xx, dtype=np.float64)
        ys.append(np.asarray(yy, dtype=np.float64))
    return xs, np.asarray(ys, dtype=np.float64)


def inject_pool(adapter, xs, ys, lo, hi):
    """Point the underlying env at a fixed candidate pool, one path per env
    (identity pin), so env i always runs path i -> exhaustive per-path evaluation."""
    e = adapter.env
    e._xs_np = np.asarray(xs, dtype=np.float64)
    e.xs = xp.asarray(e._xs_np)
    e._pool_ys_np = np.asarray(ys, dtype=np.float64)
    e._pool_ys = xp.asarray(e._pool_ys_np)
    e._pool_lo = np.asarray(lo, dtype=np.float64)
    e._pool_hi = np.asarray(hi, dtype=np.float64)
    e.path_pool_size = int(ys.shape[0])
    e._env_pool_idx = np.zeros(adapter.num_envs, dtype=np.int64)
    e._pin_pool_identity = True


def pp_pass_mask(adapter, cfg, reverse, steps):
    """(N,) bool: did PP complete the pinned path for each env within `steps`?"""
    pp = BatchedPurePursuit(cfg, reverse=reverse)
    o = adapter.reset()
    N = adapter.num_envs
    passed = np.zeros(N, dtype=bool)
    for _ in range(steps):
        a, _ = pp.predict(o)
        o, r, dones, infos = adapter.step(a)
        for i in np.nonzero(dones)[0]:
            if float(infos[i].get("is_success", 0.0)) > 0.5:
                passed[i] = True
    return passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=["lab", "truck", "shunt"], default="shunt")
    ap.add_argument("--mild", action="store_true", help="straight+gentle only")
    ap.add_argument("--fixed_speed", action="store_true",
                    help="hold speed fixed (ablation protocol; deterministic feasibility)")
    ap.add_argument("--pool_size", type=int, default=1024, help="target # safe paths")
    ap.add_argument("--batch", type=int, default=512, help="candidates validated per round")
    ap.add_argument("--steps", type=int, default=1500, help="PP rollout horizon per round")
    ap.add_argument("--max_rounds", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="path_pools/shunt_mild_safe.npz")
    args = ap.parse_args()

    fwd_cfg = cell_to_cfg(f"forward:state:multiplicative", preset=args.preset,
                          fixed_speed=args.fixed_speed, mild_paths=args.mild)
    rev_cfg = cell_to_cfg(f"reverse:state:multiplicative", preset=args.preset,
                          fixed_speed=args.fixed_speed, mild_paths=args.mild)
    rng = np.random.default_rng(args.seed)

    B = args.batch
    fwd_env = SB3BatchedVecEnv(fwd_cfg, B, path_pool_size=B)
    rev_env = SB3BatchedVecEnv(rev_cfg, B, path_pool_size=B)

    kept_ys = []
    tried = 0
    kept_lo = []
    kept_hi = []
    fwd_ok_total = 0
    rev_ok_total = 0
    both_ok_total = 0
    xs_ref = None
    for rnd in range(args.max_rounds):
        if len(kept_ys) >= args.pool_size:
            break
        xs, ys = gen_candidates(rng, fwd_cfg, B)
        if xs_ref is None:
            xs_ref = xs
        # fixed-speed feasibility: lo=hi=fixed_speed (matches the ablation protocol)
        fs = float(fwd_cfg.action.fixed_speed_m_s)
        lo = np.full(B, fs); hi = np.full(B, fs)

        inject_pool(fwd_env, xs, ys, lo, hi)
        fwd_pass = pp_pass_mask(fwd_env, fwd_cfg, reverse=False, steps=args.steps)
        inject_pool(rev_env, xs, ys, lo, hi)
        rev_pass = pp_pass_mask(rev_env, rev_cfg, reverse=True, steps=args.steps)

        both = fwd_pass & rev_pass
        tried += B
        fwd_ok_total += int(fwd_pass.sum())
        rev_ok_total += int(rev_pass.sum())
        both_ok_total += int(both.sum())
        for i in np.nonzero(both)[0]:
            kept_ys.append(ys[i]); kept_lo.append(lo[i]); kept_hi.append(hi[i])
        print(f"[round {rnd}] fwd_ok={fwd_pass.mean():.3f} rev_ok={rev_pass.mean():.3f} "
              f"both={both.mean():.3f}  kept={len(kept_ys)}/{args.pool_size}", flush=True)

    kept_ys = np.asarray(kept_ys[:args.pool_size], dtype=np.float64)
    kept_lo = np.asarray(kept_lo[:args.pool_size], dtype=np.float64)
    kept_hi = np.asarray(kept_hi[:args.pool_size], dtype=np.float64)
    print(f"\nAcceptance over {tried} candidates: fwd {fwd_ok_total/tried:.3f}  "
          f"rev {rev_ok_total/tried:.3f}  both {both_ok_total/tried:.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, xs=xs_ref, ys=kept_ys, lo=kept_lo, hi=kept_hi)
    print(f"wrote {args.out}: {kept_ys.shape[0]} safe paths, P={kept_ys.shape[1]}")

    # ---- FINAL exhaustive validation: PP over the ENTIRE saved pool, both dirs ----
    n = kept_ys.shape[0]
    fwd_v = SB3BatchedVecEnv(fwd_cfg, n, path_pool_size=n)
    rev_v = SB3BatchedVecEnv(rev_cfg, n, path_pool_size=n)
    inject_pool(fwd_v, xs_ref, kept_ys, kept_lo, kept_hi)
    fwd_final = pp_pass_mask(fwd_v, fwd_cfg, reverse=False, steps=args.steps)
    inject_pool(rev_v, xs_ref, kept_ys, kept_lo, kept_hi)
    rev_final = pp_pass_mask(rev_v, rev_cfg, reverse=True, steps=args.steps)
    print(f"\n=== VALIDATION on the saved pool ({n} paths, entire dataset) ===")
    print(f"    PP forward completion: {fwd_final.mean():.4f}  ({int(fwd_final.sum())}/{n})")
    print(f"    PP reverse completion: {rev_final.mean():.4f}  ({int(rev_final.sum())}/{n})")
    if fwd_final.all() and rev_final.all():
        print("    OK: PP completes 100% of the pool in BOTH directions.")
    else:
        print("    WARNING: some saved paths failed re-validation (nondeterminism?).")


if __name__ == "__main__":
    main()
