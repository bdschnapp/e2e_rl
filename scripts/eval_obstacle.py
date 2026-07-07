"""Difficulty-stratified evaluation for the obstacle stop-gate env (Track D, D3).

Produces the thesis table: per difficulty bucket, the % of episodes that COMPLETE
(reach the goal), STOP (fire the stop-gate), CRASH (hit an obstacle / leave the
corridor / jackknife), or TIME OUT, each with a Wilson 95% CI, plus mean
min-clearance. Because each episode is ONE obstacle event (1-2 co-located
obstacles), every outcome is cleanly attributable to that layout's difficulty --
no cascade between events.

Policies:
  --model PATH   load an SB3 policy (.zip) trained on shunt_truck_obstacle_config
  (default)      a controller baseline: pure-pursuit steering + stop iff difficulty
                 > --stop_thresh (an oracle-difficulty reference, not a learned one)

Runs on CPU (numpy backend) so it never contends with a GPU training sweep.
"""
import sys, os, argparse, warnings, math
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from tractor_trailer_rl.batched import backend
# Default to CPU (numpy) so a bare invocation never contends with a GPU training
# sweep; honour TTRL_BACKEND=cupy for a fast GPU eval when the card is free.
if not os.environ.get("TTRL_BACKEND"):
    backend.set_backend("numpy")
from tractor_trailer_rl.config import shunt_truck_obstacle_config, Direction
from tractor_trailer_rl.batched.sb3_adapter import make_batched_env
from tractor_trailer_rl.batched.pure_pursuit import BatchedPurePursuit
from tractor_trailer_rl.batched import corridor as cor
from tractor_trailer_rl.batched import obstacles as obsmod
from tractor_trailer_rl.batched import geometry as geo


def local_path_obs(env, o):
    """Copy of the observation with the path-error / curvature columns recomputed
    against the HIDDEN local-planner path (env.local_ys) instead of the centreline.
    Feeding this to pure-pursuit realises the traditional stack: global path ->
    local (APF) planner -> PP tracks the planned avoidance path. (The RL policy, by
    contrast, only ever sees the centreline obs + lidar and must infer the detour.)"""
    to_np = backend.to_numpy
    v = env.vehicle; xs = env.xs; lys = env.local_ys
    o2 = to_np(o).copy()
    e_y, e_psi = geo.path_errors(xs, lys, v.x, v.y, v.p, reverse=env.reverse,
                                 error_theta_scale=env.scale, tangent_offset=env.tan_off)
    k1 = geo.curvature_at(xs, lys, v.tx, v.ty, env.k1s, env.maxk)
    k2 = geo.curvature_at(xs, lys, v.tx, v.ty, env.k2s, env.maxk)
    o2[:, 2] = to_np(e_y); o2[:, 3] = to_np(e_psi)
    o2[:, 6] = to_np(k1); o2[:, 7] = to_np(k2)
    if not env.tractor_only:
        e_y_t, e_psi_t = geo.path_errors(xs, lys, v.tx, v.ty, v.tyaw, reverse=env.reverse,
                                         error_theta_scale=env.scale, tangent_offset=env.tan_off)
        o2[:, 4] = to_np(e_y_t); o2[:, 5] = to_np(e_psi_t)
    return o2


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion -> (p, half_width)."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center, half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="SB3 .zip policy (overrides --baseline)")
    ap.add_argument("--baseline", choices=["local_pp", "centerline_pp"], default="local_pp",
                    help="local_pp = traditional stack (global+local planner+PP on the planned "
                         "path, stop when planner infeasible); centerline_pp = PP on the raw "
                         "centreline (obstacle-blind reference)")
    ap.add_argument("--direction", choices=["forward", "reverse"], default="forward")
    ap.add_argument("--n_envs", type=int, default=512)
    ap.add_argument("--episodes", type=int, default=4000, help="target completed episodes")
    ap.add_argument("--buckets", type=int, default=5)
    ap.add_argument("--stop_thresh", type=float, default=None,
                    help="baseline: stop iff difficulty>this (default 0.95 local_pp = planner "
                         "infeasible; 0.6 centerline_pp)")
    args = ap.parse_args()
    if args.stop_thresh is None:
        args.stop_thresh = 0.95 if args.baseline == "local_pp" else 0.6

    direction = Direction.REVERSE if args.direction == "reverse" else Direction.FORWARD
    cfg = shunt_truck_obstacle_config(direction=direction)
    env = make_batched_env(cfg, args.n_envs, path_pool_size=args.n_envs)
    to_np = backend.to_numpy

    policy = None
    if args.model:
        from stable_baselines3 import PPO, TD3, DQN, SAC  # noqa
        for cls in (TD3, PPO, SAC, DQN):
            try:
                policy = cls.load(args.model, device="cpu"); break
            except Exception:
                continue
        if policy is None:
            raise SystemExit(f"could not load model {args.model}")
    else:
        pp = BatchedPurePursuit(cfg, reverse=(direction == Direction.REVERSE))

    o = env.reset(seed=0)
    N = args.n_envs
    min_clear = np.full(N, np.inf)                      # per-env running min obstacle clearance
    half_w = max(cfg.vehicle.tractor_width_m, cfg.vehicle.trailer_width_m) / 2.0
    # accumulators per episode: (difficulty, outcome, min_clearance)
    diffs, outcomes, clears = [], [], []

    def record(mask, info, mc):
        dd = info["difficulty"]; st = info["stopped"]; sc = to_np(info["success"])
        for i in np.nonzero(mask)[0]:
            if sc[i]:
                out = "complete"
            elif st[i]:
                out = "stop"
            else:
                out = "crash"       # collision / OOB / jackknife (truncation handled below)
            diffs.append(float(dd[i])); outcomes.append(out); clears.append(float(mc[i]))

    steps = 0
    while len(diffs) < args.episodes and steps < 20000:
        steps += 1
        if policy is not None:
            # SB3 predict needs a host array; under the cupy backend `o` is on-device.
            a, _ = policy.predict(to_np(o), deterministic=True)
        else:
            pp_obs = local_path_obs(env, o) if args.baseline == "local_pp" else o
            steer, _ = pp.predict(pp_obs)
            d_now = to_np(env.difficulty)
            stop_cmd = np.where(d_now > args.stop_thresh, 1.0, -1.0)  # planner-feasibility stop
            a = np.concatenate([to_np(steer), stop_cmd[:, None]], axis=1).astype(np.float32)
        # track min clearance to obstacle surfaces BEFORE stepping (pre-reset state)
        bx, by = cor.body_points(env.vehicle, cfg)
        d_surf = to_np(obsmod._body_min_surface_dist(env.ox, env.oy, env.orad, env.ovalid, bx, by)) - half_w
        min_clear = np.minimum(min_clear, d_surf)
        o, r, term, trunc, info = env.step(a)
        term = to_np(term); trunc = to_np(trunc); done = term | trunc
        if done.any():
            # a truncation that is not a success/stop = time-out (separate from crash)
            timeout = trunc & ~term
            record(done & ~timeout, info, min_clear)
            dd = info["difficulty"]
            for i in np.nonzero(timeout)[0]:
                diffs.append(float(dd[i])); outcomes.append("timeout"); clears.append(float(min_clear[i]))
            min_clear[done] = np.inf

    diffs = np.array(diffs); outcomes = np.array(outcomes); clears = np.array(clears)
    edges = np.linspace(0.0, 1.0, args.buckets + 1)
    if policy is not None:
        who = os.path.basename(args.model)
    elif args.baseline == "local_pp":
        who = f"traditional stack: local-planner + PP (stop>{args.stop_thresh})"
    else:
        who = f"centreline PP (obstacle-blind, stop>{args.stop_thresh})"
    print(f"\nObstacle stop-gate eval [{args.direction}] — {who}  ({len(diffs)} episodes)\n")
    hdr = f"{'difficulty':12} {'n':>5} {'complete%':>16} {'stop%':>16} {'crash%':>15} {'timeout%':>10} {'min_clr(m)':>11}"
    print(hdr); print("-" * len(hdr))
    for b in range(args.buckets):
        lo, hi = edges[b], edges[b + 1]
        m = (diffs >= lo) & (diffs < hi if b < args.buckets - 1 else diffs <= hi)
        n = int(m.sum())
        if n == 0:
            print(f"{lo:.2f}-{hi:.2f}    {0:>5}  (no episodes)"); continue
        def pct(name):
            k = int((outcomes[m] == name).sum()); p, h = wilson(k, n)
            return f"{100*p:5.1f}+/-{100*h:4.1f}"
        # min-clearance only over episodes that actually drove past (complete/crash);
        # a stop happens far from the obstacle so its clearance is meaningless here.
        mc = clears[m & (outcomes != "stop") & (outcomes != "timeout")]
        mc = mc[np.isfinite(mc)]
        mcs = f"{np.mean(mc):.2f}" if len(mc) else "  -"
        print(f"{lo:.2f}-{hi:.2f}    {n:>5}  {pct('complete'):>16} {pct('stop'):>16} "
              f"{pct('crash'):>15} {100*(outcomes[m]=='timeout').mean():>9.1f} {mcs:>11}")
    # aggregate stop-decision quality: on hard (>0.6) should stop; on easy (<0.3) should not
    hard = diffs > 0.6; easy = diffs < 0.3
    if hard.any():
        print(f"\nhard (>0.6): stop {100*(outcomes[hard]=='stop').mean():.1f}%  "
              f"crash {100*(outcomes[hard]=='crash').mean():.1f}%  complete {100*(outcomes[hard]=='complete').mean():.1f}%")
    if easy.any():
        print(f"easy (<0.3): complete {100*(outcomes[easy]=='complete').mean():.1f}%  "
              f"stop {100*(outcomes[easy]=='stop').mean():.1f}%  crash {100*(outcomes[easy]=='crash').mean():.1f}%")


if __name__ == "__main__":
    main()
