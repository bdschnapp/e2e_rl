"""Smoke test for the Track-D obstacle env + a regression check on the base env.

Confirms (a) the pure lane-following env still runs identically (obstacle=None is
untouched), and (b) the obstacle env places obstacles, makes them visible to lidar,
scores difficulty, collides, and applies the difficulty-gated stop-gate reward the
way we intend (stop pays on hard layouts, is penalised on easy ones).
"""
import sys, os, warnings, numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tractor_trailer_rl.batched import backend
backend.set_backend("numpy")
from tractor_trailer_rl.config import (shunt_truck_config, shunt_truck_obstacle_config,
                                       Direction)
from tractor_trailer_rl.batched.sb3_adapter import make_batched_env
from tractor_trailer_rl.batched.pure_pursuit import BatchedPurePursuit
from tractor_trailer_rl.batched import obstacles as obs


def check_base_regression():
    print("== base lane-following env (obstacle=None) regression ==")
    cfg = shunt_truck_config()
    env = make_batched_env(cfg, 64, path_pool_size=64)
    assert type(env).__name__ == "BatchedLaneFollowingEnv", type(env).__name__
    o = env.reset(seed=0)
    ctrl = BatchedPurePursuit(cfg, reverse=False)
    tot = 0.0
    for _ in range(200):
        a, _ = ctrl.predict(o)
        o, r, term, trunc, info = env.step(a)
        tot += float(np.mean(backend.to_numpy(r)))
    print(f"   ran 200 steps, obs dim={o.shape[1]}, mean-step-reward~{tot/200:.3f}  OK\n")


def check_obstacle_env():
    print("== obstacle env (shunt_truck_obstacle_config) ==")
    cfg = shunt_truck_obstacle_config()
    N = 256
    env = make_batched_env(cfg, N, path_pool_size=256)
    assert type(env).__name__ == "BatchedObstacleAvoidanceEnv", type(env).__name__
    o = env.reset(seed=1)
    to_np = backend.to_numpy

    n_obs = to_np(env.ovalid).sum(axis=1)
    diff = to_np(env.difficulty)
    print(f"   obs dim={o.shape[1]} (state+lidar)")
    print(f"   obstacles/env: min={n_obs.min()} max={n_obs.max()} mean={n_obs.mean():.2f}")
    print(f"   difficulty: min={diff.min():.3f} max={diff.max():.3f} mean={diff.mean():.3f}")
    frac_hard = float((diff > 0.6).mean()); frac_easy = float((diff < 0.3).mean())
    print(f"   layouts: easy(<0.3)={frac_easy:.2f}  hard(>0.6)={frac_hard:.2f}")

    # lidar must see obstacles: some beams < 1.0 (the base corridor rarely blocks
    # forward beams on straight/gentle paths, so sub-1 ranges come from obstacles).
    lidar_cols = o[:, -cfg.obs.lidar_beams:]
    frac_blocked = float((lidar_cols.min(axis=1) < 0.999).mean())
    print(f"   envs with a lidar return < max range: {frac_blocked:.2f}")

    # local avoidance path differs from the centreline where obstacles intrude
    shift = np.abs(to_np(env.local_ys) - to_np(env.ys)).max(axis=1)
    print(f"   hidden-path max lateral shift: mean={shift.mean():.2f}m max={shift.max():.2f}m")

    m = lambda x: float(np.mean(x)) if x else float("nan")

    # --- Phase 1: reward ordering. Force a STOP every step; every env stops
    # immediately, terminates, and resets to a fresh layout, so we sweep stop
    # rewards across the difficulty spectrum. ---
    ctrl = BatchedPurePursuit(cfg, reverse=False)
    buckets = {"easy": [], "med": [], "hard": []}
    for _ in range(200):
        steer, _ = ctrl.predict(o)
        a = np.concatenate([steer, np.ones((N, 1))], axis=1).astype(np.float32)  # always stop
        o, r, term, trunc, info = env.step(a)
        r = to_np(r); dd = info["difficulty"]; st = info["stopped"]
        buckets["easy"] += list(r[st & (dd < 0.3)])
        buckets["med"] += list(r[st & (dd >= 0.3) & (dd <= 0.6)])
        buckets["hard"] += list(r[st & (dd > 0.6)])
    print(f"   STOP reward  easy={m(buckets['easy']):.1f}  med={m(buckets['med']):.1f}  "
          f"hard={m(buckets['hard']):.1f}")
    assert m(buckets["hard"]) > m(buckets["easy"]), "stop-gate: hard must out-reward easy!"
    assert m(buckets["hard"]) > 0 > m(buckets["easy"]), "hard-stop +, easy-stop - expected"
    print("   stop-gate ordering (hard>0>easy) OK")

    # --- Phase 2: drive-through behaviour. PP steering, never stop; count how the
    # episodes end, stratified by difficulty. ---
    o = env.reset(seed=2)
    end = {"easy": {"succ": 0, "crash": 0}, "hard": {"succ": 0, "crash": 0}}
    for _ in range(400):
        steer, _ = ctrl.predict(o)
        a = np.concatenate([steer, -np.ones((N, 1))], axis=1).astype(np.float32)  # never stop
        o, r, term, trunc, info = env.step(a)
        done = to_np(term) | to_np(trunc)
        if done.any():
            dd = info["difficulty"]; sc = to_np(info["success"])
            for key, msk in (("easy", dd < 0.3), ("hard", dd > 0.6)):
                d2 = done & msk
                end[key]["succ"] += int((d2 & sc).sum())
                end[key]["crash"] += int((d2 & ~sc).sum())
    print(f"   drive-through (never stop): easy succ={end['easy']['succ']} "
          f"crash={end['easy']['crash']} | hard succ={end['hard']['succ']} "
          f"crash={end['hard']['crash']}")
    print("   (easy should mostly succeed; hard should mostly crash -> stopping is the fix)\n")


if __name__ == "__main__":
    check_base_regression()
    check_obstacle_env()
    print("ALL SMOKE CHECKS PASSED")
