"""Throughput benchmark: batched GPU env vs the scalar SB3 SubprocVecEnv baseline.

Measures pure environment throughput (env transitions per second, no policy
network) so the comparison isolates the simulator, which is what the CuPy port
accelerates. Reports:

  * batched env on the active backend (numpy CPU or cupy GPU) at several env counts
  * scalar `LaneFollowingEnv` under SB3 SubprocVecEnv at a few process counts

Run on GPU:   TTRL_BACKEND=cupy  python scripts/benchmark_batched.py
Run on CPU:   TTRL_BACKEND=numpy python scripts/benchmark_batched.py

The headline number is transitions/sec and the speedup of the batched GPU env over
the best SubprocVecEnv configuration.
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tractor_trailer_rl.batched import backend  # noqa: E402
from tractor_trailer_rl.batched.backend import to_numpy  # noqa: E402
from tractor_trailer_rl.config import lab_config  # noqa: E402


def _sync():
    if backend.get_backend() == "cupy":
        import cupy as cp
        cp.cuda.Stream.null.synchronize()


def bench_batched(cfg, n_envs, steps=200, warmup=20, pool=512):
    from tractor_trailer_rl.batched.env import BatchedLaneFollowingEnv
    xp = backend.xp
    env = BatchedLaneFollowingEnv(cfg, n_envs, path_pool_size=pool)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    acts = xp.asarray(rng.uniform(-0.4, 0.4, size=(steps + warmup, n_envs, env.action_dim)))
    for i in range(warmup):
        env.step(acts[i])
    _sync()
    t0 = time.perf_counter()
    for i in range(warmup, warmup + steps):
        env.step(acts[i])
    _sync()
    dt = time.perf_counter() - t0
    return n_envs * steps / dt


def bench_subproc(cfg, n_procs, steps=300, warmup=20):
    try:
        from stable_baselines3.common.vec_env import SubprocVecEnv
    except Exception as e:
        return None, f"sb3 unavailable ({e})"
    from tractor_trailer_rl.envs.base import LaneFollowingEnv

    def make():
        def _f():
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            return LaneFollowingEnv(cfg)
        return _f

    venv = SubprocVecEnv([make() for _ in range(n_procs)])
    venv.reset()
    adim = venv.action_space.shape[0]
    rng = np.random.default_rng(0)
    for _ in range(warmup):
        venv.step(rng.uniform(-0.4, 0.4, size=(n_procs, adim)).astype(np.float32))
    t0 = time.perf_counter()
    for _ in range(steps):
        venv.step(rng.uniform(-0.4, 0.4, size=(n_procs, adim)).astype(np.float32))
    dt = time.perf_counter() - t0
    venv.close()
    return n_procs * steps / dt, None


def main():
    cfg = lab_config()  # AgileX lab scale, lidar-24 (the realistic ablation config)
    be = backend.get_backend()
    print(f"# Backend: {be}   config: lab_config (lidar_beams={cfg.obs.lidar_beams})\n")

    print("## Batched env")
    batched = {}
    for n in [256, 1024, 4096, 16384]:
        try:
            tps = bench_batched(cfg, n)
            batched[n] = tps
            print(f"  N={n:6d}   {tps:12,.0f} transitions/s")
        except Exception as e:  # e.g. OOM at large N
            print(f"  N={n:6d}   FAILED ({type(e).__name__}: {e})")
            break

    print("\n## Scalar SB3 SubprocVecEnv baseline")
    best_sub = 0.0
    for n in [1, 8, 16]:
        tps, err = bench_subproc(cfg, n)
        if tps is None:
            print(f"  procs={n:3d}  {err}")
            break
        best_sub = max(best_sub, tps)
        print(f"  procs={n:3d}  {tps:12,.0f} transitions/s")

    if batched and best_sub:
        best_b = max(batched.values())
        print(f"\n## Speedup: best batched ({be}) / best SubprocVecEnv = "
              f"{best_b / best_sub:,.1f}x  ({best_b:,.0f} vs {best_sub:,.0f} transitions/s)")


if __name__ == "__main__":
    main()
