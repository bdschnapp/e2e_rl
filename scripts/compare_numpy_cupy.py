"""Apples-to-apples: the EXACT same batched env, numpy vs cupy (import swap only).

The batched env is written against `backend.xp`, which is numpy or cupy depending
on TTRL_BACKEND. This script benchmarks the identical vectorised code on whichever
backend is active, so running it once with TTRL_BACKEND=numpy and once with
TTRL_BACKEND=cupy is a like-for-like CPU-vs-GPU comparison of the same environment.

    TTRL_BACKEND=numpy python scripts/compare_numpy_cupy.py
    TTRL_BACKEND=cupy  python scripts/compare_numpy_cupy.py
"""

import os
import sys
import time
import numpy as np
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tractor_trailer_rl.batched import backend  # noqa: E402
from tractor_trailer_rl.config import lab_config  # noqa: E402


def _sync():
    if backend.get_backend() == "cupy":
        import cupy as cp
        cp.cuda.Stream.null.synchronize()


def bench(cfg, n_envs, steps, warmup=10):
    from tractor_trailer_rl.batched.env import BatchedLaneFollowingEnv
    xp = backend.xp
    env = BatchedLaneFollowingEnv(cfg, n_envs, path_pool_size=256)
    env.reset(seed=0)
    acts = xp.asarray(np.random.default_rng(0).uniform(
        -0.4, 0.4, size=(steps + warmup, n_envs, env.action_dim)))
    for i in range(warmup):
        env.step(acts[i])
    _sync()
    t0 = time.perf_counter()
    for i in range(warmup, warmup + steps):
        env.step(acts[i])
    _sync()
    dt = time.perf_counter() - t0
    return n_envs * steps / dt


def main():
    be = backend.get_backend()
    state = replace(lab_config(), obs=replace(lab_config().obs, lidar_beams=0))
    lidar = lab_config()
    print(f"# backend = {be}")
    for label, cfg, Ns, steps in [
        ("state", state, [256, 1024, 4096], 30),
        ("lidar24", lidar, [256, 1024, 4096], 20),
    ]:
        for n in Ns:
            tps = bench(cfg, n, steps)
            print(f"  {label:8s} N={n:5d}   {tps:14,.0f} transitions/s")


if __name__ == "__main__":
    main()
