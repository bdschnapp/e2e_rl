"""Smoke test for the Phase-3 obstacle-SDF env + obstacle-geometry label (NEW, additive).
No training — just build the env, step it, and sanity-check the image + labels.

    TTRL_BACKEND=cupy python3 scripts/smoke_geom_obstacle.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from dataclasses import replace
from tractor_trailer_rl.batched import backend
from tractor_trailer_rl.batched.backend import to_numpy
from tractor_trailer_rl.config import ActionMode, Direction, shunt_truck_obstacle_config
from tractor_trailer_rl.batched.obstacle_sdf_env import ObstacleSdfBevEnv, OBSTACLE_GEOM_NAMES
from tractor_trailer_rl.batched.bev_sdf_env import SdfObsCfg


def obstacle_cfg(direction="forward", bev_size=32):
    cfg = shunt_truck_obstacle_config(direction=Direction.REVERSE if direction == "reverse" else Direction.FORWARD)
    return replace(cfg, obs=replace(cfg.obs, bev_size=bev_size),
                   action=replace(cfg.action, mode=ActionMode.STOP_SIGNAL))


def main():
    print(f"# backend={backend.get_backend()}")
    cfg = obstacle_cfg("forward")
    env = ObstacleSdfBevEnv(cfg, 8, sdf_cfg=SdfObsCfg(coord=True))
    print("obs space:", env.single_observation_space)
    print("action dim:", env.single_action_space)
    obs = env.reset() if not hasattr(env, "reset_batch") else env.reset()
    # batched env: reset() returns (obs, info) or obs depending on API; handle both
    o = obs[0] if isinstance(obs, tuple) else obs
    img = to_numpy(o["image"]); vec = to_numpy(o["vector"])
    print(f"image shape {img.shape} range [{img.min():.2f},{img.max():.2f}]  (SDF+coord)")
    print(f"vector shape {vec.shape}")
    n_blocked = (img[:, 0] <= -0.99).sum(axis=(1, 2))   # channel 0 = SDF; obstacles/outside = -1
    print(f"blocked cells per env (obstacles+outside): {n_blocked}")
    og = to_numpy(env.obstacle_geom())
    print(f"obstacle_geom shape {og.shape} names={OBSTACLE_GEOM_NAMES}")
    print("  sample (forward, lateral, clearance) /R:")
    for i in range(min(4, og.shape[0])):
        print(f"    env{i}: {np.array2string(og[i], precision=2, floatmode='fixed')}")
    # step a few times with random actions
    A = env.single_action_space.shape[0]
    for t in range(5):
        a = np.random.uniform(-1, 1, size=(8, A)).astype(np.float32)
        step = env.step(backend.xp.asarray(a))
        og = to_numpy(env.obstacle_geom())
    print(f"stepped 5x OK; obstacle_geom finite: {np.isfinite(og).all()}")
    print("SMOKE OK")


if __name__ == "__main__":
    main()
