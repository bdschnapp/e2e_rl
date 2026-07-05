"""Parity of the batched GPU env against the scalar oracle + scipy.

Run on the numpy backend (default) so it needs no GPU:

    pytest tests/test_batched_parity.py -v

What is asserted (and why the thresholds are what they are):
  * expm/ZOH == scipy to ~1e-6 (the batched matrix exponential replacing the
    per-step scipy cont2discrete — blocker #1).
  * batched vehicle == scalar StateSpaceTractorTrailer to ~1e-10 over a rollout.
  * batched env obs == scalar env obs EXACTLY, and reward EXACTLY where the
    proximity term is zero, i.e. the whole dynamics/geometry/observation/reward
    pipeline is bit-parity; only the proximity term differs (analytic corridor vs
    rasterised grid) and only near the lane edge.
  * analytic lidar within a couple of cells of the grid raycast (blocker #3).
  * analytic collision agrees with the grid on >=95% of random states, the rest
    within one cell of the corridor boundary (blocker #2/#4).
"""

import numpy as np
import pytest
from dataclasses import replace

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tractor_trailer_rl.batched import backend
backend.set_backend("numpy")
from tractor_trailer_rl.batched.backend import xp
from tractor_trailer_rl.config import truck_config, lab_config, Direction, VehicleKind
from tractor_trailer_rl.vehicle.tractor_trailer import StateSpaceTractorTrailer
from tractor_trailer_rl.batched.vehicle import BatchedTractorTrailer
from tractor_trailer_rl.batched.expm import zoh_discretize
from tractor_trailer_rl.batched.env import BatchedLaneFollowingEnv
from tractor_trailer_rl.batched.lidar import raycast as bat_ray
from tractor_trailer_rl.batched import corridor as cor
from tractor_trailer_rl.envs.base import LaneFollowingEnv
from tractor_trailer_rl.observation.occupancy import build_occupancy_grid
from tractor_trailer_rl.observation.lidar import raycast as scal_ray, Pose
from tractor_trailer_rl.reward.proximity import proximity_penalty as scal_prox
from tractor_trailer_rl.paths import generator as pathgen


def test_zoh_matches_scipy():
    import scipy.signal
    rng = np.random.default_rng(0)
    for _ in range(50):
        A = rng.standard_normal((3, 3)) * rng.uniform(0.1, 50)
        B = rng.standard_normal((3, 2)) * rng.uniform(0.1, 50)
        Ad, Bd = zoh_discretize(A[None], B[None], 0.1)
        ref = scipy.signal.cont2discrete((A, B, np.eye(3), 0), 0.1, method="zoh")
        assert np.max(np.abs(np.asarray(Ad)[0] - ref[0])) < 1e-6
        assert np.max(np.abs(np.asarray(Bd)[0] - ref[1])) < 1e-6


@pytest.mark.parametrize("cfg", [truck_config(), lab_config()])
def test_vehicle_matches_scalar(cfg):
    rng = np.random.default_rng(1)
    sv = StateSpaceTractorTrailer(cfg.vehicle); sv.reset(4.0, x=3.0, y=45.0, p=0.1)
    bv = BatchedTractorTrailer(cfg, 2)
    bv.reset_where(xp.asarray([True, True]), xp.asarray([4.0, 2.0]),
                   xp.asarray([3.0, 9.0]), xp.asarray([45.0, 44.0]), xp.asarray([0.1, -0.2]))
    for _ in range(200):
        a = np.array([rng.uniform(-0.4, 0.4), rng.uniform(1.0, 5.0)])
        sv.loop(a)
        bv.step(xp.asarray([a[0], a[0] * 0.5]), xp.asarray([a[1], a[1] * 0.8]))
    err = max(abs(sv.x - float(bv.x[0])), abs(sv.y - float(bv.y[0])),
              abs(sv.p - float(bv.p[0])), abs(sv.xd - float(bv.xd[0])),
              abs(sv.trailer.x - float(bv.tx[0])), abs(sv.trailer.yaw - float(bv.tyaw[0])))
    assert err < 1e-10


def _setup_pair(cfg, xx, yy, speed=4.0):
    env = LaneFollowingEnv(cfg); env.reset(seed=1)
    env.xx, env.yy = xx, yy
    env.occ_grid, env.occ_meta = build_occupancy_grid(xx, yy, cfg.world)
    x, y, yaw = env._point_on_path(0.1)
    env.vehicle.reset(speed, x=x, y=y, p=yaw + (np.pi if cfg.is_reverse else 0.0))
    env.episode_speed = -speed if cfg.is_reverse else speed
    env.vehicle.fixed_speed_command = env.episode_speed
    if cfg.is_tractor_only:
        env._align_trailer()
    env.success = False; env._step_count = 0

    b = BatchedLaneFollowingEnv(cfg, 1)
    b._xs_np = xx.astype(float); b.xs = xp.asarray(b._xs_np)
    b._ys_np = yy[None].astype(float); b.ys = xp.asarray(b._ys_np); b.path_kind = ["straight"]
    i = int(np.clip(int(0.1 * (len(xx) - 1)), 0, len(xx) - 2))
    yaw = np.arctan2(yy[i + 1] - yy[i], xx[i + 1] - xx[i])
    b.vehicle.reset_where(xp.asarray([True]), xp.asarray([speed]), xp.asarray([xx[i]]),
                          xp.asarray([yy[i]]), xp.asarray([yaw + (np.pi if cfg.is_reverse else 0.0)]))
    if cfg.is_tractor_only:
        b.vehicle.align_trailer(xp.asarray([True]))
    b.episode_speed = xp.asarray([-speed if cfg.is_reverse else speed])
    b.step_count[:] = 0; b.success = xp.asarray([False])
    return env, b


@pytest.mark.parametrize("direction", [Direction.FORWARD, Direction.REVERSE])
@pytest.mark.parametrize("kind", [VehicleKind.TRAILER, VehicleKind.TRACTOR_ONLY])
def test_env_obs_reward_parity(direction, kind):
    base = truck_config(direction=direction, vehicle_kind=kind)
    cfg = replace(base, obs=replace(base.obs, lidar_beams=0),
                  path=replace(base.path, kind_probs={"straight": 1.0}))
    xx, yy, _, _ = pathgen.generate_path(np.random.default_rng(0), cfg.world, cfg.path)
    env, b = _setup_pair(cfg, xx, yy)
    max_obs = max_rew0 = 0.0
    for t in range(200):
        sr = 0.02 * np.sin(t / 12.0)
        so, srr, st, stx, _ = env.step(np.array([sr], np.float32))
        sp = scal_prox(env.occ_grid, env.occ_meta,
                       [(env.vehicle.x, env.vehicle.y),
                        (env.vehicle.trailer.x, env.vehicle.trailer.y)], cfg)
        bo, br, bt, btx, bi = b.step(np.array([[sr]], np.float32))
        fo = bi["final_observation"][0] if "final_observation" in bi else bo[0]
        max_obs = max(max_obs, float(np.max(np.abs(np.asarray(so) - np.asarray(fo)))))
        bp = float(cor.proximity_penalty(b.xs, b.ys, b.vehicle, cfg)[0])
        assert st == bool(bt[0])
        if sp == 0.0 and bp == 0.0:
            max_rew0 = max(max_rew0, abs(srr - float(br[0])))
        if st or stx:
            break
    assert max_obs < 1e-6, f"obs parity {max_obs}"
    assert max_rew0 < 1e-9, f"reward parity (proximity-free) {max_rew0}"


@pytest.mark.parametrize("cfg", [truck_config(), lab_config()])
def test_lidar_close_to_grid(cfg):
    rng = np.random.default_rng(3)
    xx, yy, _, _ = pathgen.generate_path(rng, cfg.world, cfg.path)
    grid, meta = build_occupancy_grid(xx, yy, cfg.world)
    P = len(xx); diffs = []
    for _ in range(40):
        i = int(rng.integers(int(0.1 * P), int(0.85 * P)))
        px, py, yaw = xx[i], yy[i] + rng.uniform(-1.0, 1.0), rng.uniform(-0.3, 0.3)
        sd = scal_ray(grid, meta, Pose(px, py, yaw), num_sensors=cfg.obs.lidar_beams,
                      fov_deg=cfg.obs.lidar_fov_deg, max_range_m=cfg.obs.lidar_range_m,
                      step_m=cfg.obs.lidar_step_m)
        bd = np.asarray(bat_ray(xp.asarray(xx), xp.asarray(yy[None]), cfg,
                                xp.asarray([px]), xp.asarray([py]), xp.asarray([yaw])))[0]
        diffs.append(np.mean(np.abs(sd - bd)))
    assert np.mean(diffs) < 0.02  # ~a few cells on a [0,1] normalized range


@pytest.mark.parametrize("cfg", [truck_config(), lab_config()])
def test_collision_agreement(cfg):
    rng = np.random.default_rng(4)
    xx, yy, _, _ = pathgen.generate_path(rng, cfg.world, cfg.path)
    grid, meta = build_occupancy_grid(xx, yy, cfg.world)
    P = len(xx); agree = tot = 0

    def scoll(v):
        pts = []
        tl = cfg.vehicle.tractor_length_m
        for f in (-0.5, 0.0, 0.5):
            pts.append((v.x + f * tl * np.cos(v.p), v.y + f * tl * np.sin(v.p)))
        t = v.trailer
        for f in (0.0, 0.5, 1.0):
            pts.append((t.x + f * t.L * np.cos(t.yaw), t.y + f * t.L * np.sin(t.yaw)))
        for xq, yq in pts:
            gx = int((xq - meta.origin_x) / meta.res_m); gy = int((yq - meta.origin_y) / meta.res_m)
            if gx < 0 or gx >= meta.width or gy < 0 or gy >= meta.height:
                return True
            if grid[gy, gx] == 100:
                return True
        return False

    for _ in range(300):
        i = int(rng.integers(int(0.1 * P), int(0.85 * P)))
        px, py, yaw = xx[i], yy[i] + rng.uniform(-cfg.world.corridor_half_m * 1.3,
                                                 cfg.world.corridor_half_m * 1.3), rng.uniform(-0.5, 0.5)
        sv = StateSpaceTractorTrailer(cfg.vehicle); sv.reset(3.0, x=px, y=py, p=yaw)
        bv = BatchedTractorTrailer(cfg, 1)
        bv.reset_where(xp.asarray([True]), xp.asarray([3.0]), xp.asarray([px]),
                       xp.asarray([py]), xp.asarray([yaw]))
        agree += (scoll(sv) == bool(cor.collision(xp.asarray(xx), xp.asarray(yy[None]), bv, cfg)[0]))
        tot += 1
    assert agree / tot >= 0.95
