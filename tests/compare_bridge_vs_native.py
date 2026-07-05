"""DECISIVE reverse-debug comparison: does the ROS bridge's DEPLOY observation
match the NATIVE ttrl training observation the model actually learned on?

The existing test_bridge_obs_parity only checks adapter == ros_obs (both deploy,
same v22 convention). Neither was ever checked against the native env. Here we
compare, for the SAME physical scenario:
  obs_native = tractor_trailer_rl.build_observation(xs, ys, ego, cfg)   # training obs
  obs_deploy = ROSLineFollowingAdapter.get_observation()               # bridge obs
per state component, for forward (WORKS) and reverse (BROKEN) tractor-only.
A component that sign-flips only in reverse = the bug.

Run in the e2e_rl venv.
"""
import os, sys
os.environ.setdefault("SDL_VIDEODRIVER", "dummy"); os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import numpy as np
from dataclasses import replace

E2E = "/home/ben/Ben/Thesis/e2e_rl"
BRIDGE = "/home/ben/Ben/Thesis/Electrans_project/src/electrans_rl_bridge"
for p in (E2E, BRIDGE):
    if p not in sys.path: sys.path.insert(0, p)
from electrans_rl_bridge.ros_env_adapter import install_e2e_rl_on_path
install_e2e_rl_on_path(E2E)
from e2erl_utils import config as e2erl_config
e2erl_config.window_width_px = 500; e2erl_config.window_height_px = 400
e2erl_config.meters_per_pixel = 0.05; e2erl_config.tractor_length_m = 1.0
e2erl_config.tractor_width_m = 0.65; e2erl_config.trailer_length_m = 2.8
e2erl_config.trailer_width_m = 0.65; e2erl_config.lane_centerline_half_width_m = 1.41
e2erl_config.lane_shoulder_m = 0.20; e2erl_config.grid_res_m = 0.05
e2erl_config.lane_sample_ds_m = 0.10
e2erl_config.tesla_model_s_vehicle_params = dict(
    e2erl_config.tesla_model_s_vehicle_params, lf=0.33, lr=0.32)

from electrans_rl_bridge.ros_env_adapter import ROSLineFollowingAdapter
from tractor_trailer_rl import (lab_config, Direction, VehicleKind,
                                build_observation, EgoState)

STATE = ["steer", "e_y", "e_psi", "k1", "k2"]   # tractor-only layout
CASES = {
    "forward_tractor_only": (Direction.FORWARD, VehicleKind.TRACTOR_ONLY,
                             "Environments.TractorOnly", "TractorOnlyLidarStateLineFollowingEnv"),
    "reverse_tractor_only": (Direction.REVERSE, VehicleKind.TRACTOR_ONLY,
                             "Environments.TractorOnly", "ReverseTractorOnlyLidarStateLineFollowingEnv"),
}

def deploy_cfg(direction, kind):
    base = lab_config(direction, kind)
    world = replace(base.world, width_px=500, height_px=400, meters_per_pixel=0.05,
                    grid_res_m=0.05, lane_sample_ds_m=0.10)
    veh = replace(base.vehicle, lf=0.325, lr=0.325)
    return replace(base, world=world, vehicle=veh)

def scenario(rng):
    """Path flowing roughly +X through the canvas centre; truck near it."""
    n, ds = 120, 0.1
    heading = rng.uniform(-0.3, 0.3)          # near +X so it fits the canvas
    kappa = rng.uniform(-0.04, 0.04)
    s = np.arange(n) * ds
    th = heading + kappa * s
    x0, y0 = 6.0, 10.0
    xs = x0 + np.cumsum(np.cos(th) * ds)
    ys = y0 + np.cumsum(np.sin(th) * ds)
    return xs, ys, th

def run(name):
    direction, kind, mod, cls = CASES[name]
    reverse = direction == Direction.REVERSE
    adapter = ROSLineFollowingAdapter(env_class_module=mod, env_class_name=cls,
                                      env_kwargs={"lidar_beams": 24, "fixed_speed": True})
    adapter.set_reverse_mode(reverse)
    cfg = deploy_cfg(direction, kind)
    rng = np.random.default_rng(7)
    diffs = []; signflip = np.zeros(5)
    nat_all, dep_all = [], []
    lidar_same, lidar_rev = [], []
    for _ in range(40):
        xs, ys, th = scenario(rng)
        i = int(rng.integers(20, len(xs) - 40)); tangent = th[i]
        lat = rng.uniform(-0.8, 0.8)
        nx, ny = -np.sin(tangent), np.cos(tangent)
        ex, ey = xs[i] + lat * nx, ys[i] + lat * ny
        eyaw = tangent + (np.pi if reverse else 0.0) + rng.uniform(-0.25, 0.25)
        steering = rng.uniform(-0.3, 0.3); xd = rng.uniform(0.4, 0.8)

        adapter.set_reference_path(xs, ys)
        adapter.set_ego_state(ex, ey, eyaw, steering, xd)
        adapter.set_trailer_state_from_hitch(0.0)
        dep_full = np.asarray(adapter.get_observation(), np.float64)
        dep = dep_full[:5]

        ego = EgoState(x=float(ex), y=float(ey), yaw=float(eyaw),
                       steer=float(steering), xd=float(xd))
        nat_full = build_observation(xs, ys, ego, None, cfg).vector.astype(np.float64)
        nat = nat_full[:5]

        nat_all.append(nat); dep_all.append(dep)
        diffs.append(np.abs(nat - dep))
        # lidar (indices 5:) -- under a clean Y-flip the beams REVERSE order
        dl, nl = dep_full[5:], nat_full[5:]
        d_same = float(np.abs(dl - nl).mean())
        d_rev = float(np.abs(dl - nl[::-1]).mean())
        lidar_same.append(d_same); lidar_rev.append(d_rev)
    nat_all = np.array(nat_all); dep_all = np.array(dep_all)
    diffs = np.array(diffs)
    print(f"\n=== {name} ===")
    print(f"{'comp':7s} {'max|Δ|':>9s} {'mean|Δ|':>9s} {'corr(nat,dep)':>14s}  verdict")
    for j, nm in enumerate(STATE):
        corr = np.corrcoef(nat_all[:, j], dep_all[:, j])[0, 1] if np.std(dep_all[:, j]) > 1e-9 else float("nan")
        v = "OK" if diffs[:, j].max() < 1e-2 else ("SIGN-FLIP" if corr < -0.5 else "MISMATCH")
        print(f"{nm:7s} {diffs[:,j].max():9.4f} {diffs[:,j].mean():9.4f} {corr:14.3f}  {v}")
    ls, lr = np.mean(lidar_same), np.mean(lidar_rev)
    best = "beams-REVERSED match" if lr < ls else "beams-SAME match"
    print(f"lidar   mean|Δ| same-order={ls:.3f}  reversed-order={lr:.3f}  -> {best} "
          f"(min {min(ls,lr):.3f}; >0.05 = real mismatch)")

if __name__ == "__main__":
    for nm in CASES: run(nm)
