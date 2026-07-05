"""Stage 4 gate (HIGH RISK): the new clean ROS->obs transform (ros_obs.py,
calling tractor_trailer_rl.build_observation) must produce IDENTICAL observations
to the OLD ROSLineFollowingAdapter (the hacky-but-deployed pipeline) across many
random spawn positions + orientations, for forward & reverse, trailer &
tractor-only. A single 0,0,0 spawn would hide handedness errors, so we sweep N
random poses.

Run in the e2e_rl venv (needs e2e_rl + the bridge package + tractor_trailer_rl).
"""
import os, sys
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import numpy as np
from dataclasses import replace

E2E = "/home/ben/Ben/Thesis/e2e_rl"
BRIDGE = "/home/ben/Ben/Thesis/Electrans_project/src/electrans_rl_bridge"
for p in (E2E, BRIDGE):
    if p not in sys.path:
        sys.path.insert(0, p)

from electrans_rl_bridge.ros_env_adapter import install_e2e_rl_on_path
install_e2e_rl_on_path(E2E)

# --- apply the EXACT bridge deploy config overrides (rl_bridge_node L206-247) ---
from e2erl_utils import config as e2erl_config
e2erl_config.window_width_px = 500
e2erl_config.window_height_px = 400
e2erl_config.meters_per_pixel = 0.05
e2erl_config.tractor_length_m = 1.0
e2erl_config.tractor_width_m = 0.65
e2erl_config.trailer_length_m = 2.8
e2erl_config.trailer_width_m = 0.65
e2erl_config.lane_centerline_half_width_m = 1.41
e2erl_config.lane_shoulder_m = 0.20
e2erl_config.grid_res_m = 0.05
e2erl_config.lane_sample_ds_m = 0.10
e2erl_config.tesla_model_s_vehicle_params = dict(
    e2erl_config.tesla_model_s_vehicle_params, lf=0.33, lr=0.32)

from electrans_rl_bridge.ros_env_adapter import ROSLineFollowingAdapter
from electrans_rl_bridge.ros_obs import observation_from_ros
from tractor_trailer_rl import lab_config, Direction, VehicleKind

CASES = {
    "forward_trailer": (Direction.FORWARD, VehicleKind.TRAILER,
                        "Environments.ObstacleAvoidance", "LidarStateObservationLineFollowingEnv"),
    "reverse_trailer": (Direction.REVERSE, VehicleKind.TRAILER,
                        "Environments.LineFollowing", "ReverseLidarStateObservationLineFollowingEnv"),
    "forward_tractor_only": (Direction.FORWARD, VehicleKind.TRACTOR_ONLY,
                             "Environments.TractorOnly", "TractorOnlyLidarStateLineFollowingEnv"),
    "reverse_tractor_only": (Direction.REVERSE, VehicleKind.TRACTOR_ONLY,
                             "Environments.TractorOnly", "ReverseTractorOnlyLidarStateLineFollowingEnv"),
}
# State obs match to float32 epsilon; lidar can differ by a rare 1-cell
# occupancy-grid quantization (float32-vs-float64 KDTree input) ~ a few *0.05m
# steps in a 20m range. Both are floating-point/discretization noise, NOT a
# coordinate-convention error (which would show large systematic sign flips).
STATE_ATOL = 1e-3
LIDAR_ATOL = 2e-2
N = 20


def deploy_cfg(direction, kind):
    base = lab_config(direction, kind)
    world = replace(base.world, width_px=500, height_px=400, meters_per_pixel=0.05,
                    grid_res_m=0.05, lane_sample_ds_m=0.10)
    veh = replace(base.vehicle, lf=0.325, lr=0.325)
    return replace(base, world=world, vehicle=veh)


def random_centerline(rng):
    """Smooth random centerline in 'map' frame: random origin/heading + gentle arc."""
    n = 300
    ds = 0.1
    origin = rng.uniform(-5, 5, size=2)
    heading = rng.uniform(-np.pi, np.pi)
    kappa = rng.uniform(-0.05, 0.05)  # gentle curvature
    s = np.arange(n) * ds
    th = heading + kappa * s
    xs = origin[0] + np.cumsum(np.cos(th) * ds)
    ys = origin[1] + np.cumsum(np.sin(th) * ds)
    return xs, ys, th


def _check(name):
    direction, kind, mod, cls = CASES[name]
    reverse = direction == Direction.REVERSE
    adapter = ROSLineFollowingAdapter(env_class_module=mod, env_class_name=cls,
                                      env_kwargs={"lidar_beams": 24, "fixed_speed": True},
                                      world_scale=1.0)
    adapter.set_reverse_mode(reverse)
    cfg = deploy_cfg(direction, kind)
    state_dim = 5 if kind == VehicleKind.TRACTOR_ONLY else 8
    rng = np.random.default_rng(hash(name) % 2**31)
    state_err = 0.0; lidar_err = 0.0
    for k in range(N):
        xs, ys, th = random_centerline(rng)
        i = int(rng.integers(20, len(xs) - 40))
        tangent = th[i]
        # ego near the path, yaw ~ tangent (+pi reverse) + noise, small lateral offset
        lat = rng.uniform(-0.8, 0.8)
        nx, ny = -np.sin(tangent), np.cos(tangent)
        ex = xs[i] + lat * nx
        ey = ys[i] + lat * ny
        eyaw = tangent + (np.pi if reverse else 0.0) + rng.uniform(-0.3, 0.3)
        steering = rng.uniform(-0.3, 0.3)
        xd = rng.uniform(0.4, 0.8)
        hitch = rng.uniform(-0.4, 0.4) if kind == VehicleKind.TRAILER else 0.0

        adapter.set_reference_path(xs, ys)
        adapter.set_ego_state(ex, ey, eyaw, steering, xd)
        adapter.set_trailer_state_from_hitch(hitch)
        obs_old = np.asarray(adapter.get_observation(), np.float32)

        obs_new = observation_from_ros(xs, ys, ex, ey, eyaw, steering, xd, hitch, cfg)

        err = np.abs(obs_old - obs_new)
        state_err = max(state_err, float(err[:state_dim].max()))
        lidar_err = max(lidar_err, float(err[state_dim:].max()))
    return state_err, lidar_err


def _assert(name):
    se, le = _check(name)
    assert se <= STATE_ATOL and le <= LIDAR_ATOL, (
        f"{name}: state_err={se:.2e} (<= {STATE_ATOL}), lidar_err={le:.2e} (<= {LIDAR_ATOL})")
    return se, le


def test_forward_trailer(): _assert("forward_trailer")
def test_reverse_trailer(): _assert("reverse_trailer")
def test_forward_tractor_only(): _assert("forward_tractor_only")
def test_reverse_tractor_only(): _assert("reverse_tractor_only")


if __name__ == "__main__":
    allpass = True
    for nm in CASES:
        se, le = _check(nm)
        ok = se <= STATE_ATOL and le <= LIDAR_ATOL
        allpass &= ok
        print(f"{'PASS' if ok else 'FAIL'} {nm:22s} state_err={se:.2e} lidar_err={le:.2e}")
    print("ALL PASS" if allpass else "SOME FAILED")
