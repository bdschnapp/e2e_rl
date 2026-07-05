"""Stage 0: dump e2e_rl oracle fixtures for the obs-parity gate.

Runs in the e2e_rl venv. Applies the SAME lab monkeypatches the production
pipeline uses (so the recorded obs IS the deployed obs pipeline), drives each of
the 4 concrete lidar_24 envs for K steps, and records per-step vehicle/trailer
state + the env's obs vector + the (per-episode) centerline and occupancy grid.

The parity test (test_obs_parity.py) then feeds the SAME recorded state into the
new tractor_trailer_rl.build_observation and asserts the obs match within 1e-5.
"""
import os, sys
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import numpy as np

E2E = "/home/ben/Ben/Thesis/e2e_rl"
SCRIPTS = "/home/ben/Ben/Thesis/Electrans_project/src/electrans_rl_bridge/scripts"
for p in (E2E, SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)

import train_lab_model as tlm

# Apply the obs-relevant lab patches (config dims, vehicle params, path gen).
tlm._apply_lab_config_overrides(__import__("pathlib").Path(E2E))
tlm._patch_env_vehicle_params(__import__("pathlib").Path(E2E))
tlm._patch_path_generator(__import__("pathlib").Path(E2E), mild_only=True)

import Environments.LineFollowing as lf
import Environments.ObstacleAvoidance as oa
import Environments.TractorOnly as to

ENVS = {
    "forward_trailer": lambda: oa.LidarStateObservationLineFollowingEnv(
        render_mode=None, lidar_beams=24, reward_mode="multiplicative"),
    "reverse_trailer": lambda: lf.ReverseLidarStateObservationLineFollowingEnv(
        render_mode=None, lidar_beams=24, reward_mode="multiplicative"),
    "forward_tractor_only": lambda: to.TractorOnlyLidarStateLineFollowingEnv(
        render_mode=None, lidar_beams=24, reward_mode="multiplicative"),
    "reverse_tractor_only": lambda: to.ReverseTractorOnlyLidarStateLineFollowingEnv(
        render_mode=None, lidar_beams=24, reward_mode="multiplicative"),
}

OUT = os.path.join(os.path.dirname(__file__), "fixtures")
os.makedirs(OUT, exist_ok=True)
K = 25

for name, make in ENVS.items():
    env = make()
    obs, _ = env.reset(seed=7)
    v = env.vehicle
    is_reverse = "reverse" in name
    reward_modes = (["dense", "no_hitch", "multiplicative"] if is_reverse
                    else ["dense", "tractor_focus", "multiplicative"])
    rec = {k: [] for k in ["vx", "vy", "vp", "vs", "vxd", "tx", "ty", "tyaw", "obs"]}
    for m in reward_modes:
        rec[f"reward_{m}"] = []
    # per-episode constants (path + occupancy are fixed within an episode)
    xx = np.asarray(env.xx, float).copy()
    yy = np.asarray(env.yy, float).copy()
    occ = np.asarray(env.occ_grid).copy()
    meta = env.occ_meta
    occ_res = float(meta.res_m)
    occ_origin = np.array([float(meta.origin_x), float(meta.origin_y)], float)

    def snap():
        rec["vx"].append(float(v.x)); rec["vy"].append(float(v.y))
        rec["vp"].append(float(v.p)); rec["vs"].append(float(v.s)); rec["vxd"].append(float(v.xd))
        rec["tx"].append(float(v.trailer.x)); rec["ty"].append(float(v.trailer.y))
        rec["tyaw"].append(float(v.trailer.yaw))
        rec["obs"].append(np.asarray(env._get_obs(), np.float32).copy())
        # per-mode reward at this (current) state, non-terminal running reward
        saved_mode = env.reward_mode
        for m in reward_modes:
            env.reward_mode = m
            rec[f"reward_{m}"].append(float(env._get_reward()))
        env.reward_mode = saved_mode

    # Single episode only (path + occupancy are constant within it). Small
    # actions keep the vehicle on-path so reverse/tractor-only don't terminate
    # in a few steps; stop at termination rather than resetting (a reset would
    # change the path while we keep only one copy of it).
    rng = np.random.default_rng(3)
    snap()  # at reset
    for _ in range(K):
        a = (env.action_space.sample() * 0.1).astype(env.action_space.dtype)
        obs, r, term, trunc, _ = env.step(a)
        snap()
        if term or trunc:
            break
    out = os.path.join(OUT, f"{name}.npz")
    arrays = {k: np.array(v) for k, v in rec.items() if k != "obs"}
    arrays["obs"] = np.array(rec["obs"], np.float32)
    np.savez_compressed(
        out, xx=xx, yy=yy, occ=occ, occ_res=occ_res, occ_origin=occ_origin, **arrays,
    )
    print(f"{name}: {len(rec['obs'])} snapshots, obs_dim={rec['obs'][0].shape}, "
          f"modes={reward_modes}, saved {out}")
print("DONE")
