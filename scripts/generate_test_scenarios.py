"""
Generate and save a fixed set of reproducible test scenarios for benchmarking.

Creates 50 path scenarios stratified by difficulty (straight / mild / aggressive
/ S-curve) plus 150 obstacle configurations (sparse / medium / dense) for the
obstacle avoidance task.  All scenarios are saved as .npz files under
test_scenarios/{forward,reverse,obstacles}/.

Usage
-----
    python scripts/generate_test_scenarios.py [--out test_scenarios]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Make repo root importable when run from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Environments.LineFollowing import StateObservationLineFollowingEnv
from Environments.ObstacleAvoidance import ObstacleAvoidanceEnv


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

# Each entry: (tag, seed_offset, n_paths, y_scale)
# The environment's generate_path() multiplies 6 random y ∈ [-1,1] by 5 m.
# We inject scaled control-point overrides after reset to achieve the desired
# curvature difficulty.  The seed selects the random state for everything else.
DIFFICULTY_SPECS = [
    ("straight",    0,   10, 0.1),   # seeds 0-9,   y scaled to ±0.5 m
    ("mild",       10,   15, 0.5),   # seeds 10-24, y scaled to ±2.5 m
    ("aggressive", 25,   15, 1.0),   # seeds 25-39, y ∈ full range ±5 m
    ("scurve",     40,   10, None),  # seeds 40-49, S-curve (manual control points)
]

TOTAL_PATHS = 50

# Obstacle densities for the obstacle task
OBSTACLE_DENSITIES = [
    ("sparse",  3),
    ("medium",  6),
    ("dense",  10),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_forward_env():
    return StateObservationLineFollowingEnv(render_mode=None, max_episode_steps=1000)


def _make_obstacle_env():
    env = ObstacleAvoidanceEnv(render_mode=None, max_episode_steps=1000)
    return env


def _patch_path_yscale(env, seed: int, y_scale: float):
    """
    Reset the env with the given seed so generate_path() uses self.np_random,
    then re-scale the path y-offsets around the centerline to hit the desired
    curvature difficulty.

    The cubic spline in generate_path() uses random y ∈ [-1, 1] then multiplies
    by 5.  We reset with the seed (which now correctly seeds self.np_random),
    capture the path, then re-scale the deviation from the mean y.
    """
    env.reset(seed=seed)
    mean_y = np.mean(env.yy)
    env.yy = mean_y + (env.yy - mean_y) * y_scale
    return env


def _make_scurve_path(env, seed: int):
    """
    Force an S-shaped path by overriding the spline control points directly.
    The environment's xx array is preserved; yy is recomputed.
    """
    from scipy.interpolate import CubicSpline

    env.reset(seed=seed)
    x0, x_end = env.xx[0], env.xx[-1]
    x_ctrl = np.linspace(x0, x_end, 6)
    # Alternating sign to create S-curve; amplitude ≈ ±5 m
    y_ctrl = np.array([0.0, 1.0, -1.0, 1.0, -1.0, 0.0])
    vert_offset = 45.0
    cs = CubicSpline(x_ctrl, y_ctrl, bc_type=((1, 0.0), "not-a-knot"))
    env.yy = cs(env.xx) * 5 + vert_offset
    return env


def _extract_scenario(env) -> dict:
    """Return a dict of arrays that fully specify a reproducible scenario."""
    return {
        "xx": env.xx.copy(),
        "yy": env.yy.copy(),
        "vehicle_x": np.array([env.vehicle.x]),
        "vehicle_y": np.array([env.vehicle.y]),
        "vehicle_yaw": np.array([env.vehicle.p]),
        "vehicle_speed": np.array([env.vehicle.xd]),
        "trailer_yaw": np.array([env.vehicle.trailer.yaw]),
        "hitch_angle": np.array([env.vehicle.p - env.vehicle.trailer.yaw]),
        "steering": np.array([env.vehicle.s]),
    }


# ---------------------------------------------------------------------------
# Generation routines
# ---------------------------------------------------------------------------

def generate_forward_scenarios(out_dir: Path):
    """Generate 50 path scenarios for forward and reverse tasks."""
    fwd_dir = out_dir / "forward"
    rev_dir = out_dir / "reverse"
    fwd_dir.mkdir(parents=True, exist_ok=True)
    rev_dir.mkdir(parents=True, exist_ok=True)

    env = _make_forward_env()
    count = 0

    for tag, seed_offset, n_paths, y_scale in DIFFICULTY_SPECS:
        for i in range(n_paths):
            seed = seed_offset + i
            if y_scale is None:
                env = _make_scurve_path(env, seed)
            else:
                env = _patch_path_yscale(env, seed, y_scale)

            scenario = _extract_scenario(env)
            scenario["difficulty"] = np.bytes_(tag)
            scenario["seed"] = np.array([seed])

            fname = f"path_{count:03d}_{tag}.npz"
            np.savez_compressed(fwd_dir / fname, **scenario)
            np.savez_compressed(rev_dir / fname, **scenario)  # same path, reverse controller
            count += 1
            print(f"  [{count:3d}/50] {fname}")

    env.close()
    print(f"Saved {count} forward/reverse scenarios to {fwd_dir} and {rev_dir}")


def generate_obstacle_scenarios(out_dir: Path):
    """Generate 150 obstacle scenarios (50 paths × 3 densities)."""
    obs_dir = out_dir / "obstacles"
    obs_dir.mkdir(parents=True, exist_ok=True)

    env = _make_obstacle_env()
    # Fix obstacle count to the desired density (bypass curriculum)
    count = 0

    # Reload forward paths to use the same paths
    fwd_dir = out_dir / "forward"
    scenario_files = sorted(fwd_dir.glob("path_*.npz"))

    for npz_path in scenario_files:
        base = np.load(npz_path, allow_pickle=False)
        seed = int(base["seed"][0])
        tag = base["difficulty"].tobytes().decode()

        for density_tag, n_obs in OBSTACLE_DENSITIES:
            # Use a deterministic secondary seed for obstacle placement
            obs_seed = seed * 1000 + n_obs
            env.obstacles_low = n_obs
            env.obstacles_high = n_obs + 1   # force exact count
            env.reset(seed=obs_seed)

            # Overwrite the path from the stored forward scenario so paths match
            env.xx = base["xx"].copy()
            env.yy = base["yy"].copy()

            scenario = _extract_scenario(env)
            scenario["difficulty"] = np.bytes_(tag)
            scenario["density"] = np.bytes_(density_tag)
            scenario["n_obstacles"] = np.array([n_obs])
            scenario["seed"] = np.array([seed])
            scenario["obs_seed"] = np.array([obs_seed])

            # Save obstacle list
            if hasattr(env, "obstacles") and env.obstacles:
                obs_arr = np.array(env.obstacles, dtype=np.float32)  # (N, 3): x, y, r
                scenario["obstacles"] = obs_arr

            fname = f"path_{count // 3:03d}_{tag}_{density_tag}.npz"
            np.savez_compressed(obs_dir / fname, **scenario)
            count += 1
            print(f"  [{count:3d}/150] {fname}")

    env.close()
    print(f"Saved {count} obstacle scenarios to {obs_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pre-generate fixed test scenarios")
    parser.add_argument(
        "--out", type=Path, default=Path("test_scenarios"),
        help="Output root directory (default: test_scenarios/)"
    )
    parser.add_argument(
        "--skip-obstacles", action="store_true",
        help="Skip obstacle scenario generation (faster)"
    )
    args = parser.parse_args()

    print(f"Generating scenarios → {args.out.resolve()}")

    print("\n--- Forward / Reverse paths ---")
    generate_forward_scenarios(args.out)

    if not args.skip_obstacles:
        print("\n--- Obstacle configurations ---")
        generate_obstacle_scenarios(args.out)

    print("\nDone.")


if __name__ == "__main__":
    main()
