"""
Unified benchmark runner for the E2E-RL tractor-trailer controller.

Runs one or more controllers on a fixed set of pre-generated test scenarios
and exports per-step CSVs and a summary CSV for downstream analysis.

Usage
-----
    # Forward path following — compare TD3 vs pure pursuit vs MPC vs PID
    python benchmark.py --task forward --controllers td3,fpp,pid,mpc \\
        --model models/Phase1/state_only/best_model.zip \\
        --scenarios test_scenarios/

    # Reverse — compare TD3 vs tuned reverse baselines
    python benchmark.py --task reverse --controllers td3,fpp_rev,pid_rev,mpc_rev \\
        --model models/.../best_model.zip

    # Obstacle avoidance (forward)
    python benchmark.py --task obstacle_fwd --controllers td3,fpp \\
        --model models/.../best_model.zip

Prerequisites
-------------
    python scripts/generate_test_scenarios.py   # builds test_scenarios/
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from e2erl_utils.metrics import EpisodeMetricsLogger

_TASK_ALLOWED_CONTROLLERS = {
    "forward": {"td3", "fpp", "pid", "mpc"},
    "reverse": {"td3", "fpp_rev", "pid_rev", "mpc_rev", "rpp"},
    "obstacle_fwd": {"td3"},
    "obstacle_rev": {"td3"},
}

_CONTROLLER_ALIASES = {
    "rpp": "fpp_rev",
}

# -----------------------------------------------------------------------
# Environment factories
# -----------------------------------------------------------------------

def _make_env(
    task: str,
    obs: str = "state",
    encoder: str = "scratch",
    reward: str = "dense",
    lidar_beams: int = 16,
):
    if task == "forward":
        if obs == "state":
            from Environments.LineFollowing import StateObservationLineFollowingEnv
            return StateObservationLineFollowingEnv(
                render_mode=None,
                max_episode_steps=1000,
                reward_mode=reward,
            )
        if obs == "lidar":
            from Environments.ObstacleAvoidance import LidarStateObservationLineFollowingEnv
            return LidarStateObservationLineFollowingEnv(
                render_mode=None,
                max_episode_steps=1000,
                lidar_beams=lidar_beams,
                reward_mode=reward,
            )
        if obs == "bev":
            from Environments.LineFollowing import BevObservationLineFollowingEnv
            return BevObservationLineFollowingEnv(
                render_mode=None,
                max_episode_steps=1000,
                reward_mode=reward,
            )
        raise ValueError(f"Unsupported obs={obs!r} for task={task!r}")
    elif task == "reverse":
        if obs == "state":
            from Environments.LineFollowing import ReverseStateObservationLineFollowingEnv
            return ReverseStateObservationLineFollowingEnv(
                render_mode=None,
                max_episode_steps=1000,
                reward_mode=reward,
            )
        if obs == "lidar":
            from Environments.LineFollowing import ReverseLidarStateObservationLineFollowingEnv
            return ReverseLidarStateObservationLineFollowingEnv(
                render_mode=None,
                max_episode_steps=1000,
                lidar_beams=lidar_beams,
                reward_mode=reward,
            )
        if obs == "bev":
            from Environments.LineFollowing import ReverseBevObservationLineFollowingEnv
            return ReverseBevObservationLineFollowingEnv(
                render_mode=None,
                max_episode_steps=1000,
                reward_mode=reward,
            )
        raise ValueError(f"Unsupported obs={obs!r} for task={task!r}")
    elif task == "obstacle_fwd":
        if obs == "bev":
            from Environments.ObstacleAvoidance import BevObstacleAvoidanceEnv
            env = BevObstacleAvoidanceEnv(
                render_mode=None,
                max_episode_steps=1000,
                reward_mode=reward,
            )
        else:
            from Environments.ObstacleAvoidance import ObstacleAvoidanceEnv
            env = ObstacleAvoidanceEnv(
                render_mode=None,
                max_episode_steps=1000,
                reward_mode=reward,
            )
        env.obstacles_low = 5
        env.obstacles_high = 10
        return env
    elif task == "obstacle_rev":
        if obs == "bev":
            from Environments.ObstacleAvoidance import ReverseBevObstacleAvoidanceEnv
            env = ReverseBevObstacleAvoidanceEnv(
                render_mode=None,
                max_episode_steps=1000,
                reward_mode=reward,
            )
        else:
            from Environments.ObstacleAvoidance import ReverseObstacleAvoidanceEnv
            env = ReverseObstacleAvoidanceEnv(
                render_mode=None,
                max_episode_steps=1000,
                reward_mode=reward,
            )
        env.obstacles_low = 5
        env.obstacles_high = 10
        return env
    else:
        raise ValueError(f"Unknown task: {task}")


def _scenario_dir(scenarios_root: Path, task: str) -> Path:
    mapping = {
        "forward":      "forward",
        "reverse":      "reverse",
        "obstacle_fwd": "obstacles",
        "obstacle_rev": "obstacles",
    }
    return scenarios_root / mapping[task]


# -----------------------------------------------------------------------
# Controller helpers
# -----------------------------------------------------------------------

def _load_td3(model_path: str, env):
    from stable_baselines3 import TD3
    model = TD3.load(model_path, env=env, device="auto")
    dummy_obs = env.observation_space.sample()
    for _ in range(100):
        model.predict(dummy_obs, deterministic=True)
    return model


def _td3_step(model, obs):
    t0 = time.perf_counter()
    action, _ = model.predict(obs, deterministic=True)
    return action, time.perf_counter() - t0


def _apply_env_action(env, action):
    if hasattr(env, "format_action"):
        return env.format_action(action)
    return np.asarray(action, dtype=np.float32).reshape(-1)


def _fpp_step(env, ctrl):
    """Forward pure pursuit step using PurePursuitController."""
    from Environments.LineFollowing import compute_curvature
    import e2erl_utils.config as config

    t0 = time.perf_counter()
    e_y, e_theta = env.get_vehicle_errors()
    kappa = compute_curvature(env, lookahead_steps=10)
    wheelbase = env.vehicle.lf + env.vehicle.lr
    action = ctrl.step(
        e_y=float(e_y),
        e_theta=float(e_theta),
        kappa=float(kappa),
        wheelbase=float(wheelbase),
        current_steer=float(env.vehicle.s),
        dt=float(env.vehicle.dt),
        max_steer_rate=float(np.deg2rad(config.steering_action)),
    )
    return action, time.perf_counter() - t0


def _pid_step(env, ctrl):
    """PID lane-following step using PIDLaneController."""
    from Environments.LineFollowing import compute_curvature
    import e2erl_utils.config as config

    t0 = time.perf_counter()
    e_y, e_theta = env.get_vehicle_errors()
    e_y_t, _ = env.get_trailer_errors()
    kappa = compute_curvature(env, lookahead_steps=10)
    action = ctrl.step(
        e_y=float(e_y),
        e_theta=float(e_theta),
        e_y_t=float(e_y_t),
        kappa=float(kappa),
        current_steer=float(env.vehicle.s),
        dt=float(env.vehicle.dt),
        max_steer_rate=float(np.deg2rad(config.steering_action)),
    )
    return action, time.perf_counter() - t0


def _fpp_reverse_step(env, ctrl):
    """Reverse pure pursuit step using ReverseHitchPurePursuitController."""
    import e2erl_utils.config as config

    t0 = time.perf_counter()
    from Environments.LineFollowing import compute_curvature

    e_y_t, e_theta_t = env.get_trailer_errors()
    hitch = float(env.vehicle.p - env.vehicle.trailer.yaw)
    kappa = compute_curvature(env, lookahead_steps=10)
    action = ctrl.step(
        psi2=hitch,
        e_y_t=float(e_y_t),
        e_theta_t=float(e_theta_t),
        kappa=float(kappa),
        current_steer=float(env.vehicle.s),
        dt=float(env.vehicle.dt),
        max_steer_rate=float(np.deg2rad(config.steering_action)),
    )
    return action, time.perf_counter() - t0


def _pid_reverse_step(env, ctrl):
    """Reverse PID step using ReverseHitchPIDController."""
    import e2erl_utils.config as config
    from Environments.LineFollowing import compute_curvature

    t0 = time.perf_counter()
    e_y_t, e_theta_t = env.get_trailer_errors()
    hitch = float(env.vehicle.p - env.vehicle.trailer.yaw)
    kappa = compute_curvature(env, lookahead_steps=10)
    action = ctrl.step(
        psi2=hitch,
        e_y_t=float(e_y_t),
        e_theta_t=float(e_theta_t),
        kappa=float(kappa),
        current_steer=float(env.vehicle.s),
        dt=float(env.vehicle.dt),
        max_steer_rate=float(np.deg2rad(config.steering_action)),
    )
    return action, time.perf_counter() - t0


def _mpc_step(env, mpc, prev_steer: float):
    """Forward MPC single-step action. Returns (action, elapsed_s, new_prev_steer)."""
    from controllers.mpc_traj_gen import generate_trajectory
    import e2erl_utils.config as config

    t0 = time.perf_counter()
    traj = generate_trajectory(env.xx, env.yy, env.vehicle, horizon=mpc.N)
    vx = float(env.vehicle.xd) if abs(float(env.vehicle.xd)) > 1e-3 else 1.0
    delta_opt = float(mpc.solve(traj, state=(vx, prev_steer)))

    steer_rate = float(np.clip(
        (delta_opt - float(env.vehicle.s)) / float(env.vehicle.dt),
        -np.deg2rad(config.steering_action),
        np.deg2rad(config.steering_action),
    ))
    return np.array([steer_rate], dtype=np.float32), time.perf_counter() - t0, delta_opt


def _mpc_reverse_step(env, mpc, prev_steer: float):
    """Reverse MPC single-step action. Returns (action, elapsed_s, new_prev_steer)."""
    from controllers.mpc_traj_gen import generate_trajectory
    import e2erl_utils.config as config

    t0 = time.perf_counter()
    traj = generate_trajectory(env.xx, env.yy, env.vehicle, reverse=True, horizon=mpc.N)
    vx = float(env.vehicle.xd)
    if abs(vx) < 1e-3:
        vx = -float(config.initial_xd)
    delta_opt = float(mpc.solve(traj, state=(vx, prev_steer)))

    steer_rate = float(np.clip(
        (delta_opt - float(env.vehicle.s)) / float(env.vehicle.dt),
        -np.deg2rad(config.steering_action),
        np.deg2rad(config.steering_action),
    ))
    return np.array([steer_rate], dtype=np.float32), time.perf_counter() - t0, delta_opt


def _normalize_controllers(task: str, controllers: list[str]) -> list[str]:
    normalized = [_CONTROLLER_ALIASES.get(ctrl, ctrl) for ctrl in controllers]
    invalid = [ctrl for ctrl in normalized if ctrl not in _TASK_ALLOWED_CONTROLLERS[task]]
    if invalid:
        allowed = ", ".join(sorted(_TASK_ALLOWED_CONTROLLERS[task]))
        raise ValueError(
            f"Unsupported controllers for task={task!r}: {invalid}. Allowed: {allowed}"
        )
    return normalized


# -----------------------------------------------------------------------
# Scenario loading / patching
# -----------------------------------------------------------------------

def _load_scenario(npz_path: Path) -> dict:
    data = np.load(npz_path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def _patch_env(env, scenario: dict):
    """Overwrite the environment's path (and obstacles if present) with stored scenario."""
    env.xx = scenario["xx"].copy()
    env.yy = scenario["yy"].copy()
    if "obstacles" in scenario and hasattr(env, "obstacles"):
        obs_arr = scenario["obstacles"]
        env.obstacles = [(float(r[0]), float(r[1]), float(r[2])) for r in obs_arr]


# -----------------------------------------------------------------------
# Episode runner
# -----------------------------------------------------------------------

def run_episode(env, controller: str, model=None, mpc=None, fpp=None, pid=None) -> dict:
    """Run one episode and return the summary metrics dict."""
    logger = EpisodeMetricsLogger()
    obs, _ = env.reset()

    # Reset stateful controllers at episode start
    if mpc is not None:
        mpc.reset()
    if fpp is not None:
        fpp.reset()
    if pid is not None:
        pid.reset()

    done = False
    prev_steer = 0.0

    while not done:
        if controller == "td3":
            action, elapsed = _td3_step(model, obs)
        elif controller == "fpp":
            action, elapsed = _fpp_step(env, fpp)
        elif controller == "pid":
            action, elapsed = _pid_step(env, pid)
        elif controller == "fpp_rev":
            action, elapsed = _fpp_reverse_step(env, fpp)
        elif controller == "pid_rev":
            action, elapsed = _pid_reverse_step(env, pid)
        elif controller == "mpc":
            action, elapsed, prev_steer = _mpc_step(env, mpc, prev_steer)
        elif controller == "mpc_rev":
            action, elapsed, prev_steer = _mpc_reverse_step(env, mpc, prev_steer)
        else:
            raise ValueError(f"Unknown controller: {controller!r}")

        applied_action = _apply_env_action(env, action)
        obs, reward, terminated, truncated, _ = env.step(applied_action)
        logger.log_step(env, applied_action, reward, inference_time_s=elapsed)
        done = terminated or truncated

    return logger.compute_summary(terminated=terminated, truncated=truncated)


# -----------------------------------------------------------------------
# Main benchmark loop
# -----------------------------------------------------------------------

def run_benchmark(
    task: str,
    controllers: list,
    scenarios_root: Path,
    model_path: str | None,
    output_dir: Path,
    obs: str = "state",
    encoder: str = "scratch",
    reward: str = "dense",
    lidar_beams: int = 16,
    fpp_params: dict | None = None,
    pid_params: dict | None = None,
    mpc_params: dict | None = None,
):
    scen_dir = _scenario_dir(scenarios_root, task)
    scenario_files = sorted(scen_dir.glob("path_*.npz"))

    if not scenario_files:
        print(f"No scenario files found in {scen_dir}. Run scripts/generate_test_scenarios.py first.")
        sys.exit(1)

    print(f"Task: {task} | Controllers: {controllers} | Scenarios: {len(scenario_files)}")

    env = _make_env(
        task,
        obs=obs,
        encoder=encoder,
        reward=reward,
        lidar_beams=lidar_beams,
    )

    td3_model = None
    mpc_ctrl = None
    fpp_ctrl = None
    pid_ctrl = None

    if "td3" in controllers:
        if not model_path:
            print("--model required for td3 controller")
            sys.exit(1)
        print(f"Loading TD3 model: {model_path}")
        td3_model = _load_td3(model_path, env)

    if any(ctrl in controllers for ctrl in {"mpc", "mpc_rev"}):
        if "mpc_rev" in controllers:
            from controllers.mpc import ReverseTractorTrailerMPC
            mpc_ctrl = ReverseTractorTrailerMPC()
        else:
            from controllers.mpc import TractorTrailerSteeringMPC
            mpc_ctrl = TractorTrailerSteeringMPC()
        if mpc_params:
            for k, v in mpc_params.items():
                setattr(mpc_ctrl, k, v)
        print("MPC controller initialised.")

    if any(ctrl in controllers for ctrl in {"fpp", "fpp_rev"}):
        if "fpp_rev" in controllers:
            from controllers.pure_pursuit import ReverseHitchPurePursuitController
            fpp_ctrl = ReverseHitchPurePursuitController(**(fpp_params or {}))
            print(f"FPP-rev controller: k_hitch={fpp_ctrl.k_hitch:.3f}  k_y={fpp_ctrl.k_y:.3f}  "
                  f"k_theta={fpp_ctrl.k_theta:.3f}  k_ff={fpp_ctrl.k_ff:.3f}")
        else:
            from controllers.pure_pursuit import PurePursuitController
            fpp_ctrl = PurePursuitController(**(fpp_params or {}))
            print(f"FPP controller: k_ff={fpp_ctrl.k_ff:.3f}  k_y={fpp_ctrl.k_y:.3f}  "
                  f"k_theta={fpp_ctrl.k_theta:.3f}")

    if any(ctrl in controllers for ctrl in {"pid", "pid_rev"}):
        if "pid_rev" in controllers:
            from controllers.pid import ReverseHitchPIDController
            pid_ctrl = ReverseHitchPIDController(**(pid_params or {}))
            print(f"PID-rev controller: k_hitch={pid_ctrl.k_hitch:.3f}  Kp={pid_ctrl.Kp:.3f}  "
                  f"Ki={pid_ctrl.Ki:.4f}  Kd={pid_ctrl.Kd:.3f}  k_ff={pid_ctrl.k_ff:.3f}")
        else:
            from controllers.pid import PIDLaneController
            pid_ctrl = PIDLaneController(**(pid_params or {}))
            print(f"PID controller: Kp={pid_ctrl.Kp:.3f}  Ki={pid_ctrl.Ki:.4f}  "
                  f"Kd={pid_ctrl.Kd:.3f}  Kp_t={pid_ctrl.Kp_t:.3f}  "
                  f"k_ff={pid_ctrl.k_ff:.3f}")

    raw_dir = output_dir / "raw" / task
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for scen_idx, npz_path in enumerate(scenario_files):
        scenario = _load_scenario(npz_path)
        difficulty = scenario.get("difficulty", np.bytes_(b"unknown")).tobytes().decode()
        seed = int(scenario.get("seed", np.array([scen_idx]))[0])

        for ctrl in controllers:
            print(
                f"  [{scen_idx+1:3d}/{len(scenario_files)}] "
                f"controller={ctrl:<6} difficulty={difficulty:<12} seed={seed}",
                end="  ", flush=True,
            )

            env.reset(seed=seed)
            _patch_env(env, scenario)

            try:
                summary = run_episode(
                    env, ctrl,
                    model=td3_model, mpc=mpc_ctrl, fpp=fpp_ctrl, pid=pid_ctrl,
                )
            except Exception as exc:
                print(f"ERROR: {exc}")
                summary = {"episode_length": 0, "total_reward": float("nan"), "completed": False}

            summary.update({
                "scenario_idx": scen_idx,
                "scenario_file": npz_path.name,
                "difficulty": difficulty,
                "seed": seed,
                "task": task,
                "controller": ctrl,
            })
            summary_rows.append(summary)

            print(
                f"CTE_trailer={summary.get('mean_abs_cte_trailer', float('nan')):.3f}m  "
                f"jackknife={summary.get('jackknifed', '?')}  "
                f"completed={summary.get('completed', '?')}  "
                f"steps={summary.get('episode_length', 0)}"
            )

    env.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / f"summary_{task}.csv"
    if summary_rows:
        all_keys = sorted({k for r in summary_rows for k in r.keys()})
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            for row in summary_rows:
                writer.writerow({k: row.get(k, "") for k in all_keys})

    print(f"\nSummary saved → {summary_csv}")
    _print_aggregate(summary_rows, controllers)
    return summary_rows


def _print_aggregate(rows: list, controllers: list):
    from collections import defaultdict
    import statistics

    metrics = [
        "mean_abs_cte_trailer", "rms_cte_trailer", "max_abs_cte_trailer",
        "mean_abs_cte_tractor",
        "mean_abs_hitch_angle", "max_abs_hitch_angle",
        "jackknifed", "collided", "completed",
        "episode_length", "mean_inference_ms",
    ]

    grouped: dict = defaultdict(list)
    for row in rows:
        grouped[row["controller"]].append(row)

    col_w = 26
    header = f"{'Metric':<{col_w}}" + "".join(f"{c:>{col_w}}" for c in controllers)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for m in metrics:
        line = f"{m:<{col_w}}"
        for ctrl in controllers:
            vals = [r[m] for r in grouped[ctrl] if m in r and r[m] == r[m]]
            if not vals:
                line += f"{'N/A':>{col_w}}"
            elif isinstance(vals[0], bool):
                line += f"{sum(vals)/len(vals)*100:>{col_w-1}.1f}%"
            else:
                line += f"{statistics.mean(vals):>{col_w}.4f}"
        print(line)
    print("=" * len(header) + "\n")


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark E2E-RL tractor-trailer controller vs classical baselines"
    )
    parser.add_argument(
        "--task",
        choices=["forward", "reverse", "obstacle_fwd", "obstacle_rev"],
        required=True,
    )
    parser.add_argument(
        "--controllers",
        default="td3",
        help="Comma-separated list. Forward: td3,fpp,pid,mpc. Reverse: td3,fpp_rev,pid_rev,mpc_rev. (legacy alias: rpp)",
    )
    parser.add_argument("--model", default=None, help="Path to TD3 .zip model")
    parser.add_argument(
        "--obs",
        choices=["state", "lidar", "bev"],
        default="state",
        help="Observation type for the TD3 benchmark environment (default: state).",
    )
    parser.add_argument(
        "--encoder",
        choices=["scratch", "ae_frozen", "ae_unfrozen", "unet_frozen", "unet_unfrozen"],
        default="scratch",
        help="BEV encoder variant when --obs bev is used.",
    )
    parser.add_argument(
        "--reward",
        default="dense",
        help="Reward mode used to instantiate the benchmark environment (default: dense).",
    )
    parser.add_argument(
        "--lidar_beams",
        type=int,
        default=16,
        help="Number of lidar beams when --obs lidar is used (default: 16).",
    )
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path("test_scenarios"),
        help="Root directory of pre-generated scenarios (default: test_scenarios/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
        help="Output directory for CSVs (default: results/)",
    )
    parser.add_argument(
        "--tuned_params",
        type=Path,
        default=None,
        help="JSON file with tuned controller params from tune_controllers.py",
    )

    args = parser.parse_args()
    try:
        controllers = _normalize_controllers(
            args.task,
            [c.strip() for c in args.controllers.split(",") if c.strip()],
        )
    except ValueError as exc:
        print(exc)
        sys.exit(1)

    fpp_params = pid_params = mpc_params = None
    if args.tuned_params and args.tuned_params.exists():
        import json
        with open(args.tuned_params) as f:
            all_params = json.load(f)
        if args.task == "reverse":
            fpp_params = all_params.get("fpp_rev")
            pid_params = all_params.get("pid_rev")
            mpc_params = all_params.get("mpc_rev")
        else:
            fpp_params = all_params.get("fpp")
            pid_params = all_params.get("pid")
            mpc_params = all_params.get("mpc")
        print(f"Loaded tuned params from {args.tuned_params}")

    run_benchmark(
        task=args.task,
        controllers=controllers,
        scenarios_root=args.scenarios,
        model_path=args.model,
        output_dir=args.output,
        obs=args.obs,
        encoder=args.encoder,
        reward=args.reward,
        lidar_beams=args.lidar_beams,
        fpp_params=fpp_params,
        pid_params=pid_params,
        mpc_params=mpc_params,
    )


if __name__ == "__main__":
    main()
