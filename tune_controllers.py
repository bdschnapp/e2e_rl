"""
Controller parameter tuning for tractor-trailer lane following.

For each classical controller (fpp, pid, mpc) this script:
  1. Optimises parameters using scipy differential_evolution (global, derivative-free)
  2. Validates the optimised parameters on a held-out episode set
  3. Generates a 2-D sensitivity sweep over the two most influential gains
     (for thesis figures showing the controller is properly tuned)
  4. Saves optimised parameters to results/tuned_params.json

Reverse controllers (fpp_rev, pid_rev, mpc_rev) follow the same pipeline but use
ReverseStateObservationLineFollowingEnv and the hitch-stabilising variants.

Usage
-----
    python tune_controllers.py                             # tune all forward controllers
    python tune_controllers.py --controllers fpp
    python tune_controllers.py --controllers pid,mpc
    python tune_controllers.py --controllers fpp_rev,pid_rev,mpc_rev   # reverse only
    python tune_controllers.py --controllers all           # forward + reverse
    python tune_controllers.py --quick                     # fewer episodes, faster

Outputs
-------
    results/tuned_params.json           — best params per controller
    results/sensitivity_fpp.csv         — 2-D grid sweep data (FPP, forward)
    results/sensitivity_pid.csv         — 2-D grid sweep data (PID, forward)
    results/sensitivity_mpc.csv         — 2-D grid sweep data (MPC, forward)
    results/sensitivity_fpp_rev.csv     — 2-D grid sweep data (FPP, reverse)
    results/sensitivity_pid_rev.csv     — 2-D grid sweep data (PID, reverse)
    results/sensitivity_mpc_rev.csv     — 2-D grid sweep data (MPC, reverse)
"""

import argparse
import csv
import json
import sys
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

sys.path.insert(0, str(Path(__file__).resolve().parent))

_N_WORKERS = 1
_EXECUTOR: ProcessPoolExecutor | None = None


# -----------------------------------------------------------------------
# Shared environment / evaluation helpers
# -----------------------------------------------------------------------

def _mpc_available() -> bool:
    try:
        import osqp  # noqa: F401
        return True
    except ModuleNotFoundError:
        return False


def _warn_missing_mpc_dependency():
    warnings.warn(
        "Skipping MPC tuning because the 'osqp' package is not installed in the project venv.",
        RuntimeWarning,
        stacklevel=2,
    )


def _set_parallelism(n_workers: int):
    global _N_WORKERS, _EXECUTOR
    _N_WORKERS = max(1, int(n_workers))
    if _EXECUTOR is not None:
        _EXECUTOR.shutdown(wait=True, cancel_futures=False)
        _EXECUTOR = None
    if _N_WORKERS > 1:
        start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        ctx = mp.get_context(start_method)
        _EXECUTOR = ProcessPoolExecutor(max_workers=_N_WORKERS, mp_context=ctx)


def _make_controller(kind: str, params: dict):
    if kind == "fpp":
        from controllers.pure_pursuit import PurePursuitController
        return PurePursuitController(**params)
    if kind == "pid":
        from controllers.pid import PIDLaneController
        return PIDLaneController(**params)
    if kind == "mpc":
        from controllers.mpc import TractorTrailerSteeringMPC
        ctrl = TractorTrailerSteeringMPC()
        ctrl.Q = np.array(params["Q"], dtype=float)
        ctrl.R = np.array(params["R"], dtype=float)
        ctrl.P = np.array(params["P"], dtype=float)
        return ctrl
    if kind == "fpp_rev":
        from controllers.pure_pursuit import ReverseHitchPurePursuitController
        return ReverseHitchPurePursuitController(**params)
    if kind == "pid_rev":
        from controllers.pid import ReverseHitchPIDController
        return ReverseHitchPIDController(**params)
    if kind == "mpc_rev":
        from controllers.mpc import ReverseTractorTrailerMPC
        ctrl = ReverseTractorTrailerMPC()
        ctrl.Q = np.array(params["Q"], dtype=float)
        ctrl.R = np.array(params["R"], dtype=float)
        ctrl.P = np.array(params["P"], dtype=float)
        return ctrl
    raise ValueError(f"Unknown controller kind: {kind!r}")


def _make_env_for_kind(kind: str):
    return _make_reverse_env() if kind.endswith("_rev") else _make_forward_env()


def _run_episode_by_kind(kind: str, env, ctrl, seed: int) -> dict:
    if kind == "fpp":
        return _run_episode_fpp(env, ctrl, seed=seed)
    if kind == "pid":
        return _run_episode_pid(env, ctrl, seed=seed)
    if kind == "mpc":
        return _run_episode_mpc(env, ctrl, seed=seed)
    if kind == "fpp_rev":
        return _run_episode_fpp_reverse(env, ctrl, seed=seed)
    if kind == "pid_rev":
        return _run_episode_pid_reverse(env, ctrl, seed=seed)
    if kind == "mpc_rev":
        return _run_episode_mpc_reverse(env, ctrl, seed=seed)
    raise ValueError(f"Unknown controller kind: {kind!r}")


def _rollout_worker(kind: str, params: dict, seed: int) -> dict:
    env = _make_env_for_kind(kind)
    try:
        ctrl = _make_controller(kind, params)
        return _run_episode_by_kind(kind, env, ctrl, seed=int(seed))
    finally:
        env.close()


def _evaluate_rollouts(kind: str, params: dict, seeds: list[int]) -> list[dict]:
    if _EXECUTOR is None or len(seeds) <= 1:
        env = _make_env_for_kind(kind)
        try:
            ctrl = _make_controller(kind, params)
            return [_run_episode_by_kind(kind, env, ctrl, seed=int(s)) for s in seeds]
        finally:
            env.close()

    futures = [_EXECUTOR.submit(_rollout_worker, kind, params, int(seed)) for seed in seeds]
    return [f.result() for f in futures]

def _make_forward_env():
    from Environments.LineFollowing import StateObservationLineFollowingEnv
    return StateObservationLineFollowingEnv(render_mode=None, max_episode_steps=1000)


def _make_reverse_env():
    from Environments.LineFollowing import ReverseStateObservationLineFollowingEnv
    return ReverseStateObservationLineFollowingEnv(render_mode=None, max_episode_steps=1000)


def _apply_env_action(env, action):
    if hasattr(env, "format_action"):
        return env.format_action(action)
    return np.asarray(action, dtype=np.float32).reshape(-1)


def _run_episode_fpp(env, ctrl, seed=None) -> dict:
    from Environments.LineFollowing import compute_curvature
    import e2erl_utils.config as config
    from e2erl_utils.metrics import EpisodeMetricsLogger

    logger = EpisodeMetricsLogger()
    ctrl.reset()
    obs, _ = env.reset(seed=seed)
    done = False

    while not done:
        import time
        t0 = time.perf_counter()
        e_y, e_theta = env.get_vehicle_errors()
        kappa = compute_curvature(env, lookahead_steps=10)
        action = ctrl.step(
            e_y=float(e_y), e_theta=float(e_theta),
            kappa=float(kappa), wheelbase=env.vehicle.lf + env.vehicle.lr,
            current_steer=float(env.vehicle.s), dt=float(env.vehicle.dt),
            max_steer_rate=float(np.deg2rad(config.steering_action)),
        )
        elapsed = time.perf_counter() - t0
        applied_action = _apply_env_action(env, action)
        obs, reward, terminated, truncated, _ = env.step(applied_action)
        logger.log_step(env, applied_action, reward, inference_time_s=elapsed)
        done = terminated or truncated

    return logger.compute_summary(terminated=terminated, truncated=truncated)


def _run_episode_pid(env, ctrl, seed=None) -> dict:
    from Environments.LineFollowing import compute_curvature
    import e2erl_utils.config as config
    from e2erl_utils.metrics import EpisodeMetricsLogger

    logger = EpisodeMetricsLogger()
    ctrl.reset()
    obs, _ = env.reset(seed=seed)
    done = False

    while not done:
        import time
        t0 = time.perf_counter()
        e_y, e_theta = env.get_vehicle_errors()
        e_y_t, _ = env.get_trailer_errors()
        kappa = compute_curvature(env, lookahead_steps=10)
        action = ctrl.step(
            e_y=float(e_y), e_theta=float(e_theta), e_y_t=float(e_y_t),
            kappa=float(kappa), current_steer=float(env.vehicle.s),
            dt=float(env.vehicle.dt),
            max_steer_rate=float(np.deg2rad(config.steering_action)),
        )
        elapsed = time.perf_counter() - t0
        applied_action = _apply_env_action(env, action)
        obs, reward, terminated, truncated, _ = env.step(applied_action)
        logger.log_step(env, applied_action, reward, inference_time_s=elapsed)
        done = terminated or truncated

    return logger.compute_summary(terminated=terminated, truncated=truncated)


def _run_episode_mpc(env, mpc, seed=None) -> dict:
    from controllers.mpc_traj_gen import generate_trajectory
    import e2erl_utils.config as config
    from e2erl_utils.metrics import EpisodeMetricsLogger

    logger = EpisodeMetricsLogger()
    mpc.reset()
    obs, _ = env.reset(seed=seed)
    done = False
    prev_steer = 0.0

    while not done:
        import time
        t0 = time.perf_counter()
        try:
            traj = generate_trajectory(env.xx, env.yy, env.vehicle)
            vx = float(env.vehicle.xd) if abs(float(env.vehicle.xd)) > 1e-3 else 1.0
            delta_opt = float(mpc.solve(traj, state=(vx, prev_steer)))
            prev_steer = delta_opt
        except Exception:
            delta_opt = prev_steer
        steer_rate = float(np.clip(
            (delta_opt - float(env.vehicle.s)) / float(env.vehicle.dt),
            -np.deg2rad(config.steering_action),
            np.deg2rad(config.steering_action),
        ))
        action = np.array([steer_rate], dtype=np.float32)
        elapsed = time.perf_counter() - t0
        applied_action = _apply_env_action(env, action)
        obs, reward, terminated, truncated, _ = env.step(applied_action)
        logger.log_step(env, applied_action, reward, inference_time_s=elapsed)
        done = terminated or truncated

    return logger.compute_summary(terminated=terminated, truncated=truncated)


def _run_episode_fpp_reverse(env, ctrl, seed=None) -> dict:
    from Environments.LineFollowing import compute_curvature
    import e2erl_utils.config as config
    from e2erl_utils.metrics import EpisodeMetricsLogger

    logger = EpisodeMetricsLogger()
    ctrl.reset()
    obs, _ = env.reset(seed=seed)
    done = False

    while not done:
        import time
        t0 = time.perf_counter()
        psi2 = float(env.vehicle.p - env.vehicle.trailer.yaw)
        e_y_t, e_theta_t = env.get_trailer_errors()
        kappa = compute_curvature(env, lookahead_steps=10)
        action = ctrl.step(
            psi2=psi2,
            e_y_t=float(e_y_t),
            e_theta_t=float(e_theta_t),
            kappa=float(kappa),
            current_steer=float(env.vehicle.s),
            dt=float(env.vehicle.dt),
            max_steer_rate=float(np.deg2rad(config.steering_action)),
        )
        elapsed = time.perf_counter() - t0
        applied_action = _apply_env_action(env, action)
        obs, reward, terminated, truncated, _ = env.step(applied_action)
        logger.log_step(env, applied_action, reward, inference_time_s=elapsed)
        done = terminated or truncated

    return logger.compute_summary(terminated=terminated, truncated=truncated)


def _run_episode_pid_reverse(env, ctrl, seed=None) -> dict:
    from Environments.LineFollowing import compute_curvature
    import e2erl_utils.config as config
    from e2erl_utils.metrics import EpisodeMetricsLogger

    logger = EpisodeMetricsLogger()
    ctrl.reset()
    obs, _ = env.reset(seed=seed)
    done = False

    while not done:
        import time
        t0 = time.perf_counter()
        psi2 = float(env.vehicle.p - env.vehicle.trailer.yaw)
        e_y_t, e_theta_t = env.get_trailer_errors()
        kappa = compute_curvature(env, lookahead_steps=10)
        action = ctrl.step(
            psi2=psi2,
            e_y_t=float(e_y_t),
            e_theta_t=float(e_theta_t),
            kappa=float(kappa),
            current_steer=float(env.vehicle.s),
            dt=float(env.vehicle.dt),
            max_steer_rate=float(np.deg2rad(config.steering_action)),
        )
        elapsed = time.perf_counter() - t0
        applied_action = _apply_env_action(env, action)
        obs, reward, terminated, truncated, _ = env.step(applied_action)
        logger.log_step(env, applied_action, reward, inference_time_s=elapsed)
        done = terminated or truncated

    return logger.compute_summary(terminated=terminated, truncated=truncated)


def _run_episode_mpc_reverse(env, mpc, seed=None) -> dict:
    from controllers.mpc_traj_gen import generate_trajectory
    import e2erl_utils.config as config
    from e2erl_utils.metrics import EpisodeMetricsLogger

    logger = EpisodeMetricsLogger()
    mpc.reset()
    obs, _ = env.reset(seed=seed)
    done = False
    prev_steer = 0.0

    while not done:
        import time
        t0 = time.perf_counter()
        try:
            traj = generate_trajectory(env.xx, env.yy, env.vehicle, reverse=True)
            vx = float(env.vehicle.xd)
            # Ensure vx is negative (reverse); fall back to -initial_xd if near zero
            if abs(vx) < 1e-3:
                vx = -float(config.initial_xd)
            delta_opt = float(mpc.solve(traj, state=(vx, prev_steer)))
            prev_steer = delta_opt
        except Exception:
            delta_opt = prev_steer
        steer_rate = float(np.clip(
            (delta_opt - float(env.vehicle.s)) / float(env.vehicle.dt),
            -np.deg2rad(config.steering_action),
            np.deg2rad(config.steering_action),
        ))
        action = np.array([steer_rate], dtype=np.float32)
        elapsed = time.perf_counter() - t0
        applied_action = _apply_env_action(env, action)
        obs, reward, terminated, truncated, _ = env.step(applied_action)
        logger.log_step(env, applied_action, reward, inference_time_s=elapsed)
        done = terminated or truncated

    return logger.compute_summary(terminated=terminated, truncated=truncated)


def _objective_score(summaries: list) -> float:
    """
    Scalar objective for optimisation (lower = better).
    Penalises mean CTE heavily; adds large penalties for jackknife / collision.
    """
    cte_vals = [s.get("mean_abs_cte_trailer", 10.0) for s in summaries]
    jackknife_rate = sum(s.get("jackknifed", False) for s in summaries) / len(summaries)
    collision_rate = sum(s.get("collided", False) for s in summaries) / len(summaries)
    completion_rate = sum(s.get("completed", False) for s in summaries) / len(summaries)

    mean_cte = float(np.mean(cte_vals))
    return mean_cte + 5.0 * jackknife_rate + 3.0 * collision_rate + 2.0 * (1.0 - completion_rate)


# -----------------------------------------------------------------------
# FPP tuning
# -----------------------------------------------------------------------

def tune_fpp(n_opt_episodes: int = 8, n_val_episodes: int = 20) -> dict:
    """
    Tune PurePursuitController via differential_evolution.

    Parameters optimised
    --------------------
    k_ff    [0.2, 3.0]   curvature feed-forward gain
    k_y     [0.02, 1.5]  lateral error gain
    k_theta [0.1, 3.0]   heading error gain
    """
    rng = np.random.default_rng(0)
    opt_seeds = rng.integers(1000, 9999, size=n_opt_episodes).tolist()
    val_seeds = rng.integers(10000, 19999, size=n_val_episodes).tolist()

    call_count = [0]

    def objective(x):
        k_ff, k_y, k_theta = x
        params = dict(k_ff=k_ff, k_y=k_y, k_theta=k_theta)
        summaries = _evaluate_rollouts("fpp", params, opt_seeds)
        score = _objective_score(summaries)
        call_count[0] += 1
        if call_count[0] % 10 == 0:
            print(f"  [FPP opt] eval {call_count[0]:4d}  score={score:.4f}  "
                  f"k_ff={k_ff:.3f} k_y={k_y:.3f} k_theta={k_theta:.3f}")
        return score

    bounds = [(0.2, 3.0), (0.02, 1.5), (0.1, 3.0)]
    print("\n[FPP] Starting differential_evolution optimisation …")
    result = differential_evolution(
        objective, bounds,
        seed=42, maxiter=30, popsize=8, tol=1e-3,
        mutation=(0.5, 1.0), recombination=0.7,
        workers=1, disp=False,
    )

    k_ff_opt, k_y_opt, k_theta_opt = result.x
    best_params = dict(k_ff=k_ff_opt, k_y=k_y_opt, k_theta=k_theta_opt)
    print(f"[FPP] Best: {best_params}  (obj={result.fun:.4f})")

    # Validation
    val_summaries = _evaluate_rollouts("fpp", best_params, val_seeds)
    cte_vals = [s["mean_abs_cte_trailer"] for s in val_summaries]
    comp_rate = sum(s["completed"] for s in val_summaries) / len(val_summaries)
    print(f"[FPP] Validation ({n_val_episodes} episodes): "
          f"CTE={np.mean(cte_vals):.3f}±{np.std(cte_vals):.3f} m  "
          f"completion={comp_rate*100:.1f}%")

    return best_params


# -----------------------------------------------------------------------
# PID tuning
# -----------------------------------------------------------------------

def tune_pid(n_opt_episodes: int = 8, n_val_episodes: int = 20) -> dict:
    """
    Tune PIDLaneController via differential_evolution.

    Parameters optimised
    --------------------
    Kp     [0.02, 2.0]   proportional gain on tractor e_y
    Ki     [0.0,  0.2]   integral gain on tractor e_y
    Kd     [0.1,  3.0]   derivative / heading gain
    Kp_t   [0.0,  1.0]   trailer CTE gain
    k_ff   [0.2,  3.0]   curvature feed-forward
    """
    rng = np.random.default_rng(1)
    opt_seeds = rng.integers(1000, 9999, size=n_opt_episodes).tolist()
    val_seeds = rng.integers(10000, 19999, size=n_val_episodes).tolist()

    call_count = [0]

    def objective(x):
        Kp, Ki, Kd, Kp_t, k_ff = x
        params = dict(Kp=Kp, Ki=Ki, Kd=Kd, Kp_t=Kp_t, k_ff=k_ff)
        summaries = _evaluate_rollouts("pid", params, opt_seeds)
        score = _objective_score(summaries)
        call_count[0] += 1
        if call_count[0] % 10 == 0:
            print(f"  [PID opt] eval {call_count[0]:4d}  score={score:.4f}  "
                  f"Kp={Kp:.3f} Ki={Ki:.4f} Kd={Kd:.3f} k_ff={k_ff:.3f}")
        return score

    bounds = [(0.02, 2.0), (0.0, 0.2), (0.1, 3.0), (0.0, 1.0), (0.2, 3.0)]
    print("\n[PID] Starting differential_evolution optimisation …")
    result = differential_evolution(
        objective, bounds,
        seed=42, maxiter=30, popsize=8, tol=1e-3,
        mutation=(0.5, 1.0), recombination=0.7,
        workers=1, disp=False,
    )

    Kp_opt, Ki_opt, Kd_opt, Kp_t_opt, k_ff_opt = result.x
    best_params = dict(Kp=Kp_opt, Ki=Ki_opt, Kd=Kd_opt, Kp_t=Kp_t_opt, k_ff=k_ff_opt)
    print(f"[PID] Best: {best_params}  (obj={result.fun:.4f})")

    val_summaries = _evaluate_rollouts("pid", best_params, val_seeds)
    cte_vals = [s["mean_abs_cte_trailer"] for s in val_summaries]
    comp_rate = sum(s["completed"] for s in val_summaries) / len(val_summaries)
    print(f"[PID] Validation ({n_val_episodes} episodes): "
          f"CTE={np.mean(cte_vals):.3f}±{np.std(cte_vals):.3f} m  "
          f"completion={comp_rate*100:.1f}%")

    return best_params


# -----------------------------------------------------------------------
# MPC tuning
# -----------------------------------------------------------------------

def tune_mpc(n_opt_episodes: int = 5, n_val_episodes: int = 20) -> dict:
    """
    Tune MPC cost weights via differential_evolution.

    The optimisation works in log10-space of scale factors relative to the
    MPC defaults, keeping the search bounded and numerically well-conditioned.

    Default weights
    ---------------
    Q = diag([0, 2000, 2000, 500])   (Y, psi1, psi2)
    R = diag([1.0])
    P = diag([1e5])

    Optimised scale factors (log10)
    --------------------------------
    log10_q_y    [-2, 2]  → Q[1,1] = 2000 * 10^x
    log10_q_psi1 [-2, 2]  → Q[2,2] = 2000 * 10^x
    log10_q_psi2 [-2, 2]  → Q[3,3] =  500 * 10^x
    log10_r      [-2, 2]  → R[0,0] =    1 * 10^x
    log10_p      [-2, 2]  → P[0,0] = 1e5  * 10^x
    """
    rng = np.random.default_rng(2)
    opt_seeds = rng.integers(1000, 9999, size=n_opt_episodes).tolist()
    val_seeds = rng.integers(10000, 19999, size=n_val_episodes).tolist()

    call_count = [0]

    def objective(x):
        lq_y, lq_psi1, lq_psi2, lr, lp = x
        params = dict(
            Q=np.diag([0.0,
                       2000.0 * 10**lq_y,
                       2000.0 * 10**lq_psi1,
                       500.0  * 10**lq_psi2]).tolist(),
            R=np.diag([1.0 * 10**lr]).tolist(),
            P=np.diag([1e5 * 10**lp]).tolist(),
        )
        summaries = _evaluate_rollouts("mpc", params, opt_seeds)
        score = _objective_score(summaries)
        call_count[0] += 1
        if call_count[0] % 5 == 0:
            print(f"  [MPC opt] eval {call_count[0]:4d}  score={score:.4f}  "
                  f"lq_y={lq_y:.2f} lq_psi1={lq_psi1:.2f} "
                  f"lq_psi2={lq_psi2:.2f} lr={lr:.2f} lp={lp:.2f}")
        return score

    bounds = [(-2, 2), (-2, 2), (-2, 2), (-2, 2), (-2, 2)]
    print("\n[MPC] Starting differential_evolution optimisation …")
    result = differential_evolution(
        objective, bounds,
        seed=42, maxiter=20, popsize=6, tol=1e-3,
        mutation=(0.5, 1.0), recombination=0.7,
        workers=1, disp=False,
    )

    lq_y_opt, lq_psi1_opt, lq_psi2_opt, lr_opt, lp_opt = result.x
    best_params = dict(
        Q=np.diag([0.0,
                   2000.0 * 10**lq_y_opt,
                   2000.0 * 10**lq_psi1_opt,
                   500.0  * 10**lq_psi2_opt]).tolist(),
        R=np.diag([1.0 * 10**lr_opt]).tolist(),
        P=np.diag([1e5 * 10**lp_opt]).tolist(),
    )
    print(f"[MPC] Best scale factors: lq_y={lq_y_opt:.3f}  lq_psi1={lq_psi1_opt:.3f}  "
          f"lq_psi2={lq_psi2_opt:.3f}  lr={lr_opt:.3f}  lp={lp_opt:.3f}  "
          f"(obj={result.fun:.4f})")

    val_summaries = _evaluate_rollouts("mpc", best_params, val_seeds)
    cte_vals = [s["mean_abs_cte_trailer"] for s in val_summaries]
    comp_rate = sum(s["completed"] for s in val_summaries) / len(val_summaries)
    print(f"[MPC] Validation ({n_val_episodes} episodes): "
          f"CTE={np.mean(cte_vals):.3f}±{np.std(cte_vals):.3f} m  "
          f"completion={comp_rate*100:.1f}%")

    return best_params


# -----------------------------------------------------------------------
# Reverse controller tuning
# -----------------------------------------------------------------------

def tune_fpp_reverse(n_opt_episodes: int = 8, n_val_episodes: int = 20) -> dict:
    """
    Tune ReverseHitchPurePursuitController via differential_evolution.

    Parameters optimised
    --------------------
    k_hitch [0.1, 3.0]  hitch-angle stabilisation gain
    k_y     [0.0, 1.5]  trailer lateral error gain
    k_theta [0.0, 2.0]  trailer heading error gain
    k_ff    [-2.0, 2.0] signed curvature feed-forward (can be negative)
    """
    rng = np.random.default_rng(10)
    opt_seeds = rng.integers(1000, 9999, size=n_opt_episodes).tolist()
    val_seeds = rng.integers(10000, 19999, size=n_val_episodes).tolist()

    call_count = [0]

    def objective(x):
        k_hitch, k_y, k_theta, k_ff = x
        params = dict(k_hitch=k_hitch, k_y=k_y, k_theta=k_theta, k_ff=k_ff)
        summaries = _evaluate_rollouts("fpp_rev", params, opt_seeds)
        score = _objective_score(summaries)
        call_count[0] += 1
        if call_count[0] % 10 == 0:
            print(f"  [FPP-rev opt] eval {call_count[0]:4d}  score={score:.4f}  "
                  f"k_hitch={k_hitch:.3f} k_y={k_y:.3f} k_theta={k_theta:.3f} "
                  f"k_ff={k_ff:.3f}")
        return score

    bounds = [(0.1, 3.0), (0.0, 1.5), (0.0, 2.0), (-2.0, 2.0)]
    print("\n[FPP-rev] Starting differential_evolution optimisation …")
    result = differential_evolution(
        objective, bounds,
        seed=42, maxiter=30, popsize=8, tol=1e-3,
        mutation=(0.5, 1.0), recombination=0.7,
        workers=1, disp=False,
    )

    k_hitch_opt, k_y_opt, k_theta_opt, k_ff_opt = result.x
    best_params = dict(k_hitch=k_hitch_opt, k_y=k_y_opt, k_theta=k_theta_opt, k_ff=k_ff_opt)
    print(f"[FPP-rev] Best: {best_params}  (obj={result.fun:.4f})")

    val_summaries = _evaluate_rollouts("fpp_rev", best_params, val_seeds)
    cte_vals = [s["mean_abs_cte_trailer"] for s in val_summaries]
    comp_rate = sum(s["completed"] for s in val_summaries) / len(val_summaries)
    print(f"[FPP-rev] Validation ({n_val_episodes} episodes): "
          f"CTE={np.mean(cte_vals):.3f}±{np.std(cte_vals):.3f} m  "
          f"completion={comp_rate*100:.1f}%")

    return best_params


def tune_pid_reverse(n_opt_episodes: int = 8, n_val_episodes: int = 20) -> dict:
    """
    Tune ReverseHitchPIDController via differential_evolution.

    Parameters optimised
    --------------------
    k_hitch [0.1, 3.0]  hitch-angle stabilisation gain
    Kp      [0.0, 2.0]  trailer CTE proportional gain
    Ki      [0.0, 0.2]  trailer CTE integral gain
    Kd      [0.0, 2.0]  trailer heading derivative gain
    k_ff    [-2.0, 2.0] signed curvature feed-forward
    """
    rng = np.random.default_rng(11)
    opt_seeds = rng.integers(1000, 9999, size=n_opt_episodes).tolist()
    val_seeds = rng.integers(10000, 19999, size=n_val_episodes).tolist()

    call_count = [0]

    def objective(x):
        k_hitch, Kp, Ki, Kd, k_ff = x
        params = dict(k_hitch=k_hitch, Kp=Kp, Ki=Ki, Kd=Kd, k_ff=k_ff)
        summaries = _evaluate_rollouts("pid_rev", params, opt_seeds)
        score = _objective_score(summaries)
        call_count[0] += 1
        if call_count[0] % 10 == 0:
            print(f"  [PID-rev opt] eval {call_count[0]:4d}  score={score:.4f}  "
                  f"k_hitch={k_hitch:.3f} Kp={Kp:.3f} Ki={Ki:.4f} "
                  f"Kd={Kd:.3f} k_ff={k_ff:.3f}")
        return score

    bounds = [(0.1, 3.0), (0.0, 2.0), (0.0, 0.2), (0.0, 2.0), (-2.0, 2.0)]
    print("\n[PID-rev] Starting differential_evolution optimisation …")
    result = differential_evolution(
        objective, bounds,
        seed=42, maxiter=30, popsize=8, tol=1e-3,
        mutation=(0.5, 1.0), recombination=0.7,
        workers=1, disp=False,
    )

    k_hitch_opt, Kp_opt, Ki_opt, Kd_opt, k_ff_opt = result.x
    best_params = dict(k_hitch=k_hitch_opt, Kp=Kp_opt, Ki=Ki_opt, Kd=Kd_opt, k_ff=k_ff_opt)
    print(f"[PID-rev] Best: {best_params}  (obj={result.fun:.4f})")

    val_summaries = _evaluate_rollouts("pid_rev", best_params, val_seeds)
    cte_vals = [s["mean_abs_cte_trailer"] for s in val_summaries]
    comp_rate = sum(s["completed"] for s in val_summaries) / len(val_summaries)
    print(f"[PID-rev] Validation ({n_val_episodes} episodes): "
          f"CTE={np.mean(cte_vals):.3f}±{np.std(cte_vals):.3f} m  "
          f"completion={comp_rate*100:.1f}%")

    return best_params


def tune_mpc_reverse(n_opt_episodes: int = 5, n_val_episodes: int = 20) -> dict:
    """
    Tune ReverseTractorTrailerMPC cost weights via differential_evolution.

    The hitch-angle weight q_psi2 starts from a much higher base (5000 vs 500)
    because the reverse system is open-loop unstable.

    Optimised scale factors (log10)
    --------------------------------
    log10_q_y    [-2, 2]  → Q[1,1] = 2000  * 10^x
    log10_q_psi1 [-2, 2]  → Q[2,2] = 2000  * 10^x
    log10_q_psi2 [-2, 2]  → Q[3,3] = 5000  * 10^x   (higher base than forward)
    log10_r      [-2, 2]  → R[0,0] =    1  * 10^x
    log10_p      [-2, 2]  → P[0,0] = 1e5   * 10^x
    """
    rng = np.random.default_rng(12)
    opt_seeds = rng.integers(1000, 9999, size=n_opt_episodes).tolist()
    val_seeds = rng.integers(10000, 19999, size=n_val_episodes).tolist()

    call_count = [0]

    def objective(x):
        lq_y, lq_psi1, lq_psi2, lr, lp = x
        params = dict(
            Q=np.diag([0.0,
                       2000.0 * 10**lq_y,
                       2000.0 * 10**lq_psi1,
                       5000.0 * 10**lq_psi2]).tolist(),
            R=np.diag([1.0 * 10**lr]).tolist(),
            P=np.diag([1e5 * 10**lp]).tolist(),
        )
        summaries = _evaluate_rollouts("mpc_rev", params, opt_seeds)
        score = _objective_score(summaries)
        call_count[0] += 1
        if call_count[0] % 5 == 0:
            print(f"  [MPC-rev opt] eval {call_count[0]:4d}  score={score:.4f}  "
                  f"lq_y={lq_y:.2f} lq_psi1={lq_psi1:.2f} "
                  f"lq_psi2={lq_psi2:.2f} lr={lr:.2f} lp={lp:.2f}")
        return score

    bounds = [(-2, 2), (-2, 2), (-2, 2), (-2, 2), (-2, 2)]
    print("\n[MPC-rev] Starting differential_evolution optimisation …")
    result = differential_evolution(
        objective, bounds,
        seed=42, maxiter=20, popsize=6, tol=1e-3,
        mutation=(0.5, 1.0), recombination=0.7,
        workers=1, disp=False,
    )

    lq_y_opt, lq_psi1_opt, lq_psi2_opt, lr_opt, lp_opt = result.x
    best_params = dict(
        Q=np.diag([0.0,
                   2000.0 * 10**lq_y_opt,
                   2000.0 * 10**lq_psi1_opt,
                   5000.0 * 10**lq_psi2_opt]).tolist(),
        R=np.diag([1.0 * 10**lr_opt]).tolist(),
        P=np.diag([1e5 * 10**lp_opt]).tolist(),
    )
    print(f"[MPC-rev] Best scale factors: lq_y={lq_y_opt:.3f}  lq_psi1={lq_psi1_opt:.3f}  "
          f"lq_psi2={lq_psi2_opt:.3f}  lr={lr_opt:.3f}  lp={lp_opt:.3f}  "
          f"(obj={result.fun:.4f})")

    val_summaries = _evaluate_rollouts("mpc_rev", best_params, val_seeds)
    cte_vals = [s["mean_abs_cte_trailer"] for s in val_summaries]
    comp_rate = sum(s["completed"] for s in val_summaries) / len(val_summaries)
    print(f"[MPC-rev] Validation ({n_val_episodes} episodes): "
          f"CTE={np.mean(cte_vals):.3f}±{np.std(cte_vals):.3f} m  "
          f"completion={comp_rate*100:.1f}%")

    return best_params


# -----------------------------------------------------------------------
# 2-D sensitivity sweeps  (thesis figures)
# -----------------------------------------------------------------------

def sensitivity_sweep_fpp(best: dict, output_path: Path, n_episodes: int = 5):
    """
    2-D grid: k_y × k_theta (the two most influential FPP gains).
    Writes a CSV with columns: k_y, k_theta, mean_cte, completion_rate.
    """
    rng = np.random.default_rng(99)
    seeds = rng.integers(20000, 29999, size=n_episodes).tolist()

    k_y_vals = np.linspace(0.02, 0.8, 10)
    k_theta_vals = np.linspace(0.1, 2.0, 10)

    rows = []
    total = len(k_y_vals) * len(k_theta_vals)
    done = 0
    for k_y in k_y_vals:
        for k_theta in k_theta_vals:
            params = dict(k_ff=best["k_ff"], k_y=k_y, k_theta=k_theta)
            sums = _evaluate_rollouts("fpp", params, seeds)
            cte_vals = [s["mean_abs_cte_trailer"] for s in sums]
            comp = sum(s["completed"] for s in sums) / len(sums)
            rows.append({"k_y": k_y, "k_theta": k_theta,
                         "mean_cte": np.mean(cte_vals), "completion_rate": comp})
            done += 1
            if done % 10 == 0:
                print(f"  [FPP sweep] {done}/{total}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["k_y", "k_theta", "mean_cte", "completion_rate"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[FPP] Sensitivity sweep saved → {output_path}")


def sensitivity_sweep_pid(best: dict, output_path: Path, n_episodes: int = 5):
    """
    2-D grid: Kp × Kd (most influential PID gains for lane following).
    """
    rng = np.random.default_rng(100)
    seeds = rng.integers(20000, 29999, size=n_episodes).tolist()

    Kp_vals = np.linspace(0.05, 1.0, 10)
    Kd_vals = np.linspace(0.1, 2.0, 10)

    rows = []
    total = len(Kp_vals) * len(Kd_vals)
    done = 0
    for Kp in Kp_vals:
        for Kd in Kd_vals:
            params = dict(
                Kp=Kp, Ki=best["Ki"], Kd=Kd,
                Kp_t=best["Kp_t"], k_ff=best["k_ff"]
            )
            sums = _evaluate_rollouts("pid", params, seeds)
            cte_vals = [s["mean_abs_cte_trailer"] for s in sums]
            comp = sum(s["completed"] for s in sums) / len(sums)
            rows.append({"Kp": Kp, "Kd": Kd,
                         "mean_cte": np.mean(cte_vals), "completion_rate": comp})
            done += 1
            if done % 10 == 0:
                print(f"  [PID sweep] {done}/{total}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Kp", "Kd", "mean_cte", "completion_rate"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[PID] Sensitivity sweep saved → {output_path}")


def sensitivity_sweep_mpc(best: dict, output_path: Path, n_episodes: int = 4):
    """
    2-D grid: Q_psi1 × R (the most influential MPC weights).
    Sweeps log10-scale factors relative to the optimised base weights.
    """
    rng = np.random.default_rng(101)
    seeds = rng.integers(20000, 29999, size=n_episodes).tolist()

    q_psi1_base = float(np.array(best["Q"])[2, 2])
    r_base = float(np.array(best["R"])[0, 0])

    log_q_psi1_offsets = np.linspace(-1.5, 1.5, 8)
    log_r_offsets = np.linspace(-1.5, 1.5, 8)

    rows = []
    total = len(log_q_psi1_offsets) * len(log_r_offsets)
    done = 0
    for lq in log_q_psi1_offsets:
        for lr in log_r_offsets:
            Q = np.array(best["Q"])
            Q[2, 2] = q_psi1_base * 10**lq
            params = dict(
                Q=Q.tolist(),
                R=np.diag([r_base * 10**lr]).tolist(),
                P=np.array(best["P"]).tolist(),
            )
            sums = _evaluate_rollouts("mpc", params, seeds)
            cte_vals = [s["mean_abs_cte_trailer"] for s in sums]
            comp = sum(s["completed"] for s in sums) / len(sums)
            rows.append({
                "log10_q_psi1_offset": lq, "log10_r_offset": lr,
                "q_psi1": Q[2, 2], "r": r_base * 10**lr,
                "mean_cte": np.mean(cte_vals), "completion_rate": comp,
            })
            done += 1
            if done % 8 == 0:
                print(f"  [MPC sweep] {done}/{total}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        fieldnames = ["log10_q_psi1_offset", "log10_r_offset", "q_psi1", "r",
                      "mean_cte", "completion_rate"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[MPC] Sensitivity sweep saved → {output_path}")


def sensitivity_sweep_fpp_reverse(best: dict, output_path: Path, n_episodes: int = 5):
    """
    2-D grid: k_hitch × k_y for the reverse FPP controller.
    """
    rng = np.random.default_rng(110)
    seeds = rng.integers(20000, 29999, size=n_episodes).tolist()

    k_hitch_vals = np.linspace(0.1, 2.0, 10)
    k_y_vals = np.linspace(0.0, 1.0, 10)

    rows = []
    total = len(k_hitch_vals) * len(k_y_vals)
    done = 0
    for k_hitch in k_hitch_vals:
        for k_y in k_y_vals:
            params = dict(
                k_hitch=k_hitch, k_y=k_y,
                k_theta=best["k_theta"], k_ff=best["k_ff"],
            )
            sums = _evaluate_rollouts("fpp_rev", params, seeds)
            cte_vals = [s["mean_abs_cte_trailer"] for s in sums]
            comp = sum(s["completed"] for s in sums) / len(sums)
            rows.append({"k_hitch": k_hitch, "k_y": k_y,
                         "mean_cte": np.mean(cte_vals), "completion_rate": comp})
            done += 1
            if done % 10 == 0:
                print(f"  [FPP-rev sweep] {done}/{total}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["k_hitch", "k_y", "mean_cte", "completion_rate"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[FPP-rev] Sensitivity sweep saved → {output_path}")


def sensitivity_sweep_pid_reverse(best: dict, output_path: Path, n_episodes: int = 5):
    """
    2-D grid: k_hitch × Kp for the reverse PID controller.
    """
    rng = np.random.default_rng(111)
    seeds = rng.integers(20000, 29999, size=n_episodes).tolist()

    k_hitch_vals = np.linspace(0.1, 2.0, 10)
    Kp_vals = np.linspace(0.0, 1.0, 10)

    rows = []
    total = len(k_hitch_vals) * len(Kp_vals)
    done = 0
    for k_hitch in k_hitch_vals:
        for Kp in Kp_vals:
            params = dict(
                k_hitch=k_hitch, Kp=Kp,
                Ki=best["Ki"], Kd=best["Kd"], k_ff=best["k_ff"],
            )
            sums = _evaluate_rollouts("pid_rev", params, seeds)
            cte_vals = [s["mean_abs_cte_trailer"] for s in sums]
            comp = sum(s["completed"] for s in sums) / len(sums)
            rows.append({"k_hitch": k_hitch, "Kp": Kp,
                         "mean_cte": np.mean(cte_vals), "completion_rate": comp})
            done += 1
            if done % 10 == 0:
                print(f"  [PID-rev sweep] {done}/{total}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["k_hitch", "Kp", "mean_cte", "completion_rate"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[PID-rev] Sensitivity sweep saved → {output_path}")


def sensitivity_sweep_mpc_reverse(best: dict, output_path: Path, n_episodes: int = 4):
    """
    2-D grid: Q_psi2 × R for the reverse MPC (same axes as forward MPC sweep).
    """
    rng = np.random.default_rng(112)
    seeds = rng.integers(20000, 29999, size=n_episodes).tolist()

    q_psi2_base = float(np.array(best["Q"])[3, 3])
    r_base = float(np.array(best["R"])[0, 0])

    log_q_psi2_offsets = np.linspace(-1.5, 1.5, 8)
    log_r_offsets = np.linspace(-1.5, 1.5, 8)

    rows = []
    total = len(log_q_psi2_offsets) * len(log_r_offsets)
    done = 0
    for lq in log_q_psi2_offsets:
        for lr in log_r_offsets:
            Q = np.array(best["Q"])
            Q[3, 3] = q_psi2_base * 10**lq
            params = dict(
                Q=Q.tolist(),
                R=np.diag([r_base * 10**lr]).tolist(),
                P=np.array(best["P"]).tolist(),
            )
            sums = _evaluate_rollouts("mpc_rev", params, seeds)
            cte_vals = [s["mean_abs_cte_trailer"] for s in sums]
            comp = sum(s["completed"] for s in sums) / len(sums)
            rows.append({
                "log10_q_psi2_offset": lq, "log10_r_offset": lr,
                "q_psi2": Q[3, 3], "r": r_base * 10**lr,
                "mean_cte": np.mean(cte_vals), "completion_rate": comp,
            })
            done += 1
            if done % 8 == 0:
                print(f"  [MPC-rev sweep] {done}/{total}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        fieldnames = ["log10_q_psi2_offset", "log10_r_offset", "q_psi2", "r",
                      "mean_cte", "completion_rate"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[MPC-rev] Sensitivity sweep saved → {output_path}")


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

_ALL_FORWARD = ["fpp", "pid", "mpc"]
_ALL_REVERSE = ["fpp_rev", "pid_rev", "mpc_rev"]
_ALL_CONTROLLERS = _ALL_FORWARD + _ALL_REVERSE


def main():
    parser = argparse.ArgumentParser(description="Tune classical controllers for tractor-trailer")
    parser.add_argument(
        "--controllers",
        default="fpp,pid,mpc",
        help=(
            "Comma-separated list to tune: fpp, pid, mpc, fpp_rev, pid_rev, mpc_rev.  "
            "Use 'all' for every controller.  (default: fpp,pid,mpc)"
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use fewer episodes for faster (less accurate) tuning",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
        help="Output directory (default: results/)",
    )
    parser.add_argument(
        "--sweep_only",
        action="store_true",
        help="Skip optimisation, only run sensitivity sweeps using existing tuned_params.json",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=1,
        help="Number of worker processes for per-seed rollout evaluation (default: 1)",
    )
    args = parser.parse_args()

    raw = args.controllers.strip()
    if raw == "all":
        to_tune = _ALL_CONTROLLERS
    else:
        to_tune = [c.strip() for c in raw.split(",")]

    if not _mpc_available():
        requested_mpc = [c for c in to_tune if c in {"mpc", "mpc_rev"}]
        if requested_mpc:
            _warn_missing_mpc_dependency()
            to_tune = [c for c in to_tune if c not in {"mpc", "mpc_rev"}]
            if not to_tune:
                print("Nothing to tune after removing MPC controllers that require 'osqp'.")
                return

    n_opt = 4 if args.quick else 8
    n_val = 10 if args.quick else 20
    n_sweep = 3 if args.quick else 5
    mpc_n_opt = 3 if args.quick else 5

    args.output.mkdir(parents=True, exist_ok=True)
    params_path = args.output / "tuned_params.json"

    _set_parallelism(args.n_workers)
    try:
        print(f"[parallel] rollout workers={_N_WORKERS}")

        # Load existing params if sweeping only
        all_params: dict = {}
        if params_path.exists():
            with open(params_path) as f:
                all_params = json.load(f)

        if not args.sweep_only:
            if "fpp" in to_tune:
                all_params["fpp"] = tune_fpp(n_opt_episodes=n_opt, n_val_episodes=n_val)

            if "pid" in to_tune:
                all_params["pid"] = tune_pid(n_opt_episodes=n_opt, n_val_episodes=n_val)

            if "mpc" in to_tune:
                all_params["mpc"] = tune_mpc(n_opt_episodes=mpc_n_opt, n_val_episodes=n_val)

            if "fpp_rev" in to_tune:
                all_params["fpp_rev"] = tune_fpp_reverse(n_opt_episodes=n_opt, n_val_episodes=n_val)

            if "pid_rev" in to_tune:
                all_params["pid_rev"] = tune_pid_reverse(n_opt_episodes=n_opt, n_val_episodes=n_val)

            if "mpc_rev" in to_tune:
                all_params["mpc_rev"] = tune_mpc_reverse(n_opt_episodes=mpc_n_opt, n_val_episodes=n_val)

            with open(params_path, "w") as f:
                json.dump(all_params, f, indent=2)
            print(f"\nTuned parameters saved → {params_path}")

        # Sensitivity sweeps — forward
        if "fpp" in to_tune and "fpp" in all_params:
            print("\n[FPP] Running sensitivity sweep …")
            sensitivity_sweep_fpp(
                all_params["fpp"],
                args.output / "sensitivity_fpp.csv",
                n_episodes=n_sweep,
            )

        if "pid" in to_tune and "pid" in all_params:
            print("\n[PID] Running sensitivity sweep …")
            sensitivity_sweep_pid(
                all_params["pid"],
                args.output / "sensitivity_pid.csv",
                n_episodes=n_sweep,
            )

        if "mpc" in to_tune and "mpc" in all_params:
            print("\n[MPC] Running sensitivity sweep …")
            sensitivity_sweep_mpc(
                all_params["mpc"],
                args.output / "sensitivity_mpc.csv",
                n_episodes=n_sweep,
            )

        # Sensitivity sweeps — reverse
        if "fpp_rev" in to_tune and "fpp_rev" in all_params:
            print("\n[FPP-rev] Running sensitivity sweep …")
            sensitivity_sweep_fpp_reverse(
                all_params["fpp_rev"],
                args.output / "sensitivity_fpp_rev.csv",
                n_episodes=n_sweep,
            )

        if "pid_rev" in to_tune and "pid_rev" in all_params:
            print("\n[PID-rev] Running sensitivity sweep …")
            sensitivity_sweep_pid_reverse(
                all_params["pid_rev"],
                args.output / "sensitivity_pid_rev.csv",
                n_episodes=n_sweep,
            )

        if "mpc_rev" in to_tune and "mpc_rev" in all_params:
            print("\n[MPC-rev] Running sensitivity sweep …")
            sensitivity_sweep_mpc_reverse(
                all_params["mpc_rev"],
                args.output / "sensitivity_mpc_rev.csv",
                n_episodes=n_sweep,
            )

        print("\nDone.")
        if all_params:
            print(f"\nFinal tuned parameters:")
            for ctrl, params in all_params.items():
                print(f"  {ctrl}: {params}")
        print(
            f"\nNext step: python benchmark.py --task forward "
            f"--controllers fpp,pid,mpc --tuned_params {params_path}"
        )
    finally:
        _set_parallelism(1)


if __name__ == "__main__":
    main()
