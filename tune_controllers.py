"""
Controller parameter tuning for tractor-trailer lane following.

For each classical controller (fpp, pid, mpc) this script:
  1. Optimises parameters using scipy differential_evolution (global, derivative-free)
  2. Validates the optimised parameters on a held-out episode set
  3. Generates a 2-D sensitivity sweep over the two most influential gains
     (for thesis figures showing the controller is properly tuned)
  4. Saves optimised parameters to results/tuned_params.json

Usage
-----
    python tune_controllers.py                    # tune all three
    python tune_controllers.py --controllers fpp
    python tune_controllers.py --controllers pid,mpc
    python tune_controllers.py --quick            # fewer episodes, faster

Outputs
-------
    results/tuned_params.json          — best params per controller
    results/sensitivity_fpp.csv        — 2-D grid sweep data (FPP)
    results/sensitivity_pid.csv        — 2-D grid sweep data (PID)
    results/sensitivity_mpc.csv        — 2-D grid sweep data (MPC)
"""

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

sys.path.insert(0, str(Path(__file__).resolve().parent))


# -----------------------------------------------------------------------
# Shared environment / evaluation helpers
# -----------------------------------------------------------------------

def _make_forward_env():
    from Environments.LineFollowing import StateObservationLineFollowingEnv
    return StateObservationLineFollowingEnv(render_mode=None, max_episode_steps=1000)


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
        obs, reward, terminated, truncated, _ = env.step(action)
        logger.log_step(env, action, reward, inference_time_s=elapsed)
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
        obs, reward, terminated, truncated, _ = env.step(action)
        logger.log_step(env, action, reward, inference_time_s=elapsed)
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
        speed = float(env.vehicle.xd) if abs(float(env.vehicle.xd)) > 1e-3 else float(config.initial_xd)
        action = np.array([steer_rate, speed], dtype=np.float32)
        elapsed = time.perf_counter() - t0
        obs, reward, terminated, truncated, _ = env.step(action)
        logger.log_step(env, action, reward, inference_time_s=elapsed)
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
    speed   [2.0, 9.0]   target speed (m/s)
    """
    from controllers.pure_pursuit import PurePursuitController

    env = _make_forward_env()
    rng = np.random.default_rng(0)
    opt_seeds = rng.integers(1000, 9999, size=n_opt_episodes).tolist()
    val_seeds = rng.integers(10000, 19999, size=n_val_episodes).tolist()

    call_count = [0]

    def objective(x):
        k_ff, k_y, k_theta, speed = x
        ctrl = PurePursuitController(k_ff=k_ff, k_y=k_y, k_theta=k_theta, speed=speed)
        summaries = [_run_episode_fpp(env, ctrl, seed=int(s)) for s in opt_seeds]
        score = _objective_score(summaries)
        call_count[0] += 1
        if call_count[0] % 10 == 0:
            print(f"  [FPP opt] eval {call_count[0]:4d}  score={score:.4f}  "
                  f"k_ff={k_ff:.3f} k_y={k_y:.3f} k_theta={k_theta:.3f} speed={speed:.1f}")
        return score

    bounds = [(0.2, 3.0), (0.02, 1.5), (0.1, 3.0), (2.0, 9.0)]
    print("\n[FPP] Starting differential_evolution optimisation …")
    result = differential_evolution(
        objective, bounds,
        seed=42, maxiter=30, popsize=8, tol=1e-3,
        mutation=(0.5, 1.0), recombination=0.7,
        workers=1, disp=False,
    )

    k_ff_opt, k_y_opt, k_theta_opt, speed_opt = result.x
    best_params = dict(k_ff=k_ff_opt, k_y=k_y_opt, k_theta=k_theta_opt, speed=speed_opt)
    print(f"[FPP] Best: {best_params}  (obj={result.fun:.4f})")

    # Validation
    ctrl_best = PurePursuitController(**best_params)
    val_summaries = [_run_episode_fpp(env, ctrl_best, seed=int(s)) for s in val_seeds]
    cte_vals = [s["mean_abs_cte_trailer"] for s in val_summaries]
    comp_rate = sum(s["completed"] for s in val_summaries) / len(val_summaries)
    print(f"[FPP] Validation ({n_val_episodes} episodes): "
          f"CTE={np.mean(cte_vals):.3f}±{np.std(cte_vals):.3f} m  "
          f"completion={comp_rate*100:.1f}%")

    env.close()
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
    speed  [2.0,  9.0]   target speed
    """
    from controllers.pid import PIDLaneController

    env = _make_forward_env()
    rng = np.random.default_rng(1)
    opt_seeds = rng.integers(1000, 9999, size=n_opt_episodes).tolist()
    val_seeds = rng.integers(10000, 19999, size=n_val_episodes).tolist()

    call_count = [0]

    def objective(x):
        Kp, Ki, Kd, Kp_t, k_ff, speed = x
        ctrl = PIDLaneController(Kp=Kp, Ki=Ki, Kd=Kd, Kp_t=Kp_t, k_ff=k_ff, speed=speed)
        summaries = [_run_episode_pid(env, ctrl, seed=int(s)) for s in opt_seeds]
        score = _objective_score(summaries)
        call_count[0] += 1
        if call_count[0] % 10 == 0:
            print(f"  [PID opt] eval {call_count[0]:4d}  score={score:.4f}  "
                  f"Kp={Kp:.3f} Ki={Ki:.4f} Kd={Kd:.3f} k_ff={k_ff:.3f} speed={speed:.1f}")
        return score

    bounds = [(0.02, 2.0), (0.0, 0.2), (0.1, 3.0), (0.0, 1.0), (0.2, 3.0), (2.0, 9.0)]
    print("\n[PID] Starting differential_evolution optimisation …")
    result = differential_evolution(
        objective, bounds,
        seed=42, maxiter=30, popsize=8, tol=1e-3,
        mutation=(0.5, 1.0), recombination=0.7,
        workers=1, disp=False,
    )

    Kp_opt, Ki_opt, Kd_opt, Kp_t_opt, k_ff_opt, speed_opt = result.x
    best_params = dict(Kp=Kp_opt, Ki=Ki_opt, Kd=Kd_opt, Kp_t=Kp_t_opt,
                       k_ff=k_ff_opt, speed=speed_opt)
    print(f"[PID] Best: {best_params}  (obj={result.fun:.4f})")

    ctrl_best = PIDLaneController(**best_params)
    val_summaries = [_run_episode_pid(env, ctrl_best, seed=int(s)) for s in val_seeds]
    cte_vals = [s["mean_abs_cte_trailer"] for s in val_summaries]
    comp_rate = sum(s["completed"] for s in val_summaries) / len(val_summaries)
    print(f"[PID] Validation ({n_val_episodes} episodes): "
          f"CTE={np.mean(cte_vals):.3f}±{np.std(cte_vals):.3f} m  "
          f"completion={comp_rate*100:.1f}%")

    env.close()
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
    from controllers.mpc import TractorTrailerSteeringMPC

    env = _make_forward_env()
    rng = np.random.default_rng(2)
    opt_seeds = rng.integers(1000, 9999, size=n_opt_episodes).tolist()
    val_seeds = rng.integers(10000, 19999, size=n_val_episodes).tolist()

    call_count = [0]

    def objective(x):
        lq_y, lq_psi1, lq_psi2, lr, lp = x
        mpc = TractorTrailerSteeringMPC()
        import numpy as np
        mpc.Q = np.diag([0.0,
                         2000.0 * 10**lq_y,
                         2000.0 * 10**lq_psi1,
                         500.0  * 10**lq_psi2])
        mpc.R = np.diag([1.0 * 10**lr])
        mpc.P = np.diag([1e5 * 10**lp])

        summaries = [_run_episode_mpc(env, mpc, seed=int(s)) for s in opt_seeds]
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

    mpc_best = TractorTrailerSteeringMPC()
    mpc_best.Q = np.array(best_params["Q"])
    mpc_best.R = np.array(best_params["R"])
    mpc_best.P = np.array(best_params["P"])
    val_summaries = [_run_episode_mpc(env, mpc_best, seed=int(s)) for s in val_seeds]
    cte_vals = [s["mean_abs_cte_trailer"] for s in val_summaries]
    comp_rate = sum(s["completed"] for s in val_summaries) / len(val_summaries)
    print(f"[MPC] Validation ({n_val_episodes} episodes): "
          f"CTE={np.mean(cte_vals):.3f}±{np.std(cte_vals):.3f} m  "
          f"completion={comp_rate*100:.1f}%")

    env.close()
    return best_params


# -----------------------------------------------------------------------
# 2-D sensitivity sweeps  (thesis figures)
# -----------------------------------------------------------------------

def sensitivity_sweep_fpp(best: dict, output_path: Path, n_episodes: int = 5):
    """
    2-D grid: k_y × k_theta (the two most influential FPP gains).
    Writes a CSV with columns: k_y, k_theta, mean_cte, completion_rate.
    """
    from controllers.pure_pursuit import PurePursuitController

    env = _make_forward_env()
    rng = np.random.default_rng(99)
    seeds = rng.integers(20000, 29999, size=n_episodes).tolist()

    k_y_vals = np.linspace(0.02, 0.8, 10)
    k_theta_vals = np.linspace(0.1, 2.0, 10)

    rows = []
    total = len(k_y_vals) * len(k_theta_vals)
    done = 0
    for k_y in k_y_vals:
        for k_theta in k_theta_vals:
            ctrl = PurePursuitController(
                k_ff=best["k_ff"], k_y=k_y, k_theta=k_theta, speed=best["speed"]
            )
            sums = [_run_episode_fpp(env, ctrl, seed=int(s)) for s in seeds]
            cte_vals = [s["mean_abs_cte_trailer"] for s in sums]
            comp = sum(s["completed"] for s in sums) / len(sums)
            rows.append({"k_y": k_y, "k_theta": k_theta,
                         "mean_cte": np.mean(cte_vals), "completion_rate": comp})
            done += 1
            if done % 10 == 0:
                print(f"  [FPP sweep] {done}/{total}")

    env.close()
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
    from controllers.pid import PIDLaneController

    env = _make_forward_env()
    rng = np.random.default_rng(100)
    seeds = rng.integers(20000, 29999, size=n_episodes).tolist()

    Kp_vals = np.linspace(0.05, 1.0, 10)
    Kd_vals = np.linspace(0.1, 2.0, 10)

    rows = []
    total = len(Kp_vals) * len(Kd_vals)
    done = 0
    for Kp in Kp_vals:
        for Kd in Kd_vals:
            ctrl = PIDLaneController(
                Kp=Kp, Ki=best["Ki"], Kd=Kd,
                Kp_t=best["Kp_t"], k_ff=best["k_ff"], speed=best["speed"]
            )
            sums = [_run_episode_pid(env, ctrl, seed=int(s)) for s in seeds]
            cte_vals = [s["mean_abs_cte_trailer"] for s in sums]
            comp = sum(s["completed"] for s in sums) / len(sums)
            rows.append({"Kp": Kp, "Kd": Kd,
                         "mean_cte": np.mean(cte_vals), "completion_rate": comp})
            done += 1
            if done % 10 == 0:
                print(f"  [PID sweep] {done}/{total}")

    env.close()
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
    from controllers.mpc import TractorTrailerSteeringMPC

    env = _make_forward_env()
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
            mpc = TractorTrailerSteeringMPC()
            Q = np.array(best["Q"])
            Q[2, 2] = q_psi1_base * 10**lq
            mpc.Q = Q
            mpc.R = np.diag([r_base * 10**lr])
            mpc.P = np.array(best["P"])

            sums = [_run_episode_mpc(env, mpc, seed=int(s)) for s in seeds]
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

    env.close()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        fieldnames = ["log10_q_psi1_offset", "log10_r_offset", "q_psi1", "r",
                      "mean_cte", "completion_rate"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[MPC] Sensitivity sweep saved → {output_path}")


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Tune classical controllers for tractor-trailer")
    parser.add_argument(
        "--controllers",
        default="fpp,pid,mpc",
        help="Comma-separated list to tune: fpp, pid, mpc  (default: all)",
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
    args = parser.parse_args()

    to_tune = [c.strip() for c in args.controllers.split(",")]
    n_opt = 4 if args.quick else 8
    n_val = 10 if args.quick else 20
    n_sweep = 3 if args.quick else 5
    mpc_n_opt = 3 if args.quick else 5

    args.output.mkdir(parents=True, exist_ok=True)
    params_path = args.output / "tuned_params.json"

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

        with open(params_path, "w") as f:
            json.dump(all_params, f, indent=2)
        print(f"\nTuned parameters saved → {params_path}")

    # Sensitivity sweeps
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

    print("\nDone.")
    if all_params:
        print(f"\nFinal tuned parameters:")
        for ctrl, params in all_params.items():
            print(f"  {ctrl}: {params}")
    print(f"\nNext step: python benchmark.py --task forward "
          f"--controllers fpp,pid,mpc --tuned_params {params_path}")


if __name__ == "__main__":
    main()
