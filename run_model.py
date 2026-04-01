"""
Run a trained TD3 model or a traditional controller against any environment.

Usage
-----
    # RL model — auto-resolve path from scenario/obs/reward/encoder
    python run_model.py --scenario forward --obs state --reward dense
    python run_model.py --scenario reverse --obs bev --encoder ae_frozen

    # RL model — explicit path
    python run_model.py --scenario forward --obs bev --encoder unet_frozen \
        --model ./models/forward/bev_unet_frozen/dense/best_model.zip

    # Traditional controllers
    python run_model.py --scenario forward --obs state --controller fpp
    python run_model.py --scenario reverse --obs state --controller rpp
    python run_model.py --scenario forward --obs state --controller pid
    python run_model.py --scenario reverse --obs state --controller pid
    python run_model.py --scenario forward --obs state --controller mpc
    python run_model.py --scenario reverse --obs state --controller mpc

Arguments match train.py exactly: --scenario, --obs, --reward, --encoder,
--encoder_path, --lidar_beams.  The --model flag overrides the auto-resolved
model path.

Controller/scenario compatibility
----------------------------------
    fpp   forward scenarios only  (PurePursuitController)
    rpp   reverse scenarios only  (ReverseHitchPurePursuitController)
    pid   any scenario — uses PIDLaneController (forward) or
          ReverseHitchPIDController (reverse) based on --scenario
    mpc   any scenario — uses TractorTrailerSteeringMPC (forward) or
          ReverseTractorTrailerMPC (reverse) based on --scenario
"""

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from train import (
    ENCODER_MODES,
    _PRETRAIN_ENCODERS,
    _BEV_ENCODERS,
    _UNET_ENCODERS,
    make_env,
    make_policy_kwargs,
    is_bev_obs,
    uses_obstacles,
)
import e2erl_utils.config as config
from sim_config import (
    CONTROLLER_CHOICES,
    RunModelConfig,
    SCENARIOS,
    OBS_CHOICES,
    add_config_argument,
    apply_config_file_defaults,
    obs_tag,
    validate_run_model_config,
)
from Environments.LineFollowing import (
    FORWARD_REWARD_MODES,
    REVERSE_REWARD_MODES,
    forward_pure_pursuit,
    reverse_pure_pursuit,
    compute_curvature,
)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resolve_model_path(scenario: str, obs: str, reward: str, encoder: str, lidar_beams: int = 16) -> Path:
    """Return the default best_model path under ./models/."""
    tag = obs_tag(obs, encoder, lidar_beams)
    candidate = Path(f"./models/{scenario}/{tag}/{reward}/best_model.zip")
    return candidate


# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------

def _run_episodes(env, predict_fn, n_episodes: int, render: bool, label: str):
    """Generic episode loop. predict_fn(obs) -> action."""
    rewards, lengths = [], []
    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        done = False
        total_reward, steps = 0.0, 0

        while not done:
            action = predict_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            if render:
                env.render()

        rewards.append(total_reward)
        lengths.append(steps)
        print(
            f"[{label}] Episode {ep:03d} | "
            f"Reward: {total_reward:8.2f} | "
            f"Length: {steps:4d}"
        )

    print(
        f"\n[{label}] {n_episodes} episodes — "
        f"mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}, "
        f"mean length: {np.mean(lengths):.1f}"
    )


# ---------------------------------------------------------------------------
# Tuned parameter loading
# ---------------------------------------------------------------------------

_TUNED_PARAMS_PATH = Path(__file__).parent / "controllers" / "tuned_params.json"


def _load_tuned_params() -> dict:
    """Load controllers/tuned_params.json, return empty dict on missing file."""
    if _TUNED_PARAMS_PATH.exists():
        with open(_TUNED_PARAMS_PATH) as f:
            return json.load(f)
    print(f"[warn] {_TUNED_PARAMS_PATH} not found — using controller defaults")
    return {}


def _make_mpc(ctrl_cls, params: dict):
    """Construct an MPC controller and apply tuned Q/R/P matrices if present."""
    ctrl = ctrl_cls()
    if params:
        if "Q" in params:
            ctrl.Q = np.array(params["Q"])
        if "R" in params:
            ctrl.R = np.array(params["R"])
        if "P" in params:
            ctrl.P = np.array(params["P"])
    return ctrl


# ---------------------------------------------------------------------------
# Per-controller episode runners
# ---------------------------------------------------------------------------

def _ep_fpp(env, ctrl, render: bool) -> tuple[float, int]:
    ctrl.reset()
    env.reset()
    done = False
    total_reward, steps = 0.0, 0
    while not done:
        e_y, e_theta = env.get_vehicle_errors()
        kappa = compute_curvature(env, lookahead_steps=10)
        action = ctrl.step(
            e_y=float(e_y), e_theta=float(e_theta),
            kappa=float(kappa), wheelbase=env.vehicle.lf + env.vehicle.lr,
            current_steer=float(env.vehicle.s), dt=float(env.vehicle.dt),
            max_steer_rate=float(np.deg2rad(config.steering_action)),
        )
        _, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated
        if render:
            env.render()
    return total_reward, steps


def _ep_rpp(env, ctrl, render: bool) -> tuple[float, int]:
    ctrl.reset()
    env.reset()
    done = False
    total_reward, steps = 0.0, 0
    while not done:
        psi2 = float(env.vehicle.p - env.vehicle.trailer.yaw)
        e_y_t, e_theta_t = env.get_trailer_errors()
        kappa = compute_curvature(env, lookahead_steps=10)
        action = ctrl.step(
            psi2=psi2,
            e_y_t=float(e_y_t), e_theta_t=float(e_theta_t),
            kappa=float(kappa), current_steer=float(env.vehicle.s),
            dt=float(env.vehicle.dt),
            max_steer_rate=float(np.deg2rad(config.steering_action)),
        )
        _, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated
        if render:
            env.render()
    return total_reward, steps


def _ep_pid_forward(env, ctrl, render: bool) -> tuple[float, int]:
    ctrl.reset()
    env.reset()
    done = False
    total_reward, steps = 0.0, 0
    while not done:
        e_y, e_theta = env.get_vehicle_errors()
        e_y_t, _ = env.get_trailer_errors()
        kappa = compute_curvature(env, lookahead_steps=10)
        action = ctrl.step(
            e_y=float(e_y), e_theta=float(e_theta), e_y_t=float(e_y_t),
            kappa=float(kappa), current_steer=float(env.vehicle.s),
            dt=float(env.vehicle.dt),
            max_steer_rate=float(np.deg2rad(config.steering_action)),
        )
        _, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated
        if render:
            env.render()
    return total_reward, steps


def _ep_pid_reverse(env, ctrl, render: bool) -> tuple[float, int]:
    ctrl.reset()
    env.reset()
    done = False
    total_reward, steps = 0.0, 0
    while not done:
        psi2 = float(env.vehicle.p - env.vehicle.trailer.yaw)
        e_y_t, e_theta_t = env.get_trailer_errors()
        kappa = compute_curvature(env, lookahead_steps=10)
        action = ctrl.step(
            psi2=psi2,
            e_y_t=float(e_y_t), e_theta_t=float(e_theta_t),
            kappa=float(kappa), current_steer=float(env.vehicle.s),
            dt=float(env.vehicle.dt),
            max_steer_rate=float(np.deg2rad(config.steering_action)),
        )
        _, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated
        if render:
            env.render()
    return total_reward, steps


def _ep_mpc_forward(env, mpc, render: bool) -> tuple[float, int]:
    from controllers.mpc_traj_gen import generate_trajectory
    mpc.reset()
    env.reset()
    done = False
    total_reward, steps = 0.0, 0
    prev_steer = 0.0
    while not done:
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
        _, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated
        if render:
            env.render()
    return total_reward, steps


def _ep_mpc_reverse(env, mpc, render: bool) -> tuple[float, int]:
    from controllers.mpc_traj_gen import generate_trajectory
    mpc.reset()
    env.reset()
    done = False
    total_reward, steps = 0.0, 0
    prev_steer = 0.0
    while not done:
        try:
            traj = generate_trajectory(env.xx, env.yy, env.vehicle, reverse=True)
            vx = float(env.vehicle.xd)
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
        action = np.array([steer_rate, -float(config.initial_xd)], dtype=np.float32)
        _, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated
        if render:
            env.render()
    return total_reward, steps


# ---------------------------------------------------------------------------
# Controller dispatcher
# ---------------------------------------------------------------------------

def run_controller(controller: str, scenario: str, obs: str, reward: str,
                   lidar_beams: int, n_episodes: int, render: bool):
    """Run a traditional (non-RL) controller."""
    is_reverse = "reverse" in scenario

    # Validate directional controllers
    if controller == "fpp" and is_reverse:
        raise ValueError("--controller fpp is for forward scenarios; use rpp for reverse.")
    if controller == "rpp" and not is_reverse:
        raise ValueError("--controller rpp is for reverse scenarios; use fpp for forward.")

    env = make_env(scenario, obs, render_mode="human" if render else None,
                   reward=reward, lidar_beams=lidar_beams)
    if uses_obstacles(scenario):
        env.obstacles_low  = 5
        env.obstacles_high = 10

    label = controller.upper()
    rewards, lengths = [], []
    tuned = _load_tuned_params()

    if controller == "fpp":
        from controllers.pure_pursuit import PurePursuitController
        ctrl = PurePursuitController(**tuned.get("fpp", {}))
        run_ep = lambda: _ep_fpp(env, ctrl, render)

    elif controller == "rpp":
        from controllers.pure_pursuit import ReverseHitchPurePursuitController
        ctrl = ReverseHitchPurePursuitController(**tuned.get("rpp", {}))
        run_ep = lambda: _ep_rpp(env, ctrl, render)

    elif controller == "pid":
        if is_reverse:
            from controllers.pid import ReverseHitchPIDController
            ctrl = ReverseHitchPIDController(**tuned.get("pid_reverse", {}))
            run_ep = lambda: _ep_pid_reverse(env, ctrl, render)
        else:
            from controllers.pid import PIDLaneController
            ctrl = PIDLaneController(**tuned.get("pid", {}))
            run_ep = lambda: _ep_pid_forward(env, ctrl, render)

    elif controller == "mpc":
        if is_reverse:
            from controllers.mpc import ReverseTractorTrailerMPC
            ctrl = _make_mpc(ReverseTractorTrailerMPC, tuned.get("mpc_reverse", {}))
            run_ep = lambda: _ep_mpc_reverse(env, ctrl, render)
        else:
            from controllers.mpc import TractorTrailerSteeringMPC
            ctrl = _make_mpc(TractorTrailerSteeringMPC, tuned.get("mpc", {}))
            run_ep = lambda: _ep_mpc_forward(env, ctrl, render)

    else:
        raise ValueError(f"Unknown controller: {controller!r}")

    for ep in range(1, n_episodes + 1):
        r, s = run_ep()
        rewards.append(r)
        lengths.append(s)
        print(f"[{label}] Episode {ep:03d} | Reward: {r:8.2f} | Length: {s:4d}")

    print(
        f"\n[{label}] {n_episodes} episodes — "
        f"mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}, "
        f"mean length: {np.mean(lengths):.1f}"
    )
    env.close()


def run_rl_model(model_path: Path, scenario: str, obs: str, reward: str,
                 encoder: str, encoder_path: str | None,
                 lidar_beams: int, n_episodes: int, render: bool):
    """Load and evaluate a trained TD3 model."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Train first with train.py, or pass --model <explicit_path>."
        )

    # Resolve encoder path for BEV variants with pretrained encoders
    resolved_encoder_path: str | None = None
    if encoder in _PRETRAIN_ENCODERS:
        if encoder_path:
            resolved_encoder_path = encoder_path
        else:
            tag = obs_tag(obs, encoder, lidar_beams)
            enc_type = "ae" if encoder in _BEV_ENCODERS else "unet"
            default_ep = Path(f"./models/{scenario}/{tag}/{reward}/encoder_{enc_type}.pt")
            if default_ep.exists():
                resolved_encoder_path = str(default_ep)

    _, policy_kwargs = make_policy_kwargs(obs, encoder, resolved_encoder_path)

    # BEV environments require VecTransposeImage
    if is_bev_obs(obs):
        def _env_fn():
            e = make_env(scenario, obs, render_mode="human" if render else None,
                         reward=reward, lidar_beams=lidar_beams)
            if uses_obstacles(scenario):
                e.obstacles_low  = 5
                e.obstacles_high = 10
            return Monitor(e)
        load_env = VecTransposeImage(DummyVecEnv([_env_fn]))
    else:
        load_env = make_env(scenario, obs,
                            render_mode="human" if render else None,
                            reward=reward, lidar_beams=lidar_beams)
        if uses_obstacles(scenario):
            load_env.obstacles_low  = 5
            load_env.obstacles_high = 10
        load_env = Monitor(load_env)

    # Import custom extractor classes so SB3 can deserialise them
    if is_bev_obs(obs):
        if encoder in _UNET_ENCODERS:
            from Models.UNetFeatureExtractor import UNetFeatureExtractor  # noqa: F401
        else:
            from Models.CNNFeatureExtractor import CNNFeatureExtractor  # noqa: F401

    model = TD3.load(str(model_path), env=load_env, device="auto")

    label = f"{scenario}/{obs}" + (f"/{encoder}" if obs == "bev" else "")

    if is_bev_obs(obs):
        # VecEnv interface
        rewards, lengths = [], []
        for ep in range(1, n_episodes + 1):
            obs_vec = load_env.reset()
            if render:
                load_env.render()
            done = np.array([False])
            total_reward, steps = 0.0, 0
            while not done[0]:
                action, _ = model.predict(obs_vec, deterministic=True)
                obs_vec, rew, done, _ = load_env.step(action)
                if render:
                    load_env.render()
                total_reward += float(rew[0])
                steps += 1
            rewards.append(total_reward)
            lengths.append(steps)
            print(
                f"[{label}] Episode {ep:03d} | "
                f"Reward: {total_reward:8.2f} | Length: {steps:4d}"
            )
        print(
            f"\n[{label}] {n_episodes} episodes — "
            f"mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}, "
            f"mean length: {np.mean(lengths):.1f}"
        )
    else:
        def predict(o):
            action, _ = model.predict(o, deterministic=True)
            return action

        _run_episodes(load_env, predict, n_episodes, render, label)

    load_env.close()


def run_simulation(config: RunModelConfig):
    """Shared simulation backend used by both CLI and GUI-driven launches."""
    errors = validate_run_model_config(config)
    if errors:
        raise ValueError("\n".join(errors))

    if config.controller:
        run_controller(
            controller=config.controller,
            scenario=config.scenario,
            obs=config.obs,
            reward=config.reward,
            lidar_beams=config.lidar_beams,
            n_episodes=config.episodes,
            render=config.render,
        )
        return

    model_path = (
        Path(config.model)
        if config.model
        else resolve_model_path(config.scenario, config.obs, config.reward, config.encoder, config.lidar_beams)
    )
    run_rl_model(
        model_path=model_path,
        scenario=config.scenario,
        obs=config.obs,
        reward=config.reward,
        encoder=config.encoder,
        encoder_path=config.encoder_path,
        lidar_beams=config.lidar_beams,
        n_episodes=config.episodes,
        render=config.render,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a TD3 model or traditional controller on any scenario."
    )
    add_config_argument(parser)
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS),
    )
    parser.add_argument(
        "--obs",
        choices=list(OBS_CHOICES),
    )
    parser.add_argument(
        "--reward",
        default="dense",
        help=(
            "Reward variant used during training. Forward: "
            f"{', '.join(FORWARD_REWARD_MODES)}. Reverse: "
            f"{', '.join(REVERSE_REWARD_MODES)}. (default: dense)"
        ),
    )
    parser.add_argument(
        "--encoder",
        choices=list(ENCODER_MODES),
        default="scratch",
        help="BEV encoder mode (--obs bev only). Mirrors train.py. (default: scratch)",
    )
    parser.add_argument(
        "--encoder_path",
        default=None,
        help="Path to pretrained encoder weights (.pt). Auto-resolved if omitted.",
    )
    parser.add_argument(
        "--lidar_beams",
        type=int,
        default=16,
        help="Number of lidar beams when --obs lidar (default: 16).",
    )
    parser.add_argument(
        "--controller",
        choices=list(CONTROLLER_CHOICES),
        default=None,
        help=(
            "Run a traditional controller instead of an RL model. "
            "fpp/rpp: pure pursuit (directional). "
            "pid: PIDLaneController (forward) or ReverseHitchPIDController (reverse). "
            "mpc: TractorTrailerSteeringMPC (forward) or ReverseTractorTrailerMPC (reverse)."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Explicit path to a saved model .zip file. "
            "If omitted, auto-resolved to "
            "./models/<scenario>/<obs[_encoder]>/<reward>/best_model.zip."
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes (default: 10).",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable pygame rendering.",
    )
    return parser


def parse_run_model_args(argv=None) -> RunModelConfig:
    parser = build_parser()
    apply_config_file_defaults(parser, argv, RunModelConfig)
    args = parser.parse_args(argv)
    args_dict = vars(args).copy()
    args_dict.pop("config", None)
    config = RunModelConfig.from_dict(args_dict)
    errors = validate_run_model_config(config)
    if errors:
        parser.error("\n".join(errors))
    return config


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = parse_run_model_args()
    print(
        f"\nscenario={cfg.scenario}  obs={cfg.obs}  reward={cfg.reward}  "
        f"encoder={cfg.encoder}  episodes={cfg.episodes}  render={cfg.render}"
    )
    if cfg.controller:
        print(f"controller={cfg.controller}\n")
    else:
        model_path = (
            Path(cfg.model)
            if cfg.model
            else resolve_model_path(cfg.scenario, cfg.obs, cfg.reward, cfg.encoder, cfg.lidar_beams)
        )
        print(f"model={model_path}\n")
    run_simulation(cfg)
