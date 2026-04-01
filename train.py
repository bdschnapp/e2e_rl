"""
Comprehensive training script for all scenarios and observation types.

Usage
-----
    python train.py --scenario forward --obs state --reward dense
    python train.py --scenario reverse --obs lidar --reward multiplicative
    python train.py --scenario forward_obs --obs bev --reward dense
    python train.py --scenario reverse_obs --obs bev --encoder unet_frozen --timesteps 180000

    # Pretrained AE/UNet encoder — auto-pretrained if encoder_path not found:
    python train.py --scenario forward --obs bev --encoder ae_frozen
    python train.py --scenario forward --obs bev --encoder unet_unfrozen --encoder_path ./models/my_encoder.pt

Scenarios
---------
    forward       Forward lane-following, no obstacles
    reverse       Reverse lane-following, no obstacles
    forward_obs   Forward lane-following with obstacles
    reverse_obs   Reverse lane-following with obstacles

Observation types
-----------------
    state    8-dim state vector                          (MlpPolicy)
    lidar    8-dim state + N lidar beams                 (MlpPolicy)
    bev      8-dim state + 84×84 BEV image              (MultiInputPolicy + CNN/UNet)

Encoder modes  (only apply when --obs bev)
------------------------------------------
    scratch       End-to-end CNN from random init  [default]
    ae_frozen     Pretrained NatureCNN AE, frozen during RL
    ae_unfrozen   Pretrained NatureCNN AE, fine-tuned during RL
    unet_frozen   Pretrained UNet encoder, frozen during RL
    unet_unfrozen Pretrained UNet encoder, fine-tuned during RL

    For pretrained variants, encoder weights are loaded from --encoder_path.
    If --encoder_path is omitted (or the file does not exist), the encoder is
    pretrained automatically and saved to ./models/<scenario>/encoder_<type>.pt.

Note: 'state' obs for obstacle scenarios carries no obstacle information —
included for completeness.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecTransposeImage

from Environments.LineFollowing import FORWARD_REWARD_MODES, REVERSE_REWARD_MODES
from Models.CNNFeatureExtractor import CNNFeatureExtractor
from Models.UNetFeatureExtractor import UNetFeatureExtractor
from sim_config import (
    TrainConfig,
    add_config_argument,
    apply_config_file_defaults,
    validate_train_config,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENCODER_MODES = ("scratch", "ae_frozen", "ae_unfrozen", "unet_frozen", "unet_unfrozen")
_BEV_ENCODERS  = {"ae_frozen", "ae_unfrozen"}
_UNET_ENCODERS = {"unet_frozen", "unet_unfrozen"}
_PRETRAIN_ENCODERS = _BEV_ENCODERS | _UNET_ENCODERS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def reward_choices_for_scenario(scenario: str) -> tuple:
    return REVERSE_REWARD_MODES if "reverse" in scenario else FORWARD_REWARD_MODES


def uses_obstacles(scenario: str) -> bool:
    return scenario in ("forward_obs", "reverse_obs")


def is_bev_obs(obs: str) -> bool:
    return obs == "bev"


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(scenario: str, obs: str, render_mode=None, reward: str = "dense",
             lidar_beams: int = 16):
    """Return an unwrapped env for the given (scenario, obs) combination."""

    if scenario == "forward":
        if obs == "state":
            from Environments.LineFollowing import StateObservationLineFollowingEnv
            return StateObservationLineFollowingEnv(render_mode=render_mode, reward_mode=reward)
        if obs == "lidar":
            from Environments.ObstacleAvoidance import LidarStateObservationLineFollowingEnv
            return LidarStateObservationLineFollowingEnv(
                render_mode=render_mode, reward_mode=reward, lidar_beams=lidar_beams)
        if obs == "bev":
            from Environments.LineFollowing import BevObservationLineFollowingEnv
            return BevObservationLineFollowingEnv(render_mode=render_mode, reward_mode=reward)

    elif scenario == "reverse":
        if obs == "state":
            from Environments.LineFollowing import ReverseStateObservationLineFollowingEnv
            return ReverseStateObservationLineFollowingEnv(render_mode=render_mode, reward_mode=reward)
        if obs == "lidar":
            from Environments.LineFollowing import ReverseLidarStateObservationLineFollowingEnv
            return ReverseLidarStateObservationLineFollowingEnv(
                render_mode=render_mode, reward_mode=reward, lidar_beams=lidar_beams)
        if obs == "bev":
            from Environments.LineFollowing import ReverseBevObservationLineFollowingEnv
            return ReverseBevObservationLineFollowingEnv(render_mode=render_mode, reward_mode=reward)

    elif scenario == "forward_obs":
        if obs == "state":
            from Environments.ObstacleAvoidance import ObstacleAvoidanceEnv
            return ObstacleAvoidanceEnv(render_mode=render_mode, reward_mode=reward)
        if obs == "lidar":
            from Environments.ObstacleAvoidance import ObstacleAvoidanceEnv
            return ObstacleAvoidanceEnv(render_mode=render_mode, reward_mode=reward)
        if obs == "bev":
            from Environments.ObstacleAvoidance import BevObstacleAvoidanceEnv
            return BevObstacleAvoidanceEnv(render_mode=render_mode, reward_mode=reward)

    elif scenario == "reverse_obs":
        if obs == "state":
            from Environments.ObstacleAvoidance import ReverseObstacleAvoidanceEnv
            return ReverseObstacleAvoidanceEnv(render_mode=render_mode, reward_mode=reward)
        if obs == "lidar":
            from Environments.ObstacleAvoidance import ReverseObstacleAvoidanceEnv
            return ReverseObstacleAvoidanceEnv(render_mode=render_mode, reward_mode=reward)
        if obs == "bev":
            from Environments.ObstacleAvoidance import ReverseBevObstacleAvoidanceEnv
            return ReverseBevObstacleAvoidanceEnv(render_mode=render_mode, reward_mode=reward)

    raise ValueError(f"Unknown scenario={scenario!r} or obs={obs!r}")


def _make_env_fn(scenario, obs, reward, lidar_beams, rank, base_seed=0,
                 retry_on_failure=False):
    """Return a picklable factory for SubprocVecEnv."""
    def _init():
        env = make_env(scenario, obs, render_mode=None, reward=reward,
                       lidar_beams=lidar_beams)
        if retry_on_failure:
            from Environments.wrappers import RetryOnFailureWrapper
            env = RetryOnFailureWrapper(env)
        env = Monitor(env)
        env.reset(seed=base_seed + rank)
        return env
    return _init


# ---------------------------------------------------------------------------
# Pretraining
# ---------------------------------------------------------------------------

def _default_encoder_path(scenario: str, encoder: str, save_root: Path) -> Path:
    encoder_type = "ae" if encoder in _BEV_ENCODERS else "unet"
    return save_root / f"encoder_{encoder_type}.pt"


def _policy_model_path(scenario: str, obs: str, reward: str, lidar_beams: int = 16) -> Path:
    if obs == "lidar" and lidar_beams != 16:
        obs_tag = f"lidar_{lidar_beams}"
    else:
        obs_tag = obs
    return Path(f"./models/{scenario}/{obs_tag}/{reward}/best_model.zip")


def _lidar_obs_from_bev_env(env, vector_obs: np.ndarray) -> np.ndarray:
    from Environments.ObstacleAvoidance import Pose, get_obstacle_distances

    is_reverse = "reverse" in env.__class__.__name__.lower()
    if is_reverse:
        lidar_pose = Pose(
            x=env.vehicle.trailer.x,
            y=env.vehicle.trailer.y,
            yaw=env.vehicle.trailer.yaw + np.pi,
        )
    else:
        lidar_pose = Pose(
            x=env.vehicle.x,
            y=env.vehicle.y,
            yaw=env.vehicle.p,
        )

    lidar_beams = int(getattr(env, "lidar_beams", 24 if uses_obstacles(env.unwrapped.__class__.__name__.lower()) else 16))
    lidar_distances = get_obstacle_distances(
        env.occ_grid,
        lidar_pose,
        num_sensors=lidar_beams,
    ).astype(np.float32)
    return np.concatenate([vector_obs.astype(np.float32, copy=False), lidar_distances], dtype=np.float32)


def _make_driver_obs_fn(driver_obs: str, env, lidar_beams: int):
    if driver_obs == "state":
        return lambda obs: obs["vector"]
    if driver_obs == "lidar":
        def _obs_fn(obs):
            vector_obs = obs["vector"]
            from Environments.ObstacleAvoidance import Pose, get_obstacle_distances

            is_reverse = hasattr(env.vehicle, "trailer") and "reverse" in env.__class__.__name__.lower()
            if is_reverse:
                lidar_pose = Pose(
                    x=env.vehicle.trailer.x,
                    y=env.vehicle.trailer.y,
                    yaw=env.vehicle.trailer.yaw + np.pi,
                )
            else:
                lidar_pose = Pose(
                    x=env.vehicle.x,
                    y=env.vehicle.y,
                    yaw=env.vehicle.p,
                )

            lidar_distances = get_obstacle_distances(
                env.occ_grid,
                lidar_pose,
                num_sensors=lidar_beams,
            ).astype(np.float32)
            return np.concatenate([vector_obs.astype(np.float32, copy=False), lidar_distances], dtype=np.float32)

        return _obs_fn
    raise ValueError(f"Unsupported driver observation type: {driver_obs!r}")


def resolve_pretrain_driver_models(scenario: str, reward: str, env, lidar_beams: int):
    """
    Resolve compatible pretrained RL drivers for BEV encoder pretraining.

    Preference order:
      - obstacle scenarios: lidar-24, lidar-16, state
      - regular lane-following: state, lidar-16, lidar-24, lidar-32, lidar-8
    """
    driver_specs = []
    if uses_obstacles(scenario):
        candidates = [
            ("lidar", 24),
            ("lidar", 16),
            ("state", None),
        ]
    else:
        candidates = [
            ("state", None),
            ("lidar", 16),
            ("lidar", 24),
            ("lidar", 32),
            ("lidar", 8),
        ]

    for driver_obs, beams in candidates:
        path = _policy_model_path(
            scenario=scenario,
            obs=driver_obs,
            reward=reward,
            lidar_beams=beams if beams is not None else 16,
        )
        if not path.exists():
            continue

        load_env = make_env(
            scenario=scenario,
            obs=driver_obs,
            render_mode=None,
            reward=reward,
            lidar_beams=beams if beams is not None else 16,
        )
        try:
            # Older checkpoints may reference numpy._core.numeric in pickled metadata.
            sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
            model = TD3.load(
                str(path),
                env=load_env,
                device="auto",
                custom_objects={
                    "observation_space": load_env.observation_space,
                    "action_space": load_env.action_space,
                },
            )
        except Exception as exc:
            print(f"[pretrain] Skipping driver {path} (load failed: {exc})")
            load_env.close()
            continue

        obs_fn = _make_driver_obs_fn(driver_obs, env, beams if beams is not None else lidar_beams)
        driver_specs.append((model, obs_fn))
        print(f"[pretrain] Added driver model → {path}")

    return driver_specs


def run_pretrain(scenario: str, obs: str, encoder: str, encoder_path: Path,
                 reward: str, lidar_beams: int,
                 n_collect_steps: int, epochs: int):
    """Pretrain the AE or UNet encoder and save weights to encoder_path."""
    env = make_env(scenario, obs, render_mode=None, reward=reward,
                   lidar_beams=lidar_beams)
    driver_models = resolve_pretrain_driver_models(
        scenario=scenario,
        reward=reward,
        env=env,
        lidar_beams=lidar_beams,
    )
    if encoder in _BEV_ENCODERS:
        from Models.AutoEncoder import pretrain_autoencoder
        pretrain_autoencoder(
            env=env,
            save_path=str(encoder_path),
            n_collect_steps=n_collect_steps,
            epochs=epochs,
            driver_models=driver_models,
        )
    else:
        from Models.UNet import pretrain_unet
        pretrain_unet(
            env=env,
            save_path=str(encoder_path),
            n_collect_steps=n_collect_steps,
            epochs=epochs,
            driver_models=driver_models,
        )
    env.close()


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class RenderCallback(EvalCallback.__bases__[0]):
    def __init__(self, render_freq: int = 1, verbose=0):
        super().__init__(verbose)
        self.render_freq = render_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            self.training_env.get_attr('render')[0]()
        return True


class ObstacleCallback(EvalCallback.__bases__[0]):
    def __init__(self, freq: int = 2000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obstacles_low = 0
        self.obstacles_high = 0
        self.freq = freq

    def _on_step(self) -> bool:
        if self.n_calls % self.freq == 0:
            self.obstacles_low = min(self.obstacles_low + 1, 5)
            self.training_env.set_attr('obstacles_low', self.obstacles_low)
            self.obstacles_high = min(self.obstacles_high + 2, 10)
            self.training_env.set_attr('obstacles_high', self.obstacles_high)
        return True


class NormalizedEvalCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_normalized_score = -np.inf
        self.k = 1

    def _on_step(self) -> bool:
        result = super()._on_step()

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if not self.evaluations_results:
                return result
            rewards = self.evaluations_results[-1]
            lengths = self.evaluations_length[-1]
            mean_reward = float(np.mean(rewards))
            mean_length = float(np.mean(lengths))
            if mean_length <= 0:
                return result
            normalized_score = mean_reward / mean_length
            if normalized_score > self.best_normalized_score:
                self.best_normalized_score = normalized_score
                save_path = os.path.join(self.best_model_save_path, f"norm{self.k}")
                self.model.save(save_path)
                self.k += 1
                print(
                    f"[Eval] New best normalized model | "
                    f"reward={mean_reward:.2f}, len={mean_length:.2f}, "
                    f"score={normalized_score:.3f}"
                )
        return result


# ---------------------------------------------------------------------------
# Policy / feature extractor factory
# ---------------------------------------------------------------------------

def make_policy_kwargs(obs: str, encoder: str, encoder_path: str | None) -> tuple[str, dict]:
    """Return (policy_name, policy_kwargs) for TD3."""
    if not is_bev_obs(obs):
        return "MlpPolicy", dict(net_arch=[256, 256])

    # BEV obs — MultiInputPolicy with a CNN or UNet extractor
    if encoder in _UNET_ENCODERS:
        extractor_cls = UNetFeatureExtractor
    else:
        extractor_cls = CNNFeatureExtractor

    extractor_kwargs = {}
    if encoder in _PRETRAIN_ENCODERS:
        extractor_kwargs["encoder_state_dict_path"] = encoder_path
        extractor_kwargs["freeze_encoder"] = encoder.endswith("frozen")

    return "MultiInputPolicy", dict(
        features_extractor_class=extractor_cls,
        features_extractor_kwargs=extractor_kwargs,
        net_arch=[256, 256],
        share_features_extractor=True,
    )


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def main(
    scenario: str,
    obs: str,
    reward: str,
    encoder: str = "scratch",
    encoder_path: str | None = None,
    render_mode=None,
    timesteps: int = 200_000,
    n_envs: int = 1,
    lidar_beams: int = 16,
    device: str = "auto",
    pretrain_collect_steps: int = 10_000,
    pretrain_epochs: int = 30,
    n_eval_episodes: int = 10,
    eval_freq_timesteps: int = 10_000,
    normalized_eval_freq_timesteps: int = 30_000,
    retry_on_failure: bool = False,
):
    # --- Validate args ---
    valid_rewards = reward_choices_for_scenario(scenario)
    if reward not in valid_rewards:
        raise ValueError(
            f"reward={reward!r} invalid for scenario={scenario!r}. "
            f"Choose from {valid_rewards}."
        )
    if encoder not in ENCODER_MODES:
        raise ValueError(f"encoder={encoder!r}. Choose from {ENCODER_MODES}.")
    if encoder != "scratch" and obs != "bev":
        raise ValueError(f"--encoder {encoder!r} only applies when --obs bev")

    # --- Paths ---
    if obs == "bev" and encoder != "scratch":
        obs_tag = f"{obs}_{encoder}"
    elif obs == "lidar" and lidar_beams != 16:
        obs_tag = f"lidar_{lidar_beams}"
    else:
        obs_tag = obs
    retry_tag = "_retry" if retry_on_failure else ""
    save_root = Path(f"./models/{scenario}/{obs_tag}/{reward}{retry_tag}")
    save_root.mkdir(parents=True, exist_ok=True)
    log_dir  = save_root / "logs"
    log_dir.mkdir(exist_ok=True)
    norm_dir = save_root / "normalized"
    norm_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  scenario={scenario}  obs={obs}  encoder={encoder}  reward={reward}")
    print(f"  retry_on_failure={retry_on_failure}")
    print(f"  timesteps={timesteps:,}  n_envs={n_envs}  device={device}")
    print(
        "  eval_freq="
        f"{eval_freq_timesteps:,}  normalized_eval_freq={normalized_eval_freq_timesteps:,}  "
        f"eval_episodes={n_eval_episodes}"
    )
    print(f"  save → {save_root}")
    print(f"{'='*60}")

    # --- Pretrain encoder if needed ---
    resolved_encoder_path: str | None = None
    if encoder in _PRETRAIN_ENCODERS:
        ep = Path(encoder_path) if encoder_path else _default_encoder_path(scenario, encoder, save_root)
        if ep.exists():
            print(f"[pretrain] Using existing encoder → {ep}")
        else:
            print(f"[pretrain] No encoder found at {ep} — pretraining now …")
            run_pretrain(
                scenario=scenario, obs=obs, encoder=encoder, encoder_path=ep,
                reward=reward, lidar_beams=lidar_beams,
                n_collect_steps=pretrain_collect_steps, epochs=pretrain_epochs,
            )
        resolved_encoder_path = str(ep)

    # --- Policy ---
    policy, policy_kwargs = make_policy_kwargs(obs, encoder, resolved_encoder_path)

    # --- Training env ---
    action_noise_sigma = 0.3

    if n_envs > 1:
        train_env = SubprocVecEnv(
            [_make_env_fn(scenario, obs, reward, lidar_beams, rank=i,
                          retry_on_failure=retry_on_failure)
             for i in range(n_envs)],
            start_method="fork",
        )
    else:
        base_env = make_env(scenario, obs, render_mode=render_mode,
                            reward=reward, lidar_beams=lidar_beams)
        if retry_on_failure:
            from Environments.wrappers import RetryOnFailureWrapper
            base_env = RetryOnFailureWrapper(base_env)
        train_env = Monitor(base_env)
    train_freq = (1, "step")

    # --- Eval env ---
    # SB3 auto-wraps the training VecEnv with VecTransposeImage for image obs.
    # EvalCallback skips this, so we pre-wrap the eval env for BEV variants.
    if is_bev_obs(obs):
        eval_env = VecTransposeImage(
            DummyVecEnv([_make_env_fn(scenario, obs, reward, lidar_beams, rank=n_envs)])
        )
    else:
        eval_env = Monitor(make_env(scenario, obs, render_mode=None,
                                    reward=reward, lidar_beams=lidar_beams))

    # --- TD3 model ---
    n_actions = train_env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=action_noise_sigma * np.ones(n_actions),
    )
    model = TD3(
        policy,
        train_env,
        action_noise=action_noise,
        verbose=0,
        device=device,
        policy_kwargs=policy_kwargs,
        buffer_size=300_000,
        learning_starts=5_000,
        batch_size=256,
        train_freq=train_freq,
        gradient_steps=-1,
    )

    # --- Callbacks ---
    cbs = []
    if render_mode and n_envs == 1:
        cbs.append(RenderCallback(render_freq=1))

    if uses_obstacles(scenario):
        # Warm-up: eval env starts with obstacles; training env ramps up
        if is_bev_obs(obs):
            eval_env.env_method("__setattr__", "obstacles_low",  5)
            eval_env.env_method("__setattr__", "obstacles_high", 10)
        else:
            eval_env.env.obstacles_low  = 5
            eval_env.env.obstacles_high = 10
        cbs.append(ObstacleCallback())

    cbs.append(EvalCallback(
        eval_env,
        best_model_save_path=str(save_root),
        log_path=str(log_dir),
        eval_freq=max(eval_freq_timesteps // n_envs, 1),
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    ))
    cbs.append(NormalizedEvalCallback(
        eval_env,
        best_model_save_path=str(norm_dir),
        log_path=str(norm_dir / "logs"),
        eval_freq=max(normalized_eval_freq_timesteps // n_envs, 1),
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
    ))

    # --- Train ---
    model.learn(total_timesteps=timesteps, callback=cbs, progress_bar=True)
    model.save(str(save_root / "final"))
    print(f"  Saved final model → {save_root / 'final.zip'}")

    train_env.close()
    eval_env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a TD3 agent for any scenario/obs/reward/encoder combination."
    )
    add_config_argument(parser)
    parser.add_argument(
        "--scenario",
        choices=["forward", "reverse", "forward_obs", "reverse_obs"],
    )
    parser.add_argument(
        "--obs",
        choices=["state", "lidar", "bev"],
    )
    parser.add_argument(
        "--reward",
        default="dense",
        help=(
            "Reward variant. Forward: "
            f"{', '.join(FORWARD_REWARD_MODES)}. Reverse: "
            f"{', '.join(REVERSE_REWARD_MODES)}."
        ),
    )
    parser.add_argument(
        "--encoder",
        choices=list(ENCODER_MODES),
        default="scratch",
        help=(
            "BEV encoder mode (--obs bev only). "
            "scratch: end-to-end from random init. "
            "ae_frozen/ae_unfrozen: pretrained NatureCNN AE. "
            "unet_frozen/unet_unfrozen: pretrained UNet encoder."
        ),
    )
    parser.add_argument(
        "--encoder_path",
        default=None,
        help=(
            "Path to pretrained encoder weights (.pt). "
            "If omitted for pretrained variants, auto-pretrains and saves to "
            "./models/<scenario>/encoder_<type>.pt."
        ),
    )
    parser.add_argument(
        "--timesteps", type=int, default=200_000,
        help="Total training timesteps (default: 200_000).",
    )
    parser.add_argument(
        "--n_envs", type=int, default=1,
        help=(
            "Parallel training environments via SubprocVecEnv (default: 1). "
            "Values > 1 recommended for BEV variants where rendering is the bottleneck."
        ),
    )
    parser.add_argument(
        "--lidar_beams", type=int, default=16,
        help="Number of lidar beams when --obs lidar (default: 16).",
    )
    parser.add_argument(
        "--device", default="auto",
        help="PyTorch device: auto, cuda, cpu (default: auto).",
    )
    parser.add_argument(
        "--pretrain_steps", type=int, default=10_000,
        help="Env steps to collect for AE/UNet pretraining (default: 10_000).",
    )
    parser.add_argument(
        "--pretrain_epochs", type=int, default=30,
        help="Epochs for AE/UNet pretraining (default: 30).",
    )
    parser.add_argument(
        "--eval_episodes", type=int, default=10,
        help="Eval episodes per callback check (default: 10).",
    )
    parser.add_argument(
        "--eval_freq", type=int, default=10_000,
        help=(
            "Environment timesteps between standard evaluation passes "
            "(default: 10_000)."
        ),
    )
    parser.add_argument(
        "--normalized_eval_freq", type=int, default=30_000,
        help=(
            "Environment timesteps between normalized-score evaluation passes "
            "(default: 30_000)."
        ),
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Enable rendering (single-env only).",
    )
    parser.add_argument(
        "--retry_on_failure", action="store_true",
        help=(
            "Replay the same path/spawn after a failed episode (collision or "
            "jackknife) instead of drawing a fresh random scenario. "
            "Models are saved under <reward>_retry/ to keep them separate."
        ),
    )
    return parser


def parse_train_args(argv=None) -> TrainConfig:
    parser = build_parser()
    apply_config_file_defaults(parser, argv, TrainConfig)
    args = parser.parse_args(argv)
    args_dict = vars(args).copy()
    args_dict.pop("config", None)
    config = TrainConfig.from_dict(args_dict)
    errors = validate_train_config(config)
    if errors:
        parser.error("\n".join(errors))
    return config


if __name__ == "__main__":
    args = parse_train_args()

    main(
        scenario=args.scenario,
        obs=args.obs,
        reward=args.reward,
        encoder=args.encoder,
        encoder_path=args.encoder_path,
        render_mode="human" if args.render else None,
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        lidar_beams=args.lidar_beams,
        device=args.device,
        pretrain_collect_steps=args.pretrain_steps,
        pretrain_epochs=args.pretrain_epochs,
        n_eval_episodes=args.eval_episodes,
        eval_freq_timesteps=args.eval_freq,
        normalized_eval_freq_timesteps=args.normalized_eval_freq,
        retry_on_failure=args.retry_on_failure,
    )
