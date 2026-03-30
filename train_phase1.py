"""
Phase 1: Observation modality comparison — forward lane-following, no obstacles.

Trains one or more TD3 agents with different observation types and saves models
to ./models/Phase1/<variant>/.

Variants
--------
  state_only   StateObservationLineFollowingEnv + MlpPolicy
               8-dim state [s, γ, e_y, e_ψ, e_y_t, e_ψ_t, κ₁, κ₂]

  lidar_8      ObstacleAvoidanceEnv(beams=8,  obstacles=0) + MlpPolicy
  lidar_16     ObstacleAvoidanceEnv(beams=16, obstacles=0) + MlpPolicy
  lidar_24     ObstacleAvoidanceEnv(beams=24, obstacles=0) + MlpPolicy
               state(8) + N normalised lidar distances

  cnn_bev      BevObservationLineFollowingEnv + MultiInputPolicy + CNNFeatureExtractor
               state(8) + 84×84 BEV  — encoder trains end-to-end with policy

  ae_bev       BevObservationLineFollowingEnv + MultiInputPolicy + CNNFeatureExtractor
               Encoder pretrained via ImageAutoencoder reconstruction loss,
               then frozen during TD3 training (state + image, pretrained repr.)

  unet_bev     BevObservationLineFollowingEnv + MultiInputPolicy + UNetFeatureExtractor
               Encoder pretrained via UNetAutoencoder (with skip connections),
               then frozen during TD3 training

Usage
-----
  python train_phase1.py --variant state_only
  python train_phase1.py --variant ae_bev --timesteps 300000
  python train_phase1.py                      # trains all variants sequentially
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------

VARIANTS = [
    "state_only",
    "lidar_8", "lidar_16", "lidar_24",
    "cnn_bev",
    "ae_bev",
    "unet_bev",
]

# Pretrained variants that need an AE/UNet pretraining step
_PRETRAIN_AE   = {"ae_bev"}
_PRETRAIN_UNET = {"unet_bev"}
_BEV_VARIANTS  = {"cnn_bev", "ae_bev", "unet_bev"}


# ---------------------------------------------------------------------------
# Environment factories
# ---------------------------------------------------------------------------

def make_env(variant: str, render_mode=None, max_episode_steps: int = 1000):
    """Return an env for the given variant (not Monitor-wrapped)."""

    if variant == "state_only":
        from Environments.LineFollowing import StateObservationLineFollowingEnv
        return StateObservationLineFollowingEnv(
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
        )

    if variant.startswith("lidar_"):
        beams = int(variant.split("_")[1])
        from Environments.ObstacleAvoidance import LidarStateObservationLineFollowingEnv
        return LidarStateObservationLineFollowingEnv(
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            lidar_beams=beams,
        )

    if variant in _BEV_VARIANTS:
        from Environments.LineFollowing import BevObservationLineFollowingEnv
        return BevObservationLineFollowingEnv(
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
        )

    raise ValueError(f"Unknown variant: {variant!r}. Choose from {VARIANTS}")


def _make_env_fn(variant: str, rank: int, base_seed: int = 0):
    """
    Return a picklable factory for SubprocVecEnv.

    Must be defined at module level (not inside a closure) so that Python's
    pickle can serialise it for the subprocess workers.
    Each worker gets a different seed via base_seed + rank.
    """
    def _init():
        env = make_env(variant, render_mode=None)
        env = Monitor(env)
        env.reset(seed=base_seed + rank)
        return env
    return _init


# ---------------------------------------------------------------------------
# Pretraining
# ---------------------------------------------------------------------------

# Variants whose saved models can drive a BevObservationLineFollowingEnv.
# Maps variant name → function that extracts the model's expected obs from the
# BEV Dict obs {'image': ..., 'vector': ...}.
_DRIVER_COMPATIBLE = {
    "state_only": lambda obs: obs["vector"],          # flat 8-dim
    "cnn_bev":    lambda obs: obs,                    # full dict
    "ae_bev":     lambda obs: obs,
    "unet_bev":   lambda obs: obs,
    # lidar_* variants need lidar readings not available in BevEnv — excluded
}


def _load_driver_models(save_root: Path, skip_variant: str) -> list:
    """
    Load any already-trained models that can drive BevObservationLineFollowingEnv.

    Returns a list of (model, obs_fn) pairs ready for collect_bev_images.
    Only flat-obs (state_only) and BEV-policy variants are included because
    lidar variants need sensor readings the BEV env doesn't produce.
    """
    from stable_baselines3 import TD3

    drivers = []
    for variant, obs_fn in _DRIVER_COMPATIBLE.items():
        if variant == skip_variant:
            continue
        model_dir = save_root / variant
        if not model_dir.exists():
            continue

        # Prefer best_model, fall back to final
        model_path = None
        for name in ("best_model", "final"):
            candidate = (model_dir / name).with_suffix(".zip")
            if candidate.exists():
                model_path = candidate
                break

        if model_path is None:
            continue

        try:
            # BEV policy variants need custom_objects; state_only loads cleanly
            if variant in _BEV_VARIANTS:
                from Models.CNNFeatureExtractor import CNNFeatureExtractor
                model = TD3.load(
                    model_path,
                    device="auto",
                    custom_objects={"policy_kwargs": dict(
                        features_extractor_class=CNNFeatureExtractor,
                        features_extractor_kwargs={},
                        net_arch=[256, 256],
                    )},
                )
            else:
                model = TD3.load(model_path, device="auto")

            drivers.append((model, obs_fn))
            print(f"[pretrain] Driver loaded: {variant} ← {model_path.name}")
        except Exception as exc:
            print(f"[pretrain] Could not load driver {variant}: {exc}")

    return drivers


def pretrain_encoder(variant: str, save_root: Path, n_collect_steps: int, epochs: int) -> str:
    """
    Pretrain the image encoder (AE or UNet) and return the path to the saved
    encoder state dict.  Skips pretraining if the file already exists.

    Any already-trained compatible models found in save_root are loaded and used
    as additional drivers during BEV image collection, alongside pure-pursuit.
    """
    encoder_path = str(save_root / f"{variant}_encoder.pt")

    if Path(encoder_path).exists():
        print(f"[pretrain] Found existing encoder → {encoder_path}  (skipping)")
        return encoder_path

    driver_models = _load_driver_models(save_root, skip_variant=variant)
    if not driver_models:
        print("[pretrain] No trained drivers found — using pure-pursuit only")

    # Build a temporary env for data collection
    env = make_env(variant, render_mode=None)

    if variant in _PRETRAIN_AE:
        from Models.AutoEncoder import pretrain_autoencoder
        pretrain_autoencoder(
            env=env,
            save_path=encoder_path,
            n_collect_steps=n_collect_steps,
            epochs=epochs,
            driver_models=driver_models,
        )

    elif variant in _PRETRAIN_UNET:
        from Models.UNet import pretrain_unet
        pretrain_unet(
            env=env,
            save_path=encoder_path,
            n_collect_steps=n_collect_steps,
            epochs=epochs,
            driver_models=driver_models,
        )

    env.close()
    return encoder_path


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def make_td3(
    variant: str,
    env,
    encoder_path: str | None = None,
    device: str = "auto",
    n_envs: int = 1,
) -> TD3:
    """Build a TD3 model appropriate for the variant."""
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.3 * np.ones(n_actions),
    )

    # With a single env, train after each episode (matches original behaviour).
    # With multiple envs, train every step — episode-based collection stalls
    # until every worker finishes, which wastes parallelism.
    train_freq = (1, "step") if n_envs > 1 else (1, "episode")

    common_kwargs = dict(
        env=env,
        action_noise=action_noise,
        verbose=0,
        device=device,
        buffer_size=300_000,
        learning_starts=5_000,
        batch_size=256,
        train_freq=train_freq,
        gradient_steps=-1,
    )

    if variant == "cnn_bev":
        from Models.CNNFeatureExtractor import CNNFeatureExtractor
        policy_kwargs = dict(
            features_extractor_class=CNNFeatureExtractor,
            features_extractor_kwargs={},
            net_arch=[256, 256],
        )
        return TD3("MultiInputPolicy", policy_kwargs=policy_kwargs, **common_kwargs)

    if variant == "ae_bev":
        from Models.CNNFeatureExtractor import CNNFeatureExtractor
        policy_kwargs = dict(
            features_extractor_class=CNNFeatureExtractor,
            features_extractor_kwargs=dict(
                encoder_state_dict_path=encoder_path,
                freeze_encoder=True,
            ),
            net_arch=[256, 256],
        )
        return TD3("MultiInputPolicy", policy_kwargs=policy_kwargs, **common_kwargs)

    if variant == "unet_bev":
        from Models.UNetFeatureExtractor import UNetFeatureExtractor
        policy_kwargs = dict(
            features_extractor_class=UNetFeatureExtractor,
            features_extractor_kwargs=dict(
                encoder_state_dict_path=encoder_path,
                freeze_encoder=True,
            ),
            net_arch=[256, 256],
        )
        return TD3("MultiInputPolicy", policy_kwargs=policy_kwargs, **common_kwargs)

    # Flat-vector variants: MlpPolicy
    return TD3(
        "MlpPolicy",
        policy_kwargs=dict(net_arch=[256, 256]),
        **common_kwargs,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_variant(
    variant: str,
    timesteps: int,
    save_root: Path,
    device: str = "auto",
    n_eval_episodes: int = 10,
    pretrain_collect_steps: int = 10_000,
    pretrain_epochs: int = 30,
    n_envs: int = 1,
):
    save_dir = save_root / variant
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = save_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Training: {variant}  ({timesteps:,} steps)  n_envs={n_envs}")
    print(f"  Save dir: {save_dir}")
    print(f"{'='*60}")

    # --- Pretraining (AE / UNet variants) ---
    encoder_path = None
    if variant in (_PRETRAIN_AE | _PRETRAIN_UNET):
        encoder_path = pretrain_encoder(
            variant=variant,
            save_root=save_root,
            n_collect_steps=pretrain_collect_steps,
            epochs=pretrain_epochs,
        )

    # --- Training env ---
    if n_envs > 1:
        env = SubprocVecEnv(
            [_make_env_fn(variant, rank=i) for i in range(n_envs)],
            start_method="fork",
        )
    else:
        env = Monitor(make_env(variant, render_mode=None))

    # --- Eval env (always single-process; pre-wrap BEV to match VecTransposeImage) ---
    # SB3 auto-wraps the training env with VecTransposeImage for image obs.
    # EvalCallback does not, so we pre-wrap manually for BEV variants.
    eval_env = VecTransposeImage(DummyVecEnv([_make_env_fn(variant, rank=n_envs)])) \
        if variant in _BEV_VARIANTS \
        else Monitor(make_env(variant, render_mode=None))

    model = make_td3(variant, env, encoder_path=encoder_path, device=device, n_envs=n_envs)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir),
        log_path=str(log_dir),
        eval_freq=5_000,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=timesteps, callback=eval_cb, progress_bar=True)

    model.save(str(save_dir / "final"))
    print(f"  Saved final model → {save_dir / 'final.zip'}")

    env.close()
    eval_env.close()  # works for both Monitor and VecTransposeImage


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 training: observation modality comparison"
    )
    parser.add_argument(
        "--variant",
        choices=VARIANTS + ["all"],
        default="all",
        help=f"Variant to train (default: all). Choices: {VARIANTS}",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200_000,
        help="Training timesteps per variant (default: 200_000)",
    )
    parser.add_argument(
        "--save_root",
        type=Path,
        default=Path("models/Phase1"),
        help="Root directory for saved models (default: models/Phase1/)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device: auto, cuda, cpu (default: auto)",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=10,
        help="Eval episodes per callback check (default: 10)",
    )
    parser.add_argument(
        "--pretrain_steps",
        type=int,
        default=10_000,
        help="Env steps to collect for AE/UNet pretraining (default: 10_000)",
    )
    parser.add_argument(
        "--pretrain_epochs",
        type=int,
        default=30,
        help="Epochs for AE/UNet pretraining (default: 30)",
    )
    parser.add_argument(
        "--n_envs",
        type=int,
        default=1,
        help=(
            "Number of parallel environments (default: 1). "
            "Values > 1 use SubprocVecEnv. Recommended: set to the number of "
            "physical CPU cores available. Only meaningful for BEV variants where "
            "the per-step render is the bottleneck."
        ),
    )

    args = parser.parse_args()
    to_train = VARIANTS if args.variant == "all" else [args.variant]

    for v in to_train:
        train_variant(
            variant=v,
            timesteps=args.timesteps,
            save_root=args.save_root,
            device=args.device,
            n_eval_episodes=args.eval_episodes,
            pretrain_collect_steps=args.pretrain_steps,
            pretrain_epochs=args.pretrain_epochs,
            n_envs=args.n_envs,
        )

    print("\nPhase 1 training complete.")
    print(f"Models saved under: {args.save_root}/")
    print("Next: python eval_phase1.py")


if __name__ == "__main__":
    main()
