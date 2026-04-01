"""
Evaluate a trained TD3 model and report path-tracking metrics.

CLI mirrors train.py exactly.  The model path is auto-resolved from
--scenario / --obs / --reward / --encoder, or overridden with --model.

Usage
-----
    python eval.py --scenario forward --obs state --reward dense
    python eval.py --scenario reverse --obs bev  --encoder ae_frozen --episodes 30
    python eval.py --scenario forward_obs --obs lidar --reward dense \\
        --model ./models/forward_obs/lidar/dense/best_model.zip

Output
------
    Per-episode CSV  → <output_csv>           (default: results/<label>/episodes.csv)
    Aggregate CSV    → <output_csv dir>/aggregate.csv
    Summary table    → stdout
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import TD3
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
from Environments.LineFollowing import FORWARD_REWARD_MODES, REVERSE_REWARD_MODES
from e2erl_utils.metrics import EpisodeMetricsLogger
from sim_config import (
    EvalConfig,
    add_config_argument,
    apply_config_file_defaults,
    validate_eval_config,
)


# ---------------------------------------------------------------------------
# Path helpers (mirrors run_model.py)
# ---------------------------------------------------------------------------

def _obs_tag(obs: str, encoder: str, lidar_beams: int = 16) -> str:
    if obs == "bev" and encoder != "scratch":
        return f"{obs}_{encoder}"
    if obs == "lidar" and lidar_beams != 16:
        return f"lidar_{lidar_beams}"
    return obs


def _default_model_path(scenario: str, obs: str, reward: str, encoder: str) -> Path:
    return Path(f"./models/{scenario}/{_obs_tag(obs, encoder)}/{reward}/best_model.zip")


def _default_encoder_path(scenario: str, obs: str, reward: str, encoder: str) -> Path | None:
    if encoder not in _PRETRAIN_ENCODERS:
        return None
    enc_type = "ae" if encoder in _BEV_ENCODERS else "unet"
    return Path(f"./models/{scenario}/{_obs_tag(obs, encoder)}/{reward}/encoder_{enc_type}.pt")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: Path, obs: str, encoder: str, encoder_path: str | None, env):
    """Load a TD3 model, importing custom extractor classes so SB3 can deserialise."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Train first with train.py, or pass --model <explicit_path>."
        )

    if is_bev_obs(obs):
        if encoder in _UNET_ENCODERS:
            from Models.UNetFeatureExtractor import UNetFeatureExtractor  # noqa: F401
        else:
            from Models.CNNFeatureExtractor import CNNFeatureExtractor  # noqa: F401

    model = TD3.load(str(model_path), env=env, device="auto")

    # GPU warm-up: reduces first-step latency spikes in timing measurements
    dummy_obs = env.observation_space.sample()
    for _ in range(50):
        model.predict(dummy_obs, deterministic=True)

    return model


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def _run_episode_vecenv(vec_env, model) -> dict:
    """Episode loop for BEV variants using VecEnv (required for channel transposing)."""
    raw_env = vec_env.venv.envs[0]   # underlying raw env for metrics
    logger = EpisodeMetricsLogger()

    obs = vec_env.reset()
    done = False
    terminated = truncated = False

    while not done:
        t0 = time.perf_counter()
        action, _ = model.predict(obs, deterministic=True)
        elapsed = time.perf_counter() - t0

        applied_action = raw_env.format_action(action[0]) if hasattr(raw_env, "format_action") else action[0]
        obs, rew, dones, infos = vec_env.step(action)
        logger.log_step(raw_env, applied_action, float(rew[0]), inference_time_s=elapsed)

        if dones[0]:
            done = True
            truncated = bool(infos[0].get("TimeLimit.truncated", False))
            terminated = not truncated

    return logger.compute_summary(
        terminated=terminated,
        truncated=truncated,
        completed=bool(getattr(raw_env, "success", False)),
    )


def _run_episode(env, model) -> dict:
    """Episode loop for non-BEV variants using the standard Gym interface."""
    logger = EpisodeMetricsLogger()
    obs, _ = env.reset()
    done = False
    terminated = truncated = False

    while not done:
        t0 = time.perf_counter()
        action, _ = model.predict(obs, deterministic=True)
        elapsed = time.perf_counter() - t0

        applied_action = env.format_action(action) if hasattr(env, "format_action") else action
        obs, reward, terminated, truncated, _ = env.step(action)
        logger.log_step(env, applied_action, float(reward), inference_time_s=elapsed)
        done = terminated or truncated

    return logger.compute_summary(
        terminated=terminated,
        truncated=truncated,
        completed=bool(getattr(env, "success", False)),
    )


# ---------------------------------------------------------------------------
# Aggregate & display
# ---------------------------------------------------------------------------

def _aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {}
    metrics = [
        "mean_abs_cte_tractor", "mean_abs_cte_trailer",
        "rms_cte_trailer", "max_abs_cte_trailer",
        "mean_abs_hitch_angle", "max_abs_hitch_angle",
        "completed", "jackknifed", "collided",
        "episode_length", "total_reward",
        "mean_inference_ms", "p95_inference_ms",
    ]
    agg = {"n_episodes": len(rows)}
    for m in metrics:
        vals = [r[m] for r in rows if m in r and r[m] == r[m]]
        if not vals:
            agg[m] = float("nan")
        elif isinstance(vals[0], bool):
            agg[m] = float(sum(vals) / len(vals))
        else:
            agg[m] = float(np.mean(vals))
    return agg


_DISPLAY_METRICS = [
    ("mean_abs_cte_trailer",       "CTE trailer mean (m)"),
    ("rms_cte_trailer",            "CTE trailer RMS  (m)"),
    ("max_abs_cte_trailer",        "CTE trailer max  (m)"),
    ("mean_abs_cte_tractor",       "CTE tractor mean (m)"),
    ("mean_abs_hitch_angle",       "Hitch angle mean (rad)"),
    ("max_abs_hitch_angle",        "Hitch angle max  (rad)"),
    ("completed",                  "Completion rate"),
    ("jackknifed",                 "Jackknife rate"),
    ("collided",                   "Collision rate"),
    ("episode_length",             "Episode length (steps)"),
    ("total_reward",               "Total reward"),
    ("mean_inference_ms",          "Inference mean (ms)"),
    ("p95_inference_ms",           "Inference p95  (ms)"),
]


def _print_summary(label: str, agg: dict, n_episodes: int):
    w = 36
    sep = "=" * (w + 18)
    print(f"\n{sep}")
    print(f"  Evaluation: {label}  ({n_episodes} episodes)")
    print(sep)
    for key, display in _DISPLAY_METRICS:
        val = agg.get(key, float("nan"))
        if val != val:
            print(f"  {display:<{w}}  {'N/A':>10}")
        elif key in ("completed", "jackknifed", "collided"):
            print(f"  {display:<{w}}  {val*100:>9.1f}%")
        else:
            print(f"  {display:<{w}}  {val:>10.4f}")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    scenario: str,
    obs: str,
    reward: str,
    encoder: str = "scratch",
    encoder_path: str | None = None,
    model_path: Path | None = None,
    lidar_beams: int = 16,
    n_episodes: int = 20,
    output_csv: Path | None = None,
    retry_on_failure: bool = False,
):
    # --- Resolve model and encoder paths ---
    if model_path is None:
        model_path = _default_model_path(scenario, obs, reward, encoder)

    resolved_encoder_path: str | None = encoder_path
    if resolved_encoder_path is None and encoder in _PRETRAIN_ENCODERS:
        ep = _default_encoder_path(scenario, obs, reward, encoder)
        if ep and ep.exists():
            resolved_encoder_path = str(ep)

    tag = _obs_tag(obs, encoder, lidar_beams)
    label = f"{scenario}/{tag}/{reward}"

    print(f"\n{'='*60}")
    print(f"  Evaluating: {label}")
    print(f"  Model:      {model_path}")
    print(f"  Episodes:   {n_episodes}")
    print(f"{'='*60}")

    # --- Build eval environment ---
    if is_bev_obs(obs):
        def _env_fn():
            e = make_env(scenario, obs, render_mode=None,
                         reward=reward, lidar_beams=lidar_beams)
            if uses_obstacles(scenario):
                e.obstacles_low  = 5
                e.obstacles_high = 10
            if retry_on_failure:
                from Environments.wrappers import RetryOnFailureWrapper
                e = RetryOnFailureWrapper(e)
            return e
        eval_env = VecTransposeImage(DummyVecEnv([_env_fn]))
        run_ep = lambda m: _run_episode_vecenv(eval_env, m)
    else:
        eval_env = make_env(scenario, obs, render_mode=None,
                            reward=reward, lidar_beams=lidar_beams)
        if uses_obstacles(scenario):
            eval_env.obstacles_low  = 5
            eval_env.obstacles_high = 10
        if retry_on_failure:
            from Environments.wrappers import RetryOnFailureWrapper
            eval_env = RetryOnFailureWrapper(eval_env)
        run_ep = lambda m: _run_episode(eval_env, m)

    # --- Load model ---
    model = load_model(
        model_path, obs, encoder, resolved_encoder_path, eval_env
    )
    print(f"  Loaded:     {model_path}\n")

    # --- Run episodes ---
    all_rows: list[dict] = []
    for ep in range(1, n_episodes + 1):
        summary = run_ep(model)
        summary["episode"] = ep
        summary["label"] = label
        all_rows.append(summary)

        cte = summary.get("mean_abs_cte_trailer", float("nan"))
        print(
            f"  ep {ep:3d}/{n_episodes}  "
            f"CTE={cte:.3f}m  "
            f"completed={summary.get('completed', '?')}  "
            f"steps={summary.get('episode_length', 0):4d}"
        )

    eval_env.close()

    # --- Aggregate & display ---
    agg = _aggregate(all_rows)
    _print_summary(label, agg, n_episodes)

    # --- Save CSVs ---
    if output_csv is None:
        output_csv = Path(f"results/{label}/episodes.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    all_keys = sorted({k for r in all_rows for k in r})
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, "") for k in all_keys})
    print(f"Per-episode CSV  → {output_csv}")

    agg_csv = output_csv.parent / "aggregate.csv"
    agg["label"] = label
    agg_keys = sorted(agg.keys())
    write_header = not agg_csv.exists()
    with open(agg_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=agg_keys, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({k: agg.get(k, "") for k in agg_keys})
    print(f"Aggregate CSV    → {agg_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained TD3 model with full path-tracking metrics."
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
        "--model",
        default=None,
        help=(
            "Explicit path to a saved model .zip. "
            "If omitted, resolved to "
            "./models/<scenario>/<obs[_encoder]>/<reward>/best_model.zip."
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes (default: 20).",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help=(
            "Path for per-episode CSV output. "
            "Default: results/<scenario>/<obs[_encoder]>/<reward>/episodes.csv."
        ),
    )
    parser.add_argument(
        "--retry_on_failure", action="store_true",
        help=(
            "Replay the same path/spawn after a failed episode during evaluation. "
            "Useful for stress-testing a model on hard scenarios, but note that "
            "metrics will reflect performance on repeated (not i.i.d.) layouts."
        ),
    )
    return parser


def parse_eval_args(argv=None) -> EvalConfig:
    parser = build_parser()
    apply_config_file_defaults(parser, argv, EvalConfig)
    args = parser.parse_args(argv)
    args_dict = vars(args).copy()
    args_dict.pop("config", None)
    config = EvalConfig.from_dict(args_dict)
    errors = validate_eval_config(config)
    if errors:
        parser.error("\n".join(errors))
    return config


if __name__ == "__main__":
    args = parse_eval_args()

    main(
        scenario=args.scenario,
        obs=args.obs,
        reward=args.reward,
        encoder=args.encoder,
        encoder_path=args.encoder_path,
        model_path=Path(args.model) if args.model else None,
        lidar_beams=args.lidar_beams,
        n_episodes=args.episodes,
        output_csv=args.output_csv,
        retry_on_failure=args.retry_on_failure,
    )
