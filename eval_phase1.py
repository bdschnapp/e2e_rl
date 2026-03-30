"""
Phase 1 evaluation: compare observation modalities on forward lane-following.

Loads the best saved TD3 model for each variant from models/Phase1/<variant>/
and evaluates over N episodes, reporting path-tracking and stability metrics.

Usage
-----
  python eval_phase1.py                          # evaluate all variants
  python eval_phase1.py --variants state_only lidar_16 cnn_bev
  python eval_phase1.py --model_root models/Phase1 --episodes 30
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from e2erl_utils.metrics import EpisodeMetricsLogger
from train_phase1 import VARIANTS, make_env


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _find_model(model_dir: Path) -> Path | None:
    """
    Return path to the best saved model (.zip) in model_dir.
    Prefers 'best_model.zip' (EvalCallback output), falls back to 'final.zip'.
    """
    for name in ("best_model.zip", "best_model", "final.zip", "final"):
        candidate = model_dir / name
        if candidate.with_suffix(".zip").exists():
            return candidate.with_suffix(".zip")
        if candidate.exists():
            return candidate
    return None


def load_model(variant: str, model_dir: Path, model_root: Path, env):
    from stable_baselines3 import TD3

    model_path = _find_model(model_dir)
    if model_path is None:
        raise FileNotFoundError(
            f"No model found in {model_dir}. "
            "Run train_phase1.py first."
        )

    if variant in ("cnn_bev", "ae_bev"):
        from Models.CNNFeatureExtractor import CNNFeatureExtractor
        # ae_bev: load with encoder path so weights are available for policy rebuild
        encoder_path = str(model_root / f"{variant}_encoder.pt")
        extractor_kwargs = (
            dict(encoder_state_dict_path=encoder_path, freeze_encoder=True)
            if variant == "ae_bev" and Path(encoder_path).exists()
            else {}
        )
        model = TD3.load(
            model_path,
            env=env,
            device="auto",
            custom_objects={
                "policy_kwargs": dict(
                    features_extractor_class=CNNFeatureExtractor,
                    features_extractor_kwargs=extractor_kwargs,
                    net_arch=[256, 256],
                )
            },
        )
    elif variant == "unet_bev":
        from Models.UNetFeatureExtractor import UNetFeatureExtractor
        encoder_path = str(model_root / "unet_bev_encoder.pt")
        extractor_kwargs = (
            dict(encoder_state_dict_path=encoder_path, freeze_encoder=True)
            if Path(encoder_path).exists()
            else {}
        )
        model = TD3.load(
            model_path,
            env=env,
            device="auto",
            custom_objects={
                "policy_kwargs": dict(
                    features_extractor_class=UNetFeatureExtractor,
                    features_extractor_kwargs=extractor_kwargs,
                    net_arch=[256, 256],
                )
            },
        )
    else:
        model = TD3.load(model_path, env=env, device="auto")

    # GPU warm-up
    dummy_obs = env.observation_space.sample()
    for _ in range(50):
        model.predict(dummy_obs, deterministic=True)

    return model, model_path


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env, model) -> dict:
    logger = EpisodeMetricsLogger()
    obs, _ = env.reset()
    done = False

    while not done:
        t0 = time.perf_counter()
        action, _ = model.predict(obs, deterministic=True)
        elapsed = time.perf_counter() - t0

        obs, reward, terminated, truncated, _ = env.step(action)
        logger.log_step(env, action, reward, inference_time_s=elapsed)
        done = terminated or truncated

    return logger.compute_summary(terminated=terminated, truncated=truncated)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_variant(variant: str, model_dir: Path, model_root: Path, n_episodes: int) -> list[dict]:
    env = make_env(variant, render_mode=None, max_episode_steps=1000)

    try:
        model, model_path = load_model(variant, model_dir, model_root, env)
    except FileNotFoundError as e:
        print(f"  [SKIP] {variant}: {e}")
        env.close()
        return []

    print(f"  Loaded: {model_path}")

    results = []
    for ep in range(n_episodes):
        summary = run_episode(env, model)
        summary["variant"] = variant
        summary["episode"] = ep
        results.append(summary)

        cte = summary.get("mean_abs_cte_trailer", float("nan"))
        completed = summary.get("completed", False)
        print(
            f"    ep {ep+1:3d}/{n_episodes}  "
            f"CTE_trailer={cte:.3f}m  "
            f"completed={completed}  "
            f"steps={summary.get('episode_length', 0)}"
        )

    env.close()
    return results


def _aggregate(rows: list[dict], variant: str) -> dict:
    """Compute mean of key metrics over episodes for a variant."""
    if not rows:
        return {"variant": variant}

    metrics = [
        "mean_abs_cte_tractor",
        "mean_abs_cte_trailer",
        "rms_cte_trailer",
        "max_abs_cte_trailer",
        "mean_abs_hitch_angle",
        "max_abs_hitch_angle",
        "completed",
        "jackknifed",
        "collided",
        "episode_length",
        "mean_inference_ms",
    ]

    agg = {"variant": variant, "n_episodes": len(rows)}
    for m in metrics:
        vals = [r[m] for r in rows if m in r and r[m] == r[m]]
        if not vals:
            agg[m] = float("nan")
        elif isinstance(vals[0], bool):
            agg[m] = sum(vals) / len(vals)  # rate
        else:
            agg[m] = float(np.mean(vals))

    return agg


def _print_table(agg_rows: list[dict], variants: list[str]):
    metrics = [
        ("mean_abs_cte_trailer",  "CTE trailer (m)"),
        ("rms_cte_trailer",       "RMS CTE trailer (m)"),
        ("max_abs_cte_trailer",   "Max CTE trailer (m)"),
        ("mean_abs_cte_tractor",  "CTE tractor (m)"),
        ("mean_abs_hitch_angle",  "Mean |hitch| (rad)"),
        ("max_abs_hitch_angle",   "Max |hitch| (rad)"),
        ("completed",             "Completion rate"),
        ("jackknifed",            "Jackknife rate"),
        ("episode_length",        "Episode length (steps)"),
        ("mean_inference_ms",     "Inference (ms)"),
    ]

    by_variant = {r["variant"]: r for r in agg_rows}
    col_w = 22

    header = f"{'Metric':<30}" + "".join(f"{v:>{col_w}}" for v in variants)
    sep = "=" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)

    for key, label in metrics:
        line = f"{label:<30}"
        for v in variants:
            val = by_variant.get(v, {}).get(key, float("nan"))
            if val != val:  # NaN
                line += f"{'N/A':>{col_w}}"
            elif key in ("completed", "jackknifed", "collided"):
                line += f"{val*100:>{col_w-1}.1f}%"
            else:
                line += f"{val:>{col_w}.4f}"
        print(line)

    print(sep + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 evaluation: observation modality comparison"
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=None,
        help="Variants to evaluate (default: all found in model_root)",
    )
    parser.add_argument(
        "--model_root",
        type=Path,
        default=Path("models/Phase1"),
        help="Root directory of trained models (default: models/Phase1/)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Evaluation episodes per variant (default: 20)",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("results/phase1_eval.csv"),
        help="Path for per-episode CSV output (default: results/phase1_eval.csv)",
    )
    args = parser.parse_args()

    # Determine which variants to evaluate
    if args.variants:
        to_eval = args.variants
    else:
        # Auto-discover variants that have saved models
        to_eval = [v for v in VARIANTS if (args.model_root / v).exists()]
        if not to_eval:
            print(f"No trained variants found in {args.model_root}/")
            print("Run train_phase1.py first.")
            sys.exit(1)

    print(f"\nPhase 1 Evaluation")
    print(f"  Variants : {to_eval}")
    print(f"  Episodes : {args.episodes}")
    print(f"  Model dir: {args.model_root}\n")

    all_rows: list[dict] = []
    agg_rows: list[dict] = []

    for variant in to_eval:
        print(f"[ {variant} ]")
        model_dir = args.model_root / variant
        rows = evaluate_variant(variant, model_dir, args.model_root, args.episodes)
        all_rows.extend(rows)
        agg_rows.append(_aggregate(rows, variant))

    # Print comparison table
    _print_table(agg_rows, to_eval)

    # Save per-episode CSV
    if all_rows:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        all_keys = sorted({k for r in all_rows for k in r.keys()})
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            for row in all_rows:
                writer.writerow({k: row.get(k, "") for k in all_keys})
        print(f"Per-episode results saved → {args.output_csv}")

    # Save aggregate CSV
    agg_csv = args.output_csv.parent / "phase1_aggregate.csv"
    if agg_rows:
        agg_keys = sorted({k for r in agg_rows for k in r.keys()})
        with open(agg_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=agg_keys, extrasaction="ignore")
            writer.writeheader()
            for row in agg_rows:
                writer.writerow({k: row.get(k, "") for k in agg_keys})
        print(f"Aggregate results saved   → {agg_csv}")


if __name__ == "__main__":
    main()
