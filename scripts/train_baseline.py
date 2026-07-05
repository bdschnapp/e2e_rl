"""Mild-paths baseline trainer for tractor_trailer_rl (the proven recipe).

2-D stop_signal action ALWAYS (portable to the future obstacle model), with the
stop dim strongly suppressed so it stays dormant in the no-obstacle baseline:
  high stop_threshold, low stop-action noise, high (terminal) stop_penalty.
Mild paths (straight+gentle) + fixed lookahead = the recipe that previously
generalized to the lab corner. Reverse / forward-trailer fine-tuning (longer
trailer) is a later step.

Usage:
  python scripts/train_baseline.py                # all 4 models, 200k each
  MODELS="forward_trailer" STEPS=50000 python scripts/train_baseline.py   # smoke

Env knobs: MODELS, STEPS, N_ENVS, OUT, DEVICE, STOP_THRESHOLD, STOP_NOISE,
STOP_PENALTY, DRIVE_PENALTY, SPEED_MIN, SPEED_MAX.
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

from dataclasses import replace
import numpy as np
from tractor_trailer_rl import lab_config, Direction, VehicleKind, ActionMode, LookaheadMode
from tractor_trailer_rl.train.runner import train

MODELS = {
    "forward_trailer": (Direction.FORWARD, VehicleKind.TRAILER),
    "reverse_trailer": (Direction.REVERSE, VehicleKind.TRAILER),
    "forward_tractor_only": (Direction.FORWARD, VehicleKind.TRACTOR_ONLY),
    "reverse_tractor_only": (Direction.REVERSE, VehicleKind.TRACTOR_ONLY),
}


def build_cfg(direction, kind):
    return replace(
        lab_config(direction, kind),
        action=replace(
            lab_config(direction, kind).action,
            mode=ActionMode.STOP_SIGNAL,
            stop_threshold=float(os.environ.get("STOP_THRESHOLD", 0.8)),
            stop_noise_sigma=float(os.environ.get("STOP_NOISE", 0.05)),
            stop_penalty=float(os.environ.get("STOP_PENALTY", 500.0)),
        ),
        reward=replace(lab_config(direction, kind).reward, mode="multiplicative"),
        speed_random=replace(
            lab_config(direction, kind).speed_random,
            explicit_min=float(os.environ.get("SPEED_MIN", 0.5)),
            explicit_max=float(os.environ.get("SPEED_MAX", 0.8)),
        ),
        path=replace(lab_config(direction, kind).path, mild_only=True),
        # Speed-scaled lookahead: at the slow deploy speed (~0.6 m/s) a FIXED
        # 10/20-sample (=10/20 m) lookahead previews the corner 16-33 s ahead, so
        # the policy turns off the lane long before reaching it. Speed-scaled puts
        # kappa1/kappa2 at ~v*1.5s / v*3s ahead (~1 m/2 m at 0.6 m/s). Heading
        # stays local. LOOKAHEAD=fixed reverts to the parity behaviour.
        lookahead=(replace(lab_config(direction, kind).lookahead,
                           mode=LookaheadMode.SPEED_SCALED,
                           preview_time_s=(1.5, 3.0), heading_preview_s=0.0)
                   if os.environ.get("LOOKAHEAD", "speed_scaled") == "speed_scaled"
                   else lab_config(direction, kind).lookahead),
    )


def main():
    steps = int(os.environ.get("STEPS", 200_000))
    n_envs = int(os.environ.get("N_ENVS", 8))
    device = os.environ.get("DEVICE", "auto")
    out_root = os.environ.get("OUT", os.path.expanduser(
        "~/Ben/Thesis/Electrans_project/lab_models_ttrl_baseline"))
    want = os.environ.get("MODELS", " ".join(MODELS)).split()

    for name in want:
        direction, kind = MODELS[name]
        cfg = build_cfg(direction, kind)
        out = os.path.join(out_root, name)
        print(f"\n===== TRAIN {name}  steps={steps} n_envs={n_envs}  out={out} =====",
              flush=True)
        train(cfg, timesteps=steps, n_envs=n_envs, out_dir=out,
              eval_freq=20000, n_eval_episodes=10, device=device)
        d = np.load(os.path.join(out, "logs", "evaluations.npz"))
        res = d["results"].mean(1); ln = d["ep_lengths"].mean(1)
        best = int(np.argmax(res))
        print(f"  {name}: best eval reward={res[best]:.1f} (len={ln[best]:.0f}) "
              f"at t={int(d['timesteps'][best])}; final={res[-1]:.1f}/len{ln[-1]:.0f}",
              flush=True)
    print("\nBASELINE TRAINING DONE", flush=True)


if __name__ == "__main__":
    main()
