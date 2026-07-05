"""Reverse-trailer fine-tune on the small-env tanh CHICANE with the additive
'corridor' reward.

Motivation (2026-06-30): the synthetic lab_corner is mostly straight (one brief
jog in a 140 m path) so a mild-only base never gets corner signal; and the
multiplicative reward collapses to 0 through any un-trackable turn. This script
trains on the compact 28x20 m lab_chicane (two P*tanh(N*(x-c)) turns, 10 m apart,
long flat middle so a long rig can't cut both knees) using the 'corridor' reward
(additive bounded center bonus + progress + anti-jackknife; no proximity term).

Warm-starts the DEPLOYED phase-2 base (stage2_recover_spd_old); obs are local/
scale-free so the big-env policy transfers. Completion-aware best-model selection.

FEASIBILITY: a reversing 2.8 m trailer jackknifes below ~trailer-length turn
radius. min radius ~ 1/(P*N^2): N=3 -> 0.44 m (infeasible, forced jackknife),
N=0.5 -> ~3.5 m (hard but drivable) at P=5. Use CHICANE_N to calibrate; CURVE_SCALE
scales P (amplitude) as the curriculum knob.

Env knobs: STEPS, N_ENVS, DEVICE, SEED, INIT (warm-start .zip), OUT,
  CHICANE_N, CHICANE_P, CHICANE_SPACING, CURVE_SCALE, PATH_TIGHTNESS,
  SPEED_MIN, SPEED_MAX, LR, KIND_PROBS ("straight:0.2,gentle:0.15,lab_chicane:0.65"),
  CRASH_PENALTY, STOP_PENALTY.

Usage:
  python scripts/train_reverse_chicane.py                       # N=3 confirm from base
  CHICANE_N=0.5 STEPS=250000 python scripts/train_reverse_chicane.py
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

from dataclasses import replace
import numpy as np
from tractor_trailer_rl import (
    lab_chicane_config, Direction, VehicleKind, ActionMode, LookaheadMode)
from tractor_trailer_rl.train.runner import finetune


def _f(name, default):
    return float(os.environ.get(name, default))


def build_cfg():
    kinds = ["straight", "gentle", "sharp", "winding", "lab_seam", "lab_corner", "lab_chicane"]
    kp_env = os.environ.get("KIND_PROBS", "straight:0.2,gentle:0.15,lab_chicane:0.65")
    raw = {k: 0.0 for k in kinds}
    for tok in kp_env.split(","):
        name, val = tok.split(":"); raw[name.strip()] = float(val)
    tot = sum(raw.values()); probs = {k: raw[k] / tot for k in kinds}

    crash_pen = -abs(_f("CRASH_PENALTY", 500.0))
    stop_pen = abs(_f("STOP_PENALTY", 300.0))
    c = lab_chicane_config(Direction.REVERSE, VehicleKind.TRAILER)
    c = replace(
        c,
        action=replace(c.action, mode=ActionMode.STOP_SIGNAL, stop_threshold=0.8,
                       stop_noise_sigma=0.05, stop_penalty=stop_pen),
        lookahead=replace(c.lookahead, mode=LookaheadMode.FIXED_SAMPLES),
        speed_random=replace(c.speed_random,
                             explicit_min=_f("SPEED_MIN", 0.3),
                             explicit_max=_f("SPEED_MAX", 0.5)),
        reward=replace(c.reward, mode="corridor",
                       path_tightness=_f("PATH_TIGHTNESS", 2.0),
                       terminal_fail=crash_pen),
        path=replace(c.path, kind_probs=probs,
                     chicane_amplitude_m=_f("CHICANE_P", 5.0),
                     chicane_slope=_f("CHICANE_N", 3.0),
                     chicane_turn_spacing_m=_f("CHICANE_SPACING", 10.0),
                     curve_scale=_f("CURVE_SCALE", 1.0)),
    )
    return c, probs


def main():
    cfg, probs = build_cfg()
    n_envs = int(os.environ.get("N_ENVS", 8))
    device = os.environ.get("DEVICE", "auto")
    seed = int(os.environ.get("SEED", 0))
    lr = _f("LR", 2e-4)
    steps = int(os.environ.get("STEPS", 150_000))
    init = os.environ.get("INIT", os.path.expanduser(
        "~/Ben/Thesis/previous_models/lab_models_ttrl_rev_curriculum/"
        "stage2_recover_spd_old/best_model.zip"))
    out = os.environ.get("OUT", os.path.expanduser(
        "~/Ben/Thesis/previous_models/lab_models_ttrl_rev_curriculum/chicane_corridor"))

    p = cfg.path
    r_min = 1.0 / (p.chicane_amplitude_m * p.curve_scale * p.chicane_slope ** 2 + 1e-9)
    print(f"===== CHICANE  P={p.chicane_amplitude_m}*cs{p.curve_scale} N={p.chicane_slope} "
          f"spacing={p.chicane_turn_spacing_m}  (~min radius {r_min:.2f} m; trailer {cfg.vehicle.trailer_length_m} m)\n"
          f"      reward=corridor tight={cfg.reward.path_tightness}  speed=[{cfg.speed_random.explicit_min},"
          f"{cfg.speed_random.explicit_max}]  lr={lr}  steps={steps}\n"
          f"      kind_probs={ {k: round(v,2) for k,v in probs.items() if v>0} }\n"
          f"      from {init}\n      out={out} =====", flush=True)

    finetune(init, cfg, timesteps=steps, n_envs=n_envs, out_dir=out, seed=seed,
             eval_freq=20000, n_eval_episodes=10, device=device, learning_rate=lr)

    d = np.load(os.path.join(out, "logs", "evaluations.npz"))
    res = d["results"].mean(1); ln = d["ep_lengths"].mean(1); ts = d["timesteps"]
    comp = d["completions"].mean(1) if "completions" in d.files else np.zeros_like(res)
    b = int(np.argmax(comp))
    print(f"\nDONE best-by-completion {comp[b]:.2f}@{int(ts[b])//1000}k "
          f"(R={res[b]:.0f}/len{ln[b]:.0f})  final comp={comp[-1]:.2f}", flush=True)


if __name__ == "__main__":
    main()
