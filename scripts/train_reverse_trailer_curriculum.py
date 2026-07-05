"""3-stage recovery curriculum for the hard REVERSE-TRAILER case.

The proven mild baseline plateaus at ~256/len506 and then REGRESSES: the policy
never learns to hold the trailer stable in reverse for a full episode (reverse
hitch dynamics are non-minimum-phase + the jackknife wall at |hitch|>pi/2). This
curriculum warm-starts through three stages, each continuing from the previous
best, to bootstrap a robust reverse-trailer tracker:

  Stage 1 — STABILIZE. Easy paths (straight+gentle), SLOW speed (hitch dynamics
            yaw_rate=v/L*sin(dhitch) are slower/controllable at low v), spawn
            on-path. Goal: drive in reverse keeping the hitch stable for the full
            1000 steps. FIXED lookahead + multiplicative reward (anti-jackknife
            term exp(-2|hitch|)).
  Stage 2 — RECOVER. Warm-start S1. Still easy paths + slow, but spawn in BAD
            poses (off-lane / mis-heading / open hitch) so it learns to recover.
  Stage 3 — GENERALIZE. Warm-start S2. Full path mixture incl. lab_corner, speed
            ramped back to the deploy range (0.5-0.8 m/s), mild residual spawn
            perturbation to keep the recovery skill.

Validate natively on lab_corner (scratchpad/rev_trailer_eval.py) AFTER stage 3,
target ~len1000 / positive reward like reverse_tractor_only, BEFORE deploy.

Usage:
  python scripts/train_reverse_trailer_curriculum.py                 # full run
  STAGES="1" STEPS1=50000 python scripts/train_reverse_trailer_curriculum.py   # smoke S1
  STAGES="2 3" python scripts/train_reverse_trailer_curriculum.py    # resume from S1 best

Env knobs: STAGES, STEPS1/2/3, N_ENVS, OUT, DEVICE,
  SPEED (fixed) or SPEED_MIN/SPEED_MAX (range — the winning recipe is 0.3-0.5),
  S2_LAT, S2_HEAD, S2_HITCH (stage-2 spawn half-ranges),
  S3_LAT, S3_HEAD, S3_HITCH (stage-3 residual spawn half-ranges),
  INIT3 (stage-3 warm-start .zip; default = this run's stage-2 best),
  S3_OUT (stage-3 output dir; default = <OUT>/stage3_generalize),
  LAB_FOCUS=1 (stage-3: drop synthetic sharp/winding, focus the real lab curves).

Fine-tune the DEPLOYED phase-2 model on phase-3 corners (0.3-0.5 range):
  STAGES=3 SPEED_MIN=0.3 SPEED_MAX=0.5 \
  INIT3=~/Ben/Thesis/previous_models/lab_models_ttrl_rev_curriculum/stage2_recover_spd_old/best_model.zip \
  S3_OUT=~/Ben/Thesis/previous_models/lab_models_ttrl_rev_curriculum/stage3_corner_from_spd_old \
  python scripts/train_reverse_trailer_curriculum.py
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
    lab_config, Direction, VehicleKind, ActionMode, LookaheadMode, SpawnConfig)
from tractor_trailer_rl.train.runner import train, finetune

DIR, KIND = Direction.REVERSE, VehicleKind.TRAILER


def _f(name, default):
    return float(os.environ.get(name, default))


def base_cfg(speed_min, speed_max, mild_only, spawn, reward_mode):
    """Shared knobs across stages: FIXED lookahead, stop_signal action with the
    stop dim suppressed (deploy-layout parity). reward_mode is per-stage:
      'dense'          — additive: +3·clip(-xd) − path_quad − 10·max(0,|hitch|-0.4)²
                         (well-shaped, learnable base signal; bootstraps stability).
      'multiplicative' — deploy-matched: 5·pt·exp(-2|hitch|) + floor (hard to learn
                         cold, but the shaping the other 3 deployed models use).
    Reward mode affects ONLY the training scalar, not obs/action — so a dense-
    trained base is export/deploy-identical to a multiplicative one."""
    c = lab_config(DIR, KIND)
    # Penalty hierarchy: crash/jackknife WORSE than a deliberate stop, so the safe
    # failure (stop) is preferred over the dangerous one (crash), driving well best.
    # crash -500 < stop -300 < drive(+per-step) / success +200. NOTE: crash MUST stay
    # at the proven -500 — raising it to -1000 floods cold-start training with huge
    # negatives (everything jackknifes early) and the policy never escapes (collapsed
    # Stage-1 to -945/len41 on 2026-06-29). Env knobs CRASH_PENALTY / STOP_PENALTY.
    crash_pen = -abs(_f("CRASH_PENALTY", 500.0))
    stop_pen = abs(_f("STOP_PENALTY", 300.0))
    return replace(
        c,
        action=replace(c.action, mode=ActionMode.STOP_SIGNAL,
                       stop_threshold=0.8, stop_noise_sigma=0.05, stop_penalty=stop_pen),
        reward=replace(c.reward, mode=reward_mode, terminal_fail=crash_pen),
        lookahead=replace(c.lookahead, mode=LookaheadMode.FIXED_SAMPLES),
        speed_random=replace(c.speed_random, explicit_min=speed_min, explicit_max=speed_max),
        path=replace(c.path, mild_only=mild_only),
        spawn=spawn,
    )


def summarize(out, tag):
    d = np.load(os.path.join(out, "logs", "evaluations.npz"))
    res = d["results"].mean(1); ln = d["ep_lengths"].mean(1); ts = d["timesteps"]
    b = int(np.argmax(res))
    print(f"  [{tag}] best={res[b]:.1f}/len{ln[b]:.0f}@{int(ts[b])//1000}k  "
          f"final={res[-1]:.1f}/len{ln[-1]:.0f}", flush=True)


def main():
    n_envs = int(os.environ.get("N_ENVS", 8))
    device = os.environ.get("DEVICE", "auto")
    # Per-stage LR: 1e-3 for the from-scratch base (proven to reach 2382 on S1),
    # 3e-4 for the warm-start finetune stages (damps the TD3 divergence that
    # collapsed the first Stage-3). Single LR= overrides all three.
    lr_all = os.environ.get("LR")
    lr1 = float(lr_all) if lr_all else float(os.environ.get("LR1", 1e-3))
    lr2 = float(lr_all) if lr_all else float(os.environ.get("LR2", 3e-4))
    lr3 = float(lr_all) if lr_all else float(os.environ.get("LR3", 3e-4))
    out_root = os.environ.get("OUT", os.path.expanduser(
        "~/Ben/Thesis/previous_models/lab_models_ttrl_rev_curriculum"))
    stages = os.environ.get("STAGES", "1 2 3").split()
    s1 = os.path.join(out_root, "stage1_stabilize")
    s2 = os.path.join(out_root, "stage2_recover")
    s3 = os.path.join(out_root, "stage3_generalize")

    # Reward per stage. multiplicative is the deploy-matched shaping AND it proved
    # learnable here (Stage-1 base hit 2382/straight-8-of-8, gentle 5/8 @260k), so
    # it's the default throughout. 'dense' (additive +3·clip(-xd) − path_quad −
    # 10·max(0,|hitch|-0.4)²) is the smoother fallback if a stage won't stabilize.
    rw1 = os.environ.get("REWARD_S1", "multiplicative")
    rw2 = os.environ.get("REWARD_S2", "multiplicative")
    rw3 = os.environ.get("REWARD_S3", "multiplicative")

    # Training speed. SPEED sets a single fixed speed; SPEED_MIN/SPEED_MAX override
    # it with a RANGE (within-episode randomization). Diagnostic 2026-06-29: a
    # 0.3-0.5 RANGE base tracks straights at 3.2° hitch vs 10.9° for fixed-0.5 —
    # randomization regularizes out the brittle fixed-speed limit cycle. The
    # 0.5->0.8 JUMP between stages is the separate failure (collapsed Stage-3); a
    # consistent range avoids both. SEED lets us run controlled A/Bs.
    spd = _f("SPEED", 0.5)
    spd_min = _f("SPEED_MIN", spd)
    spd_max = _f("SPEED_MAX", spd)
    seed = int(os.environ.get("SEED", 0))

    if "1" in stages:
        cfg = base_cfg(spd_min, spd_max, mild_only=True, spawn=SpawnConfig(), reward_mode=rw1)
        steps = int(os.environ.get("STEPS1", 300_000))
        print(f"\n===== STAGE 1 STABILIZE  speed=[{spd_min},{spd_max}] seed={seed}  steps={steps}  out={s1} =====", flush=True)
        train(cfg, timesteps=steps, n_envs=n_envs, out_dir=s1, seed=seed,
              eval_freq=20000, n_eval_episodes=10, device=device, learning_rate=lr1)
        summarize(s1, "S1")

    if "2" in stages:
        spawn = SpawnConfig(lateral_offset_m=_f("S2_LAT", 0.6),
                            heading_offset_rad=_f("S2_HEAD", 0.25),
                            hitch_offset_rad=_f("S2_HITCH", 0.30))
        cfg = base_cfg(spd_min, spd_max, mild_only=True, spawn=spawn, reward_mode=rw2)
        steps = int(os.environ.get("STEPS2", 200_000))
        init = os.path.join(s1, "best_model.zip")
        print(f"\n===== STAGE 2 RECOVER  speed=[{spd_min},{spd_max}]  from {init}  steps={steps}  out={s2} =====", flush=True)
        finetune(init, cfg, timesteps=steps, n_envs=n_envs, out_dir=s2, seed=seed,
                 eval_freq=20000, n_eval_episodes=10, device=device, learning_rate=lr2)
        summarize(s2, "S2")

    if "3" in stages:
        spawn = SpawnConfig(lateral_offset_m=_f("S3_LAT", 0.15),
                            heading_offset_rad=_f("S3_HEAD", 0.08),
                            hitch_offset_rad=_f("S3_HITCH", 0.10))
        cfg = base_cfg(spd_min, spd_max, mild_only=False, spawn=spawn, reward_mode=rw3)
        # Path mix: KIND_PROBS ("straight:0.3,gentle:0.3,lab_seam:0.2,lab_corner:0.2")
        # gives full control; else LAB_FOCUS=1 uses a lab-curves preset; else the
        # base v18 mix. Missing kinds default to 0.0; the list is renormalized.
        kinds = ["straight", "gentle", "sharp", "winding", "lab_seam", "lab_corner"]
        kp_env = os.environ.get("KIND_PROBS")
        if kp_env:
            raw = {k: 0.0 for k in kinds}
            for tok in kp_env.split(","):
                name, val = tok.split(":"); raw[name.strip()] = float(val)
            tot = sum(raw.values()); probs = {k: raw[k] / tot for k in kinds}
            cfg = replace(cfg, path=replace(cfg.path, kind_probs=probs))
            print(f"  [S3] KIND_PROBS -> {probs}", flush=True)
        elif os.environ.get("LAB_FOCUS", "0") == "1":
            probs = {"straight": 0.20, "gentle": 0.15, "sharp": 0.0,
                     "winding": 0.0, "lab_seam": 0.25, "lab_corner": 0.40}
            cfg = replace(cfg, path=replace(cfg.path, kind_probs=probs))
            print(f"  [S3] LAB_FOCUS on -> kind_probs={probs}", flush=True)
        # CURVE_SCALE (<1.0) softens lab_seam/lab_corner bend amplitude — the curvature
        # curriculum rung: ramp 0.4 -> 0.7 -> 1.0 so the mild-only base meets corners
        # gradually instead of the full-sharpness jump that diverged TD3.
        curve_scale = _f("CURVE_SCALE", 1.0)
        if curve_scale != 1.0:
            cfg = replace(cfg, path=replace(cfg.path, curve_scale=curve_scale))
            print(f"  [S3] curve_scale={curve_scale}", flush=True)
        steps = int(os.environ.get("STEPS3", 300_000))
        # INIT3 overrides the warm-start source (default = this run's Stage-2 output).
        # For a corner fine-tune of the DEPLOYED phase-2 model, point INIT3 at
        # stage2_recover_spd_old (the RViz-validated 0.4deg-hitch base), NOT the
        # noisier same-dir stage2_recover. S3_OUT overrides the output dir so a
        # fine-tune run doesn't clobber a prior stage3_generalize.
        init = os.environ.get("INIT3") or os.path.join(s2, "best_model.zip")
        out3 = os.environ.get("S3_OUT") or s3
        print(f"\n===== STAGE 3 GENERALIZE  speed=[{spd_min},{spd_max}]  from {init}  steps={steps}  out={out3} =====", flush=True)
        finetune(init, cfg, timesteps=steps, n_envs=n_envs, out_dir=out3, seed=seed,
                 eval_freq=20000, n_eval_episodes=10, device=device, learning_rate=lr3)
        summarize(out3, "S3")

    print("\nCURRICULUM DONE", flush=True)


if __name__ == "__main__":
    main()
