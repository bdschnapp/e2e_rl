# Phase-2B — Geometry-Distillation Encoder (vision rescue)

**Date:** 2026-07-07/08. **Machine:** remote RTX 3090 (mvslab), cupy backend, venv `ttrl_venv2`.
**Code:** `tractor_trailer_rl_cupy` @ commit `481c4b6` — `scripts/{geom_encoder,geom_pretrain,geom_rl,geom_obs_rl,geom_verify_cupy}.py`, `src/tractor_trailer_rl/batched/{bev_sdf,bev_sdf_env,geom_encoder_cupy,geom_obs_env}.py`. New files only; `run_stage1_sweep.py` (2A) untouched.
See `../../PHASE2_GEOMETRY_ENCODER.md` for the full method/theory write-up.

## Method (one line)
Pretrain a CNN to regress the physical state from the SDF-BEV image alone (privileged distillation); freeze it; train TD3 **from scratch** on `[true proprio δ,γ + predicted exteroceptive geometry]`.

## Results (FORWARD)

**1. Per-component recoverability** (`recoverability_forward.csv`) — test R² of predicting each
state component from the image alone (SDF+coord, ~205k samples, 25 epochs):
the exteroceptive geometry is recovered near-perfectly (R² ≥ 0.97); only internal steering `δ`
resists (0.66). → *the ego-centric BEV is a near-sufficient statistic for the path geometry.*

**2. RL from scratch on the frozen encoder** (`geom_rl_seed0_curve.csv`) — completion climbs
monotonically to **1.0 by ~300k**, no BC / warm-up / unfreeze / collapse (matches the 2A `state`
trajectory). CTE at parity ≈ 0.5–0.8 m (looser than state's ~0.25, from the ~3% encoder error).

## Status / caveats
- **Confirmed:** seed 0 → 1.0 (the recipe works).
- **Pending:** clean 3-seed (forward) and the **reverse direction** — both blocked by a stochastic
  native segfault under the geom per-step-encoder workload (NOT a torch/cupy issue; the cupy-native
  path segfaults too). Mitigations added to the code (im2col cache, smaller eval env) but not yet
  run on hardware. See `../../PHASE2_GEOMETRY_ENCODER.md §6`.
- This is a **statistical-rigor** gap, not a validity gap.

## Files
- `recoverability_forward.csv` — per-component test R².
- `geom_rl_seed0_curve.csv` — completion vs timesteps (seed 0, forward).
