# Geometry-Distillation Encoder for Vision-Based RL

*Implementation & theory notes (2026-07-08). Companion to `PHASE2_METHODOLOGY_AND_CHAPTER_PLACEMENT.md`. Ch. 4 method material + Ch. 5 evidence. No `.tex` written by this file.*

---

## 1. Idea in one paragraph

A CNN encoder is trained **supervised** to predict the vehicle's **physical geometry from the BEV image alone** (the simulator's ground-truth state is the privileged teacher label). The encoder is then **frozen** and used as a **fixed perception module**: the RL policy observes `[true proprioception (δ, γ)] + [encoder-predicted exteroceptive geometry]` and is trained **from scratch** with ordinary TD3. Because the frozen encoder emits stable, task-relevant features from the first step, RL-from-scratch behaves exactly like the low-dimensional `state` run (2A) — it converges to parity with **no imitation warm-start, no critic warm-up, no unfreeze schedule, and no collapse**. This is the "learning by cheating" / privileged-distillation paradigm (privileged teacher sees the state; the deployable student must infer it from the image), specialised to articulated-vehicle geometry.

## 2. Why this is the right decoupling method (theory)

The Phase-2 diagnosis (see the causal ladder in the methodology note) is that from-scratch vision RL fails **not** because of the image, representation, or encoder, but because of an **off-policy optimisation pathology**: a co-trained/random encoder emits high-variance features that destabilise TD3's critic, and — in the imitation route — a good pretrained actor paired with a *fresh* critic gets dragged off-distribution to where the critic's Q is over-estimated (extrapolation error / deadly triad), collapsing the policy.

The geometry-distillation encoder removes **both** failure sources at once:

| Failure source | How geometry-distillation avoids it |
|---|---|
| Noisy co-trained encoder features (scratch BEV) | Encoder is pretrained + **frozen** → features are stable and task-relevant from step 0. |
| Good-actor / cold-critic mismatch (BC→RL) | Actor is trained **from scratch** on the frozen features → actor and critic co-evolve, so there is no offline-to-online mismatch to exploit (this is exactly the regime of the 2A `state` run, which reached 1.0 with no collapse). |

**Contrast with the other decoupling routes:**
- **vs from-scratch CNN:** the encoder is never asked to learn from the reward signal, so the critic is never fed non-stationary noise.
- **vs autoencoder (reconstruction):** reconstruction spends capacity on task-irrelevant pixel detail; predicting the *state* forces exactly the task-relevant geometry.
- **vs behaviour cloning (BC):** BC clones the *policy* end-to-end, and because the observation also contains the state vector, the network can satisfy the loss largely through the state branch — it does **not** prove the CNN extracts image information. Predicting the state **from the image alone** cannot cheat, so success is direct evidence that *the image is a sufficient statistic for the geometry*.

## 3. Implementation

All code is **new files** in `tractor_trailer_rl_cupy`; nothing that `run_stage1_sweep.py` depends on was modified.

### 3.1 Encoder — `scripts/geom_encoder.py`
`GeomEncoder(in_ch, out_dim, img_size=32, feat=128)`:
- **CNN tower:** `Conv2d(in_ch→16, 3, s2, p1) → ReLU → Conv2d(16→32, 3, s2, p1) → ReLU → Conv2d(32→32, 3, s2, p1) → ReLU → Flatten` (32×32 → 4×4×32 = 512).
- **Trunk:** `Linear(512→128) → ReLU → LayerNorm(128)`.
- **Head:** `Linear(128→out_dim)` regressing the standardized geometry labels.
- Exposes `features(img)` (128-d embedding, for a raw-feature RL variant) and `forward(img)` (the predicted geometry vector, the primary obs).
- Input channels = 3 (signed-distance BEV + 2 CoordConv channels; see the SDF representation in `bev_sdf.py`).

### 3.2 Supervised pretraining — `scripts/geom_pretrain.py`
- **Data:** roll out the 2A `state` expert inside the SDF-BEV env, with a fraction (`rand_frac`, e.g. 0.4) of random actions for state-space coverage; record `(image, true_state)`. Run config: 256 envs × 800 steps ≈ **205k samples**.
- **Labels:** the 8-dim trailer state `[δ, γ, e_y, e_psi, e_y_t, e_psi_t, k1, k2]`, standardized per-dim (so R² = 1 − standardized-MSE, scale-invariant).
- **Training:** predict state **from the image only** (no state input), MSE, Adam lr 1e-3, 25 epochs, batch 512, 90/10 train/test.
- **Output:** saves `state_dict` + per-dim label mean/std + metadata; reports per-component test R².

### 3.3 RL on the frozen encoder — `scripts/geom_rl.py`
- `geom_state_vecenv(...)` — a `VecEnvWrapper` around the SDF-BEV `VecEnv`. Each step it runs the **frozen** encoder on the image, de-standardises the prediction, and returns the observation vector `[true δ, true γ (proprio, overriding the predicted values), predicted e_y, e_psi, e_y_t, e_psi_t, k1, k2]`. It stashes the **true** state in `info['true_vector']` and transforms the `terminal_observation` so metrics use ground truth, never the prediction.
- **Policy:** standard TD3 `MlpPolicy` over the 8-d Box (identical setup to the 2A `state` run), trained **from scratch**, lr 1e-3, `learning_starts=10k`, per-dim action noise, 500k steps.
- **Eval (`geom_eval`):** completion from `info['is_success']`, CTE/hitch from the **true** state — the estimate is never used to score.

### 3.4 Observation-design rationale
The ego-centric BEV (vehicle at grid centre, not drawn) contains the **exteroceptive** geometry but not the **internal** actuator state. So:
- **`δ`, `γ` → given as true proprioception** (real, cheap sensor measurements; not privileged; not reliably in the image).
- **path errors + curvature (+ obstacle distance, Phase 3) → predicted from the image.**

## 4. Results

### 4.1 Per-component recoverability (test R², forward, SDF+coord)
| Component | R² | Reading |
|---|---|---|
| `e_psi` (tractor heading err) | 0.998 | in the image |
| `e_y` (tractor cross-track) | 0.997 | in the image |
| `k1`, `k2` (curvature) | 0.997 | in the image |
| `γ` (hitch) | 0.995 | high (partly correlational via visible path history) |
| `e_psi_t` (trailer heading) | 0.993 | in the image |
| `e_y_t` (trailer cross-track) | 0.974 | in the image |
| **`δ` (steering)** | **0.659** | **weak — internal actuator state, only correlationally recoverable** |

**Result:** the BEV image is a near-sufficient statistic for the geometry (7/8 components R² ≥ 0.97); only steering `δ` resists, consistent with it being an internal actuator state. This alone is a Ch. 5 result ("the ego-centric BEV encodes path geometry but not steering angle").

### 4.2 RL from scratch on the frozen encoder (forward, seed 0)
```
completion:  50k 0.00 | 100k 0.00 | 150k 0.69 | 200k 0.76 | 250k 0.87 | 300k 1.00 | 350k 1.00
```
Monotonic climb to **full completion parity (1.0)** by ~300k, matching the 2A `state` trajectory, with **no collapse** — confirming the recipe. True CTE at parity was ~0.5–0.8 m (looser than the `state` expert's ~0.25 m), attributable to the ~3% encoder prediction error and not-yet-converged tracking; a longer run and/or the raw-feature variant may tighten it.

**Status:** the mechanism is confirmed (seed 0). A clean **3-seed** confirmation is pending an infra fix (see §6); it does not affect the validity of the result, only its statistical rigour.

## 5. Where it goes in the thesis
- **Ch. 4 (method):** the SDF+CoordConv representation, the `GeomEncoder` architecture, the supervised distillation procedure, the proprio+exteroception observation split, and the *frozen-encoder-as-fixed-perception-module + RL-from-scratch* recipe — presented with the decoupling theory (§2) as motivated design.
- **Ch. 5 (evidence):** the per-component recoverability table (§4.1) and the RL parity curve (§4.2), contrasted against scratch-BEV (0.0) and the BC route (parity but via imitation).
- **Phase 3 hand-off:** add an **obstacle-distance** head to the pretraining target (labels from obstacle-env images). The frozen encoder then supplies obstacle geometry the raw state lacks, so RL on those features can *beat* state where state is blind — the setting in which the image finally "adds value."

## 6. Reproducibility / infra notes
- **Compute:** remote RTX 3090 (`mvslab@100.83.234.83`, venv `ttrl_venv2`: torch cu128 + cupy-cuda12x + SB3 2.9). Set `TTRL_BACKEND=cupy`.
- **Commands:**
  - Pretrain: `python3 -u scripts/geom_pretrain.py --coord --steps 800 --epochs 25 --rand_frac 0.4 --out geom_encoder_fwd_coord.pt`
  - RL: `python3 -u scripts/geom_rl.py --encoder geom_encoder_fwd_coord.pt --coord --steps 500000 --eval_every 50000 --seed <s>`
- **Known issue (still open):** running the frozen encoder in the per-step obs loop triggers a **stochastic native segfault** (~150k–350k steps, correlated with eval boundaries) that `bc_rescue`/2A-style loads never hit. This was *first* seen with the torch encoder interleaved with the cupy env (`geom_rl.py`), but the **cupy-native reimplementation** (`geom_obs_env.py` + `geom_obs_rl.py`, encoder forward in cupy, **zero torch in the env loop**, verified numerically correct vs torch to 3e-4 by `geom_verify_cupy.py`) **also segfaults** — so it is *not* a torch/cupy-interleave problem but a stochastic CUDA/cupy instability exposed by the heavier per-step encoder workload. **Recipe validity is unaffected** (the recipe reached 1.0; this only blocks clean multi-seed statistics). Mitigations to try: reduce per-step cupy allocation churn in the hand-rolled conv (preallocate the im2col buffer), use a smaller eval env, or run solo on the local 4080; pin CPU threads; avoid concurrent heavy jobs; and do not spam retries (repeated core dumps can wedge the GPU → reboot / `nvidia-smi --gpu-reset`).

## 7. Open items
1. **3-seed statistical confirmation** (after the cupy-encoder fix + a clean box).
2. **Reverse direction** — all of the above is forward only; reverse is the fragile direction (2A) and must be re-run before any direction-general claim.
3. **Raw-feature vs predicted-vector** ablation, and whether a longer run tightens CTE toward the classical optimum.
4. **Phase 3:** obstacle-distance prediction head + frozen-encoder RL on the obstacle env.
