# Phase-2 Perception Study — Methodology, Findings, and Chapter Placement

*Working notes (2026-07-08). Reference doc for writing up the observation-space / high-dimensional-perception work. No `.tex` is written or modified by this file — it is a planning/summary artifact.*

---

## 0. TL;DR — where does the new material go?

- **Ch. 4 (Learning Framework):** the **theory and methods** we designed — the SDF/graded observation representation, the encoder architecture, and a new **training-methodology** section (why from-scratch high-dim RL fails, and the imitation-warm-start / frozen-encoder / safe-adaptation methods that follow). Present the failure *mechanism* here as **motivated hypothesis**.
- **Ch. 5 (Results & Discussion):** the **experiments** — the 2A dimensionality ablation and the 2B **causal ladder** + rescue numbers that *confirm* the Ch. 4 mechanism.
- **Rule applied:** *"what we designed and why" → framework chapter; "what we ran and found" → results chapter.*
- **Ch. 4 title:** broaden it to signal it now owns *how we train perception policies*, not just observation/reward design (candidates in §4).

---

## 1. Chapter-5 narrative arc (Phase 1 → 2 → 3; each phase motivates the next)

The results chapter is a single build-up — each phase answers a question and *sets up* the next:

1. **Phase 1 — model & reward structure.** Which algorithm and reward formulation induce stable articulated-vehicle tracking (esp. reverse)? → TD3 + multiplicative reward. *Motivates:* a fixed, trusted recipe to hold constant while varying observations.
2. **Phase 2 — observation structure & training stability.**
   - *2A:* a fair observation ablation shows reliability degrades **monotonically with observation dimensionality** (worst in reverse); BEV-image-from-scratch fails outright.
   - *2B:* the failure is an **RL-optimization pathology, not a perception limit** — a causal ladder isolates it to off-policy actor-vs-cold-critic instability, and **decoupling the encoder** (imitation warm-start, and more cleanly **geometry distillation** → a frozen state-estimating encoder) restores parity with RL-from-scratch.
   - *Motivates:* a **trainable high-dimensional perception→control stack** and the frozen-encoder recipe Phase 3 needs.
3. **Phase 3 — the integration: obstacle-aware reverse maneuvering.** Put it together where it matters: obstacles make the low-dim state **insufficient** (it is blind to them), so the image must carry information the state cannot. Extend the geometry encoder with an **obstacle-distance head** and train the frozen-encoder policy on the obstacle env, **in reverse**. This is the payoff — high-dimensional observation spaces enabling **end-to-end planning *and* control**, not merely a fancier controller.

**Scope (confirmed):** *reverse is in scope throughout* (the thesis is about a *reversing* shunt truck; 2A already shows reverse is the fragile direction), and *Phase 3 / obstacles is in scope as the culmination of this chapter*, not future work.

---

## 2. What we established (the work done)

### 2.1 2A — observation-space ablation (fair, apples-to-apples)
Fixed recipe (TD3 + multiplicative reward, `shunt` preset, mild paths, 3 seeds, PP-safe eval pool), **only the observation varied**, **no rescue tricks**. Data archived in `data/phase2_obs_ablation_2A/` (summary/curves/manifest + README).

Best-checkpoint completion (mean over 3 seeds):

| Obs (dim) | Forward | Reverse |
|---|---|---|
| state (8) | **1.00** | **0.97** |
| lidar8 (16) | **1.00** | 0.90 |
| lidar24 (32) | **1.00** | 0.63 (1 seed → 0.0) |
| lidar64 (72) | 0.67 (1 seed → 0.0) | 0.31 (2 seeds → 0.0) |
| bev (32×32 image) | **0.00** | (deferred to 2B) |

**Finding:** reliability degrades **monotonically with observation dimensionality**, and the effect **amplifies in the harder reverse direction**. Low-dimensional observations are not merely *sufficient* but more *robust*; adding input dimensions raises seed variance and then causes outright seed collapses, culminating in total failure for the image. This is the motivating result for 2B.

### 2.2 2B — rescuing the BEV image (methods + causal diagnosis)
All 2B code is **new files only** (nothing `run_stage1_sweep.py` depends on was touched, so 2A remains byte-for-byte reproducible): `src/tractor_trailer_rl/batched/{bev_sdf.py, bev_sdf_env.py}`, `scripts/{sb3_stability.py, screen_bev_rescue.py, bc_rescue.py}`.

**Representation.** Replaced binary occupancy with a **signed-distance (graded) BEV** (+1 at corridor centreline → 0 at edge → negative outside), optionally with **2 CoordConv channels**. Rationale: binary occupancy is flat almost everywhere, so the task-relevant quantity (clearance) lives only in sparse edge gradients a random CNN turns into critic-destabilising noise; the SDF is a dense, task-aligned field. (Independently corroborated by a collaborator's Nav2 project, where *graded* channels were the single largest contributor to breaking a training plateau.)

**The causal ladder** (forward, SDF+coord BEV; seed 0 unless noted):

| Configuration | Completion | What it isolates |
|---|---|---|
| BEV occupancy, scratch (from 2A) | 0.00 | RL-from-scratch on vision fails |
| + SDF | 0.00 | representation alone is not the fix |
| + SDF + CoordConv | 0.00 | — |
| + stability extractor (LayerNorm + separate actor/critic encoders) | 0.00 | light stability alone is not the fix |
| **BC warm-start (clone the 2A state expert into the vision actor)** | **1.00** (CTE ≈ 0.26) | **imitation rescues vision → full state parity** |
| BC + naive TD3 fine-tune (all trainable) | 0.00 | RL *destroys* the good policy |
| BC + RL, **encoder frozen** | 0.00 | …the encoder is *not* the culprit |
| BC + RL, **actor frozen (critic-only)** | **1.00** (3-seed 1.00/1.00/0.92) | **the actor chasing the cold critic is the sole destroyer** |

**Diagnosis (the key result):** vision RL does not fail because of the image, the representation, or the encoder. It fails because **TD3's actor is dragged into collapse by a randomly-initialised critic** — an off-policy extrapolation-error / deadly-triad instability. Imitation supplies a perfect policy; standard RL then re-destroys it, and the destruction is entirely attributable to the actor update (freezing the actor holds parity; freezing only the encoder does not).

**BC = behavior cloning of the *policy* (not a standalone encoder pretrain).** We roll out the 2A `state` expert inside the SDF-BEV env, record `(Dict obs {vector,image}, expert action)`, and supervised-regress the TD3 actor (encoder + head, end-to-end) onto the expert actions. *Caveat:* the observation contains the full state vector and the expert's actions derive from it, so on this state-saturated task BC may satisfy the loss largely via the state branch — i.e. it yields a competent, non-collapsing policy but does **not prove the CNN learned strong image features**. That question is a Phase-3 concern.

### 2.3 Safe-adaptation exploration (and its honest limits)
We tested whether RL can fine-tune the BC policy *without* collapse:
- **freeze-actor (critic-only RL):** holds parity reliably (3-seed 1.00/1.00/0.92) — but the policy never changes.
- **warm the critic (100k, actor frozen) → unfreeze actor head, encoder frozen throughout:**
  - at actor LR 1e-4: **permanent collapse, 3/3 seeds** (too slow to recover after the unfreeze dip).
  - at actor LR 1e-3: **unreliable** — 1/3 seeds permanent collapse, 2/3 recover but oscillate; CTE did **not** improve (post-unfreeze CTE 0.42–0.92 vs BC ≈ 0.26).
- **Conclusion:** *unconstrained* actor unfreezing is not a usable recipe. Warming the critic does not help because it improves Q only where the actor already is; it does nothing to stop the actor walking off to off-distribution actions where Q is over-estimated.

**Two distinct claims that must not be conflated:**

| Claim | Improvement axis | Testable on |
|---|---|---|
| **Safe adaptation** — RL improves reward/tracking *without* collapse | CTE (headroom ~0.25 → ~0.12) | the **lane task** (no obstacles needed) |
| **Image adds value** — image supplies info the state cannot | completion on obstacle courses | **obstacles (Phase 3)** |

**Open, honest caveat on lane-task headroom:** pure-pursuit reaches ~0.12 m CTE, but that is a *different controller class* (continuous steering directly minimising cross-track); our RL agent uses steering-*rate* actions + a stop channel + exploration noise + a saturating multiplicative reward, and 2A's from-scratch `state` TD3 converged to CTE ≈ 0.27–0.46. So it is **unproven that RL has headroom below ~0.25**. A principled anti-collapse method (**TD3+BC** action-anchor, or **frozen-base+residual**) would *prevent the collapse* (it constrains the actor to the on-distribution trust region where Q is reliable — mechanistically different from warming the critic), but preventing collapse ≠ producing improvement. On the lane task it may simply reproduce parity. **Cheap way to decide before investing:** inspect the 2A `state` learning curves — if CTE *plateaued* at ~0.25, there is no lane headroom and safe-adaptation belongs to Phase 3; if still *dropping*, TD3+BC on the lane task is worth running.

### 2.4 Known gaps
- **Everything in 2B so far is FORWARD only.** Reverse (non-minimum-phase, and already the fragile direction in 2A) is untested; the BC init is likely weaker and the fine-tune collapse likely worse. The reverse `state` expert exists (`results_stage2_obs/models/td3__reverse__state__multiplicative__s0.zip`) to replicate the pipeline. This must be run before any 2B claim is stated as direction-general.

---

## 3. Why this placement (Ch. 4 vs Ch. 5)

- The **representation design, encoder architecture, and training methodology** are things we *built and justified* → **framework** material (Ch. 4). They are reusable method, independent of the specific numbers.
- The **failure mechanism** (actor-vs-cold-critic) is best introduced in Ch. 4 as **motivated hypothesis** grounded in the off-policy RL literature (extrapolation error / deadly triad; TD3+BC; SAC-AE stop-gradient encoder), *then confirmed empirically* by the Ch. 5 causal ladder. Do **not** assert the empirical conclusion in Ch. 4 before Ch. 5 presents the evidence.
- The **ablations and curves** (2A dimensionality sweep; 2B causal ladder; safe-adaptation runs) are experiments → **results** (Ch. 5). The causal ladder is the Ch. 5 highlight: it is the evidence for the theory stated in Ch. 4.

### Proposed Ch. 4 section additions
1. *Observation & representation design* (extend existing perception section): state / lidar / BEV, **+ signed-distance (graded) BEV + CoordConv**, with rationale; encoder = CNN + LayerNorm + separate actor/critic extractors.
2. *Training high-dimensional-observation policies* (new): the off-policy instability mechanism (hypothesis) → imitation warm-start to decouple the encoder → frozen encoder → **safe adaptation** (freeze-actor / TD3+BC anchor / frozen-base+residual).

### Ch. 4 title candidates (broaden from "Observation Design, Reward Formulation")
- *Learning Framework: Observation & Representation Design, Reward Formulation, and Policy Training*
- *Learning Framework: Observations, Rewards, and Training High-Dimensional Policies*
- *Learning Framework: Perception, Reward, and Training Methodology*

---

## 4. Provenance (data, code, compute)

- **2A data (archived):** `e2e_rl/thesis/data/phase2_obs_ablation_2A/` — `summary.csv`, `curves.csv`, `manifest.jsonl`, `README.md`. Model `.zip` checkpoints excluded (large; on the remote).
- **2B code (new files):** in `tractor_trailer_rl_cupy` — `src/tractor_trailer_rl/batched/{bev_sdf.py, bev_sdf_env.py}`, `scripts/{sb3_stability.py, screen_bev_rescue.py, bc_rescue.py}`. Nothing `run_stage1_sweep.py` uses was modified.
- **2B logs / curves (on the remote):** `bc_sdf_*.log`, per-seed `bc_curve_s*_warm*_lr*.csv`.
- **Expert policies (on the remote):** `results_stage2_obs/models/td3__{forward,reverse}__state__multiplicative__s0.zip`.
- **Compute:** long training on the remote RTX 3090 (`mvslab@100.83.234.83`, venv `ttrl_venv2`: torch cu128 + cupy-cuda12x + SB3 2.9); local RTX 4080 reserved for development (avoid concurrent heavy jobs — one heavy job at a time).

---

## 5. Suggested next steps (not yet run)
1. **Decide lane headroom** (free): read the 2A `state` CTE curves — plateaued vs still-dropping — to know whether "safe adaptation improves reward" is even testable on the lane task.
2. **Reverse replication** of the 2B BC-rescue (and the winning safe-adaptation method), expecting it to be harder; the forward↔reverse contrast is itself a result.
3. **Phase 3 (obstacles):** the frozen-encoder + TD3+BC / residual toolkit applied where the image supplies information the state cannot — the setting in which vision can finally beat state and safe adaptation is both necessary and measurable.
