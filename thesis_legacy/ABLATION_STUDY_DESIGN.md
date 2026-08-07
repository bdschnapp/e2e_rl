# Ablation study design (precise, justified)

Purpose: define an ablation whose goal is defensible — each experiment isolates one
design axis on a task where differences are *attributable to that axis*, not to an
unsolvable corner or a mis-specified plant.

## 0. Scope decisions (locked) + coherence fixes

**Decisions (user):**
- **Plant model:** keep the **dynamic bicycle**. Tesla-S params are a **placeholder**
  to be replaced with **real full-size shunt-truck params before the final ablation /
  draft** (a `VehicleConfig` swap: wheelbase, trailer length, mass, inertia, tyre
  stiffness). Not kinematic — deployment (AgileX) is validated separately and left as-is.
- **Scale:** the **whole thesis uses full-size params**; **AgileX is referenced only
  in Ch.6 (sim-to-real).** The scalar `tractor_trailer_rl` env stays as-is (it works
  for sim-to-real); the **CuPy env is free to change for the ablations.**

| Issue | Current | Decision |
|---|---|---|
| Plant model | dynamic bicycle, Tesla params | **Keep dynamic.** Tesla → **real shunt-truck params (TODO before final run)**. |
| Vehicle params | Tesla + semi geometry | **Full-size shunt truck** (terminal tractor: source real wheelbase ≈4 m, trailer ≈12 m, mass, tyre). Tesla is the interim placeholder. |
| World / path scale | 150×90 m, 5 m lane, 10 m trailer | **Keep full-size** (already semi-scale). AgileX small-world config is Ch.6 only. |
| Path distribution | mix incl. lab-specific tight kinds (`lab_seam/corner/chicane`) | **Use full-size-appropriate kinds only** — Stage 1: straight+gentle; Stage 2: +sharp+winding. Drop the AgileX `lab_*` kinds from the full-size ablation (they belong to Ch.6). |
| Speed | `speed_random` on → false ~56% completion ceiling (low-speed episodes time out) | **Fixed speed** for the no-obstacle ablation (matches SB3); scale `max_episode_steps` so a good policy always completes. |
| `dt` = 0.1 (10 Hz) | — | **Keep** — standard embedded rate. |

## 1. Stage 1 — Stabilization (PRIMARY, controlled ablation)

**Task.** Straight + gentle paths only (min radius comfortably above the vehicle's
turning limit, e.g. ≥8–10× wheelbase), fixed speed, small spawn perturbation
(cross-track + heading + hitch) so the policy must *stabilize*, not just hold a line.
Solvable from scratch → clean attribution. Run **forward and reverse** (reverse
stage-1 = stabilize the hitch while backing; the non-minimum-phase case is the
interesting one).

**Metrics** (per config, ≥3 seeds, mean ± 95% CI): completion, trailer CTE
(mean/RMS), max |hitch|, jackknife rate, and **sample-efficiency** (env-steps to a
stabilization threshold). Report seed variance explicitly.

**Ablated axes (one at a time; everything else at the locked baseline):**

| Axis | Levels | Research question |
|---|---|---|
| Observation | state / lidar-{4,8,16,24,32} / BEV{scratch,scaled-CNN,AE,UNet} | Does richer sensing help base stabilization, and is the gain from the modality or the representation? |
| Reward mode | dense / tractor_focus / multiplicative / no_hitch / guided | Which formulation best induces stable articulated tracking (esp. reverse hitch)? |
| Hitch/jackknife penalty coeff | e.g. exp(−k·|γ|), k∈{0,1.5,2,3} + jackknife-pen on/off | Flagged counter-productive in reverse (`lab_deployment_findings.md`); test it. |
| Policy network | width×depth (e.g. [256,256] vs [400,300] vs [512,512,256]) | Capacity vs sample-efficiency (and reconcile the two codebases). |
| Action parameterization | continuous vs discrete (evenly-spaced steer-rate) × steer-rate limit | Does discretization help/hurt; where is the agility limit? |
| Exploration | action-noise σ, `target_policy_noise` (0.05 vs paper 0.2) | Audit flagged noise decisive; justify the non-standard value. |
| Look-ahead | fixed 10/20 samples vs speed-scaled preview | Does a speed-scaled curvature preview improve tracking? |

**Baseline (locked while ablating others):** best-known deployed config = `lidar_24`
+ `multiplicative`, network TBD, fixed speed. Select the winner per axis, then form
the **best Stage-1 config**.

**Classical baselines** (pure pursuit / PID / MPC) are run on the *same* Stage-1
task at the same fixed speed — this is the fair, information-matched controller
comparison (Ch.5 §Information Parity). MPC's path-preview advantage noted separately.

## 2. Stage 2+ — Curriculum to the hard corner (SECONDARY)

**Task.** Warm-start from the best Stage-1 policy; ramp path curvature (via
`curve_scale` / kind mixture) toward the ~2 m lab corner and the chicane.

**Ablated axes:** curriculum schedule (ramp rate / stage boundaries), recovery-spawn
distribution (near-jackknife starts), failure-replay on/off, anti-jackknife override
on/off.

**Metrics:** max solvable curvature (smallest completed radius), completion on the
lab-corner / chicane, sample-efficiency of the extension, forward-vs-reverse gap.

**Goal:** show *what curriculum is necessary/sufficient* to reach the physical lab
corner — this is where the reverse-instability narrative and the sim-to-real
difficulty story live. Not a from-scratch comparison, by design.

## 3. Vehicle regime

- **Full-size shunt truck = the thesis-wide scale** for all of Ch.3–5 (the ablation).
  Source real terminal-tractor specs (wheelbase, trailer length, mass, tyre) and swap
  them into `VehicleConfig` before the final run; the Tesla params are an interim
  placeholder that keeps the pipeline running now.
- **AgileX appears only in Ch.6 (sim-to-real).** The scalar `tractor_trailer_rl` env
  and the deployed models stay as-is.

## 4. Justified without testing (state the reason, don't ablate)

Vehicle geometry beyond wheelbase/trailer (overhangs, width) → from the real robot;
`dt`=0.1 → 10 Hz embedded rate; lane width → scaled to vehicle; obs Box bounds &
lidar FOV/range → normalization/sensor spec; reward *coefficients* other than the
hitch term → mode ablation covers the structural choice; standard TD3 γ/τ/lr/buffer
→ cite TD3 paper (with the two flagged deviations tested). See `PARAMETER_AUDIT.md`.

## 5. Why this design is defensible (one line)

Every primary result is measured on a **solvable, plant-coherent, deployment-scaled**
base task, so each number reflects the *design axis under test*; task difficulty
(the corner) and vehicle mis-specification (Tesla) are removed as confounds, and the
hard-corner capability is studied separately as an explicit curriculum question.

---

# Full study walkthrough (staged, with propagation)

**Metric philosophy.** On the easy Stage-1 task final completion saturates (~1.00
for every algorithm), so the *primary discriminator is sample-efficiency*
(steps-to-90%-completion / area-under-eval-curve). Every cell: forward AND reverse,
≥3 seeds → mean ± 95% CI. Selection axes PROPAGATE their winner downstream;
reported axes don't. Screen at 1 seed (includes every combo) → multi-seed the top-k
(Hyperband-style). Framing per Google tuning_playbook: scientific (ablated) /
nuisance (tuned once, frozen) / fixed (justified) hyperparameters.

**Frozen infrastructure (nuisance, tuned once):** N=256, minibatch 4096, per-algo LR,
UTD via gradient_steps, 250k-step screen / 500k final, buffer 300k, net [256,256]
until Stage 3, fixed speed, gentle paths, dt=0.1, SB3 vetted implementations.

**Stage 0 — Preconditions.** Requires: full-size dynamic env (Tesla placeholder →
real shunt-truck params at Stage 8); discretized-action wrapper (steer-rate bins;
MultiDiscrete for 2-D); SB3 VecEnv adapter; steps-to-threshold logging; 3-seed
harness. Tests nothing — establishes the controlled, attributable substrate.

**Stage 1 — Algorithm × Reward (foundational joint grid). PRIMARY.**
RQ: which update-regime × action-representation and which reward best learn stable
articulated stabilization, and does the best reward depend on the algorithm?
Design: {PPO, TD3, PPO-disc, DQN} × reward{4 per dir} × {fwd,rev} = 32 cells,
fixed obs = lidar-24. 1-seed screen → ×3 the top pairs.
Tests: on/off-policy, continuous/discrete, value-based(DQN) vs actor-critic, reward
shaping, and the algo×reward interaction. Propagates: winning (algo, reward)/dir.

**Stage 2 — Observation (perception study). PRIMARY + contribution.**
RQ: does richer sensing beat engineered state, and is any gain from the modality or
the learned representation? Requires best (algo,reward). 
  2a Lidar beam-count: state + lidar-{4,8,16,24,32} → resolution vs perf/latency.
  2b BEV representation (vision, later): scratch / scaled-CNN / AE / UNet (frozen &
     unfrozen) → modality-vs-representation; scaled-CNN diagnostic isolates a weak
     representation from a broken image path.
Plus a reward×obs mini-grid (top-2 × top-2) to verify the greedy pick.
Propagates: best obs. Metric adds inference latency (sensor trade-off).

**Stage 3 — Policy/optimization refinements (OFAT around best algo+reward+obs).**
Levels: network width/depth; exploration + target_policy_noise (justify the 0.05
deviation from the TD3 paper's 0.2); look-ahead (fixed 10/20 vs speed-scaled);
hitch/jackknife penalty coefficient. Propagate only clear winners.

**Stage 4 — SAC secondary (hard/reverse only).**
RQ: does max-entropy stochastic exploration rescue the unstable reverse case where
deterministic TD3 plateaus? SAC vs TD3 vs Stage-1 winner on reverse, ×3 seeds.

**Stage 5 — Curriculum to hard corners. SECONDARY.**
RQ: what curriculum extends the base policy to high curvature (sharp/winding, and
the deployment corner)? Warm-start from best Stage-1 config; ablate schedule /
curve_scale ramp / recovery-spawn / failure-replay / anti-jackknife override.
Metric: max solvable curvature (min radius completed), completion on hard kinds,
jackknife rate, forward-vs-reverse gap (headline result).

**Stage 6 — Classical benchmark (endpoint, not propagated).**
Final RL config vs tuned pure-pursuit / PID / MPC on the same held-out scenarios at
matched information (parity table); MPC = informed upper bound (path preview).
Metric: trailer CTE, max hitch, completion, latency.

**Stage 7 — Obstacle / stop-gate (separate task).**
RQ: go-around easy / stop hard, difficulty-gated, via the binary stop action.
Requires obstacle rasterization into the corridor + stop-gate reward (stop iff
difficulty>threshold). Eval stratified by density (sparse/medium/dense).
Metric: go-around-vs-stop correctness by difficulty, min clearance, collision.

**Stage 8 — Finalize with real params.**
Swap real shunt-truck VehicleConfig; re-run the WINNERS only (not the grid) ×3
seeds; regenerate tables/figures with CIs. Report throughput/systems result too.
