# Thesis First-Draft Checklist

## Context

MASc thesis (Benjamin Schnapp, UWaterloo MME): *"Autonomous Articulated Vehicle
Control via Deep Reinforcement Learning and High-Dimensional Perception."*
Manuscript: `/home/ben/Ben/Thesis/e2e_rl/thesis/` (LaTeX, `main.tex`).

The manuscript is methodologically ~80% written (Ch.1–4, 6-methods, 7 are solid
prose). The gaps are **empirical scaffolding** (results tables/figures, multi-seed
data, deployment numbers), a **near-empty bibliography**, and **stub appendices**.
Two engineering items remain, per the user's decisions:

**User decisions driving this plan:**
- **Strategy:** During a 1-week lab-free focus window, write the paper to
  *completion as if all experiments are done*, using **placeholder numbers** for
  real-robot results; update those placeholders after returning to the lab.
- **CuPy GPU env: REQUIRED before the draft is "complete."** Motivation: run a few
  **seeds** (supervisor only requires one) for better statistics — this both
  fulfills the manuscript's existing "Statistical Rigor" claims and is a standalone
  systems/engineering contribution. Role = **systems contribution + multi-seed
  regeneration** of the key ablations.
- **Obstacle work:** integration should be done but **placeholder info is OK** for
  the first draft; real on-robot obstacle tests are delayed. Get the **new
  sim system working** (fixed eval speed + **binary stop-gate**: reward stopping
  when path difficulty > threshold, penalize stopping when difficulty is low). The
  old obstacle results were weak because they used speed control and never learned
  to stop on hard cases; the stop-gate makes this learnable.
- **Smith predictor: DONE.** Implemented, **tuned, and enabled on the real robot**
  via a flag on the run script (off by default). (The exploration snapshot that
  called it "disabled/untuned" was stale; supersedes the old memory note that the
  Smith predictor was blocked on a dynamics bag.) The *learned neural-dynamics
  augmentation* (`DynamicsMLP`) remains optional/future and is the only part that
  needs a real full-topic bag.

---

## Repo state at a glance (from full exploration)

| Repo | Role | State |
|---|---|---|
| `Thesis/e2e_rl/` | Core RL research (pygame Gym env, TD3, ablations, results tables) | Forward/reverse lane-following results solid but **single-seed**; 2 metric bugs; some un-run eval/table rows; obstacle results weak (being redone) |
| `Thesis/tractor_trailer_rl/` | Clean headless numpy successor; lab curves; ROS-bridge obs source | Single-env **scalar** (Python floats); the **CuPy port base**. 5 vectorization blockers identified |
| `Thesis/Electrans_project/` | AgileX ROS2 sim-to-real deploy (RL bridge, Smith predictor, hitch estimator) | Working; real-robot testing done. Smith predictor tuned+flag-enabled. No obstacle hooks yet; training scripts live here (should move) |
| `Thesis/IndoorPerception/` | **ROS1 (Noetic)** infra sensor nodes → obstacle list | Obstacle list works (`/mec/lidar_euclidean/tracked`, `DynamicObjectArray`, frame `mec_1`); consumer is bed-centric; **needs ROS1→ROS2 bridge + ego-in-infra-frame** |
| `Electrans/rudislabs_gazebo_worlds/` | Blender+GIS→Gazebo digital-twin worlds (dc, dayton) | Done. Maps to Ch.6 §High-Fidelity Geographic Simulation. Force-driven vehicle (fidelity caveat) |
| `Electrans/Electrans_project/` | Autoware classical truck-trailer planning/control baseline | Done (robot-scale, not full-size — **note timeline/scale label mismatch**). Produces reusable planner figures on real MVSL map. Background/related-work, not core results |

---

## THE CHECKLIST (prioritized)

### Track A — Manuscript completion (the 1-week write-to-completion pass)
Highest priority; unblocks the draft. Most items are writing, not experiments.

- [ ] **A1. Bibliography (biggest scholarly defect).** Only **1 `\cite`** exists in the
  whole thesis; `thesis_refs.bib` is empty and `references.bib` has an empty
  `@article{placeholder}` (source of the `.blg` error). Add citations throughout —
  Ch.2 names DDPG, PPO, TD3, Decision Transformers, SOTIF, CBFs, Pacejka, pure
  pursuit, MPC/NMPC, Autoware, Gymnasium, SB3, hybrid A*, Zhang L-shape — all
  currently uncited. Remove the empty placeholder entry. Target ≥30–50 refs.
  Build workflow: bibtex, then copy `build/main.bbl` → root `main.bbl` (per memory).
- [ ] **A2. Ch.6 §Deployment Results — replace all `[TODO]`.** Section is literal
  placeholder prose (`06_sim_to_real.tex:183–192`): test location, forward CTE,
  γ bound, reverse outcome, Smith-predictor effect. Write it now with **placeholder
  numbers clearly marked**, structured to be swapped for real values post-lab.
  Fold in the three findings from `lab_deployment_findings.md` (vehicle-scale
  transfer / curvature-to-wheelbase ratio; pygame↔Autoware actuator + path-dist
  gap; forward-vs-reverse trainability) — these are written up nowhere yet.
- [ ] **A3. Fill the four stub appendices:** `appendix_ros2_stack` (Gym class
  hierarchy, pygame pipeline, training config), `appendix_hyperparameters` (fill
  the blank TD3 table — LR, batch, buffer, noise, target rate — read from
  `train.py`/SB3 defaults), `appendix_additional_plots`, `appendix_a_reward_derivation`
  (full reward-shaping derivation).
- [ ] **A4. Resolve the four in-chapter `% TODO` markers:** `04:98` (architecture
  diagrams, UNet desc, BEV anchor, attention viz), `04:148` (TD3 HP table),
  `05:159` (statistical tests + reward-weight ablation), `03:184` (training curves).
- [ ] **A5. Fix the BEV image-size doc mismatch.** Code uses **32×32** (`OBS_WIDTH=
  OBS_HEIGHT=32`); CLAUDE.md, `train.py` docstring, `UNet.py`, the audit file, and
  some notes say 84×84. Confirm which the thesis quotes and make it consistent (the
  thesis chapters already say 32×32 — verify none say 84).
- [ ] **A6. Rewrite speculative future-tense prose in Ch.5 into past-tense
  reporting** once the numbers are final (currently "is expected to", "are intended
  to"). Draw actual conclusions from the tables.
- [ ] **A7. Clear `thesis/pending_updates.md`** (25+ flagged env edits) — reconcile
  LineFollowing/ObstacleAvoidance/wrappers/pid changes against the manuscript.
- [ ] **A8. Admin (from `main.tex` header TODO):** choose 2 readers, schedule
  seminar (late July), plan the mandatory 15-day display period. *Start reader
  selection early — it gates submission independently of the writing.*

### Track B — Data-integrity fixes + table/figure generation
Fix before quoting any numbers. These are mostly re-eval + regenerate, not retrain.

- [ ] **B1. Fix the benchmark completion metric bug.** `benchmark_results.tex` shows
  **0% completion for every controller** (forward & reverse, incl. TD3) —
  inconsistent with obs-ablation's 100%. Root-cause in `benchmark.py` /
  `eval.py` completion logic (likely the `max_progress≥95%` threshold vs episode
  definition), then regenerate.
- [ ] **B2. Fix the BEV completion under-reporting.** `bev_scaled_cnn` shows 0%
  completion despite full reward (614) and full episode length (221). Same
  progress-threshold artifact noted in `eval.py`. Fix threshold, re-eval BEV rows.
- [ ] **B3. Complete the failure-replay table.** Model exists
  (`reverse/lidar_24/multiplicative_retry`); only needs eval + the "steps to 50%
  completion" metric computed, then `generate_results_tables.py`.
- [ ] **B4. Reconcile models/ vs results/ drift.** Forward BEV encoder result CSVs
  (`ae_frozen`, `ae_unfrozen`, `unet_frozen`) exist without matching `models/`
  dirs — decide whether to retrain those (→ CuPy multi-seed, Track C) or drop the
  rows. Also fix the `run_ablations.sh:~359` shell bug that silently drops forward
  BEV plot curves.
- [ ] **B5. Render the 5 missing figures** (only `forward_obs_ablation.pdf` exists):
  `reverse_obs_ablation`, `forward_reward_ablation`, `reverse_reward_ablation`,
  `failure_replay_curve`, and the `infrastructure_sensor_nodes` concept diagram.
  Use `plot_learning_curves.py` / `thesis/scripts/`. Plus the `lab_deployment_findings`
  suggested figures (steering-response histograms, fwd-vs-rev learning curves).
- [ ] **B6. Regenerate all tables** via `python thesis/scripts/generate_results_tables.py`
  after B1–B4 and after CuPy multi-seed runs land.

### Track C — CuPy GPU-parallel training env (REQUIRED; systems contribution)
Built in the copy **`/home/ben/Ben/Thesis/tractor_trailer_rl_cupy/`** (new
`batched/` subpackage). See `README_CUPY.md` there for full details. **BUILT &
VALIDATED THIS SESSION (C1–C6); C7 infra ready, needs GPU-time to run.**

- [x] **C1. Copied** `tractor_trailer_rl` → `tractor_trailer_rl_cupy/`.
- [x] **C2. `(N_env,)` batch axis** through vehicle/geometry/reward/observation/env
  with masks + internal auto-reset (`batched/vehicle.py`, `geometry.py`, `reward.py`,
  `env.py`). Backend switch numpy↔cupy via `batched/backend.py` (`TTRL_BACKEND`).
- [x] **C3. All 5 blockers replaced:** batched matrix-exp ZOH (`batched/expm.py`,
  matches scipy ~1e-8); analytic distance-to-centerline + on-device **path pool**
  replacing the cKDTree grid (`batched/corridor.py`); vectorised lidar march
  (`batched/lidar.py`); batched collision/proximity; masked termination/stop/spawn.
- [x] **C4. Parity suite** `tests/test_batched_parity.py` — **11 pass**: obs EXACT
  (0.0), reward ~1e-14 (proximity-free), vehicle ~1e-14, numpy==cupy (0.0),
  lidar ~1 cell, collision 97–98.5% agreement. (Proximity/lidar intentionally
  analytic, not rasterised — documented.)
- [x] **C5. GPU TD3** (`batched/td3.py`, DLPack zero-copy) — trains end-to-end on
  GPU (`scripts/train_batched_td3.py`); smoke run improved return −94→−52.
- [x] **C6. Throughput benchmark** (`scripts/benchmark_batched.py`): **state-obs
  ~1.7M transitions/s @ N=16384; lidar-24 ~64k tps**; scalar single-core ~2.1k
  (state) / ~470 (lidar). Honest caveats on the SubprocVecEnv baseline in README.
- [ ] **C7. Multi-seed ablation reruns** — runner **ready** (`scripts/
  run_ablations_gpu.py`, mean ± 95% CI). TODO for the user: tune TD3 HPs/step
  budget to match SB3 policy quality, then run the campaign to regenerate the
  ablation tables with CIs (fulfils Ch.5 §"Statistical Rigor").
- [ ] **C-opt (optional throughput):** fused CuPy `RawKernel` lidar ray-march (the
  remaining lever to bring lidar-obs throughput toward the state-obs env's).

### Track D — Obstacle-in-lane RL with the binary stop-gate (sim results for draft)
Get a *working sim* result (go around easy / stop hard); real-robot obstacle tests
are deferred with placeholders.

- [ ] **D1. Implement/confirm the stop-gate reward** in `tractor_trailer_rl` (+ CuPy
  copy): fixed eval speed, small training speed distribution for regularization,
  binary stop action (`STOP_SIGNAL` mode already exists in `actions/modes.py`),
  reward stop iff `path_difficulty > threshold`, penalize stop when difficulty low.
  Reuse `compute_path_difficulty()` (`e2e_rl/Environments/ObstacleAvoidance.py:287`).
- [ ] **D2. Add obstacle rasterization into the occupancy grid** for the headless env
  (the pygame `ObstacleMixin` logic ported to numpy/CuPy occupancy).
- [ ] **D3. Train + evaluate** obstacle policies, **stratified by difficulty**
  (easy/medium/hard, mirroring `generate_test_scenarios.py` sparse/medium/dense);
  report go-around vs stop success separately + the missing **min-clearance** metric
  (currently `---` in `obstacle_results.tex`).
- [ ] **D4. Write Ch.5 §Obstacle Navigation** around the new results + difficulty
  framing; frame real-robot obstacle validation as placeholder/future.

### Track E — IndoorPerception → AgileX obstacle wiring (placeholder in draft)
Real end-to-end obstacle test delayed; wire enough to describe it + get the Ch.6
§Infrastructure Sensor Nodes section out of "placeholder" state.

- [ ] **E1. Bridge ROS1→ROS2.** IndoorPerception is **ROS1 Noetic**; the AgileX
  deploy is **ROS2 Humble**. Stand up a `ros1_bridge` (or DDS/MQTT bridge) to carry
  the obstacle list across. This is the crux and is currently absent.
- [ ] **E2. Consume `/mec/lidar_euclidean/tracked`** (`DynamicObjectArray`, frame
  `mec_1`) in a new RL perception-adapter node. Avoid the bed-centric
  `/perception_to_mpc` path unless you strip the bed-ego coupling.
- [ ] **E3. Ego pose in the infra frame.** Publish TF `mec_1 → agilex_base_link`
  (AprilTag on the robot — infra already supports AprilTags — or robot odometry),
  then transform obstacles into the robot frame.
- [ ] **E4. Rasterize obstacles into the bridge's occupancy grid.** Seam already
  exists: `rl_bridge_node` builds the occ grid from lane boundaries only; the unused
  `ros_obs.observation_from_ros` already accepts `occ_grid`/`occ_meta` — either wire
  a subscriber+rasterizer into `rl_bridge_node`, or migrate the node to `ros_obs`.
- [ ] **E5. Fix hardcoded lab config** in IndoorPerception: `interested_node_list=[3]`
  (only node 3 fused), the bed-specific ROI filter in `MOTWithFeedback.py`, MQTT
  broker / `MEC_ID`, per-node TF calibration.
- [ ] **E6. Write Ch.6 §Infrastructure Sensor Nodes** to reflect what's actually
  wired; keep the concept figure (B5) + placeholder results.

### Track F — Code cleanup / relocation
- [ ] **F1. Move training + dynamics-ID scripts** out of the ROS deploy package
  (`Electrans_project/src/electrans_rl_bridge/scripts/`) into `e2e_rl` /
  `tractor_trailer_rl` where they belong.
- [ ] **F2. Resolve duplicates/dead code:** duplicate `smith_predictor.py` (package
  vs `scripts/`); decide on unused `ros_obs.py` (migrate to it or delete); commit or
  drop untracked `docs/localization_fixes_2026-07-01/` (has 4 useful analysis PNGs).
- [ ] **F3. De-hardcode dev paths** (`/home/ben/...` vs `/home/electrans_robot/...`)
  via params/env (flagged in `SETUP_NOTES.md`).
- [ ] **F4. Fix placeholder package metadata** (`todo@electrans.com`, `description:
  TODO`) in the classical-planning packages if any are cited by the thesis.
- [ ] **F5 (optional).** Learned neural-dynamics augmentation (`DynamicsMLP`) — record
  a **real full-topic dynamics bag** (`start_dynamics_bag.sh`: the 7 topics incl.
  `/control/command/control_cmd`, `/localization/kinematic_state`,
  `/vehicle/trailer_state`; the existing bag has only 3 and an all-zero action
  array), retrain, evaluate. Future work — the analytic Smith predictor is already
  tuned and working, so this is not draft-blocking.

---

## CuPy port — implementation detail (Track C)

Base: `tractor_trailer_rl` is clean and small (~2,500 LOC, one env class), but
**fundamentally single-env scalar** — state stored as Python floats; parallelism
today is only SB3 `SubprocVecEnv` over CPython copies. GPU-parallel means adding a
leading `(N_env,)` axis everywhere and converting scalar branches to masked ops.

**Easy (already array math → add batch axis):**
- `vehicle/tractor_trailer.py::TrailerModel.update()` — pure trig.
- `geometry.py` (`nearest_index` → `argmin(axis=-1)`, `path_errors`, `curvature_at`,
  `wrap_to_pi`); convert per-env `if` branches to `cp.where`.
- `reward/composer.py` — scalar arithmetic + `exp`/`clip`; mode is a static config
  string (safe to branch once per batch).

**Hard (the 5 real blockers):**
1. **`vehicle/bicycle.py:110` `scipy.signal.cont2discrete(...,"zoh")` every step** —
   biggest item. Replace with a batched matrix-exponential ZOH (Padé/`expm` on
   `(N,3,3)`) or precompute/analytic ZOH per env.
2. **`observation/lidar.py::raycast()`** — double Python loop with early-exit break;
   rewrite as a vectorized fixed-length march over `(N_env, N_beam, N_step)`, first
   hit via cumulative mask (default 24 beams × 400 steps).
3. **`observation/occupancy.py::build_occupancy_grid()`** — `scipy.spatial.cKDTree`
   per reset; replace with batched point-to-polyline distance (or keep grids on CPU
   and transfer). Grids are large (lab ~1500×900).
4. **`reward/proximity.py` + `envs/base.py::_collision()`** — Python loops with fancy
   indexing/early-return → batched gather + `any`.
5. **`envs/base.py` control flow** (`step`/`reset`/`_terminated`: jackknife, bounds,
   success, stop-signal early return, spawn perturbation) → per-env boolean masks
   with internal **auto-reset** (a batched env must implement the auto-reset that SB3
   VecEnv currently provides).

**Training:** SB3 TD3 won't consume a single batched-GPU env. Simplest path: a
compact torch TD3 that reads the CuPy env via **DLPack** (zero-copy) and keeps
replay/updates on-GPU. Keep the numpy env + SB3 as the parity oracle.

**Fallback:** if CuPy proves heavy, JAX (`jax.vmap` + `lax.scan` for the ray-march /
integration, `jit`) is an alternative that the user explicitly accepts ("CuPy or
other Jax style GPU acceleration"). Either satisfies the contribution.

---

## Verification

- **Manuscript:** `cd thesis && make` compiles to `build/main.pdf` with **no
  "Figure unavailable" boxes** (all figures present) and **no undefined-citation /
  bibtex warnings** in `main.log`/`main.blg`. Sanity-check page count (target 80–120).
- **Data integrity (B1/B2):** after the completion-metric fix, benchmark + BEV
  completion rates are internally consistent with the obs-ablation tables (no
  all-zero completion column; BEV rows that finish full episodes report >0%).
- **CuPy (C4/C6):** parity test passes ≤1e-4 vs the numpy oracle; throughput
  benchmark shows the envs/sec + wall-clock speedup vs `SubprocVecEnv`; a multi-seed
  ablation reproduces the single-seed trends within CI.
- **Obstacle (D3):** trained policy demonstrably **goes around easy** obstacles and
  **stops on hard** ones, stratified success reported with min-clearance.
- **Perception wiring (E):** obstacle list from IndoorPerception is visible on the
  ROS2 side (bridge up) and rasterized into the bridge occupancy grid in a bag
  replay — full on-robot test deferred with placeholders.

## Suggested sequence for the 1-week window
1. Days 1–2: **Track A** writing (A1 bibliography, A2 deployment placeholders, A3–A5
   appendices/TODOs) + kick off **B1/B2** metric fixes.
2. Days 2–4: **Track C** CuPy port (C1–C5) — the required, riskiest item; start early.
3. Days 4–5: **C6/C7** benchmark + multi-seed reruns; **D1–D3** obstacle stop-gate in
   sim; **B3–B6** regenerate tables/figures.
4. Days 5–7: **D4/E6** results/deployment write-up; **Track E** wiring to the extent
   possible off-robot (bag replay); **F** cleanup; full `make` + proofread pass.
5. Post-lab: swap placeholder real-robot numbers (A2), run E end-to-end on hardware,
   optional F5 learned dynamics.
