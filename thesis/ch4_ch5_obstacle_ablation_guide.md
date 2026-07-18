# Obstacle Navigation and GPU Ablation: Ch4/Ch5 Writeup Guide

*Session log (2026-07-07/08) plus a placement guide for what belongs in Chapter 4
(Learning Framework / Methodology) versus Chapter 5 (Results and Discussion, incl. the
ablation study). Companion evidence file: `obstacle_difficulty_calibration.md`. Corrected
plan of record: `FIRST_DRAFT_CHECKLIST_AUDIT.md`.*

---

## 0. What this session produced (at a glance)

Artifacts now on disk:
- **Stage-1 ablation (obs = state)**: `tractor_trailer_rl_cupy/results_stage1_V1/` (algorithm x reward x direction, 3 seeds, real shunt-truck params). Also the earlier `results_stage1_v4/` (obs = lidar24) with drop-in `.tex` tables already staged into `thesis/tables/` (`reward_ablation_{forward,reverse}.tex`, `algorithm_comparison_{forward,reverse}.tex`) and figures (`figures/experiment_results/{forward,reverse}_reward_ablation.pdf`).
- **Obstacle stop-gate policies**: `results_obstacle/stageB_td3_{forward_s0,reverse_s2}.zip`, with difficulty-stratified evaluation in `results_obstacle/eval_gpu2.log`.
- **PP drivability calibration**: `results_obstacle/pp_drivability_{fwd,rev}.log`, written up in `obstacle_difficulty_calibration.md`.
- **Code**: fixed obstacle trainer recipe (`scripts/train_obstacle.py`); GPU-capable obstacle eval (`scripts/eval_obstacle.py`); new stop-gate calibration analysis (`scripts/analysis_stopgate.py`); **branching hidden-path reward** (`config.py`, `batched/obstacles.py::side_offsets`, `batched/obstacle_env.py`).

Headline findings:
- Forward obstacle stop-gate **works cleanly** (goes around easy layouts, stops on hard, near-zero collisions).
- Reverse **over-stops**, but calibration shows this is **near-rational**, not a training failure.
- The `difficulty` rating is a **geometric clearance proxy**; a pure-pursuit drivability sweep calibrates it to a per-direction threshold (`d*_forward` around 0.8, `d*_reverse` below 0.2).
- **Vision-based driving now works** via a staged recipe (behaviour cloning, then critic-only, then unfreeze), which unblocks the observation-space ablation.

---

## 1. What belongs in Chapter 4 (Learning Framework / Methodology)

Chapter 4 should present the *formulations, designs, and training procedures*. Nothing here
is a result; it is how the system is built and trained.

### 1.1 GPU-vectorized simulation environment (systems contribution)
The batched CuPy environment: a leading `(N_env,)` axis through vehicle dynamics, geometry,
reward, observation, and internal auto-reset, with a numpy/cupy backend switch and a
parity-tested equivalence to the scalar reference. Motivates the multi-seed statistics used
in Ch5. (Detail and throughput belong here; the throughput numbers themselves are a Ch5/appendix
result.)

### 1.2 Observation spaces
Define the three observation modalities the ablation compares:
- **state** (tracking errors, hitch, curvature look-aheads),
- **lidar-N** (N-beam range scan; default 24 beams, 120 deg FOV, 20 m range),
- **BEV** (analytic top-down occupancy image).

Note the observation image size (32x32) and that obstacles are sensed *only* through
lidar/BEV, never through the planned avoidance path.

### 1.3 Obstacle-avoidance task formulation
- Obstacles as circles placed near the centreline; one obstacle "event" (cluster of 1-2)
  per episode so every outcome is cleanly attributable to that layout's difficulty.
- Layout categories (clear / shift / squeeze / blocked) drawn per episode so the difficulty
  distribution is controlled and stratifiable.
- The **hidden local planner**: the reward tracks a planned avoidance path the agent never
  observes, so the policy must infer the detour from lidar/BEV alone. This is the core
  design that makes obstacle avoidance a perception problem rather than a path-following one.

### 1.4 The difficulty metric (geometric clearance proxy)
State the definition exactly (see `obstacle_difficulty_calibration.md` Section 1):
`difficulty = clip(1 - (max_gap - veh_w)/(lane_w - veh_w), 0, 1)`, where 0 is a fully open
lane and 1.0 is a gap at or below the vehicle width (impassable). Emphasize the *design
rationale*: it is a **cheap static geometric proxy** for drivability, because evaluating true
kinematic feasibility for a nonholonomic tractor-trailer inside the training loop is too slow.
Flag the two properties that matter later: the impossible regime is clipped to 1.0, and the
metric does not account for the manoeuvring / trailer-swing room a real trajectory needs
(which is direction-dependent). The *quantification* of that proxy gap is a Ch5 result.

### 1.5 Stop-gate action and reward
- The `STOP_SIGNAL` action: a binary stop appended to the steering action, so the policy can
  decline to attempt a layout.
- The difficulty-keyed stop-gate reward: stopping pays on an impassable layout (must beat a
  crash) and is penalized on an easy one (must lose to driving on), via
  `stop_reward = exp_scale(d)*R_hard - (1-exp_scale(d))*P_easy + progress*bonus`, with
  `exp_scale(d) = (e^{kd}-1)/(e^k-1)`.
- **Design decision (this session): difficulty is kept global** (max over the whole path,
  static per episode). Justification worth stating: since the agent never observes difficulty
  and must infer risk from its own perception, the global scalar only sets the *prize* for a
  correct stop; the agent's perception sets the *timing*, and the progress term rewards
  driving through as much of the layout as it can before stopping. (A position-windowed
  look-ahead variant was considered and deferred; it only matters for multi-cluster paths,
  which are out of the single-cluster training regime.)

### 1.6 Branching hidden-path reward (new this session)
Present as a methodological refinement: instead of a single greedy avoidance corridor, the
env builds **two** hidden corridors (pass-left and pass-right) and rewards tracking whichever
the agent is closer to (`max(track_left, track_right)`). This makes the **side-of-obstacle
choice an emergent decision** driven by perception rather than a greedy planner pick; the
globally-infeasible side leads into the blocking obstacle and crashes. Motivation: even when
both sides are feasible, a single prescribed corridor over-constrains the agent (it is
penalized for choosing an equally valid other side); branching removes that. It also handles
the "staggered blockers force a global choice" dead-end case for free. (`side_offsets` builds
a one-sided clearance corridor; toggle via `ObstacleConfig.branching_reward`.)

### 1.7 Training procedures
- **Obstacle curriculum**: Stage A trains lane-following in the stop-capable action space,
  Stage B warm-starts from it with obstacles on. Reverse Stage B is warm-started from a
  pre-validated reverse lane-follower rather than trained from scratch (reverse from scratch
  is high-variance; see 1.8).
- **Vision training recipe (new, works)**: behaviour cloning to initialize, then freeze the
  actor and encoder and train the critic, then unfreeze the actor and let the full system
  continue. This is the recipe that makes BEV viable (vision-from-scratch fails). Document the
  freeze/unfreeze rationale; it is a genuine methodological contribution.
- **Off-policy training protocol**: learning rate 1e-3 (2e-3 diverges once the stop action is
  present), n_envs = 32 for off-policy (the stable regime), evaluation on a pure-pursuit
  validated feasible path pool with the stop action disabled at eval (so completion measures
  tracking, not stop-misfires).

### 1.8 Classical baselines and the drivability oracle
- Classical controllers (pure pursuit, PID, LQR, MPC) as comparison controllers (defer full
  formulations/gains to an appendix per the audit's A3b).
- **PP as a drivability oracle (method)**: run the traditional stack (global path, APF local
  planner, pure-pursuit tracking) with the stop action disabled, stratified by difficulty, to
  measure empirical drivability `P_clear(d)`. Used to calibrate the geometric difficulty into
  a reward-relevant, per-direction threshold. State the break-even math here: with terminal
  rewards success = +100 and crash = -500, attempting is positive expected value only where
  `P_clear >= 5/6 (83.3%)`, so the drivability threshold `d*` is where `P_clear` crosses 83%.

### 1.9 Evaluation methodology
- Difficulty-stratified outcomes (complete / stop / crash / timeout) with Wilson 95% CIs.
- **crash-conditional-on-attempt** (`crash / (complete + crash)`) as the honest avoidance
  metric, reported alongside the raw crash rate (raw crash is deflated by a high stop rate).
- Minimum-clearance metric.
- Multi-seed mean +/- 95% CI throughout (the systems contribution enabling it).

---

## 2. What belongs in Chapter 5 (Results and Discussion / Ablation)

Chapter 5 is the numbers and their interpretation. Every table/figure below has data on disk.

### 2.1 Stage-1 ablation: algorithm x reward x direction
Source: `results_stage1_V1/summary.csv` (obs = state) and `results_stage1_v4/` (obs = lidar24,
staged `.tex` in `thesis/tables/`). Story:
- **Forward**: several algorithms reach 100% completion; **TD3 + multiplicative** is the robust
  choice and DQN is strong; on-policy is weaker.
- **Reverse**: **TD3 + multiplicative is the clear, stable winner (about 0.97 completion, tight
  CI)**; on-policy (PPO) fails reverse; reward conditioning governs trainability.
- Conclusion that fixes the rest of the study: **TD3 + multiplicative** is the (algorithm,
  reward) held fixed for the observation-space ablation.

### 2.2 Stage-2 observation-space ablation (PENDING, run on the remote GPU)
Fix TD3 + multiplicative, vary observation: state / lidar-{4,8,16,24,32} / BEV, both
directions, 3 seeds. Produces the two currently-missing Ch5 tables (`obs_ablation`,
`lidar_beams_ablation`). This is the perception-axis pillar and is now unblocked by the
working BEV recipe (2.5). It should run on the remote 3090 (the local box is unstable under
sustained load; see Section 3).

### 2.3 Obstacle navigation results
Source: `results_obstacle/eval_gpu2.log` (4000 episodes/direction, real shunt-truck params,
lidar-24). Report the difficulty-stratified table for the RL stop-gate policy and the
traditional-planner baseline, per direction. Key numbers:
- **Forward RL stop-gate works**: goes around easy layouts (about 89% complete, difficulty
  0.2-0.4), stops on hard (about 99% stop, difficulty 0.8-1.0), and **crashes at most about
  0.7% at every difficulty**. A clean, discriminative, safety-first stop-gate.
- **Forward RL vs planner**: the planner drives through more but crashes 8-25% on mid/hard
  because it only stops on planner infeasibility; the RL trades a little throughput for
  near-zero collisions.
- **Reverse RL over-stops** (about 86-90% stop across all difficulties, even easy), and the
  reverse planner under-stops and crashes 20-50%. Neither solves reverse cleanly.
- Report **crash-conditional-on-attempt** to be honest: the RL's low raw crash rate is partly
  from declining to attempt; per attempt its avoidance is weaker than the planner's
  (e.g. reverse RL about 54% crash-per-attempt on easy). This is the fair comparison.

### 2.4 Difficulty calibration results (drivability curves)
Source: `obstacle_difficulty_calibration.md` and `results_obstacle/pp_drivability_{fwd,rev}.log`.
- Forward drivability holds above the 83% break-even up to about 0.8, then collapses:
  **`d*_forward` is about 0.8**. Geometrically-hard-but-below-0.8 layouts are genuinely
  drivable.
- Reverse never clears 83% at any difficulty (only 78% even at the easiest), and expected
  value of attempting is negative in every bin: **`d*_reverse` is below 0.2**.
- Present the forward/reverse `P_clear(d)` and `E[attempt](d)` tables from the calibration doc.

### 2.5 Vision-based driving result
The staged recipe (behaviour cloning, critic-only, unfreeze) makes BEV viable where
from-scratch fails. Present the working BEV curve/result and fold it into 2.2 as a real
observation-space row (not the previous "expected to fail" footnote).

### 2.6 Discussion themes (the interpretation Ch5 should draw)
1. **Reverse is a drivability / training-reliability problem, not a control-capability one.**
   Reverse lane-following has high seed variance (a fraction of seeds collapse), reverse
   obstacle drivability is low even for a classical stack, and the RL's reverse over-stopping
   is *near-rational* given `d*_reverse < 0.2` (attempting is negative expected value almost
   everywhere). This is a stronger, more honest framing than "the reverse policy failed".
2. **Geometric difficulty is a good drivability proxy forward and a poor one reverse.** The
   same geometric gap is comfortably drivable forward and near-hopeless in reverse because
   reverse needs far more trailer-swing room than the body width. This motivates a
   direction-aware or swept-area difficulty metric as future work.
3. **The forward reward is already well-calibrated** (its stop break-even, about 0.785,
   essentially coincides with `d*_forward` about 0.8), so the forward mid-range over-stopping
   is a mild policy conservatism, not a reward-calibration fault.
4. **Honest metrics matter**: raw crash rate understates the difficulty because the stop-gate
   avoids attempting; crash-conditional-on-attempt is the fair avoidance measure.
5. **Perception drives the hard decisions**: choosing the feasible side (branching) and
   knowing when to stop both require seeing far enough ahead, linking the obstacle results to
   the observation-space ablation.

---

## 3. Status: done / pending / deferred

**Done this session:**
- Stage-1 ablation results + staged CuPy tables/figures.
- Obstacle stop-gate trained + evaluated (forward seed 0, reverse seed 2), difficulty
  calibration measured and documented.
- Branching hidden-path reward implemented and verified (applies on the next retrain).
- Design decision: difficulty stays global.

**Pending (next actions):**
- **Stage-2 observation-space ablation on the remote GPU** (the big lever; produces
  `obs_ablation` + `lidar_beams`, now with a working BEV row).
- **Next obstacle retrain** with branching reward + the smoother local planner, then a
  multi-seed pass (forward seeds 1-2; reverse warm-started from good seeds) for mean +/- CI.
- Regenerate `obstacle_results.tex` in the new stop-gate format (difficulty-stratified,
  RL vs planner, both directions, with crash-per-attempt).

**Deferred (with reasons):**
- **Reverse reward reshape to `d*_reverse`**: deferred deliberately. Reshaping now would
  hard-code the current (artificially low) reverse drivability into the reward. First improve
  reverse drivability (smoother planner, branching, richer perception), re-measure `d*_reverse`,
  then reshape to the achievable frontier.
- **Smoother local planner**: the current APF planner is greedy and non-smooth, which hurts
  both the drivability oracle and the RL reward target (reverse especially). A batched
  smoothing (anticipatory lead-in/out, wider smoothing, curvature cap) is the recommended next
  code change after branching.
- **Real on-robot obstacle validation**: placeholder in the draft (per the audit).

**Environment caveat:** the local training box (24 cores, 30 GB RAM) crashed twice this
session under concurrent heavy jobs (thread oversubscription + RAM pressure). Run one heavy
job at a time and pin BLAS/torch threads; prefer the remote 3090 for the Stage-2 sweep.

---

## 4. Pointers

| Item | Location |
|---|---|
| Difficulty definition + drivability calibration | `thesis/obstacle_difficulty_calibration.md` |
| Corrected plan of record | `thesis/FIRST_DRAFT_CHECKLIST_AUDIT.md` |
| Stage-1 ablation (state obs, winner) | `tractor_trailer_rl_cupy/results_stage1_V1/summary.csv` |
| Stage-1 ablation (lidar24) + staged tables | `results_stage1_v4/thesis_export/`, `thesis/tables/` |
| Obstacle stratified eval (RL + baselines) | `results_obstacle/eval_gpu2.log` |
| PP drivability curves | `results_obstacle/pp_drivability_{fwd,rev}.log` |
| Obstacle trainer / eval / calibration scripts | `scripts/train_obstacle.py`, `eval_obstacle.py`, `analysis_stopgate.py` |
| Branching reward | `batched/obstacles.py::side_offsets`, `batched/obstacle_env.py`, `config.py::ObstacleConfig` |
