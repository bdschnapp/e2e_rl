# Phase 3 — Obstacle-Aware Maneuvering via Hidden-Avoidance-Path Distillation

Culminating phase: show that a high-dimensional observation (BEV image) enables
**planning-and-control**, not just control, by distilling the *avoidance path* from
perception rather than encoding obstacles explicitly. Forward + trailer + obstacles.

## The arc (how we got here)

1. **Flicker/ordering ruled out.** First we tested whether encoding obstacles as ordered
   polar slots destabilises off-policy training (the "flicker" hypothesis). Result: all
   orderings (nearest / fixed / random) collapse identically — ordering is NOT the
   instability source. So we abandoned obstacle-slot encoding.
   (data: `phase3_obstacle_ordering/`.)

2. **The idea (key insight): observe the hidden avoidance path, not the obstacles.**
   The env already computes a hidden obstacle-diverting reference path `local_ys`
   (potential-field offset from the centreline) that the reward tracks. Instead of
   feeding obstacle coordinates to the policy, feed it the **errors + curvature to the
   avoidance path**. Obstacle avoidance then reduces to path-following — the solved lane
   task — and the obstacle information is "pre-digested" as the path to follow. The image
   predicts this path (learning-by-cheating: privileged in sim, image-derived at deploy).

3. **Ground-truth smoke test (control decoupled from perception).** An EXISTING forward
   lane-following state policy, fed avoidance-path errors instead of centreline errors
   with NO retraining, cut obstacle collisions ~in half (8.5 -> 4.5 /1k steps) and lifted
   completion 0.55 -> 0.67. Mechanism confirmed: the controller follows whatever path it
   is given.

4. **Fresh GT training + the difficulty channel.** Trained from scratch on the
   avoidance-path observation (curriculum: obstacles-off warmup -> on; train 3-5 m/s,
   eval fixed 5). Added a 9th observation dim: **windowed difficulty** (free-gap
   difficulty over only the obstacles inside the observable window <= bev_range_m), so the
   policy can make a causal stop decision.

5. **The windowed-reward fix.** The 9-dim first collapsed to degenerate stop-everywhere.
   Cause: the stop-gate reward was keyed on GLOBAL episode difficulty while the agent
   observes WINDOWED difficulty — so early (obstacle-not-yet-visible) stops paid +60 on a
   secretly-blocked episode and -60 on a passable one, i.e. the stop value was
   unpredictable from the observation (critic label-noise -> collapse). Keying the reward
   on windowed difficulty made the stop value a deterministic function of the observation
   and fixed the collapse.

6. **Calibration sharpening.** With the windowed reward the 9-dim still over-stopped on
   moderate layouts. Sharpening the stop-gate calibration (stop_difficulty_k 3->6,
   stop_easy_penalty 60->100) — NOT lowering the reward magnitude (that re-collapsed) —
   concentrated stopping on genuinely-blocked layouts.

## Result (forward — STATIC difficulty, the RETAINED forward recipe)

Difficulty-stratified outcomes, forward, full layouts (incl. blocked), eval fixed 5 m/s.
Forward keeps this STATIC-difficulty recipe: the later multi-seed study showed dynamic
difficulty HURTS forward (mean crash 0.087 vs 0.027 static) because forward does not jackknife
and its CTE is small, so the state-dependent width inflation is just noise. Dynamic difficulty is
a REVERSE-only tool. See `DYNAMIC_DIFFICULTY_STOP_GATE.md` sections 7-9 for the reverse gate, the
articulation limitation, dd image-recoverability, and the full multi-seed study.

**Best model — 9-dim, windowed reward, sharpened (k=6, easy=100): SAFE 0.907, crash 0.019**

| difficulty | success | stop | crash |
|---|---|---|---|
| [0.2,0.4) easy | 0.831 | 0.155 | 0.014 |
| [0.4,0.6) | 0.530 | 0.408 | 0.063 |
| [0.6,0.8) | 0.000 | 1.000 | 0.000 |
| [0.8,1.0) blocked | 0.000 | 1.000 | 0.000 |

Drives through easy layouts, stops on 100% of hard layouts with zero crash on the
hardest. Beats: 8-dim no-difficulty (SAFE 0.819, crash 0.055) and un-sharpened 9-dim
(SAFE 0.739). Weak spot: the [0.4,0.6) "tight-but-maybe-passable" band (split decisions,
most residual crash).

## Observation design (9-dim, all image-predictable except proprio)
- Proprioceptive (sensors, GT): delta (steer), gamma (hitch) — NOT predicted.
- Exteroceptive (image-predicted): avoidance-path errors e_y, e_psi, e_y_t, e_psi_t;
  avoidance-path curvature k1, k2; windowed difficulty d_win.
- Curvature kept at the control-justified lookahead (k1 = 2.5 m is the tuned pure-pursuit
  feedforward `atan(L*k1)`); difficulty anticipation lives in d_win, not in curvature.

## Framing (thesis positioning)
Not monolithic end-to-end — a learned interpretable geometric bottleneck (justified by the
Phase-2A instability of image->action RL). "Why not a classical planner?": the value is (a)
the learned attempt-vs-abort stop decision a geometric planner doesn't make, and (b) the
reverse/articulated case where a geometric path is not trackable. Branching (two-corridor
side-choice) is deferred; smoke tests use the single greedy `local_ys`.

## Code / data
- Env: `batched/obstacle_hidden_path_env.py` (avoidance-path obs, windowed difficulty,
  windowed stop reward, curriculum). Helper: `obstacles.difficulty_per_obstacle`.
- Train: `scripts/train_hidden_path.py` (`--difficulty --stop_k 6 --stop_easy 100`).
- Analyze: `scripts/analyze_hidden_path.py` (difficulty-stratified success/stop/crash).
- Data: `phase3_hidden_path/gt_avoidance_9v8_seed0.csv`.
- Best model: `results_hidden_path/hp_forward_avoidance_d_win_sharp_s0.zip`.

## Status (2026-07-17): Phase 3 substantively complete
Everything below is DONE; details in `DYNAMIC_DIFFICULTY_STOP_GATE.md`.
- **Forward (static):** SAFE 0.907 / crash 0.019 GT; vision closed-loop SAFE 0.903 / crash 0.041
  (`vision_closed_loop.csv`). This is the forward result to cite (NOT the discarded forward-dynamic).
- **Reverse (dynamic difficulty):** the culmination. dyn_s0 crash ~0.05, clean binary stop gate at
  dynamic difficulty ~0.5; multiplicative reward + mild paths + no-stop warmup. Vision closed-loop
  SAFE 0.749 / crash 0.083.
- **Image->state predictor:** done both directions; dynamic d_win recoverable (fwd R^2 0.891,
  rev R^2 0.954). Reverse avoidance-path curvature is soft (0.43/0.28); forward strong (0.83/0.71).
- **Multi-seed (from scratch, 3 seeds each):** reverse crash 0.079+/-0.048 (2/3 seeds ~0.05, s1
  outlier 0.135); forward-dynamic worse than forward-static (negative result -> keep static).
- **Accepted limitation:** reverse over-stops on easy paths when the hitch degrades; correct
  reverse-only behaviour (pull-forward recovery is out of scope).

Open (optional): tighten reverse reproducibility via a critic-warmup at the reward-change boundary
+ SAFE-based checkpoint selection (the s1 outlier is the only soft spot); then write the chapter.
