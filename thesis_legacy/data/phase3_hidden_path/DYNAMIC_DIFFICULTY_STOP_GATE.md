# Dynamic Difficulty and the Binary Stop Gate (Reverse Obstacle Maneuvering)

Sub-chapter: how the reverse obstacle-avoidance stop decision was made *provably* clean by
replacing a static, per-layout difficulty with a **state-dependent (dynamic) difficulty**,
after which stopping reduces to a single learnable threshold.

## 1. The problem: over-confidence in the ambiguous band

The reverse hidden-path model (avoidance-path observation + windowed difficulty + windowed
stop reward + sharpened calibration) drove easy layouts and stopped on clearly-hard ones,
but in the ambiguous [0.4,0.6) *static*-difficulty band it **over-attempted and clipped**:
of the attempts there, ~40-45% crashed (vs ~11% for the forward model). Inflating the
static difficulty (a uniform `veh_w + 2*CTE` tolerance) barely helped — crash-per-attempt
stayed ~38-45% even at large inflation, and easy-layer completion started to erode.

## 2. Why static difficulty fails: the stop is state-dependent

Forensic (deterministic policy, decision = max-`d_win` step, difficulty band [0.4,0.6)):

| signal at decision | success | stop | crash |
|---|---|---|---|
| \|e_y_t\| (trailer off-path) | 0.362 | 0.572 | 0.557 |
| d_win (perceived difficulty) | 0.401 | 0.480 | 0.366 |

The stop is **not** a flat difficulty gate (a pure gate would give 0%/100% per band, not the
observed ~50/50 mix). It is a function of the full state:
- **Success** = trailer **on-path** (low |e_y_t|) -> attempt succeeds.
- **Stop** = trailer **off-path** (high |e_y_t|) AND high d_win -> abort.
- **Crash** = trailer **off-path** (high |e_y_t|, like a stop) BUT d_win reads **low** -> the
  policy is fooled into attempting, and the off-path correction around the obstacle clips it.

Root cause: the static difficulty describes the *layout* (the free gap), not the *situation*.
"I am tracking well now" does not imply "I will clear this gap," because reverse off-tracking
is systematic (~0.8 m). The policy's confidence is miscalibrated for it.

## 3. The fix: dynamic (state-dependent) difficulty

Define the effective vehicle width using the **current** tracking error rather than an
average: `w_eff = veh_w + cte_margin + 2*|current trailer error to the avoidance path|`.
When the trailer is off-path (into the obstacle) the usable gap shrinks *now*, so difficulty
rises *now* and can trigger a stop. Implementation: cache the free gap per obstacle at reset
(`free_gap_per_obstacle`), and per step apply the state-dependent `w_eff`
(`gap_to_difficulty`), max'd over the in-window obstacles
(`ObstacleHiddenPathEnv.dynamic_windowed_difficulty`).

## 4. The windowing anomaly and its fix

Bucketing crashes by max dynamic difficulty first showed a clean rising crash-per-attempt
(0.00 -> 0.27 -> 0.68) EXCEPT a spurious 34% crash in the lowest [0,0.2) bucket. Root cause,
by instrumenting the collision geometry (tractor = body points 0-2, trailer = 3-5):
- 84% of collisions are trailer clips, 16% tractor.
- **63% of culprit obstacles were BEHIND the difficulty window** (mean -4.6 m), and **100% of
  the anomalous low-bucket crashes were behind it** (mean -7.7 m).

In reverse the trailer *leads*, but the difficulty window looked only *ahead of the lead
point*, so obstacles the trailing tractor/body was still clearing were invisible (difficulty
= 0). Fix: anchor the window to the **whole vehicle body extent** (trailing edge - 1 m ->
leading edge + look-ahead) via `body_points`, not just ahead of the lead point
(`ObstacleHiddenPathEnv._in_window_mask`).

## 5. Result: a provably clean binary crash gate

After the window fix, measured at the collision moment over 479 collisions:
- culprit obstacle in-window: **1.000** (was 0.37).
- dynamic difficulty at collision: **min 0.511**, p5 0.593, median 1.000, mean 0.943.
- fraction of collisions with dynamic difficulty < 0.4: **0.000**.

**No reverse collision ever occurs below dynamic difficulty ~0.5.** The [0,0.2) crashes in the
episode-bucketed table were a one-step sampling lag (episode-max sampled just before the final
approach). So the dynamic difficulty is a clean binary crash gate at **X approximately 0.5**.

## 6. The stop reward (binary gate)

Because no crash occurs below X, a stop below X is unambiguously wrong (the layout is
passable) and a stop at/above X is the best achievable (attempting would crash). So the stop
reward becomes a clean binary keyed on the **dynamic** difficulty at the stop moment:
- **stop with dynamic difficulty >= X (~0.5) -> reward = terminal success (+200 reverse).**
- **stop with dynamic difficulty <  X -> penalty = terminal crash (-500).**
- complete -> +200; crash -> -500 (unchanged); running/tracking reward unchanged
  (multiplicative), so completion stays strictly preferred over a same-difficulty stop.

The observation's difficulty channel is switched from static to **dynamic** windowed
difficulty so the agent observes the exact signal it is gated on. This removes the
over-conservative-vs-over-confident tradeoff: the decision is "stop iff dynamic difficulty >=
X", a single threshold the policy can learn.

Watch item: making stop@(dd>=X) pay exactly like success could weakly incentivize deliberately
going off-path to trigger the stop reward on a passable layout; the intact running reward
(completion earns more) and the crash risk of going off-path mitigate it. Reduce the stop
reward slightly below success if over-stopping on passable layouts appears.

## 7. Residual limitation: articulation-driven stops on easy paths (accepted; out of scope to fix)

The binary-gate model (`hp_reverse_avoidance_d_win_dyn_s0`) is safe (overall crash 0.052) and
its obstacle gate is clean, but on the lowest dynamic-difficulty layouts it stops far more than
it needs to. Bucketed by max dynamic difficulty (n=11740):

| dyn bucket | n | complete | stop | crash | stop mean \|hitch\| (rad) |
|---|---|---|---|---|---|
| **[0.00,0.20)** | 762 | 0.013 | **0.866** | 0.121 | 0.204 |
| [0.20,0.40) | 2239 | 0.981 | 0.016 | 0.003 | 0.105 |
| [0.40,0.50) | 584 | 0.327 | 0.628 | 0.045 | 0.039 |
| [0.50,0.60) | 1160 | 0.104 | 0.799 | 0.097 | 0.101 |
| [0.60,0.80) | 1217 | 0.055 | 0.781 | 0.164 | 0.119 |
| [0.80,1.00) | 5778 | 0.080 | 0.890 | 0.030 | 0.062 |

**These stops are articulation-driven, not spurious.** Every crash is a collision (0 jackknife-
terminations, 0 out-of-bounds), but the collisions split by vehicle state: the low-difficulty
collisions occur at high hitch (mean |hitch| = 0.75 rad approximately 43 deg, deep in the
danger zone) while the high-difficulty collisions occur at low hitch (approximately 0.09). In
the [0.00,0.20) bucket the terminal |hitch| orders cleanly by outcome:

    success 0.011  <  stop 0.204  <  crash 0.753    (rad)

so the agent completes when articulation is clean, stops once the hitch has begun to degrade,
and crashes on the cases where it kept manoeuvring and the hitch ran away. The stops sit at an
intermediate hitch (approximately 0.20, roughly 20x the success level), well below the
unrecoverable regime (approximately 0.75), so the policy errs conservative: it aborts as the
articulation starts to degrade rather than at the point of no return.

**Why this is the correct reverse-only behaviour (and why we do not fix it).** A reversing
tractor-trailer cannot unwind a growing hitch by continuing to reverse; the reverse articulation
dynamics are unstable, so more reverse motion increases the hitch. The geometrically-required
recovery is to **pull forward**, re-stabilise the articulation on the (stable) forward dynamics,
and re-approach in reverse: a shunting / multi-point manoeuvre. That recovery needs a
forward-reverse switching action space, which is outside the scope of this continuous-reverse
study. Given a reverse-only action space, halting once the hitch is degrading is the safe and
correct response. Extending the action space to permit pull-forward recovery is future work.

Draft limitation paragraph (thesis-ready):

> On predominantly-straight reverse paths the agent occasionally halts where a continued
> manoeuvre would have completed. This is not spurious stopping: at these halts the trailer's
> hitch articulation is measurably degrading (terminal |gamma| approximately 0.20 rad, versus
> approximately 0.01 rad on completed paths), trending toward, though stopping short of, the
> unrecoverable regime that produces the collision failures (|gamma| approximately 0.75 rad). A
> reversing tractor-trailer cannot unwind a growing hitch angle by continuing in reverse; the
> geometrically-required recovery is to pull forward, re-stabilise the articulation, and
> re-approach (a shunting manoeuvre). As that recovery demands a forward-reverse switching action
> space outside the scope of this continuous-reverse study, halting is the correct and safe
> reverse-only response. The policy errs conservative, aborting as articulation begins to degrade
> rather than at the point of no return, trading completion throughput for safety. Permitting
> pull-forward recovery is left to future work.

## 8. Dynamic difficulty is image-recoverable (vision pipeline is unblocked)

The stop gate keys on *dynamic* difficulty, which depends on the current trailer CTE, so it is a
harder image target than the static per-layout difficulty validated earlier. Distilling the
9-dim avoidance state from the obstacle SDF-BEV (reverse, dynamic target, expert = the binary
model) recovers it well:

| dim | test R^2 | | dim | test R^2 |
|---|---|---|---|---|
| e_y | 0.967 | | e_psi_t | 0.882 |
| e_psi | 0.876 | | k1 | 0.425 |
| e_y_t | 0.977 | | k2 | 0.277 |
| **d_win (dynamic)** | **0.954** | | | |

Dynamic difficulty is a function of the free gap (image-visible) and the trailer CTE (recovered
at R^2 = 0.977), so it inherits their recoverability (R^2 = 0.954). The avoidance-path curvature
(k1, k2) is soft (0.43 / 0.28), softer than the forward case, but the errors and the difficulty
channel carry the control. Conclusion: the pixels->state->plan->control pipeline can gate on the
predicted dynamic difficulty; there is no perception roadblock.

## 9. Final multi-seed study (from scratch; 2026-07-17)

Six runs trained from scratch (fair: the observation space gained the dd channel between Phase 2
and Phase 3, so no shared warm-start checkpoint). Binary dyn_s0 recipe. Raw:
`final_multiseed_{analysis,vision,timeline}.txt`. Bucketed by max DYNAMIC difficulty.

**Reverse (dynamic recipe, 3 seeds):**

| seed | crash | complete | stop |
|---|---|---|---|
| s0 | 0.052 | 0.260 | 0.688 |
| s1 | **0.135** | 0.368 | 0.497 |
| s2 | 0.050 | 0.328 | 0.623 |
| mean +/- sd | **0.079 +/- 0.048** | 0.319 | 0.603 |

Finding: 2 of 3 seeds reproduce the safe approximately 0.05 result; s1 converged to a less
conservative transition-band policy (crash 0.36 / 0.42 in the [0.5,0.8) buckets vs s0/s2 stopping
there). The spread traces to the reward-change sensitivity of Section (critic-shock at the
warmup->obstacle boundary): the setup is usable but not tightly reproducible, and the
best-completion checkpoint selection can favour the less-safe policy. A critic-warmup (freeze
actor, refit critic on the new reward, then unfreeze) and a safety-aware checkpoint criterion are
the levers to tighten it. The accepted `dyn_s0` matches the two good seeds.

**Forward (dynamic vs static baseline):**

| model | crash |
|---|---|
| static sharp (baseline) | **0.027** |
| dynamic s0 / s1 / s2 | 0.092 / 0.138 / 0.032  (mean 0.087) |

Finding: dynamic difficulty **hurts** forward (mean 0.087 vs 0.027 static) and is high-variance.
Forward keeps the STATIC recipe. Confirmed reason: forward does not jackknife and its CTE is small
(~0.35 m), so the state-dependent width inflation adds noise without the reverse benefit. The
unified-recipe hope does not hold: the two directions have different dynamics and want different
stop signals.

**Tail-swing (collision geometry) -- a reverse-only phenomenon:**

| direction | tractor-clip | trailer-clip |
|---|---|---|
| reverse (all seeds) | 0.17-0.20 | **0.80-0.84** |
| forward (static AND dynamic) | **1.000** | 0.000 |

Finding: reverse crashes are 80%+ trailer-clips (the leading trailer sweeps into obstacles -- what
the body-extent window fixes). Forward crashes are **100% tractor-clips** (the leading tractor hits
tight gaps head-on); the trailing forward trailer follows cleanly and never sweeps in. So the
body-window fix is essential for reverse and irrelevant to forward -- the forward tail-swing
hypothesis is refuted by the data.

**Vision closed-loop (pixels -> predicted state -> plan -> control):**

| direction | encoder d_win R^2 | SAFE | crash | note |
|---|---|---|---|---|
| reverse | 0.954 | 0.749 | 0.083 | curvature soft (k1 0.43 / k2 0.28) |
| forward | 0.891 | 0.836 | 0.088 | curvature strong (k1 0.83 / k2 0.71) |

Finding: the full vision pipeline holds in both directions; predicted-state crash sits ~0.03-0.05
above the GT-state policy (expected prediction noise). Dynamic difficulty is recoverable both ways.
Reverse avoidance-path curvature is much harder to see in the image than forward, but the errors +
difficulty channel carry the control.

NOTE for the write-up: the forward vision row above (SAFE 0.836) is on the forward-DYNAMIC model,
which we DISCARD (dynamic hurts forward). The forward result to cite is the STATIC model's closed
loop -- **SAFE 0.903 / crash 0.041** (`vision_closed_loop.csv`) -- which is better, consistent with
static being the correct forward recipe. So the thesis pair is reverse-dynamic 0.749 / forward-static
0.903. (Optional cosmetic: re-run the forward-static closed loop under the identical harness as the
reverse one for a like-for-like table.)

## Code / data
- `batched/obstacles.py`: `free_gap_per_obstacle`, `gap_to_difficulty` (additive).
- `batched/obstacle_hidden_path_env.py`: `_in_window_mask` (body-extent window),
  `dynamic_windowed_difficulty`, dynamic-diff observation + threshold stop reward
  (`dd_pen_max=dd_pen_min=500` binary; `_hard_gate=False` = the un-regressed dyn_s0 recipe).
- Analyses: `scripts/phase3_analyze.py` (outcome-by-dd + crash-cause + collision geometry +
  hitch-at-stop, one shot per model); scratchpad `forensic.py`, `rootcause{,2,3}.py`,
  `crashcause.py`, `stopstate.py`.
- Recoverability: `scripts/geom_pretrain_obstacle.py --dynamic_diff` (d_win R^2 = 0.954).
- Final multi-seed study: `scratchpad/final_phase3_study.sh` (reverse x3, forward x3, from
  scratch; analysis + vision closed-loop). Results appended on completion.
