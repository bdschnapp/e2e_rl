# Reverse safety-gate filter: attempt / stop / crash by difficulty

A post-hoc **two-gate safety filter** on the trained reverse dynamic-difficulty policy (no
retraining), and the resulting attempt/stop/crash breakdown. Regenerate with
`scripts/safety_gate_sweep.py`; raw tables: `dd_sweep.csv`, `attempted_split.csv`.

## Model & evaluation
- Model: `hp_reverse_avoidance_d_win_dyn_s0` (reverse, dynamic-difficulty stop gate, binary reward).
- Env: reverse, mild paths, fixed 5 m/s, obstacles ON, policy's own learned stop ENABLED.
- Rollout: 256 envs x 4000 steps, seed 7 -> **11,740 episodes**. Bucketing is by DYNAMIC difficulty.
- Baseline (no filter): **success 0.260, stop 0.688, crash 0.052.**

## The safety filter
At every step (pre-action) the filter stops the vehicle if **either** gate trips:

> **STOP if  peak dynamic difficulty > X  OR  peak |hitch| > Y**

Two gates for two crash modes:
- **`dd > X`** targets **obstacle** crashes (tight gap; the vehicle tracks well but clips).
- **`|hitch| > Y`** targets **articulation** crashes (reverse hitch runaway). These are the
  [0,0.2)-dynamic-difficulty anomaly: dd reads ~0 because the obstacle gap is wide, yet the trailer
  swings in. The engineering response to a large hitch is to **pull forward to reset the articulation,
  then resume in reverse** (a shunt) -- out of scope here, so within a reverse-only action space the
  gate STOPS instead. 100% of those crashes occur at peak |hitch| >= 0.71 rad and were already above
  0.6 before impact, so the hitch gate catches all of them.

**Chosen gates: Y = 0.6 rad, X = 0.375.**

### How Y = 0.6 rad was chosen
Peak |hitch| reached during the episode, by outcome (rad): successful maneuvers stay low, crashes
run away.

| outcome | p50 | p90 | p99 | MAX |
|---|---|---|---|---|
| success | 0.040 | 0.204 | 0.400 | **0.659** |
| articulation crash | -- | -- | -- | (min 0.71) |

Y = 0.6 sits above the 99th-percentile of successful hitch (0.40 rad) and below the articulation-crash
floor (0.71 rad). It very occasionally stops an otherwise-successful high-hitch maneuver (those in
0.60-0.66), which is an accepted, safe cost.

### How X = 0.375 was chosen (sweep)
The filter is applied post-hoc to the model's rollouts; for each episode we track the peak dd and
peak |hitch| over the pre-action observations (what a real-time gate would see) and mark it gated if
the condition ever holds. Sweeping X (Y fixed at 0.6); `class3` is the RESIDUAL crash rate (crashes
the gate never caught), `succ_lost_to_gate` is would-succeed episodes the gate stopped:

| X (dd) | 1 success | 2 model-stop | 3 **crash (residual)** | 4 gate-stop | succ lost to gate |
|---|---|---|---|---|---|
| 0.300 | 0.074 | 0.057 | **0.0000** | 0.869 | 0.185 |
| 0.325 | 0.114 | 0.057 | 0.0001 | 0.828 | 0.145 |
| 0.350 | 0.152 | 0.058 | 0.0002 | 0.790 | 0.107 |
| **0.375** | **0.177** | 0.058 | **0.0004** | 0.764 | 0.083 |
| 0.400 | 0.188 | 0.059 | 0.0006 | 0.752 | 0.072 |
| 0.450 | 0.197 | 0.067 | 0.0014 | 0.735 | 0.063 |
| 0.500 | 0.204 | 0.093 | 0.0039 | 0.699 | 0.056 |

The obstacle-type crashes (hitch <= Y) have a minimum peak-dd of **0.314**, so a *literal* zero
residual requires **X < 0.314** -- but that costs most of the throughput (completions collapse to
0.074, because dynamic difficulty routinely spikes above 0.3 while legitimately passing an obstacle).
**X = 0.375** reduces crashes 0.052 -> **0.0004** (99.2%; ~5 episodes in 11,740) while keeping 2.4x
the completions of the literal-zero setting -- the recommended operating point.

## Result: outcome by peak dynamic difficulty (gate X = 0.375, Y = 0.6)

Overall only **23.6%** of episodes are ATTEMPTED (gate never fires); the other 76.4% are auto-stopped,
which is mostly correct -- **75.6%** of the eval layouts have peak dd > 0.375 (half sit at 0.7-1.0),
where stopping is the right answer. Among the attempted episodes: **success 0.750, model-stop 0.248,
crash 0.002.**

> NOTE: this peak-dd view is CONFOUNDED for comparison -- "attempted" here still counts the RL's
> voluntary model-stops (0.248 of it), and stopping early caps an episode's peak dd, inflating the
> low buckets. For the un-confounded picture (model-stop as its own class, layout-intrinsic buckets),
> and the pure-pursuit comparison, see "Comparison to a pure-pursuit baseline" below.

| peak-dd bucket | n | gated % | \| among ATTEMPTED: | success | model-stop | crash |
|---|---|---|---|---|---|---|
| [0.00, 0.10) | 756 | 12.3% | \| | **0.000** | 1.000 | 0.000 |
| [0.10, 0.20) | 10 | 0% | \| | 1.000 | 0.000 | 0.000 |
| [0.20, 0.30) | 867 | 0% | \| | **0.994** | 0.006 | 0.000 |
| [0.30, 0.375) | 1228 | 0.2% | \| | **0.981** | 0.015 | 0.004 |
| [0.375, 0.50) | 777 | 100% | \| | -- (auto-gated) | | |
| [0.50, 0.70) | 1908 | 100% | \| | -- (auto-gated) | | |
| [0.70, 1.00) | 6194 | 100% | \| | -- (auto-gated) | | |

## Interpretation
- **On genuine mild-obstacle layouts we attempt (peak dd 0.1-0.375): completion is 98-99%**, crash ~0.
  This is the real capability figure.
- The low overall completion (~0.18) is NOT a failure rate: it is (a) a hard-layout-dominated eval
  (76% of layouts correctly gated) plus (b) the **[0,0.10) bucket, where the policy stops 100% and
  completes 0%**. Those are the articulation-conservatism stops (the policy bails as the hitch begins
  to degrade on a curve, capping the episode's peak dd near 0); forced-drive completes ~92% of gentle
  paths, so most are recoverable throughput, not necessary stops.
- **Crash is effectively eliminated (0.052 -> 0.0004).**

## Comparison to a pure-pursuit baseline (same gate)

To separate the learned policy's value into "tracking" vs "stop decision", we run pure pursuit (PP)
through the identical pipeline: same env, same avoidance path, same eval pool, same gate
(X = 0.375, Y = 0.6). PP tracks the avoidance path (steer only) and has NO learned stop, so the
deterministic gate is its only stop mechanism. Data: `safety_gate_4class_{model,pp}.csv`.

Bucketed by **static (layout) difficulty** -- the layout-intrinsic axis, identical for both
controllers on the same layouts, which avoids the peak-dd confound (see Caveats). 4 classes; the gate
takes precedence over the natural end.

**Model (RL), n = 11,740 episodes:**

| static difficulty | success | model-stop | crash | gate-stop |
|---|---|---|---|---|
| [0.0,0.2) | 0.738 | 0.095 | 0.024 | 0.143 |
| [0.2,0.4) | 0.596 | 0.060 | 0.001 | 0.343 |
| [0.4,0.6) | 0.000 | 0.055 | 0.000 | 0.945 |
| [0.6,0.8) | 0.000 | 0.060 | 0.000 | 0.940 |
| [0.8,1.0) | 0.000 | 0.059 | 0.000 | 0.941 |
| **overall** | **0.177** | **0.058** | **0.000** | 0.764 |

**Pure pursuit, n = 8,031 episodes:**

| static difficulty | success | model-stop | crash | gate-stop |
|---|---|---|---|---|
| [0.0,0.2) | 0.719 | 0.000 | 0.000 | 0.281 |
| [0.2,0.4) | 0.610 | 0.000 | 0.003 | 0.387 |
| [0.4,0.6) | 0.000 | 0.000 | 0.001 | 0.999 |
| [0.6,0.8) | 0.000 | 0.000 | 0.004 | 0.996 |
| [0.8,1.0) | 0.000 | 0.000 | 0.002 | 0.998 |
| **overall** | **0.181** | **0.000** | **0.002** | 0.817 |

*Sample counts differ but the eval set is the same.* The eval is a fixed STEP budget (256 envs x 4000
steps), not a fixed episode count, so the two controllers complete different numbers of episodes only
because the RL's voluntary stops make its episodes shorter. Both draw from the same 512-path pool and
the per-bucket fractions match (e.g. [0.2,0.4): 0.292 vs 0.293; [0.8,1.0): 0.421 vs 0.428), confirming
an identical eval distribution -- compare per-episode rates, not counts.

**Reading the comparison** (model-stop is a distinct class -- a decision NOT to attempt -- so it is
not counted as an attempt; PP has none):
- **Per-bucket success is nearly identical** (e.g. [0.2,0.4): 0.596 vs 0.610) -- the two controllers
  track the same feasible layouts equally well.
- **Commit-success** = success / (success + crash), i.e. "when it drives to a terminal, does it make
  it": **RL 0.998 vs PP 0.990.**
- **Under the gate both are safe** (crash 0.000 vs 0.002); the RL is marginally safer because its
  voluntary stop catches the ~0.2 % articulation cases PP jackknifes.
- **PP tracks at least as tightly** (median CTE on completed episodes 0.077 vs the RL's 0.328).
- **Without the gate, the RL is decisively safer: crash 0.052 vs PP 0.540** -- PP has no way to decline
  an infeasible layout, so it drives in and crashes on 54 % of the (hard-layout-dominated) mix.

**Conclusion.** Given the perception layer (avoidance path + difficulty), the control layer is largely
interchangeable: PP + gate matches the learned policy on completion (0.181 vs 0.177) and is nearly
equal on safety. The learned policy's distinctive value is (a) the learned attempt-vs-abort **stop
decision when no hand-designed gate is available** (crash 0.052 vs 0.540), and (b) a sliver of extra
safety with the gate. The reverse avoidance path itself is trackable by a classical controller (PP
completes the feasible band at ~0.99 commit-success), so the thesis contribution is best framed as the
**perception -> plan (hidden-path + difficulty distillation) plus the difficulty gate**, which either a
learned or a classical controller can then execute safely.

## Caveats
- Post-hoc filter on an already-conservative policy. Retraining with the articulation term folded into
  the difficulty (`dd = max(obstacle_dd, jackknife_risk(|hitch|))`) would let the policy learn to avoid
  the gated region and recover much of the lost throughput.
- Peak-dd bucketing has a confound: stopping early caps an episode's peak dd, so the [0,0.10) bucket
  collects the preemptive/articulation stops, and voluntary model-stops inflate the "attempted" count.
  The static-difficulty bucketing in the pure-pursuit comparison (with model-stop as its own class)
  disentangles it, and is the version to cite for the RL-vs-classical comparison.
- Numbers are the reference seed (dyn_s0); the 0.314 ceiling may shift slightly across seeds.
