# Obstacle Difficulty Metric — Definition and PP-Drivability Calibration

*Generated 2026-07-07. Evidence for the Ch.5 obstacle-navigation section: how the
`difficulty` rating is defined, why it is a geometric proxy, and how a pure-pursuit
drivability sweep calibrates it into a per-direction, reward-relevant threshold.*

---

## 1. Difficulty as implemented

`difficulty` is computed once per reset (`batched/obstacles.py::difficulty`) as a
**static geometric clearance ratio**, not a dynamic drivability estimate:

```
difficulty = clip( 1 − (max_gap − veh_w) / (lane_w − veh_w),  0, 1 )
```

- `max_gap` — widest contiguous free lateral gap at the tightest blocked cross-section,
- `veh_w` = max(tractor width, trailer width),
- `lane_w` = full lane width.

Semantics (exactly the intended design):

| difficulty | meaning |
|---|---|
| **0.0** | lane fully open (`max_gap = lane_w`) |
| **0 < d < 1** | partially blocked but the body geometrically fits; margin shrinks as d rises |
| **1.0** | `max_gap ≤ veh_w` — gap is at or below the vehicle width → **impassable** |

Two properties to keep in mind:

1. **The `impossible` regime is clipped to 1.0.** Any layout with `gap ≤ veh_w` — whether
   exactly critical or a solid wall — reads as `difficulty = 1.0`. The metric cannot
   distinguish degrees of impossibility.
2. **It is a *cheap geometric proxy* for drivability, by design.** Evaluating true
   kinematic feasibility (does a collision-free trajectory exist for a nonholonomic
   tractor-trailer?) inside the training loop is too slow, so difficulty measures whether
   the body *width* fits the gap. It does **not** account for the manoeuvring / trailer-swing
   room a real trajectory needs — which is direction-dependent and much larger in reverse.

The calibration below quantifies that proxy gap.

---

## 2. Calibration method: pure pursuit as a drivability oracle

To map geometric `difficulty` → actual drivability we run the **traditional stack**
(global path → APF local planner → pure-pursuit tracking) **with the stop action
disabled** (`--stop_thresh 1.1`, so it always attempts), stratified by difficulty. The
per-bin completion rate is the empirical drivability `P_clear(d)`.

```
TTRL_BACKEND=cupy python3 scripts/eval_obstacle.py --baseline local_pp \
    --direction {forward,reverse} --stop_thresh 1.1 --buckets 10 \
    --episodes 8000 --n_envs 4096
```

Real shunt-truck (Kalmar Ottawa T2) params, 8000 episodes/direction, GPU (cupy) backend.

**Reward-relevant break-even.** With terminal `success = +100`, `crash = −500`, the expected
value of attempting a layout is

```
E[attempt](d) = P_clear(d)·100 − (1 − P_clear(d))·500
```

`E[attempt] = 0` at **`P_clear = 500/600 = 83.3%`**. Above that clear-rate attempting is
positive-EV (the policy *should* drive); below it, stopping is the rational choice. The 5×
crash-vs-success asymmetry makes this frontier appropriately conservative. The difficulty
where `P_clear` crosses 83.3% is the **true drivability threshold `d*`** — and it is what the
stop-gate reward should key on, per direction, instead of the raw geometric scale.

---

## 3. Forward drivability (8004 episodes)

| difficulty | n | P_clear (%) | crash (%) | E[attempt] |
|---|---|---|---|---|
| 0.2–0.3 | 1707 | 99.8 | 0.2 | **+98.8** |
| 0.3–0.4 | 329 | 99.1 | 0.9 | **+94.6** |
| 0.4–0.5 | 992 | 93.5 | 6.5 | **+61.0** |
| 0.5–0.6 | 881 | 83.5 | 16.5 | **+1.0** |
| 0.7–0.8 | 230 | 86.8 | 13.2 | **+20.8** |
| 0.8–0.9 | 984 | 73.1 | 26.9 | −61.4 |
| 0.9–1.0 | 2841 | 4.8 | 95.2 | −471.2 |

**Forward `d* ≈ 0.8`.** Attempting stays positive-EV up to ~0.8, then falls off a cliff
(73% clear at 0.8–0.9, 4.8% at 0.9–1.0). Geometrically-hard-but-`<0.8` layouts are genuinely
drivable forward. (Bins 0.6–0.7 are sparse — the difficulty distribution is bimodal, most mass
at 0.2–0.3 and 0.9–1.0 — so the mid curve is slightly noisy but the trend is clear.)

---

## 4. Reverse drivability (8020 episodes)

| difficulty | n | P_clear (%) | crash (%) | E[attempt] |
|---|---|---|---|---|
| 0.2–0.3 | 1611 | 78.3 | 21.7 | **−30.2** |
| 0.3–0.4 | 327 | 71.9 | 28.1 | −68.6 |
| 0.4–0.5 | 977 | 58.2 | 41.8 | −150.8 |
| 0.5–0.6 | 969 | 24.4 | 75.6 | −353.6 |
| 0.7–0.8 | 228 | 48.3 | 51.7 | −210.2 |
| 0.8–0.9 | 1025 | 44.2 | 55.8 | −234.8 |
| 0.9–1.0 | 2843 | 3.3 | 96.7 | −480.2 |

**Reverse `d* < 0.2`** — off the bottom of the scale. PP never clears ≥83% at *any*
difficulty; even the geometrically-easiest layouts (0.2–0.3) clear only 78%, and
**E[attempt] is negative in every bin.** For reverse, stopping is the rational choice at
essentially all difficulties.

---

## 5. Why this matters (interpretation)

1. **The difficulty scale is direction-blind, but drivability is not.** Forward `d*≈0.8` vs
   reverse `d*<0.2`: the same geometric gap is comfortably drivable forward and hopeless in
   reverse, because reverse needs far more trailer-swing room than the body width. The metric
   labels a reverse layout "easy" (d=0.25) when it is, in practice, at its drivability limit.

2. **It reframes the trained RL results (see `results_obstacle/eval_gpu2.log`).**
   - *Forward RL over-stops at mid* (stops 68% at d=0.4–0.6) even though those layouts are
     drivable (E[attempt] +1…+61). This is a genuine **policy suboptimality** — leaving reward
     on the table by stopping on passable layouts.
   - *Reverse RL stops ~86–90% everywhere*, which we first read as degenerate. The drivability
     curve shows it is **near-rational**: E[attempt] is negative at every difficulty, so
     stopping almost always *is* the higher-value action. Reverse is a fundamentally harder
     problem, not a training failure.

3. **Actionable calibration (no retraining to derive; one retrain to apply).** Reshape the
   stop-gate reward's `exp_scale` to span `[0, d*]` **per direction** so the stop incentive's
   break-even sits at the true drivability frontier (forward ≈0.8; reverse ≈ its low frontier).
   Forward, this penalises stopping across the drivable mid-range, pushing the policy to attempt
   d≤0.8. Reverse, it makes the near-always-stop behaviour explicitly correct rather than a
   pathology — and motivates a better reverse driver / curriculum / richer perception as the
   real lever, since reward shaping alone cannot make an undrivable layout drivable.

---

## 6. Caveats

- **PP is an imperfect oracle, especially in reverse.** Pure pursuit is a weak reverse
  tractor-trailer controller, so the reverse curve is best read as a **lower bound** on true
  drivability. However, the trained RL policy did not beat it (it also over-stopped), so PP's
  poor reverse performance is consistent with the task being genuinely hard, not merely a
  controller artifact. A stronger reverse planner would sharpen `d*_reverse`.
- **Geometric proxy + clip at 1.0** (Section 1): difficulty conflates all `gap ≤ veh_w`
  layouts and ignores dynamic feasibility — that is the entire reason this calibration is
  needed.
- **Bimodal difficulty distribution:** most episodes fall at 0.2–0.3 or 0.9–1.0; the 0.6–0.7
  bins are sparse (n≈10–15) and noisy. Frontiers are read from the well-populated bins.
- **Params:** real Kalmar Ottawa T2 shunt-truck geometry; `success=+100`, `crash=−500`,
  `stop_hard_reward=stop_easy_penalty=60`, `stop_difficulty_k=3.0` (current config).

*Raw logs: `tractor_trailer_rl_cupy/results_obstacle/pp_drivability_{fwd,rev}.log`.
Reproduce with the command in Section 2.*
