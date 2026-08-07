# Thesis-worthy findings from the Electrans lab deployment

Notes captured 2026-05-28 while integrating the e2e_rl TD3 policies with the
AgileX 1/8-scale lab rig via the Autoware planning_simulator and the
`electrans_rl_bridge` ROS 2 package.

The integration produced three observations that are general enough to be
worth writing into the thesis (probably across the Simulation, Deployment,
and Results chapters). Implementation-level noise (config-patching bugs,
specific tuning values, drive-direction inference formula, etc.) is
intentionally excluded.

---

## 1. Vehicle-scale transfer of RL controllers (Simulation / Training chapter)

**Finding.** A TD3 policy trained at the *training-default* semi-truck scale
(lf=1.2, lr=1.6, trailer=10 m, lane half-width 5 m) transferred to the lab
better than a policy retrained at the *actual* lab-scale vehicle dimensions
(lf=0.33, lr=0.32, trailer=2 m, lane half-width 1.41 m), despite the
retraining being intuitively the right move.

The failure mode of the lab-scale policy is **understeer at sharp corners**
(specifically the MVSL ~2 m-radius 90° corner at 0.6 m/s driving speed).
The semi-truck-scale policy handles the same corner adequately even though
it was never trained on the lab vehicle.

**Mechanism.** Both policies were trained on the same path generator (cubic
spline through 6 random control points spread over 140 m, y ∈ ±5 m, min
radius ≈ 40 m). What differs is the path radius *relative to the trained
vehicle's wheelbase*:

| Policy            | Wheelbase | Trained-min radius | Ratio (radius / wheelbase) |
|-------------------|-----------|--------------------|----------------------------|
| Semi-truck scale  | 2.8 m     | ~40 m              | ~14×                       |
| Lab scale         | 0.65 m    | ~40 m              | ~60×                       |
| Lab deployment    | 0.65 m    | ~2 m  (MVSL corner)| ~3×                        |

For the lab-scale policy, the trained-curvature regime is ~20× more
permissive than what it faces at deployment. For the semi-truck-scale
policy, the mismatch is only ~5×.

The deeper effect is on the **action distribution**:

- A sluggish semi-truck only makes progress on its training curves if the
  policy outputs *sustained large* steering rates. The training reward
  (multiplicative, with `exp(-1.5·|hitch|)` hitch penalty) doesn't punish
  the magnitude — it punishes the result.
- The lab-scale vehicle handles 40 m radius effortlessly with small
  corrections. The same reward function pushes the policy toward small,
  precise steering inputs.

When deployed against a 2 m-radius corner, the larger-action policy
extrapolates in the useful direction; the small-action policy doesn't have
enough authority in its action distribution to track the corner.

**Lesson.** Naive "train at deployment scale for best transfer" reasoning
is wrong when the path-distribution stays fixed. What matters is the
*ratio* of path curvature to vehicle wheelbase — i.e. the relative tightness
the policy was trained against. Either the path distribution must be
adapted to the vehicle, or the action distribution must be regularised
some other way (action-noise schedule, action-magnitude bonus, domain
randomisation over vehicle scale, etc.).

**Suggested artefacts for thesis.**

- Histogram of `error_theta` and steering-rate magnitudes from rollouts of
  both policies at training time. Should show a clear distribution shift.
- A short table like the one above showing the relative-tightness ratio
  argument.
- A figure showing the policy's steering response at the lab corner: old
  policy reaches near-saturation, new policy stays near zero.

---

## 2. Pygame training env ↔ Autoware deployment env distribution gap (Deployment / Integration chapter)

**Finding.** Even when the trained policy works perfectly in its native
pygame eval environment, two specific gaps to the Autoware
`simple_planning_simulator` cause non-trivial deployment regressions. Both
are generic to anyone bridging a Python RL env to an Autoware planning
stack.

### 2.1 Actuator dynamics

- Pygame's `StateSpaceTractorTrailer.loop()` applies steering changes
  instantaneously; the env has no actuator model.
- Autoware's `DELAY_STEER_ACC_GEARED*` vehicle models apply a dead-time +
  first-order-lag on tire angle (`steer_time_delay`,
  `steer_time_constant`). The default values inherited from the
  `sample_vehicle` config are 0.24 s dead time + 0.27 s time constant —
  calibrated for full-size cars with hydraulic steering.
- At lab driving speed (0.6 m/s) those defaults dominate closed-loop
  behaviour: a policy expecting instant response keeps stacking corrections
  for ~500 ms of arrival delay, producing the observed oscillation /
  saturation. A hobby-class servo on the actual AgileX rig responds in
  ~50 ms with no dead time, so the autoware defaults aren't just wrong for
  training-transfer — they're wrong for the physical system too.

The fix is a one-line YAML edit, but the diagnostic is worth one
paragraph: a policy that works *in its own env* and fails *in the
deployment env* with no other changes is almost always an actuator-model
mismatch, not a controller bug.

### 2.2 Path distribution

- Training paths are randomly-generated cubic splines through 6 control
  points (smooth, low-curvature).
- Deployment paths are constructed from hand-authored Lanelet2 centerlines
  (piecewise linear at 0.5 m resolution, with explicit sharp corners that
  the cubic-spline distribution never produces).

The bridge resamples the lanelet to a denser polyline before passing it
into the env, but the *shape distribution* (in particular the local
curvature spectrum) cannot be reconciled by resampling. The training-time
curvature spectrum is set by the env's path generator, which is a
deployment-blind design choice.

**Lesson.** When training in pygame for ROS deployment, the actuator
model and the path distribution are the two highest-impact discrepancies.
Both deserve explicit mention. The bridge's job is to translate state
formats (ROS ↔ env), not to compensate for distribution shifts in either
direction.

---

## 3. Forward vs reverse RL trainability with a trailer (Results / Discussion chapter)

**Finding.** Forward trailer-following is open-loop stable: the trailer
naturally tracks the truck. Reverse trailer-driving is open-loop
unstable (non-minimum-phase: steering right makes the trailer go left,
with delay). The same TD3 + multiplicative-reward setup that produces
monotonically-improving forward training produces *fast initial
improvement followed by plateau or collapse* in reverse training.

This is consistent across both scales:

- Semi-truck-scale reverse training: best eval reward 211 at 20 k steps,
  then degrades.
- Lab-scale reverse training: best eval reward ~-570 at 200 k steps —
  never escapes the early-termination basin (episode_length 65 vs
  forward's 222).

**Lesson.** A vanilla actor-critic RL setup that works for forward
lane-following is not a viable choice for reverse trailer driving without
substantive modification. Plausible directions:

- Action reparameterisation: the policy outputs target hitch angle (or
  target trailer yaw) rather than steering rate, and an inner controller
  closes the loop. Shifts the unstable inversion out of the learned
  function.
- Curriculum: start with very gentle curves, increase curvature only after
  the policy has learned to stabilise the hitch.
- Reward redesign: the forward `multiplicative` reward's hitch penalty
  `exp(-1.5·|hitch|)` is reasonable for forward but actively
  counter-productive for reverse where hitch deflection *is* the control
  authority.

**Suggested artefacts for thesis.**

- Learning curves for forward and reverse on the same axes — the
  qualitative shape difference is the figure.
- A brief stability analysis showing the non-minimum-phase nature of
  reverse hitch dynamics.

---

## What NOT to document

- Specific bugs found during integration (e.g. `TractorTrailerEnv.__init__`
  hardcoding `lf=1.2, lr=1.6` and bypassing the config dict). Implementation
  noise.
- Specific numerical values we tried during tuning (lane half-width 1.41
  vs 2.0, steer_time_constant 0.05 vs 0.15, etc.). Pick representative
  numbers for thesis figures and don't chronicle the loop.
- The drive-direction inference formula (XOR of canonical-aligned heading
  and goal-downstream). Useful for the ROS package's docs, not the
  thesis.
- The lanelet bidirectional collapse procedure. Same — package doc, not
  thesis.
