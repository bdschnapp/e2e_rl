# Parameter / config audit — what deserves ablation vs. what is justified

Scan of the "magic numbers" and design choices across `e2e_rl` (`e2erl_utils/config.py`,
`train.py`) and `tractor_trailer_rl` (`config.py`), plus the TD3 setup. Verdicts:

- **ABLATE** — a design choice that plausibly changes results and has no first-principles
  justification; a reader/reviewer would ask, or we already know it matters.
- **JUSTIFIED** — defensible by a clear reason (physical, standard practice, matched-to-
  deployment, or required for a fair comparison); state the reason, no test needed.
- **PLANNED** — already in the ablation campaign.

Prioritised so the top items are the highest-value additions to the study.

---

## Highest-value NEW ablations (do these)

| Parameter / choice | Where | Current | Verdict | Why |
|---|---|---|---|---|
| **Policy network architecture** (width/depth) | `train.py:521` net_arch `[256,256]`; batched TD3 `[400,300]` | fixed | **ABLATE** | You flagged this. Capacity vs. sample-efficiency is untested and cheap on GPU; also the two codebases disagree (256×256 vs 400×300) — pick one and show sensitivity. |
| **Discrete vs continuous action space** | `actions/modes.py` | continuous `[δ̇,v]` | **ABLATE** | You flagged this. Evenly-spaced discretisation of steer-rate (and speed) is a real alternative; note it changes the algorithm (discretised-TD3 / DQN-style), so it's a scoped experiment, not a config flip. |
| **Steering-rate limit** `steering_action_deg` | `config.py:146` | 25 °/s | **ABLATE** | Directly bounds agility; the lab-transfer finding showed the *action distribution* is where scale mismatch bites — the steer-rate ceiling is exactly that knob. Pairs naturally with the discrete-action study. |
| **Hitch-penalty coefficient** `exp(-1.5·|γ|)` fwd / `exp(-2·|γ|)` rev | `reward/composer.py` | 1.5 / 2.0 | **ABLATE** | `lab_deployment_findings.md` §3 explicitly says the hitch penalty is *counter-productive in reverse* (hitch deflection is the control authority). This coefficient is both magic and known-impactful. |
| **Exploration noise** σ (steer 0.05, speed 0.5) + `target_policy_noise` | `train.py:528,685` | 0.05 / 0.5 / 0.05 | **ABLATE** | `bev_regular_driving_audit_recommendations.txt` found action-noise magnitude decisive for BEV. Also `target_policy_noise=0.05` **deviates from the TD3 paper's 0.2** with no stated reason — justify or test. |
| **Training path distribution** `kind_probs` + bend geometry | `config.py:182`, `paths/generator.py` | mixture (straight .18 … lab_corner .25) | **ABLATE / discuss** | The single biggest sim-to-real lever per `lab_deployment_findings.md` (curvature-to-wheelbase ratio). At minimum a sensitivity study; the curvature spectrum *is* the transfer story. |
| **Curvature look-ahead** k1=10, k2=20 samples; fixed vs speed-scaled | `config.py:132-139` | fixed 10/20 | **ABLATE** | The `SPEED_SCALED` mode already exists in code but is off by default — testing fixed-vs-speed-scaled preview (and the distances) is a clean, cheap ablation with a clear hypothesis. |
| **BEV image resolution / crop / anchor** | `e2e_rl` config | 32×32, 20 m crop, rear-axle anchor, zoom 3 | **ABLATE (resolution) / JUSTIFY (rest)** | Resolution 32 vs 64/84 is untested (and the docs still disagree 32 vs 84). The encoder ablation already covers *representation*; add one resolution point. Anchor/crop are defensible design choices. |

---

## Reward shaping — structural choice PLANNED, coefficients mostly JUSTIFIED

| Parameter | Current | Verdict | Reason |
|---|---|---|---|
| Reward **mode** (dense/tractor_focus/multiplicative/guided/no_hitch) | per-dir | **PLANNED** | This is the reward ablation. Covers the big structural decision. |
| Terminal reward fwd ±100 / rev +200,−500 | asymmetric | **ABLATE (small)** | The rev −500 vs +200 asymmetry is magic; it interacts with the stop-penalty ordering (crash<stop<success). Worth one check. |
| Path-error weights 0.5·e_ψ, 0.75·e_y(trailer), 0.5·e_ψ(trailer) | fixed | **JUSTIFIED** | Encodes the stated thesis principle "trailer tracking prioritised over tractor"; the exact 0.75 is arbitrary but a full weight sweep is out of scope given the mode ablation. |
| Multiplicative scale 4.0/5.0, speed-floor 0.5 | fixed | **JUSTIFIED** | The floor's *existence* is justified (documented anti-"suicidal-agent" fix); magnitudes are secondary to the mode choice. |
| `path_tightness` 1.0 (lab used 2.0) | 1.0 | **JUSTIFIED / note** | A reward temperature; note the lab value differed. Low priority. |
| Reverse jackknife penalty `10·max(0,|γ|−0.4)²` | 10, 0.4 | **ABLATE (with hitch coeff)** | Same family as the hitch coefficient above; fold into that study. |

---

## Vehicle / plant — mostly JUSTIFIED (with one real caveat)

| Parameter | Current | Verdict | Reason |
|---|---|---|---|
| Tesla-S params m,Iz,Cf,Cr,lf,lr,Cd,A | fixed | **JUSTIFIED** | Representative passenger-vehicle plant; the study is control/perception, not vehicle ID. |
| Cf = Cr (equal cornering stiffness) | equal | **JUSTIFIED** | Standard linear-tyre simplification. |
| `dt` = 0.1 s (10 Hz) | 0.1 | **JUSTIFIED** | Matches the deployed RL-bridge 10 Hz control loop — deliberate, not arbitrary. |
| **Vehicle / trailer scale** (wheelbase, trailer length 10 m) | fixed | **ABLATE / discuss** | `lab_deployment_findings.md` §1: transfer depends on path-curvature-to-wheelbase *ratio*. This is a headline sim-to-real result — needs the scale discussion even if not a full sweep. |
| Longitudinal PID kp3/ki0.3/kd0.05, `e_prev` never updated | fixed | **JUSTIFIED / FIX** | Inner speed loop, not the object of study (fixed-speed tasks). BUT the `e_prev=0` parity quirk is a latent bug — document or fix; don't ablate. |

---

## Simulation / world — JUSTIFIED

| Parameter | Current | Verdict | Reason |
|---|---|---|---|
| World extent, lane half-width (5 m truck / 1.41 m lab), shoulder | scale-matched | **JUSTIFIED** | Lane sized to the vehicle; lab value = real rig. |
| `grid_res` 0.1 m, lidar_step 0.05 m | fixed | **JUSTIFIED** | Resolution/compute trade-off; the analytic CuPy corridor removes the grid anyway. |
| `max_episode_steps` 1000 | fixed | **JUSTIFIED** | Long enough to complete the route. |
| success line `x > 0.85·W` | 0.85 | **JUSTIFIED (note)** | Avoids world-edge effects; the 0.85 is mildly arbitrary but low-impact. |
| Obstacle curriculum (ramp 0→5/0→10 every 2000 steps) | fixed | **JUSTIFIED** | Standard difficulty ramp; the *stop-gate* redesign is the real obstacle change. |

---

## Observation — beam count PLANNED, rest JUSTIFIED

| Parameter | Current | Verdict | Reason |
|---|---|---|---|
| `lidar_beams` (4/8/16/24/32) | 24 | **PLANNED** | Beam-count ablation. |
| lidar FOV 120°, range 20 m | fixed | **JUSTIFIED** | Reasonable forward cone / sensor range; low-priority to sweep. |
| obs Box bounds (steering π/6, hitch π/2, CTE 100 …) | fixed | **JUSTIFIED** | Normalisation/clipping ranges, not physical parameters. |
| `error_theta_scale` 1.0 | 1.0 | **JUSTIFIED** | Heading-error normalisation. |

---

## TD3 algorithm — standard = JUSTIFIED, deviations = ABLATE

| Parameter | Current | Verdict | Reason |
|---|---|---|---|
| γ 0.99, τ 0.005, policy_delay 2, lr 1e-3, batch 256, buffer 300k, learning_starts 5k | defaults | **JUSTIFIED** | Standard TD3 / SB3 defaults; cite the TD3 paper. (Large-batch/high-LR is being explored separately as a *GPU-throughput* study, not a claim about the base result.) |
| `target_policy_noise` 0.05 | 0.05 | **ABLATE / JUSTIFY** | TD3 paper uses 0.2 — this is an unexplained deviation. |
| action noise σ 0.05/0.5 | fixed | **ABLATE** | See exploration-noise row above (audit flagged it decisive). |
| `total_timesteps` 100k–200k | fixed | **JUSTIFIED (per-config)** | Convergence budget; validate per config rather than sweep. |

---

## One-line summary

**Add to the campaign:** network architecture, discrete-vs-continuous actions + steer-rate
limit, the hitch/jackknife penalty coefficient, exploration/target-policy noise, the
look-ahead (fixed vs speed-scaled), and a BEV-resolution point — plus a **discussion/
sensitivity of the training path-curvature distribution and vehicle scale** (the two
things `lab_deployment_findings.md` says drive sim-to-real). **Justify without testing:**
the vehicle plant params (representative + `dt` matches 10 Hz deployment), lane/world
geometry (scale-matched), obs bounds & lidar FOV/range, the reward *coefficients* (mode
ablation covers the structural choice; trailer-weighting encodes a stated principle), and
the standard TD3 hyperparameters (cite the paper) — with the two noted deviations
(`target_policy_noise`, action-noise) called out.
