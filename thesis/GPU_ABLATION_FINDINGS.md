# GPU ablation — findings, numbers, and thesis-writing notes

Consolidated record of the results and decisions from the CuPy/GPU ablation work, so
nothing is lost when writing the manuscript. Companion docs in this folder:
`FIRST_DRAFT_CHECKLIST.md` (overall plan), `ABLATION_STUDY_DESIGN.md` (the staged
study design + walkthrough), `PARAMETER_AUDIT.md` (what to ablate vs justify).
Code + full detail: `/home/ben/Ben/Thesis/tractor_trailer_rl_cupy/README_CUPY.md`.

All numbers below measured on an **RTX 4080 SUPER (16 GB)**, backend cupy 14.1.1,
SB3 2.7.0, torch 2.9/CUDA 12.8, unless noted. Vehicle params were the interim
**Tesla placeholder** (see §6) — treat numbers as relative, re-run final with real
shunt-truck params.

---

## 1. Systems contribution — GPU-parallel batched env (Ch.5 / methods)

A CuPy port of `tractor_trailer_rl` runs **N environments at once on the GPU** with a
leading `(N,)` batch axis (`tractor_trailer_rl_cupy/src/tractor_trailer_rl/batched/`).
The five scalar blockers were replaced (writeup in README_CUPY):
1. per-step `scipy.cont2discrete` ZOH → batched matrix-exponential (Padé-13); matches
   scipy to ~1e-8 abs / 1e-12 rel.
2. per-reset `scipy cKDTree` occupancy grid → analytic distance-to-centerline + an
   on-device path pool (removes the reset bottleneck).
3. sequential per-beam lidar ray-march → vectorised fixed-length march (float32,
   local segment window).
4. Python-loop collision/proximity → batched gather.
5. scalar termination/stop/spawn branches → boolean masks + internal auto-reset.

**Parity vs the scalar oracle** (`tests/test_batched_parity.py`, 11 pass): observation
**exact (0.0)**, reward **~1e-14** (proximity-free), dynamics **~1e-14** over 200
steps, **numpy backend ≡ cupy backend (0.0)**. Intentionally analytic (not bit-exact):
proximity differs ~0.1–0.5 only near the lane edge; lidar within ~1 cell of the grid
raycast; collision decisions agree with the grid on **97–98.5%** of random states
(disagreements within one cell of the boundary).

### Throughput (env stepping, transitions/s), lab_config, lidar-24
| Observation | Batched GPU (best N) | Scalar single-core |
|---|---|---|
| state (no lidar) | **~1.7 M** (N=16384) | ~2.1 k |
| lidar-24 | **~64 k** (N=1024–16384) | ~470 |

### numpy vs cupy — same code, import swap only (`scripts/compare_numpy_cupy.py`)
| obs | N | numpy (CPU) | cupy (GPU) | speedup |
|---|---|---|---|---|
| state | 256 | 87,060 | 30,821 | 0.35× (CPU wins) |
| state | 4096 | 72,200 | 479,141 | **6.6×** |
| lidar24 | 256 | 2,676 | 24,815 | 9.3× |
| lidar24 | 4096 | 1,005 | 62,601 | **62.3×** |

**Reading:** clear GPU crossover — at small N the GPU's per-kernel launch overhead
loses to CPU numpy; the GPU pulls away with N, and for the compute-heavy lidar obs it
wins at every N (up to 62×). Argues for large N + lidar to exploit the accelerator.

**Concurrency (measured):** 1 job 57 s vs 5 concurrent jobs 195 s → only **~1.5×**
throughput (the big-batch recipe already uses the GPU well). Plan sweeps at ~1.5×, not
÷5. Remaining throughput lever: a fused CuPy `RawKernel` lidar march.

**Trainer note:** the hand-rolled `batched/td3.py` (all-on-GPU, DLPack) is the
throughput demo; the **algorithm ablation uses SB3** via a `VecEnv` adapter
(`batched/sb3_adapter.py`) so implementations are vetted.

---

## 2. Reward conditioning — a real Ch.5 finding

TD3 with the **`dense` reward diverges** from scratch: its penalties are *unbounded
quadratics* (`e_y²+e_ψ²`), returns run to the −1000s and the value function can't fit
(observed return −107 → −447 and worse). The **`multiplicative` reward is bounded**
(~[0, 4.5]/step) and the identical trainer converges cleanly (**−43 → +1544**, stable).
Multiplicative is also the deployed reward (§7). **Takeaway for the thesis:** reward
*conditioning* (bounded vs unbounded), not just shaping, governs trainability — and it
interacts with the algorithm, so reward is selected jointly with the algorithm (S1 of
the study design).

---

## 3. Off-policy batch size & update ratio — for the systems/methods discussion

- Naive large N is *sample-inefficient* for off-policy TD3: at matched UTD≈1 with a
  small (256) minibatch, **N=16 reached ~945 return by 96k steps while N=256 reached
  only ~4 by 102k**. Large batches make replay samples correlated.
- **Fix = scale the update minibatch, not just N.** With **N=256, minibatch 4096,
  lr 2e-3** the large-N run converges *faster in wall-clock* than N=16 (higher return
  in ~70 s vs ~243 s) — a batch-4096 gradient step costs ~the same as batch-256 on the
  GPU but does ~16× the work. Recipe: big env-batch + big update-minibatch + higher LR.
- The bottleneck in off-policy RL is the gradient updates, not env stepping — so the
  GPU env's headline value for *off-policy* is (a) many concurrent jobs and (b)
  enabling on-policy PPO (below). For *on-policy* PPO the fast env pays off directly.

---

## 4. Algorithm comparison — preliminary (full study = ABLATION_STUDY_DESIGN.md)

Head-to-head, truck fwd, lidar-24, multiplicative, fixed speed, gentle paths, 500k
steps, N=256 (3 jobs concurrent → ~1.5× contention-inflated):

| Algorithm | Wall-clock | Completion |
|---|---|---|
| **SB3 PPO** | **34 s** | 1.00 |
| custom TD3 (all-GPU, solo) | 127 s | 1.00 |
| SB3 TD3 | 220 s | 1.00 |
| SB3 SAC | 343 s | 1.00 |

Discrete (2×2 completed): **DQN 1.00** (50 s @120k); **PPO-discrete 1.00** at 2M steps.

Key points for the write-up:
- **PPO is ~6–10× faster** because it is on-policy (no replay buffer, big rollout
  batches) — it *actually exploits* the GPU env. Validates "use on-policy for N≫100."
- **On the easy Stage-1-forward task every algorithm saturates at completion 1.00** →
  final completion does not discriminate. **Rank on sample-efficiency
  (steps-to-threshold), reverse, and the hard curriculum stages.**
- **Per-algorithm learning rate is essential:** PPO needs **1e-3** (collapses at 2e-3,
  too slow at 3e-4); off-policy TD3/SAC/DQN use **2e-3** with the big minibatch. lr is
  a per-algorithm nuisance parameter (tune once each). Baked into `scripts/train_sb3.py`.
- **Discrete vs continuous, early result:** PPO-discrete *completes* (1.00) but at much
  lower return (118 vs ~700) — the coarse 7-bin steering tracks less precisely (higher
  CTE). Discrete trades control precision for completion; report trailer CTE alongside
  completion to capture this.

Algorithm 2×2 = {PPO (on-cont), TD3 (off-cont), PPO-disc (on-disc), DQN (off-disc)};
SAC is a reverse-only secondary study (max-entropy exploration on the unstable case).
Discrete action space = 7 evenly-spaced steer bins `[-25,-16.7,-8.3,0,8.3,16.7,25]°` ×
{go, stop}; stop reproduces the continuous stop-signal (terminate + −penalty).

---

## 5. Metric pitfalls discovered — report correctly

Three artifacts produced a false ~56% completion "ceiling" before they were fixed;
document so the manuscript's numbers are trustworthy:
1. **Auto-reset clears `success` before eval reads it** (same class as the e2e_rl
   B1/B2 benchmark bug). Fix: capture success in `info` *before* auto-reset.
2. **Variable speed × episode horizon:** low sampled speeds can't reach the goal within
   `max_episode_steps`, so they truncate and count as incomplete. The SB3 no-obstacle
   ablation used **fixed speed** — match that; scale the horizon to world/speed.
3. **AgileX-specific tight path kinds** (`lab_corner`) in the full-size mix are near the
   turning limit → unsolvable → cap completion. Use full-size-appropriate kinds.

With fixed speed + gentle (Stage-1) paths, the validated recipe reaches **100%
completion**: truck fwd lidar-24 multiplicative, N=256, minibatch 4096, lr 2e-3, 500k
steps → completion 1.00 in **127 s** (reproduces the SB3 baseline).

---

## 6. Vehicle model & scale

- **DONE — real shunt-truck params applied.** `shunt_truck_config()` (now the default
  `--preset shunt`) uses the **Kalmar Ottawa T2 diesel bobtail**: m=6577, Iz=16000,
  Cf=190000, Cr=200000, lf=1.33, lr=1.62, wheelbase 2.95 m, max_steer 0.87 rad (50°),
  Cd=0.8, A=7.0, + a **kinematic 53′ trailer (L=12.5 m)**. Primary-sourced mass/
  wheelbase/steer; estimated Iz/Cf/Cr (flag as estimates — see
  `shunt_truck_parameters_research.md`). Validated: trains to completion 1.00.
  The old Tesla Model S set (m=1500 …) was an ungrounded placeholder, now retired for
  the ablation (still present as `truck_config` for reference/parity).
- **Model mapping:** dynamic params = the BOBTAIL TRACTOR body; the trailer is a
  kinematic hitch link (no trailer mass / load transfer). So "empty vs loaded trailer"
  is an *optional secondary robustness variant* (`shunt_truck_config(loaded=True)`
  folds the fifth-wheel load into the tractor dynamics: m=36740, Iz=93000, rear-biased
  lf/lr, higher Cr) — flag the kinematic-trailer limitation when using it.
- **No-trailer (tractor-only) is NOT an ablation axis** — it's the rigid-bicycle case
  (solved), carrying none of the articulation challenge. Keep at most as a single
  reference comparison (trailer vs no-trailer at the best config) to quantify the
  articulation penalty; otherwise a Ch.6 deployment convenience.
- The **deployed Autoware simulator is kinematic** (`DELAY_STEER_ACC_GEARED`: kinematic
  bicycle + steering/accel first-order lag — no mass/tyre); low-speed articulated motion
  is kinematic-dominated. Useful justification: the sim's dynamic terms are second-order
  at operating speed and the deployed model is effectively kinematic + actuator lag.
- **Scale decision:** the whole thesis (Ch.3–5) uses **full-size** params; **AgileX
  appears only in Ch.6 (sim-to-real).** The scalar `tractor_trailer_rl` env and the
  deployed models stay as-is; the CuPy env is the one changed for the ablation.
- Real geometries in-repo: AgileX `electrans_robot` wheelbase **0.65 m**; the only
  large vehicle is `orangebus` (2.82 m, a bus). **No shunt-truck param set exists** —
  source real terminal-tractor specs (≈4 m wheelbase, ≈12 m trailer).

---

## 7. Deployed / best-known config (reference)

`Electrans_project/lab_models/models/{forward,reverse}/lidar_24/multiplicative/` →
the deployed best config is **`lidar_24` observation + `multiplicative` reward, both
directions.** Used as the Stage-1 baseline while ablating other axes.

---

## 8. Where each finding lands in the thesis

- Ch.3/4 (modelling/framework): fix the plant-param description (§6); reward
  conditioning motivation (§2).
- Ch.5 (results/discussion): the GPU env as a systems/throughput contribution (§1);
  the algorithm 2×2 + reward + obs ablation with sample-efficiency + CIs
  (ABLATION_STUDY_DESIGN.md); discrete-vs-continuous precision trade-off (§4);
  the metric-pitfall note as an experimental-rigor paragraph (§5).
- Ch.6 (sim-to-real): AgileX scale only; deployment kinematic-model justification (§6).
- Methodology: tuning_playbook scientific/nuisance/fixed framing; screen-then-multiseed;
  per-algorithm lr as a nuisance parameter (§4).
- Still TODO before final numbers: real shunt-truck params (§6); steps-to-threshold
  logging; run the staged sweep; then regenerate tables/figures with 3-seed CIs.

---

## 9. Path scale & eval metrics (verified + built)

**Path scale — verified accurate for the shunt truck.** The stock `sharp`/`winding`
kinds are small-vehicle geometries (~2.5 m radius) — impossible for a 12.5 m trailer
(would jackknife; a 53′ trailer's real min radius is ~13 m). Fix: dropped
`winding`/`lab_*`, added `PathConfig.bend_scale` (scales bend width → radius ∝ width²
with bounded excursion), set `bend_scale=4.5` in `shunt_truck_config`. Result:
- Stage-1 (straight+gentle): min radius **21 m** (all trackable).
- Curriculum (straight/gentle/sharp): median **32 m**, min **14.7 m**, p10 **16 m**.
- sharp alone: median **18 m**, min **12 m** (hardest realistic yard turn).
Speed **5 m/s = 18 km/h** is a realistic yard speed (245 steps to goal < 1000 budget).

**Eval now logs the full thesis metric set** (`scripts/train_sb3.py`): completion,
**trailer CTE**, **max |hitch|**, **jackknife rate**, ep_return — read from the obs
vector during a deterministic rollout — plus a periodic-eval **learning curve** (CSV)
giving **steps-to-90%-completion**. Evidence it's needed: on Stage-1 completion
saturates (~240k steps) but **trailer CTE keeps falling (0.64→0.41 m) afterwards** —
so final CTE/hitch discriminate configs even when completion ties; steps-to-threshold
is a secondary efficiency dimension (useful mainly for the algorithm comparison).

---

## 10. Reverse feasibility + pure-pursuit oracle (Stage-1 diagnosis)

The 1-seed→3-seed Stage-1 screen (shunt truck, gentle paths, 1.5M steps) showed reverse
capping ~0.6 mean. Per-seed data revealed this is **training instability, not path
difficulty**: TD3+multiplicative reverse got **seed0=1.00 (hitch 0.16), seeds1,2=0.00
(hitch 0.95, jackknifed)**; td3+no_hitch 0.93/0.91/0.00. So gentle reverse IS solvable
to 1.0 but training collapses into a jackknife basin on most seeds. **PPO fails reverse
on every seed** (on-policy can't stabilise the non-minimum-phase hitch) — a genuine
algorithm result. Likely contributors to the collapse: no anti-jackknife/recovery aid
(deferred to curriculum) and possibly the over-long 1.5M budget over-training TD3.

**Pure-pursuit oracle (feasibility check + guided-reward reference).** A batched PP
controller (`batched/pure_pursuit.py`, a fixed proportional law on the obs errors)
confirms path feasibility: **forward 1.00** (CTE 0.31 m), **reverse 0.995** (CTE 0.97 m,
hitch 0.14). The e2e_rl reverse gains destabilise the shunt truck (jackknife) — the
**hitch-stabilisation gain sign flips at this geometry** (k_hitch +0.8 → −2.59); re-tuned
gains recover ~1.0. So the reverse paths are fine; reverse needs training stabilisation
(curriculum / shorter TD3 budget / more seeds), which is the thesis's reverse-instability
story.

**Sweep fairness fixes identified:** `guided` reward is not yet ported to the batched
composer (24 crash-isolated failures — needs the PP reference above); per-algorithm step
budgets are needed (TD3/DQN ~400–500k, PPO ~2M) so the fixed-budget comparison doesn't
under-train PPO. `n_envs`=256 is the env-parallelism (separate from the 4096 gradient
minibatch); lidar throughput plateaus by N~1024, so 256–1024 is the useful range (PPO
benefits from the top end, off-policy does not).

---

## 11. Classical controllers (batched) + guided-teacher decision

Batched PP (`batched/pure_pursuit.py`, stateless) and PID (`batched/pid.py`, per-env
integral reset on episode boundary) ported from e2e_rl and re-tuned for the shunt
truck (e2e_rl gains give 0.00 — reverse hitch-gain sign flips at this geometry).
Shunt-truck comparison (gentle paths, best over a random-search tune):

| controller | forward compl / CTE | reverse compl / CTE |
|---|---|---|
| PP  | 1.00 / **0.310** | **1.00** / 0.970 |
| PID | 1.00 / 0.323 | 0.91 / 1.012 |

**PP >= PID** (forward tied, PP tighter CTE; reverse PP wins 1.00 vs 0.91) and PP is
stateless — so **PP stays the guided-reward teacher** (no switch to PID needed). Both
serve as tuned classical baselines for the RL-vs-classical benchmark (Stage 6).

**Controller eval harness** (`scripts/eval_controllers.py`): runs *any* controller
(incl. stateful) in the batched env with done-masked resets — this is the mechanism
to evaluate/tune an **unbatched MPC in the same env for a fair comparison** (run at
N=1 / small N via an `MPCController.predict` wrapper around the e2e_rl OSQP MPC; the
remaining work is exposing the local reference trajectory to it — MPC itself is not
batched, by design).

**MPC (unbatched) in the batched env** (`batched/mpc_adapter.py`): the e2e_rl OSQP MPC
(standalone) driven per-env inside the batched env via `MPCController.predict` (small N;
one QP solve/env/step), with a copied `generate_trajectory` and the vehicle geometry/
limits set to the shunt truck. The **mechanism works** (runs, solves, ~13-15 s for
N=16x500). Out-of-box completion is poor (forward 0.33, reverse 0.00) because the MPC's
Q/R/P weights are tuned for the old vehicle — it needs shunt re-tuning (same story as
PP/PID), which this harness now enables. So the classical-baseline set for Stage-6 is:
PP (tuned, best), PID (tuned), MPC (harness ready, tuning pending).

---

## 12. SESSION STATE / RESUME GUIDE (read this to continue)

Code lives in **`/home/ben/Ben/Thesis/tractor_trailer_rl_cupy/`** (the CuPy port).

### Ablation vehicle & task (all decided)
- `shunt_truck_config()` = **Kalmar Ottawa T2 bobtail** (m=6577, Iz=16000, Cf=190000,
  Cr=200000, lf=1.33, lr=1.62 → wheelbase 2.95, Cd=0.8, A=7.0) + **kinematic 53' trailer
  (L=12.5 m)**, **max_steer 0.87 rad (50°)**, `bend_scale=4.5`, paths {straight, gentle,
  sharp} (winding/lab_* dropped). Iz/Cf/Cr are estimates (flag in thesis); rest primary.
- Stage-1 task: fixed speed 5 m/s, gentle paths (`--fixed_speed --mild_paths`), obs=lidar24,
  spawn on-path aligned (perturbation = later curriculum, OFF).

### Study design (definitive Stage-1)
- **Algorithm 2×2:** ppo, td3, ppo_disc, dqn (SB3, via `batched/sb3_adapter.py`). SAC = reverse-only secondary.
- **Reward (4):** fwd {dense, tractor_focus, multiplicative, guided}; rev {dense, no_hitch, multiplicative, guided}. ALL implemented in `batched/reward.py`.
- **guided → multiplicative** (decided): PP-imitation warmup, α decays 1→0 over 100k env-steps; guide_rew = progress − 5·steer_diff² (rev: −jackknife_pen); teacher = **PP** (PP≥PID, and stateless). Wired in `env.py` (PP reference from `pure_pursuit.FWD/REV_GAINS`).
- **Best-checkpoint** (decided, done): `EvalCurveCallback` tracks best by (completion↓, CTE↑), saves best model; `train_sb3`/sweep report best + final; **summary uses best_**. Compare each run's BEST, not final.
- **Per-algo learning rate:** PPO 1e-3, off-policy 2e-3 (in `train_sb3.build_model` + sweep).
- **Per-algo STEP BUDGET (decided, NOT YET WIRED):** TD3/DQN/SAC ~500k, PPO(+disc) ~2M (PPO under-trains at fixed budget; discrete PPO needs ~2M). TODO: in `run_stage1_sweep.train_one` set `steps = args.ppo_steps if sb3_algo=='ppo' else args.steps`, add `--ppo_steps` (default 2_000_000), `--steps` default 500_000, and compute `eval_every` from the run's steps. Currently uses a single `--steps`.
- Recipe: n_envs 256, batch 4096. Resumable + crash-isolated (skips done run_ids from manifest).

### Launch (definitive, after wiring per-algo budgets)
`TTRL_BACKEND=cupy python scripts/run_stage1_sweep.py --preset shunt --fixed_speed --mild_paths --algos ppo td3 ppo_disc dqn --obs lidar24 --directions forward reverse --seeds 0 1 2 --steps 500000 --ppo_steps 2000000 --n_envs 256 --outdir results_stage1_v2`
Then **Stage-2 obs ablation** at the WINNING (algo,reward): `--algos <win> --rewards <win> --obs state lidar4 lidar8 lidar16 lidar24 lidar32 --outdir results_stage2_obs`.

### Classical controllers (Stage-6 baselines) — `scripts/eval_controllers.py` harness (numpy/CPU)
- **PP** `batched/pure_pursuit.py` (stateless): tuned shunt gains; **reverse k_hitch sign FLIPS** (+0.8→−2.591). fwd 1.00/cte0.31, rev 1.00/cte0.97. = guided teacher + baseline.
- **PID** `batched/pid.py` (per-env integral, reset on done): tuned shunt gains (e2e_rl gains give 0.00). fwd 1.00/cte0.32, rev 0.91.
- **MPC** `batched/mpc_adapter.py` (unbatched OSQP per env, small N; copied `generate_trajectory`; geometry/limits set to shunt): mechanism works; DEFAULT gains poor on shunt (fwd 0.33/rev 0.00). **TUNING IN PROGRESS** (Q/R/P search on CPU; suspected fix: P=5e5 rate penalty too heavy). Formulation = condensed **LTV kinematic path-tracking MPC** + curvature FF + hard input/rate constraints + OSQP → defensible industry-standard baseline (caveats: LTV not NMPC; kinematic — matches deployment; fixed weights). MPC is CPU → can tune in parallel with the GPU ablation.

### Reverse finding (important for the thesis)
Reverse is NOT too hard: PP completes 0.995, TD3 seed0 hit 1.0. The ~0.6 mean was **training instability** — jackknife collapse on 2/3 seeds (final metric hid the peak; best-checkpoint recovers it). **PPO fails reverse entirely** (on-policy, all seeds jackknife) = real algorithm result. **Guided reward fixes reverse**: 0.91–0.96 @120k, clean hitch (PP warmup escapes the jackknife basin).

### Preliminary results (results_stage1/, DONE — old code: final-metric, no guided, uniform 1.5M budget)
- Recompute best from curves.csv (best≈final for td3+mult). Forward: state **0.67** (insufficient!), any lidar 1.00 (cte~0.32). Reverse (obs): lidar16 1.00, fewer/state ~0.66 → lidar necessary, more beams help reverse. Algo (Study A): TD3 wins forward (1.0, tight CTE); PPO fails reverse; reverse high-variance.

### File map (tractor_trailer_rl_cupy)
- `src/tractor_trailer_rl/batched/`: backend, expm (ZOH), vehicle, geometry, corridor (analytic lane), lidar, reward (all modes incl guided), env (BatchedLaneFollowingEnv), td3 (custom GPU TD3=throughput demo), sb3_adapter, pure_pursuit, pid, mpc_adapter.
- `scripts/`: train_sb3.py (SB3 trainer: best-checkpoint + curve CSV + per-algo lr), run_stage1_sweep.py (sweep: resumable, best-checkpoint summary, curves/summary/manifest), run_overnight.sh (orchestrator), plot_stage1.py (figures from CSVs), compare_numpy_cupy.py, benchmark_batched.py, eval_controllers.py (classical eval/tune), run_ablations_gpu.py (has cell_to_cfg + custom-TD3 runner).
- `README_CUPY.md` (full port detail). `tests/test_batched_parity.py` (11 pass).
- Outputs: `results_stage1/{A_algo_reward,B_obs}/` (preliminary). Definitive → `results_stage1_v2/`.

### Still open (not started / blocked)
- **summarize_from_curves.py** utility (recompute best-checkpoint summary from any curves.csv) — not built.
- **B1/B2** e2e_rl completion-metric fix (benchmark all-zero + BEV under-report) — for results that stay on e2e_rl (BEV, since batched has no renderer).
- **D1/D2** stop-gate obstacle env (obstacle rasterization into corridor + difficulty-gated stop reward) — not started.
- **BEV** obs ablation — needs the pygame e2e_rl env or a batched renderer (batched env is state+lidar only).
- **Thesis LaTeX** — NOT to be edited by assistant (user constraint); bibliography (1 cite), Ch6 deployment [TODO]s, stub appendices remain the user's to write.

### MPC tuning outcome (update)
Q/R/P random search (CPU) gave only: forward 0.57→**0.67** (cte worsened), reverse
**0.00** (unchanged — still jackknifes). MPC is NOT yet a fair baseline; needs finer
tuning + a structural look at the reverse variant for the 12.5 m trailer (the reverse
LTV linearization / delta_ff / solve_psi2 may need attention at this geometry). Not
blocking. The **definitive Stage-1 is RUNNING → `results_stage1_v2/`** (per-algo budgets
wired: off-policy 500k, PPO 2M; guided + best-checkpoint + tuned reverse PP).
