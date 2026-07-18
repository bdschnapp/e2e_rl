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

## 4b. Phase-1 algorithm × reward × direction study (results_stage1_v4, lidar-24)

> **SUPERSEDED for Phase-1 of record by V1 (state obs) — see §4c below.** v4 (lidar-24)
> is retained for provenance / the obs-bridge discussion; the canonical Phase-1 ablation
> we carry forward is the **state-obs V1 sweep**. Full record persisted at
> `e2e_rl/thesis/data/phase1_ablation_V1/` (summary.csv, curves.csv, manifest.jsonl,
> classical baselines, champion models). Live working copy:
> `tractor_trailer_rl_cupy/results_stage1_V1/`.

**Study:** from-scratch RL, trailer, **2-D stop-capable action space** (continuous
`[steer, stop]` for TD3/PPO; `Discrete(7×{go,stop})` for DQN/ppo-disc), forward +
reverse, obs = lidar-24, shunt-truck preset, **3 seeds**, best-checkpoint policy.
96/96 runs, 0 errors. Forward off-policy 500k / reverse off-policy 1.5M / PPO 2M steps;
eval every 100k. **This supersedes §4 (preliminary).**

### The headline (mean best-completion over 3 seeds; "best-seed" = max over seeds)

| dir | TD3 (off-cont) | DQN (off-disc) | PPO (on-cont) | ppo-disc (on-disc) |
|---|---|---|---|---|
| forward | 1.00 (mult, tractor_focus); dense/guided fragile | **1.00 all rewards, all seeds** | 0.71 (can hit 1.0, unreliable) | 0.43 (~0.9 max) |
| reverse | **0.99 best-seed** (mult); mean 0.6 (one dead seed/cell) | 0.90 cap; robust (no dead seed) | 0.04 (**fails**) | 0.15 (**fails**) |

### Findings (what actually goes in the chapter)

1. **Forward is an off-policy vs on-policy split — NOT "model choice doesn't matter."**
   Off-policy (TD3, DQN) solve forward reliably (~1.0); on-policy (PPO, ppo-disc) are
   weaker and higher-variance even forward (PPO 0.71, ppo-disc 0.43). **DQN is the most
   robust forward** (1.00 on all 4 rewards, all seeds); TD3 ties at its best rewards.
   *Correction to the earlier "TD3 best forward" phrasing — DQN edges it on robustness.*

2. **Reward shaping is a first-class Phase-1 result (for the continuous algos).**
   `multiplicative` & `tractor_focus` are collapse-proof for TD3 (1.00 ± 0.00, all
   seeds); `dense` & `guided` admit a degenerate sustained-turn basin (max_hitch≈0.8,
   **jackknife=0** — a bad basin, not a jackknife) that some seeds fall into. `guided`
   is worst because its PP-teacher→multiplicative α-decay shifts the objective mid-run.
   DQN's coarse discrete steering can't reach that fine-turn basin → DQN is collapse-free.

3. **The 2-D stop-gate does NOT break the gradient — it is dormant in lane-following.**
   The stop is thresholded IN THE ENV (`stop = a₁>0.75`), so TD3 sees two *continuous*
   actions and the DPG is well-defined. Decisive evidence: **DQN carries the same binary
   stop and never collapses**. The stop never fires here (no stop-terminations). It adds
   mild early-training variance (random pre-threshold stops → −penalty noise), not
   breakage — and this validates that the stop-capable action space is safe to carry
   into the obstacle task unchanged.

4. **Reverse separates the field — and it is a ceiling-vs-reliability trade.**
   On-policy **fails** reverse (PPO 0.04, ppo-disc 0.15). **TD3 = highest ceiling**
   (best-seed **0.99** mult; also guided s2 0.96, mult s2 0.94) but **exactly one
   collapsed seed per cell** (e.g. mult = 0.99/0.94/**0.00**) → mean dragged to ~0.6.
   **DQN = reliable but capped** (~0.90, no dead seed). The low reverse *means* are a
   *training-reliability* story, not a capability ceiling.

5. **RL matches/exceeds classical reverse at best-seed, but not on average.**
   Classical PP oracle reverse = 0.97 (see §11); RL reverse *means* ≈ 0.6 but *best-seed*
   TD3 = 0.99. → the deployment framing (train N seeds, deploy the best) + the motivation
   for the classical baseline.

6. **We carry TD3 + DQN into downstream experiments; drop PPO/ppo-disc (keep in the
   Phase-1 table only).** TD3 — continuous steering needed for smooth obstacle go-around,
   only continuous method competitive both directions, highest ceiling. DQN — robust
   discrete baseline / foil (no seed collapses, strong forward, decent reverse).

### Reverse combos reaching deployable completion (best-checkpoint, any seed)
TD3 mult s0 **0.99**; TD3 guided s2 0.96; TD3 mult s2 0.94; TD3 guided s0 0.94; DQN mult s2 0.90.

### Wall-clock caveat (systems note)
Total ~11.5 h; **DQN reverse alone = 5.2 h** (1.5M steps × 16 grad-steps @ 32 envs).
GPU sat at ~34% — **SB3 off-policy *learning* is serial small-batch** (rollouts are GPU-fast,
the optimizer loop is latency-bound). The batched-env speedup is on the environment, not
SB3's update loop. Lever for future runs: cut reverse budget or DQN `gradient_steps`.

### WHERE THE RESULTS ARE SAVED
`tractor_trailer_rl_cupy/results_stage1_v4/`:
- `manifest.jsonl` — one line/run: full config, vehicle params, wall-clock, `best_*`/`final_*` metrics (96 runs).
- `curves.csv` — per-checkpoint metric vs timesteps (learning curves / figures).
- `summary.csv` — one row per (algo,dir,obs,reward): mean ± 95% CI over seeds, from best-checkpoint (32 groups).

### HOW TO GENERATE THE PAPER TABLES/FIGURES (do NOT run yet — pending user go-ahead)
From `tractor_trailer_rl_cupy/`:
```bash
# LaTeX tables (reward ablation + algorithm comparison, fwd/rev, mean ± 95% CI) + curve figures
python scripts/export_thesis_results.py --outdir results_stage1_v4
#   -> results_stage1_v4/thesis_export/{reward_ablation_{forward,reverse}.tex,
#      algorithm_comparison_{forward,reverse}.tex, *.pdf}  + an index with \input lines.
#   Optional: --algo td3 / --reward multiplicative to fix the pivot; --thesis_tables <dir> to also copy .tex in.

# Learning-curve plots straight from the CSVs
python scripts/plot_stage1.py --outdir results_stage1_v4
```
Copy into the thesis manually (never auto-edit the .tex): `cp results_stage1_v4/thesis_export/*.tex <thesis>/tables/`.

**TODO before final tables:** the exporter currently emits **mean ± CI only**. Per the
agreed format, add a **"best-seed" column** (max over seeds, labelled "deployment-selected")
and a **robustness column** (e.g. # seeds ≥ 0.9) so the ceiling-vs-reliability trade
(finding 4) and the deployment framing (finding 5) are visible in the table itself.
`export_thesis_results.py::agg` already has per-seed values in scope — add `max(vals)` +
a threshold count there.

---

## 4c. Phase-1 DEFINITIVE study — STATE obs, V1 (results_stage1_V1) — the Ch.5 story of record

**This is the canonical Phase-1 ablation carried forward.** Full record persisted at
`e2e_rl/thesis/data/phase1_ablation_V1/` (with its own README); live working copy at
`tractor_trailer_rl_cupy/results_stage1_V1/`.

**Study:** from-scratch RL, trailer, **state obs** (8-dim), 4 algos (td3/dqn/ppo/ppo_disc)
× 4 rewards/dir × 2 directions × **3 seeds** = 96 runs (96/96, 0 failures). Fixed 5 m/s,
train-random / **eval on safe pool** (`shunt_mild_safe.npz`) / **eval-no-stop** (fair vs
classical), guided = **pure PP clone** (`guide_transition=-1`). Off-policy fwd 500k / rev
1.3M / PPO 2M; eval every 100k; best-checkpoint policy (completion desc, CTE asc).

### Reverse — the decisive task (mean best-completion ± 95% CI, 3 seeds)
| algo | reward | completion | CTE |
|---|---|---|---|
| **TD3** | **multiplicative** | **0.969 ± 0.011** | 0.90 |
| TD3 | guided | 0.875 ± 0.060 | 1.19 |
| TD3 | no_hitch | 0.830 ± 0.144 | 0.97 |
| TD3 | dense | 0.810 ± 0.058 | 1.11 |
| PPO | guided | 0.814 ± 0.283 | 1.34 |
| ppo_disc | guided | 0.536 ± 0.081 | 1.62 |
| DQN | guided | 0.464 ± 0.557 | 1.35 |
| PPO | multiplicative | 0.434 ± 0.010 | 1.44 |
| ppo_disc | dense | 0.423 ± 0.024 | 1.03 |
| ppo_disc | no_hitch | 0.289 ± 0.283 | 1.19 |
| DQN | dense | 0.147 ± 0.286 | 1.54 |
| ppo_disc | multiplicative | 0.145 ± 0.284 | 0.89 |
| PPO | dense | 0.143 ± 0.278 | 1.64 |
| PPO | no_hitch | 0.135 ± 0.263 | 1.44 |
| DQN | no_hitch | 0.020 ± 0.038 | 1.28 |
| DQN | multiplicative | 0.0005 ± 0.001 | 1.12 |

### Forward — saturated for continuous, reward-sensitive for discrete (mean, 3 seeds)
| algo | dense | tractor_focus | multiplicative | guided |
|---|---|---|---|---|
| TD3 | 1.00 | 1.00 | 1.00 | 1.00 |
| PPO | 1.00 | 1.00 | 1.00 | 1.00 |
| DQN | 1.00 | 1.00 | 0.82 | 0.57 |
| ppo_disc | 0.81 | 0.89 | 1.00 | 0.42 |

### Findings (what goes in Ch.5)
1. **Champion = `(TD3, multiplicative)` reverse 0.969 ± 0.011** — tight CI, no dead seeds,
   ~3% under PP oracle (1.0 on safe pool). The (algo, reward) carried into Phase-2.
2. **TD3 is the only algorithm that solves reverse** (all TD3 rewards ≥ 0.81; best non-TD3
   = PPO+guided 0.814 but ±0.283 = one lucky seed).
3. **Reward winner is algorithm-conditional:** TD3 → multiplicative ≫ guided; PPO (on-policy)
   → the ordering *inverts*, guided is the only thing that works (mult collapses to 0.434);
   discrete → guided best-but-mediocre, **multiplicative catastrophic for DQN (0.0005)**.
4. **Guided fails both discrete algos in both directions** — reward-space behavioural cloning
   only works when the action space can represent the continuous PP teacher (clean negative
   control; strengthens guided-as-continuous-clone framing).
5. **Forward saturated for continuous control** (TD3 & PPO = 1.000 under every reward),
   consistent with the classical baselines (all four hit 100% forward).

### Classical yardstick (same safe pool, eval-no-stop equivalent)
Fwd: all four = 1.000. Rev: PP 1.000, MPC 0.990, LQR 0.967 (best CTE 0.568), PID 0.898.
→ RL champion (TD3 mult, 0.969) sits between LQR and MPC on completion; classical leads on
average, RL matches at best-seed. See `classical_baselines_safepool.log`.

### Generate the paper tables/figures (do NOT run until user go-ahead)
```bash
cd tractor_trailer_rl_cupy
python scripts/export_thesis_results.py --outdir results_stage1_V1   # now has best-seed + robustness cols
python scripts/plot_stage1.py --outdir results_stage1_V1
```

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

## Track D — Obstacle avoidance + binary stop-gate (batched, BUILT)

**Status: env built & validated (D1/D2 ⟨CODE ✓⟩); training/eval/write-up (D3/D4) pending.**
Ch.5 §Obstacle Navigation / Ch.6 §Simulation.

**Isolation (important):** the obstacle work does NOT touch the lane-following ablation.
It is a *subclass* (`BatchedObstacleAvoidanceEnv(BatchedLaneFollowingEnv)`) selected only
when `Config.obstacle` is set (default `None` → the ablation env is byte-for-byte the same;
**all 11 parity tests still pass** after the refactor). New files only:
`batched/obstacles.py`, `batched/obstacle_env.py`, preset `shunt_truck_obstacle_config`,
`scripts/smoke_obstacle.py`. The base `env.py` got two behaviour-preserving hooks
(`_lidar`, `_compute_reward`) — the pre-refactor inline code is the base implementation.
`sb3_adapter.make_batched_env` is the factory (obstacle env iff `cfg.obstacle`).

**The "hidden local planner" reward (the mechanism that worked well forward in e2e_rl).**
At reset a batched artificial-potential-field planner (`obstacles.plan_offsets`) produces an
obstacle-avoiding *reference path* = centreline + smoothed lateral shift (push away from any
intruding obstacle, clipped to the corridor). The **reward tracks that path**, but the agent
**never observes it** — its only obstacle signal is lidar. So the policy must rediscover the
avoidance manoeuvre from lidar to collect the tracking reward. Observation errors stay vs the
*centreline* (base `_observe`); reward errors are recomputed vs `local_ys` in `_compute_reward`.

**Difficulty metric** (`obstacles.difficulty`, batched port of `compute_path_difficulty`): at each
obstacle station, scan the lane laterally, take the widest contiguous free gap, and map
`difficulty = clip(1 − (gap − veh_w)/(lane_w − veh_w), 0, 1)` (0 = lane clear, 1 = gap ≤ vehicle
width = impassable). Scalar = max over stations. Surfaced per-episode in `info["difficulty"]`
(pre-auto-reset snapshot) for **difficulty-stratified evaluation**.

**Layout categories** (one event cluster/episode, controlled + stratifiable mix, `layout_probs`):
`clear` (edge obstacle → drive straight), `shift` (mid-lane → go around), `squeeze` (opposing
pair, passable central gap), `blocked` (big central obstacle → impassable). Default 0.30/0.25/0.20/0.25.

**Stop-gate reward (the new stopping mechanism, difficulty-keyed).** Uses the existing
`STOP_SIGNAL` action. On stop: `exp = exp_scale(difficulty,k=3)`, `stop_reward =
exp·stop_hard_reward − (1−exp)·stop_easy_penalty + progress·stop_progress_bonus`
(defaults 60 / 60 / 20). **Keyed on difficulty, not progress** → position-robust (unlike e2e_rl's
progress×MAX_FAIL×exp form, which makes an early-obstacle stop net-negative even when correct).
Ordering it enforces: hard-stop **+**, easy-stop **−**, and a deliberate stop beats a crash
(−100) on impassable layouts while success (+100) still beats a stop on passable ones.
Plus a slow-down-on-difficulty running bonus (`difficulty·(1−norm_speed)·0.1`).

**Obstacles reach the env analytically** (no occupancy grid, matching corridor.py):
`obstacles.lidar` = ray-circle ranges min-combined with the corridor raycast; `obstacles.collision`
= body-point-in-circle; `obstacles.proximity` = quadratic ramp on obstacle clearance. All batched,
GPU-verified (numpy + cupy).

**Smoke validation** (`scripts/smoke_obstacle.py`, numpy + cupy): base env regression OK;
difficulty spread easy 0.25 / med 0.35 / hard 0.40; stop reward easy −49.8 / med −36.8 / **hard
+39.2**; never-stop drive-through succeeds on easy (46 succ / 1 crash) but crashes on hard (21 /
93) → stopping is the correct policy exactly where difficulty is high.

**Classical obstacle baseline — the RL-vs-traditional-planning bar** (`scripts/eval_obstacle.py`,
difficulty-bucketed, Wilson CIs; the honest comparison for pure RL). Three arms:
`--baseline centerline_pp` (obstacle-blind PP on the raw centreline), `--baseline local_pp`
(the **traditional stack**: global path -> local APF planner -> PP tracking the *planned* path,
with a planner-feasibility stop when difficulty>0.95), and `--model <rl.zip>`. Forward baseline
(complete / crash / stop %):

| difficulty | centreline PP | traditional stack (planner+PP) |
|---|---|---|
| 0.2-0.4 | 97.7 / 0.3 / 0.3 | 94.8 / 0.7 / 0.5 |
| 0.4-0.6 | 25.2 / 74.0 / 0.3 | 84.9 / 10.0 / 0.5 |
| 0.6-0.8 | 1.5 / 1.5 / 98.5 | 88.8 / 7.4 / 3.6 |
| 0.8-1.0 | 0.1 / 0.1 / 99.9 | 23.3 / 20.9 / 55.7 |

The local planner recovers the middle (medium-obstacle completion 25%->85%; 0.6-0.8 1.5%->89%). The
residual: even the full stack only completes 23% (21% crash) on the hardest *passable* squeezes --
PP can't thread a tight gap with the 12.5 m trailer, and the feasibility stop only catches the truly
blocked ones. That hard-squeeze bucket is where a learned policy can beat the classical stack. NOTE
this is the PP-controller arm; the RL policy sees only the centreline obs + lidar (never the planned
path). Refinements pending: difficulty-targeted eval for even bucket coverage (0-0.2 underpopulated);
mid-bucket timeouts ~4-5% (PP wanders on the sharp planned detour at fixed speed).

**Still TODO (D3/D4):** train on `shunt_truck_obstacle_config` (needs the GPU free from the v3
sweep); evaluate stratified by difficulty (go-around vs stop success + min-clearance); write the
chapter. Multi-obstacle-per-episode and BEV-obstacle variants are possible extensions.

---

## Stage-2 BEV — vectorized top-down render (no pygame), BUILT & validated

The e2e_rl BEV is a per-env pygame raster (a CPU bottleneck that doesn't
parallelise); the batched env therefore shipped state+lidar only. **A top-down
occupancy image is just a membership query at a grid of sample points** — "is this
point inside the lane corridor? on an obstacle? in-world?" — and those primitives
are already analytic (`corridor.min_dist_to_centerline`, obstacle circle tests). So
BEV is one big vectorized gather, the same idea as the lidar march but on an S×S
grid instead of along rays. Runs entirely on the active backend (numpy **and** cupy,
verified identical: free-fraction 0.25 both).

- **`batched/bev.py::render_bev`** — lays an S×S grid in the VEHICLE frame (forward =
  image-up, lateral = cols), rotates+translates it to world coords by the anchor pose,
  evaluates membership at all N·S·S points at once → **(N,S,S) float32**, 1.0=free /
  0.0=blocked. Anchor = the same reverse-aware `_lidar_pose()`, so the image faces the
  direction of travel. Cost ≈ S·S pts/env (1024 for 32×32) ≈ a single lidar frame.
  Verified: corridor renders as the expected stripe (~9 cells ≈ 2·corridor_half);
  behind the start reads blocked (out-of-world); an obstacle carves a clean circular
  hole at the right range/size (12 cells for r=2.5 m at 10 m ahead).
- **Dict obs, additive & off-by-default.** `ObsConfig.bev_size` (0 = disabled →
  vector obs unchanged; **11/11 parity still pass**). When >0 the env obs becomes
  `Dict{vector, image=(1,S,S)}`; `sb3_adapter` handles the Dict (incl. per-env
  `terminal_observation` on auto-reset), and `train_sb3.build_model` auto-switches to
  **MultiInputPolicy + `BevCombinedExtractor`** — a small 3×3/stride-2 CNN (NatureCNN's
  8×8/s4 tower collapses below ~36 px, so we use a 32→16→8→4 stack) → 128 feats
  concatenated with the state vector. Obstacle env overrides `_bev_obstacles()` to draw
  its circles. Cell token: `bev` (or `bevN` for S=N) in `cell_to_cfg`.
- **Validated end-to-end on CPU**: SB3 PPO(MultiInputPolicy)+CNN learns through the
  Dict obs (rollout buffer → CNN forward/backward → update) on the real batched env.
- **Thesis fit**: gives Stage-2 the perception axis `state / lidar-{4..32} / BEV` on
  ONE consistent batched env (the e2e_rl BEV was a separate pygame pipeline) — cleaner
  cross-modality attribution. Matches the thesis's 32×32 BEV (not the stale 84×84 in
  CLAUDE.md — see plan A5).

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

### Launch — DEFINITIVE STAGE-1 COMPLETE → `results_stage1_v4/` (96/96, 0 errors; see §4b)
The definitive Stage-1 study is DONE. Full findings + save location + table-gen instructions in **§4b**.
Command that produced it (per-algo budgets: fwd off-policy 500k / rev 1.5M / PPO 2M; off-policy n_envs=32, PPO n_envs=256):
`TTRL_BACKEND=cupy python scripts/run_stage1_sweep.py --preset shunt --algos td3 ppo dqn ppo_disc --seeds 0 1 2 --outdir results_stage1_v4`
Then **Stage-2 obs ablation** at the WINNING (algo,reward): `--algos <win> --rewards <win> --obs state lidar4 lidar8 lidar16 lidar24 lidar32 bev --outdir results_stage2_obs`. The `bev` cell now runs natively on the batched GPU env (vectorized 32x32 render, no pygame — see §Stage-2 BEV below); it auto-switches SB3 to MultiInputPolicy + a small CNN extractor.

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
- Outputs: `results_stage1/{A_algo_reward,B_obs}/` (preliminary, superseded). **DEFINITIVE → `results_stage1_v4/` (DONE, 96/96 — see §4b).**

### Still open (not started / blocked)
- **summarize_from_curves.py** utility (recompute best-checkpoint summary from any curves.csv) — not built.
- **B1/B2** e2e_rl completion-metric fix (benchmark all-zero + BEV under-report) — for results that stay on e2e_rl (BEV, since batched has no renderer).
- **D1/D2** stop-gate obstacle env (obstacle rasterization into corridor + difficulty-gated stop reward) — not started.
- **BEV** obs ablation — ~~needs pygame or a batched renderer~~ **RESOLVED**: vectorized 32x32 BEV render built for the batched GPU env (§Stage-2 BEV); no pygame, runs on numpy+cupy.
- **Thesis LaTeX** — NOT to be edited by assistant (user constraint); bibliography (1 cite), Ch6 deployment [TODO]s, stub appendices remain the user's to write.
- **Failure-replay curriculum** (retry the same failed scenario) — exists in e2e_rl, NOT ported to CuPy, NOT in the V1 sweep. See dedicated section below.

### Failure-replay (hard-case retry) curriculum — GAP: exists in e2e_rl, not ported to CuPy
**Mechanism (as requested by the user):** when an actor fails an episode (stop / crash /
jackknife / OOB *before* the goal), instead of discarding that env and drawing a fresh
random path, it **replays the exact same scenario** (path + spawn) for up to a fixed number
of consecutive attempts (or until it succeeds), then forces a fresh draw. Successes and
timeouts always draw fresh. This over-weights hard layouts the agent would otherwise escape
by getting an easier random reset — most beneficial for open-loop-unstable tasks (reverse).

**Where it already exists (e2e_rl only):**
- `e2e_rl/Environments/wrappers.py::RetryOnFailureWrapper` — gym wrapper, seed-replay,
  `retry_on_failure` / `retry_on_timeout=False` / `max_retries=5`, retries only *feasible*
  failures. Wired through `train.py` + `eval.py` (`--retry_on_failure`), models saved under
  `<reward>_retry/`.
- Was a *planned but unfinished ablation row*: `run_study.py:115-117` (commented) =
  `reverse, lidar_24, multiplicative, retry_on_failure=True`; thesis-plan **B3** references a
  `reverse/lidar_24/multiplicative_retry` model that "exists; only needs eval +
  steps-to-50%-completion metric." **This is the failure-replay results table that was never
  completed.**

**Status in the CuPy repo: ABSENT.** Not implemented in `batched/env.py`; NOT a dimension of
the running V1 sweep (`algo × reward × direction × seed`); not scheduled as any Track-C /
Phase-2 CuPy stage.

**Architecture readiness: HIGH — both hard prerequisites already exist.** Port is small,
additive, default-off, no SB3/adapter change (retry lives inside the batched auto-reset, so
SB3 sees normal step/reset):
- ✅ Per-env scenario identity: `self._env_pool_idx` (N,) tracks each env's pool path
  (`_reset_idxs`, env.py:186). Retry = *don't* redraw it for failed envs.
- ✅ Per-env success at terminal: `succ` captured BEFORE auto-reset (env.py:435);
  `_diag_causes` already derives jackknife/oob/success/timeout. `failed = terminated & ~succ`
  is in scope in the auto-reset block (env.py:429-454) — the single localized insertion point.
- ✅ Feasibility guard trivially satisfied — the safe pool is 100% PP-feasible (no impassable
  layouts to get stuck on); `max_retries` stays as a safety valve.
- ✅ True same-scenario replay holds for the CURRENT config: shunt uses `SpawnConfig` defaults
  (all offsets 0.0) and fixed speed (`pool_lo==pool_hi`), so a scenario is fully determined by
  `_env_pool_idx` alone. ⚠️ CAVEAT: if spawn perturbation or variable speed is later enabled,
  you must ALSO store/replay per-env spawn offsets + speed draw for a true replay.

**What to add (~15-30 lines):** a per-env `_consecutive_retries` int array + `_max_retries`,
and a retry mask in the auto-reset block that gates the random redraw at env.py:186. Zero
parity-test impact (default off).

**How it slots as an ablation:** a training-curriculum knob ORTHOGONAL to (algo, reward, obs)
— a single-flag on/off comparison, best applied only to the hard case (reverse) at the
*winning* (algo, reward) once V1 lands. Metrics: completion lift + **steps-to-threshold**
(the B3 "steps to 50% completion"). MUST be **train-only** — eval stays an unbiased safe-pool
pass with no retry, else the completion number is inflated. Resurrects the unfinished B3 table.

### MPC tuning outcome (superseded — see next section)
Q/R/P random search on the inherited e2e_rl OSQP MPC gave only forward 0.67 / reverse
0.00. That MPC was then **root-caused and replaced** (below).

### Classical controller baselines — RESOLVED (all four work both directions)
Full investigation (2026-07-05) fixed the classical baselines into a defensible suite
spanning the control spectrum: **pure pursuit (geometric) → PID (classical feedback) →
LQR (optimal linear) → MPC (optimal + constraints + preview) → RL (learned)**. Final
completion / trailer-CTE (128 envs, shunt truck, mild paths, spacing 0.25 m):

| Controller | Forward | Reverse |
|---|---|---|
| Pure pursuit | 1.00 / 0.16 | 0.97 / 0.95 |
| PID | 1.00 / 0.18 | 0.87 / 0.93 |
| LQR | 1.00 / 0.30 | 0.96 / 0.66 |
| MPC (tractor-tracking / virtual-trailer) | 1.00 / 0.31 | 0.97 / 0.93 |

**Three root causes found (none were "reverse is a hard control problem" — PP/PID/LQR/MPC
all solve reverse; the reverse *difficulty is in RL training*, cured by guided reward):**

1. **Inherited MPC structure.** e2e_rl's OSQP MPC servos the *trailer pose* with the
   open-loop-unstable reverse hitch dynamics inside the QP — stable for its ~10 m trailer
   (L2/L1≈2.5) but unstable for the shunt truck's 12.5 m trailer (L2/L1≈4.2): it corrects
   the trailer offset then oscillates into the corridor wall (fails *all* curves fwd, and
   reverse). Replaced by `batched/mpc_tractor.py` — a kinematic **error-dynamics MPC**:
   forward tracks the *tractor* errors (tractor leads, trailer follows — MPC analogue of
   pure pursuit); reverse uses the **virtual-trailer / flatness reformulation** (from the
   working Autoware `trailer_ltv_mpc`, `Electrans/Electrans_project`): the QP plans the
   trailer as a *stable* bicycle with a virtual steer δ_T (hitch enters only via the
   inner-loop-closed stable dynamics γ̇=K(δ_T−γ)), and an analytic inner map δ_T→δ_f
   provides the anti-jackknife counter-steer.
2. **Cross-track observation artifact.** `geometry.path_errors` computes e_y as distance
   to the nearest path *point* (not perpendicular-to-segment); with 1 m samples and 0.5 m
   steps this injects a **~0.5 m period-2 ripple** into e_y/e_y_t. Robust controllers
   (PP/PID/RL/forward-MPC) tolerate it; high-gain optimal controllers (LQR, aggressive
   MPC) chase it and go bang-bang. **Fix: `shunt_truck_config` path `spacing_m=0.25`**
   (shrinks the ripple ~4×). Proof: forward LQR 0.00→1.00 with the finer path. This also
   cleans the RL observation → **Stage-1 re-run as `results_stage1_v3/`** (spacing 0.25;
   v2 with 1 m spacing is superseded).
3. **Reverse path-following speed sign.** In reverse the *trailer leads* and progresses
   in +s (toward the goal) at along-path speed |V|, but the error models used the signed
   tractor speed V<0 for the trailer path-following terms → the derived lateral gain came
   out with the wrong sign (vs. the known-good PP reverse gains). Fix: path-following
   kinematics use |V|; only the yaw dynamics (tyaw_rate=(V/L2)sinγ) keep signed V. Also
   the env's tractor yaw response to steering does **not** flip in reverse (`vehicle._ct_matrices`
   uses `|xd|`, `s_eff=−s`), so the inner map / input uses |V|/L1.

**New/added files:** `batched/mpc_tractor.py` (KinematicTractorMPC, fwd tractor-tracking +
rev virtual-trailer cascade, per-env OSQP), `batched/lqr.py` (BatchedLQR, DARE gain,
vectorised — fast). Tuned weights baked in (FWD/REV). `scripts/eval_controllers.py` now
benchmarks PP/PID/LQR/MPC together. Reverse tuned weights: LQR q=[0.05,2.87,22.6] R=83;
MPC K=4.4 q=[0.12,0.73,3.97] R=12.9 Rd=54.3 N=30 (heavy rate penalty avoids ripple-chasing).

**Narrative guidance for the thesis:** do NOT argue "reverse is a hard control problem"
(false — all four classical controllers solve it). The defensible framing: reverse is
hard for *RL training* (non-minimum-phase instability makes naive reward shaping fail;
in the v2 sweep PPO reverse = 0.00 on all rewards, TD3 reverse high-variance) → motivates
the **guided reward**. Classical controllers are baselines the RL policy is measured
against, spanning geometric→feedback→optimal-linear→optimal-constrained.

### Definitive Stage-1 sweep
**RUNNING → `results_stage1_v3/`** (spacing 0.25 m; per-algo budgets off-policy 500k /
PPO 2M; guided + best-checkpoint). Supersedes v2 (1 m spacing). Export via
`scripts/export_thesis_results.py`.

### RL vs classical — tracking-accuracy comparison & framing (grounded in v3 data)
Forward trailer-CTE (m), converged seeds, spacing 0.25 m:

| Controller | Forward CTE | Notes |
|---|---|---|
| Pure pursuit | 0.16 | tightest; near the observation floor |
| PID | 0.18 | |
| **RL (PPO, best rewards)** | **0.19–0.22** | multiplicative/guided ≈ 0.20 |
| LQR | 0.30 | |
| MPC | 0.31 | |

**Grounded finding: on forward tracking, RL (~0.20) BEATS the model-based optimal
controllers (LQR/MPC ~0.30) but sits just behind the best-tuned geometric/feedback
controllers (PP 0.16 / PID 0.18).** RL is *competitive, not dominant*, for three concrete
reasons — important so the thesis does not overclaim:
1. **Observation-noise floor.** Even at 0.25 m spacing the nearest-point cross-track
   ripple is ~0.125 m; PP at 0.16 is already near it, so there is little headroom below
   for anyone. (Validated by the fix: v2 noisy-obs forward RL cte ≈ 0.37 → v3 clean-obs
   ≈ 0.20 — the finer path dropped the measured floor, not the policy quality.)
2. **PP/PID are near-optimal for simple lane-following** — tight tractor tracking is
   exactly what pure pursuit is built for.
3. **RL optimizes a composite reward** (progress × hitch × heading × CTE), not pure CTE,
   so it trades a little tracking for stability/progress; a CTE-only-tuned controller
   wins on CTE by construction.

**Reverse** is where RL has genuine room: classical reverse CTE is loose (LQR 0.66,
PP/PID/MPC ~0.93) because the non-minimum-phase plant is hard for linear/geometric
control. A learned nonlinear, anticipatory policy could beat 0.66 **if it trains
reliably** — the open question, since reverse RL training is the unstable part (v2: PPO
reverse 0.00 all rewards; TD3 reverse high-variance) and is exactly what guided reward +
best-checkpoint address. v3 TD3-reverse numbers pending — the one place a clean RL CTE
*win* may appear.

**Thesis narrative (do NOT claim RL beats optimal control on CTE — reviewers would be
skeptical, and it's not what the data shows):** the defensible claim is that RL is
*competitive with (and on forward, beats the model-based-optimal) classical baselines on
tracking accuracy, while being model-free and perception-driven* — one lidar policy, no
per-vehicle model derivation or path projection, extensible to obstacles/BEV, and
handling the unstable reverse regime that required a bespoke MPC reformulation. The four
classical controllers (PP/PID/LQR/MPC) span geometric → feedback → optimal-linear →
optimal-constrained, so the comparison is well-justified.
