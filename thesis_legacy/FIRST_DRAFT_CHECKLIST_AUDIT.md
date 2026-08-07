# First-Draft Checklist — Audit & Re-Prioritization (2026-07-06)

Full re-verification of `FIRST_DRAFT_CHECKLIST.md` against the *current* state of all
repos (4 parallel read-only audits). **The original checklist is substantially stale:**
its two headline "biggest defects" (A1 empty bibliography, A5 84×84 mismatch) are already
resolved, and several items marked TODO are actually built-and-unrun rather than unbuilt.

The real remaining work splits cleanly into two kinds:
1. **GPU-time experiments** that produce *new numbers* (long lead, gate Ch.5) — start first, run in background.
2. **Writing + regeneration** that runs from existing/placeholder data — fill the gaps while GPU runs.

---

## Corrected status (what's ACTUALLY left)

### Track A — Manuscript
| Item | Original claim | TRUE status | Draft-blocking |
|---|---|---|---|
| A1 bibliography | "only 1 cite / empty bib / .blg error" | **DONE** — 79 entries, 44+ cites, clean build (0 undefined, 0 warnings). Root `main.blg` was a stale pre-restructure artifact; real build in `build/` is clean. | no |
| A2 Ch.6 Deployment Results | replace `[TODO]` | **NOT STARTED** — `06_sim_to_real.tex:183-192` still 6 raw `[TODO]`. 3 findings in `lab_deployment_findings.md` unwritten (only #2 reflected). | **YES** (writing, placeholders OK) |
| A3 stub appendices | fill 4 | **NOT STARTED** — 3 stubs + blank TD3 HP table (`tables/hyperparameters.tex`). | **YES** |
| A3b classical-controllers appendix | new | **NOT STARTED** — impls done (`GPU_ABLATION_FINDINGS.md`), only write-up. Do *with* Ch.5. | no |
| A4 in-chapter TODOs | 4 markers | **PARTIAL** — only `04:98` (architecture diagrams) still literal; `05:159`/`03:184` reworded, gated on final results. | partial |
| A5 BEV 32×32 | "code says 84×84" | **DONE** — thesis uniformly 32×32. Only stale text is in `e2e_rl/CLAUDE.md` (code-doc, trivial). | no |
| A6 future→past tense Ch.5 | rewrite | deferred until Ch.5 results final | no |
| A7 clear `pending_updates.md` | 25 edits | still full; bookkeeping | no |
| A8 admin (readers/seminar/display) | choose readers | **NOT STARTED** — gates *submission* not draft; **start now** (external lead time). | no (but start) |

### Track B — Data integrity / figures
| Item | TRUE status | Note |
|---|---|---|
| B1 benchmark 0% completion | **code FIXED, data STALE** | `metrics.py:201` fixed Jul 5; CSVs are Apr-1 (`max_progress=nan`). **Just re-run `benchmark.py` + `generate_results_tables.py`.** |
| B2 BEV 0% completion | **code FIXED, data STALE** | e.g. `bev_scaled_cnn` has max_progress 0.998 but completed=0 (Jun-26 data). Re-eval flips to non-zero. |
| B3 failure-replay table | PARTIAL | model+CSV exist; needs steps-to-50% metric + generator wiring. Small. |
| B4 models/results drift | PARTIAL / decision | 3 orphaned forward BEV-encoder result rows w/o models → drop or retrain (ties to Track C). |
| B5 five missing figures | **NOT STARTED** | 5 `\safegraphic` placeholder boxes in PDF: `reverse_obs_ablation`, `forward_reward_ablation`, `reverse_reward_ablation`, `failure_replay_curve`, `infrastructure_sensor_nodes`. **Draft-blocking.** |
| B6 regenerate tables | blocked on B1–B4 + Track C | one command once inputs ready |

### Track C — CuPy GPU env  ⚑ *far more advanced than checklist*
- **C1–C6 DONE.** Plus, built since the checklist: `batched/bev.py` (BEV render), `obstacle_env.py`+`obstacles.py` (stop-gate), 4 classical baselines, `sb3_adapter.py`.
- **C7 Stage-1 multi-seed = DONE & Ch.5-ready.** `results_stage1_v4/`: **96/96 runs, 4 algos × 4 rewards × 2 dirs × 3 seeds**, real **shunt-truck** params (Tesla retired), `summary.csv` with **mean±95% CI**, and **drop-in `.tex` tables** in `results_stage1_v4/thesis_export/` (reward + algorithm comparison, fwd/rev, w/ best-seed + robustness cols).
  - New story (motivates the expected Ch.5 rewrite): **DQN most robust forward / TD3 highest reverse ceiling but 1 collapsed seed/cell / on-policy fails reverse / reward-conditioning governs trainability**; reverse is a *training-reliability* problem (PP/PID/LQR/MPC all solve it).
- **C7 remainder = Stage-2 observation ablation NOT run** (state / lidar-{4,8,16,24,32} / BEV). This is the perception axis Ch.5 needs. Env+BEV tooled; needs GPU time + HP match.
- Data **not yet copied** into thesis `tables/`/`figures/` (deliberate manual step).

### Track D — Obstacle + BEV  ⚑ *the single highest-leverage GPU-day*
- **D1 stop-gate reward: code DONE, never run.** `obstacle_env.py:124-131` implements "reward stop iff difficulty>threshold, penalize stop when low"; `STOP_SIGNAL` action live; `ObstacleConfig` + `shunt_truck_obstacle_config()` set.
- **D2: DONE analytically** (obstacles as circles + lidar/BEV, not a ported occ grid). Fine unless thesis specifically claims grid rasterization in headless env.
- **D3: infra DONE, NO obstacle policy ever trained.** `eval_obstacle.py` already emits the exact Ch.5 table (per-difficulty complete/stop/crash + **min-clearance** + Wilson CIs + traditional APF+PP baseline). Every obstacle number in the thesis is still the **old weak pygame result**. → **~1 GPU-day: `train_obstacle.py` (warm-start from a Stage-1 policy) → `eval_obstacle.py` → new headline result.**
- **D4 Ch.5 §Obstacle Navigation: NOT STARTED** (write around new results).
- **BEV: DONE** — real rotating camera + 3 encoders + results (pygame) *and* clean analytic BEV (CuPy). Only needs B2 hygiene + B4 drop-decision. Marginal value ≈ cleanup, not new results.

### Track E — Perception wiring (real obstacle test DEFERRED w/ placeholders)
- **E6 write Ch.6 §Infrastructure Sensor Nodes: draft-blocking, WRITING ONLY** — can be written accurately from ground truth today (see below).
- **E1–E5: NOT STARTED, all deferrable.** No ROS1→ROS2 bridge; `rl_bridge_node` occ grid is lane-centerline-only; the `ros_obs.observation_from_ros` occ seam exists but is **dead** (no importers). IndoorPerception hardcoded to node 3 + bed ROI (`MOTWithFeedback.py:52`).

### Track F — Cleanup (all deferrable, none draft-blocking)
- F1 training/dynamics scripts still in ROS deploy pkg; F2 byte-identical duplicate `smith_predictor.py` + dead `ros_obs.py` + untracked `docs/localization_fixes_2026-07-01/` (**4 PNGs = good Ch.6 figures**); F3 hardcoded `/home/ben` vs `/home/electrans_robot` paths; F4 `lidar_hitch_angle/package.xml` desc TODO (fix only if cited); F5 DynamicsMLP done but real full-topic bag unrecorded (future work).

### ⚠ Correction to fold into the manuscript
Checklist "User decisions" says the **Smith predictor is flag-enabled via the run script**. FALSE:
it is implemented, wired, and off-by-default, but **no launch/`start_robot.sh` flag exposes it** —
enabling requires `--ros-args -p use_smith_predictor:=true`. Ch.6 must either say that, or a
pass-through flag must be added before the sentence is true. (Also supersedes the old memory note
that it was "blocked on a dynamics bag" — that's the optional `DynamicsMLP`, not the Smith predictor.)

---

## Highest-value chunk & order of operations

**The bottleneck is new numbers for Ch.5, not prose.** The author is handling chapter prose
(Ch.3/4/6 cleanup, eventual Ch.5 rewrite); the highest-leverage help is the **empirical
scaffolding**: run the two remaining experiments, then regenerate/ingest all tables & figures.

### Phase 0 — kick off long-running GPU jobs FIRST (they run unattended, gate Ch.5)
1. **D3 obstacle train+eval** (~1 GPU-day) — the only genuinely *new* empirical result, warm-startable. Highest single-action value.
2. **C7 Stage-2 observation ablation** (state/lidar-N/BEV, 3 seeds) — fills Ch.5's perception axis with CIs.

### Phase 1 — quick wins in parallel (immediate, no deps, mostly CPU/mechanical)
3. **B1 + B2**: re-run `benchmark.py` (fwd+rev) + BEV `eval.py` rows → `generate_results_tables.py`. Kills the all-zero completion columns (code already fixed). *Highest value-per-effort.*
4. **Copy `results_stage1_v4/thesis_export/*.tex` into thesis `tables/`** → real multi-seed Ch.5 tables now.
5. A3 **HP table** fill (from `train.py`/SB3); A5 **CLAUDE.md** doc fix; delete empty `thesis_refs.bib`; A8 **start reader selection**.

### Phase 2 — writing that needs no new experiments (placeholders where noted)
6. **A2** Ch.6 Deployment Results (placeholders) + fold 3 `lab_deployment_findings.md` findings + **Smith-predictor flag correction**.
7. **E6** Ch.6 §Infrastructure Sensor Nodes (honest as-is description + concept figure).
8. **A3** remaining appendices (reward derivation, sim-software arch, additional plots); **A4 04:98** architecture content.

### Phase 3 — after GPU jobs land
9. **B5** figures (obs/reward-ablation curves from CuPy `curves.csv`; failure-replay curve; sensor-node concept diagram) + **B3** failure-replay table + **B4** drop-vs-retrain decision + **B6** regenerate all tables.
10. **D4** Ch.5 §Obstacle Navigation write-up; **A6** future→past-tense Ch.5 pass and rewrite around the new DQN/TD3/reward-conditioning story.

### Deferred (post-lab / future work, placeholders in draft)
- E1–E5 real perception wiring + on-robot obstacle test; F1–F5 cleanup; F5 DynamicsMLP with a real dynamics bag.

---

## Verification targets (unchanged from original, still valid)
- `cd thesis && make` → `build/main.pdf`, **no "Figure unavailable" boxes**, no undefined-cite/bibtex warnings, 80–120 pp.
- Completion columns internally consistent (no all-zero column; BEV rows that finish full episodes report >0%).
- Obstacle policy demonstrably goes-around-easy / stops-on-hard, stratified + min-clearance.
