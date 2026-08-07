# Phase-1 ablation — DEFINITIVE results (V1)

**This is the Phase-1 ablation of record.** Multi-seed (3 seeds) algorithm × reward ×
direction sweep on the CuPy GPU-parallel batched env, **state observation**, fixed 5 m/s,
train-random / eval-safe / eval-no-stop, guided = pure PP clone. Supersedes the earlier
`cupy_stage1_backup_2026-07-06/` snapshots (those were the lidar-24 v4 study, kept only for
provenance). Persisted 2026-07-07.

## Configuration
- **Env:** CuPy batched, shunt truck (2.95 m wheelbase, 12.5 m trailer, max steer 0.87 rad),
  world 150×90 m, dt 0.1 s, **fixed 5 m/s**, spawn at 10% of path.
- **Design:** 4 algos (td3, dqn, ppo, ppo_disc) × 4 rewards/dir × 2 directions × 3 seeds
  = **96 runs, 96/96 complete, 0 failures.**
- **Rewards:** forward = {dense, tractor_focus, multiplicative, guided}; reverse =
  {dense, no_hitch, multiplicative, guided}.
- **Budgets:** off-policy fwd 500k / rev 1.3M; PPO 2M; eval every 100k.
- **Eval:** on the PP-validated safe pool (`path_pools/shunt_mild_safe.npz`), **eval-no-stop**
  (stop action ignored at eval for a fair RL-vs-classical comparison). Metric = best-checkpoint
  completion (completion desc, trailer-CTE asc), captured pre-auto-reset.

## Files
- `summary.csv`     — per (algo,dir,reward): completion/CTE/hitch/jackknife/return mean ± 95% CI,
                      steps-to-threshold. **Primary source for the thesis tables.**
- `manifest.jsonl`  — per-run record (best + final metrics, wall time, vehicle params).
- `curves.csv`      — eval curves (every 100k steps) for the learning-curve figures.
- `classical_baselines_safepool.log` — PP/PID/LQR/MPC on the same safe pool (the RL-vs-classical
                      yardstick). Fwd: all 4 = 1.000 compl. Rev: PP 1.000, MPC 0.990,
                      LQR 0.967 (best CTE 0.568), PID 0.898.
- `sweep.log`       — full run log (numpy-import warnings are non-fatal noise).
- `champion_models/` — the 3 champion policies (TD3 reverse multiplicative, s0/s1/s2) carried
                      into Phase-2. Full 96-model set stays in
                      `tractor_trailer_rl_cupy/results_stage1_V1/models/` (240M, not copied here).

## The champion
**`(TD3, multiplicative)` reverse = 0.969 ± 0.011** — the single best cell, tight CI, no dead
seeds, ~3% under the PP oracle (1.0 on the safe pool). This is the (algo, reward) carried into
Phase-2 (observation/perception ablation).

## Headline reverse results (mean best-completion over 3 seeds)
| algo | reward | completion (mean ± 95% CI) |
|---|---|---|
| **TD3** | **multiplicative** | **0.969 ± 0.011** |
| TD3 | guided | 0.875 ± 0.060 |
| TD3 | no_hitch | 0.830 ± 0.144 |
| TD3 | dense | 0.810 ± 0.058 |
| PPO | guided | 0.814 ± 0.283 |
| ppo_disc | guided | 0.536 ± 0.081 |
| DQN | guided | 0.464 ± 0.557 |
| PPO | multiplicative | 0.434 ± 0.010 |
| ppo_disc | dense | 0.423 ± 0.024 |
| ppo_disc | no_hitch | 0.289 ± 0.283 |
| DQN | dense | 0.147 ± 0.286 |
| ppo_disc | multiplicative | 0.145 ± 0.284 |
| PPO | dense | 0.143 ± 0.278 |
| PPO | no_hitch | 0.135 ± 0.263 |
| DQN | no_hitch | 0.020 ± 0.038 |
| DQN | multiplicative | 0.0005 ± 0.001 |

## Headline forward results (mean best-completion over 3 seeds)
| algo | dense | tractor_focus | multiplicative | guided |
|---|---|---|---|---|
| TD3 | 1.00 | 1.00 | 1.00 | 1.00 |
| PPO | 1.00 | 1.00 | 1.00 | 1.00 |
| DQN | 1.00 | 1.00 | 0.82 | 0.57 |
| ppo_disc | 0.81 | 0.89 | 1.00 | 0.42 |

## Findings (for Ch.5)
1. **`(TD3, multiplicative)` reverse = 0.969 ± 0.011** — Phase-1 champion; ~3% under PP oracle.
2. **TD3 is the only algorithm that solves reverse** (all TD3 rewards ≥ 0.81; best non-TD3 cell
   PPO+guided 0.814 but with a huge CI = one lucky seed).
3. **Reward winner is algorithm-conditional:** TD3 → multiplicative ≫ guided; PPO (on-policy) →
   guided is the *only* thing that works (ordering inverts); discrete → guided is best-but-
   mediocre and multiplicative is catastrophic for DQN (0.0005).
4. **Guided fails both discrete algorithms in both directions** — reward-space behavioural
   cloning only works when the action space can represent the continuous PP teacher (clean
   negative control; strengthens guided-as-continuous-clone framing).
5. **Forward is saturated for continuous control** (TD3 & PPO = 1.000 under every reward),
   matching the classical baselines (all four hit 100% forward). Only discreteness + a wrong
   reward breaks forward.

## Regenerate the LaTeX tables/figures (when ready)
```
cd tractor_trailer_rl_cupy
python scripts/export_thesis_results.py --outdir results_stage1_V1
python scripts/plot_stage1.py --outdir results_stage1_V1
```
