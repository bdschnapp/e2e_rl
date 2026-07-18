# Phase-2A — Observation-space ablation (fair, apples-to-apples)

**Date:** 2026-07-07. **Machine:** remote RTX 3090 (mvslab), cupy backend.
**Source:** `tractor_trailer_rl_cupy` @ branch `tractor_trailer_rl_cupy`, via `scripts/run_stage1_sweep.py` (frozen Phase-1 harness, only `--obs` varied). Identical recipe for every observation — NO curriculum / failure-replay / feature-warmup — so the comparison is pure modality-vs-dimensionality.

**Fixed config:** TD3 + `multiplicative` reward, `shunt` preset, fixed speed, mild paths, 3 seeds (0/1/2), eval on the PP-validated safe pool (`shunt_mild_safe.npz`), `eval_no_stop`, `guide_transition=-1`. Forward budget 500k steps; reverse 1.3M. Metric = best-checkpoint eval completion (completion ↑ then trailer-CTE ↓).

**Command (equivalent):**
```
# forward: state lidar8 lidar24 lidar64 bev   |   reverse: state lidar8 lidar24 lidar64 (lidar-only)
run_stage1_sweep.py --algos td3 --rewards multiplicative --obs state lidar8 lidar24 lidar64 bev \
    --directions forward --seeds 0 1 2      # + reverse w/o bev
```

## Result (mean completion over 3 seeds; per-seed in `manifest.jsonl`)

| Obs (dim)        | Forward           | Reverse            |
|------------------|-------------------|--------------------|
| state (8)        | **1.00** (1/1/1)  | **0.97** (.97/.97/.97) |
| lidar8 (16)      | **1.00** (1/1/1)  | **0.90** (.96/.96/.77) |
| lidar24 (32)     | **1.00** (1/1/1)  | **0.63** (.95/.94/**.00**) |
| lidar64 (72)     | **0.67** (1/1/**0**) | **0.31** (.94/**0**/**0**) |
| bev (32x32 img)  | **0.00** (0/0/0)  | (deferred to 2B)   |

**Headline:** reliability degrades **monotonically with observation dimensionality**, and the effect amplifies on the harder reverse task. Low-dimensional observations are not merely sufficient but more *robust* (plain `state` is the most reliable in both directions); adding input dimensions raises seed variance and then causes outright seed collapses (lidar64 fwd 1/3, lidar24 rev 1/3, lidar64 rev 2/3), culminating in total failure for BEV-from-scratch. This "dimensionality → reliability" curve motivates Phase-2B (rescuing the BEV image representation).

## Files
- `summary.csv` — per-(algo,dir,obs,reward) mean ± 95% CI + steps-to-threshold (thesis tables/bars).
- `curves.csv` — per-eval-checkpoint learning curves (all runs).
- `manifest.jsonl` — one line per run: full config, vehicle params, wall-clock, best/final metrics (provenance + per-seed values).
- Model checkpoints (`models/*.zip`) intentionally NOT copied (large; on the remote box).
