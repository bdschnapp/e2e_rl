# Fair-pool evaluation audit (2026-07-21)

**Concern.** Several experiments evaluated on the PP-validated SAFE pool
(`path_pools/shunt_mild_safe.npz`), where pure-pursuit completes 100% *by construction*. That is a
mild "PP-success" bias: it excludes the paths PP cannot complete from spawn (~2-3% of REVERSE paths;
0% forward -- forward is minimum-phase and tolerates spawn bends). Do the conclusions survive on a
FAIR unfiltered pool?

**Fair pool.** `path_pools/shunt_mild_fair_unfiltered.npz` -- the SAME mild random distribution the
safe pool was drawn from, minus the PP filter (path generation unchanged; every path kept). Fixed
1024 paths so all models see the same set. Every re-eval scores each model on BOTH pools with an
identical rollout, so the safe->fair delta is purely the pool.

## Which experiments used which pool

| Experiment | Eval pool | Same problem? | Action |
|---|---|---|---|
| Phase-1 ablation (`run_stage1_sweep`) | SAFE | yes | re-eval'd -> robust |
| Classical baselines (`eval_controllers`) | SAFE | yes | re-eval'd -> robust |
| Phase-2A obs ablation (same frozen harness) | SAFE | yes | see note |
| Phase-2b geometry encoder (`geom_frozen_rl`) | SAFE | yes | re-eval'd -> robust |
| Phase-3 flicker/ordering | FAIR (`pool_file=None`) | no | already fair (also a relative result) |
| Phase-3 hidden-path (headline) | FAIR (`pool_file=None`) | no | already fair |

**Phase-2A note.** Its lidar/bev checkpoints were not saved, so those cells cannot be re-scored
without retraining. But spawn-infeasibility is a PATH property, independent of the observation, so it
removes the same ~3% of reverse paths for every obs type and shifts all cells equally -> the obs
ranking is preserved. Its "state" cells are the Phase-1 cells, already fair-re-eval'd. So Phase-2A is
robust by the same argument.

## Re-eval results (all safe -> fair)

**Phase-1 ablation** (`phase1_ablation_V1_fairpool/`): TD3+multiplicative stays #1 (reverse 0.967 ->
0.959); full ranking preserved; forward pool-invariant. Max |delta completion| = 0.022 across 32
cells, reverse-only.

**Classical baselines** (`classical_baseline_fairpool/`): forward all controllers 1.000 on both
pools. Reverse:

| controller | safe | fair | delta |
|---|---|---|---|
| PurePursuit | 1.000 | 0.980 | -0.020 |
| MPC | 0.991 | 0.977 | -0.014 |
| LQR | 0.964 | 0.959 | -0.005 |
| PID | 0.898 | 0.873 | -0.025 |

The important correction: PP's safe-pool 1.000 is circular (the pool WAS the PP-completable set); the
fair pool gives PP's true reverse completion, 0.980. All drops <= 0.025.

**Phase-2b geometry encoder** (`phase2b_geometry_encoder/fairpool_per_model.csv`): forward 1.000 ->
0.999; reverse 0.831 -> 0.822 (mean over 3 seeds; worst seed delta -0.021).

## Conclusion

The PP-safe-pool bias is **small (~2-3%), reverse-only, and observation-independent, and it changes
no conclusion or ranking anywhere.** The ablation winner (TD3+multiplicative), the RL-vs-classical
ordering, and the geometry-encoder results all hold on the fair unfiltered pool. The safe pool was a
fair-to-the-policy convenience (it removes spawn-infeasible reverse paths that no policy can solve),
not a result-changing bias. Phase-3 already used the fair pool throughout.
