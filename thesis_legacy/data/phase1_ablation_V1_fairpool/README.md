# Phase-1 ablation: fair-pool re-evaluation (2026-07-20)

**Question.** The locked Phase-1 ablation (`results_stage1_V1/`) evaluated on the PP-validated
SAFE pool (`path_pools/shunt_mild_safe.npz`), where pure-pursuit completes 100% by construction.
That excludes the paths PP cannot complete from spawn -- a mild "PP-success" bias. Do the ablation
conclusions (TD3 + multiplicative wins; the algo x reward ranking) survive on an unbiased pool?

**Method (additive; the frozen `run_stage1_sweep.py` is untouched).** `scripts/reeval_stage1_fairpool.py`
re-scores all 96 saved best-checkpoints (`results_stage1_V1/models/`) on BOTH pools with an
identical rollout, so the safe->fair delta is purely the pool:
- **Fair pool** = `path_pools/shunt_mild_fair_unfiltered.npz`: the SAME mild random distribution the
  safe pool was drawn from, minus the PP filter (path generation unchanged -- every path kept).
  Fixed 1024-path .npz so all models see the same paths.
- Eval protocol matches the locked ablation: shunt preset, mild paths, fixed 5 m/s, `guide_transition=-1`,
  eval WITHOUT the stop action, best checkpoint per run.

**Result: the ablation is robust to the pool. Nothing changes.**
- **Forward: pool-invariant.** Every forward cell 1.000 -> 1.000 (delta 0.000). Forward is minimum-phase,
  so it tolerates spawn bends -- 0% spawn-infeasible.
- **Reverse: the winner and the ranking hold.** TD3+multiplicative stays #1 (completion 0.967 -> 0.959),
  and the top ordering is unchanged (td3-mult > td3-guided > ppo-guided > td3-no_hitch > td3-dense).
- **Effect is tiny and reverse-only:** max |delta completion| = **0.022** across all 32 cells; mean |delta| =
  0.005. Jackknife rates essentially unchanged. Consistent with only ~2-3% of reverse paths being
  spawn-infeasible (`path_pools_build.log`: PP fwd 100%, PP rev 96.7-98.4%).

**Takeaway for the thesis.** The Phase-1 model/reward selection is not an artifact of the PP-filtered
eval pool; the same conclusions hold on the full unfiltered path distribution. The safe pool was a
fair-to-the-policy convenience (it removes ~3% spawn-infeasible reverse paths), not a result-changing
bias.

## Files
- `summary_safe_vs_fair.csv` -- per (algo, direction, reward): completion/cte/hitch/jackknife mean +/- CI
  on each pool, plus the delta.
- `per_model.csv` -- per checkpoint, both pools.
- Script: `scripts/reeval_stage1_fairpool.py`. Fair pool: `path_pools/shunt_mild_fair_unfiltered.npz`.
