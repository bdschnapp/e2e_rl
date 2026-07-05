#!/bin/bash
# Overnight Stage-1 screen (1 seed) on the shunt truck. Two INDEPENDENT, RESUMABLE,
# crash-isolated studies to separate outdirs — if one dies, the other and all
# already-written results survive; re-running this script resumes each where it left
# off. Budget 1.5M steps so PPO/discrete-PPO converge (fair vs sample-efficient
# off-policy); steps-to-threshold + wall-clock capture the efficiency axes.
# BEV/image obs is NOT included (the batched env has no renderer — separate track).
set -u
cd "$(dirname "$0")/.."
export TTRL_BACKEND=cupy PYTHONUNBUFFERED=1
OUT=${OUT:-results_stage1}
STEPS=${STEPS:-1500000}
SEEDS="${SEEDS:-0}"          # 1-seed screen; re-run with SEEDS="0 1 2" to add seeds
NENVS=${NENVS:-256}
mkdir -p "$OUT"

echo "=== [$(date)] Study A: algorithm 2x2 x reward  (obs=lidar24) ==="
python3 scripts/run_stage1_sweep.py --preset shunt --fixed_speed --mild_paths \
    --algos ppo td3 ppo_disc dqn --obs lidar24 --directions forward reverse \
    --seeds $SEEDS --steps "$STEPS" --n_envs "$NENVS" --outdir "$OUT/A_algo_reward" \
    2>&1 | tee -a "$OUT/A_algo_reward.log"
echo "=== [$(date)] Study A done (rc=${PIPESTATUS[0]}) ==="

echo "=== [$(date)] Study B: observation ablation  (td3 + multiplicative) ==="
python3 scripts/run_stage1_sweep.py --preset shunt --fixed_speed --mild_paths \
    --algos td3 --rewards multiplicative --obs state lidar4 lidar8 lidar16 lidar24 lidar32 \
    --directions forward reverse --seeds $SEEDS --steps "$STEPS" --n_envs "$NENVS" \
    --outdir "$OUT/B_obs" \
    2>&1 | tee -a "$OUT/B_obs.log"
echo "=== [$(date)] Study B done (rc=${PIPESTATUS[0]}) ==="

echo "=== [$(date)] Overnight sweep complete. Plot with:"
echo "    python3 scripts/plot_stage1.py --outdir $OUT/A_algo_reward"
echo "    python3 scripts/plot_stage1.py --outdir $OUT/B_obs"
