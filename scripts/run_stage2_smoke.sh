#!/usr/bin/env bash
# Stage-2 perception SMOKE (morning, 1 seed). Validates the obs-ablation pipeline for
# state / lidar24 / bev across TD3 (both directions) + DQN (forward spot-check), fixed
# reward=multiplicative (Phase-1 winner, robust both directions). NOT definitive:
#   - 1 seed only (definitive = 3 seeds)
#   - reverse budget reduced to 500k (definitive = 1.5M; reverse converges late)
#   - BEV is trained FROM SCRATCH and is EXPECTED TO FAIL (per prior experience: BEV
#     needs curriculum state/lidar->vision or AE pretraining). This run confirms that
#     baseline failure to motivate the curriculum build.
set -u
cd "$(dirname "$0")/.."
OUT=results_stage2_smoke
export TTRL_BACKEND=cupy

echo "### TD3 (both directions) — state/lidar24/bev, multiplicative, seed 0"
python3 scripts/run_stage1_sweep.py --preset shunt \
    --algos td3 --directions forward reverse \
    --obs state lidar24 bev --rewards multiplicative \
    --seeds 0 --steps 500000 --rev_steps 500000 --eval_every 100000 \
    --outdir "$OUT"

echo "### DQN (forward spot-check) — state/lidar24/bev, multiplicative, seed 0"
python3 scripts/run_stage1_sweep.py --preset shunt \
    --algos dqn --directions forward \
    --obs state lidar24 bev --rewards multiplicative \
    --seeds 0 --steps 500000 --eval_every 100000 \
    --outdir "$OUT"

echo "### DONE — results in $OUT/{manifest.jsonl,curves.csv,summary.csv}"
