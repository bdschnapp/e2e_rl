#!/usr/bin/env bash
# Stage-2A perception ablation (the FAIR, apples-to-apples study).
# Fixed recipe = Phase-1 winner (TD3 + multiplicative), IDENTICAL for every observation,
# NO tricks (no curriculum / failure-replay / feature-warmup) so the comparison is pure
# modality-vs-representation. 3 seeds + 95% CI, matching Phase-1 V1 rigor.
#
#   Forward (500k): state, lidar8, lidar24, lidar64, bev(CNN-from-scratch)
#   Reverse (1.3M): state, lidar8, lidar24, lidar64   (bev-reverse deferred to 2B best-vision;
#                   reverse-from-scratch vision is expected ~0% and is the most expensive cell)
#
# CNN-from-scratch (bev) is EXPECTED TO FAIL here — that failure is the motivation for 2B.
# All other defaults (n_envs=32, eval pool, eval_no_stop, guide_transition=-1, save_models)
# already match Phase-1 V1, so this is V1's harness with only --obs swapped.
set -eu
cd "$(dirname "$0")/.."
export TTRL_BACKEND=cupy
OUT=results_stage2_obs

echo "### 2A FORWARD — state/lidar8/lidar24/lidar64/bev, td3, multiplicative, seeds 0-2"
python3 scripts/run_stage1_sweep.py \
    --algos td3 --rewards multiplicative --directions forward \
    --obs state lidar8 lidar24 lidar64 bev --seeds 0 1 2 \
    --outdir "$OUT"

echo "### 2A REVERSE — state/lidar8/lidar24/lidar64 (lidar-only), td3, multiplicative, seeds 0-2"
python3 scripts/run_stage1_sweep.py \
    --algos td3 --rewards multiplicative --directions reverse \
    --obs state lidar8 lidar24 lidar64 --seeds 0 1 2 \
    --outdir "$OUT"

echo "### 2A DONE -> $OUT/{summary.csv,curves.csv,manifest.jsonl}"
