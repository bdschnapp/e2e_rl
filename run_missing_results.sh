#!/usr/bin/env bash
# run_missing_results.sh — Targeted runner for thesis experiments that are
# still missing from the current results snapshot.
#
# This script mirrors the helper syntax used in run_ablations.sh, but only
# touches experiments that are currently absent from results/ or models/.

set -euo pipefail

if [[ -f "venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "venv/bin/activate"
fi

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_ENVS_FAST=16
N_ENVS_BEV=12
TIMESTEPS=100000
EVAL_EPISODES=30
LIDAR_BEAMS_DEFAULT=16
RUN_TRAIN_MISSING="${RUN_TRAIN_MISSING:-1}"
RUN_EVAL_MISSING="${RUN_EVAL_MISSING:-1}"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

model_path_for() {
    local scenario="$1" obs_tag="$2" reward="$3" retry="${4:-0}"
    local suffix=""
    [[ "$retry" == "1" ]] && suffix="_retry"
    printf 'models/%s/%s/%s%s/best_model.zip' "$scenario" "$obs_tag" "$reward" "$suffix"
}

result_path_for() {
    local scenario="$1" obs_tag="$2" reward="$3" retry="${4:-0}"
    local suffix=""
    [[ "$retry" == "1" ]] && suffix="_retry"
    printf 'results/%s/%s/%s%s/aggregate.csv' "$scenario" "$obs_tag" "$reward" "$suffix"
}

obs_tag_from_cli() {
    local obs="$1"
    shift
    local encoder="scratch"
    local lidar_beams="$LIDAR_BEAMS_DEFAULT"
    while [[ "$#" -gt 0 ]]; do
        case "$1" in
            --encoder)
                shift
                [[ "$#" -gt 0 ]] && encoder="$1"
                ;;
            --lidar_beams)
                shift
                [[ "$#" -gt 0 ]] && lidar_beams="$1"
                ;;
        esac
        shift || true
    done

    if [[ "$obs" == "bev" && "$encoder" != "scratch" ]]; then
        printf 'bev_%s' "$encoder"
    elif [[ "$obs" == "lidar" && "$lidar_beams" != "$LIDAR_BEAMS_DEFAULT" ]]; then
        printf 'lidar_%s' "$lidar_beams"
    else
        printf '%s' "$obs"
    fi
}

train() {
    local scenario="$1" obs="$2" reward="$3"
    shift 3
    local extra=("$@")
    local obs_tag retry=0
    obs_tag="$(obs_tag_from_cli "$obs" "${extra[@]}")"
    for arg in "${extra[@]}"; do
        [[ "$arg" == "--retry_on_failure" ]] && retry=1
    done

    local model_path
    model_path="$(model_path_for "$scenario" "$obs_tag" "$reward" "$retry")"
    if [[ -f "$model_path" ]]; then
        log "SKIP train (exists): $model_path"
        return 0
    fi

    log "TRAIN  scenario=$scenario  obs=$obs_tag  reward=$reward  extra=${extra[*]:-}"
    python3 train.py \
        --scenario "$scenario" \
        --obs "$obs" \
        --reward "$reward" \
        --timesteps "$TIMESTEPS" \
        "${extra[@]}"
}

eval_() {
    local scenario="$1" obs="$2" reward="$3"
    shift 3
    local extra=("$@")
    local obs_tag
    obs_tag="$(obs_tag_from_cli "$obs" "${extra[@]}")"
    local model_path result_path
    model_path="$(model_path_for "$scenario" "$obs_tag" "$reward")"
    result_path="$(result_path_for "$scenario" "$obs_tag" "$reward")"

    if [[ -f "$result_path" ]]; then
        log "SKIP eval (exists): $result_path"
        return 0
    fi
    if [[ ! -f "$model_path" ]]; then
        log "SKIP eval (missing model): $model_path"
        return 0
    fi

    log "EVAL   scenario=$scenario  obs=$obs_tag  reward=$reward  extra=${extra[*]:-}"
    python3 eval.py \
        --scenario "$scenario" \
        --obs "$obs" \
        --reward "$reward" \
        --model "$model_path" \
        --episodes "$EVAL_EPISODES" \
        "${extra[@]}"
}

eval_retry() {
    local scenario="$1" obs="$2" reward="$3"
    shift 3
    local extra=("$@")
    local obs_tag
    obs_tag="$(obs_tag_from_cli "$obs" "${extra[@]}")"
    local model_path result_path
    model_path="$(model_path_for "$scenario" "$obs_tag" "$reward" 1)"
    result_path="$(result_path_for "$scenario" "$obs_tag" "$reward" 1)"

    if [[ -f "$result_path" ]]; then
        log "SKIP eval retry (exists): $result_path"
        return 0
    fi
    if [[ ! -f "$model_path" ]]; then
        log "SKIP eval retry (missing model): $model_path"
        return 0
    fi

    log "EVAL   scenario=$scenario  obs=$obs_tag  reward=${reward}_retry  extra=${extra[*]:-}"
    python3 eval.py \
        --scenario "$scenario" \
        --obs "$obs" \
        --reward "$reward" \
        --model "$model_path" \
        --episodes "$EVAL_EPISODES" \
        --output_csv "results/${scenario}/${obs_tag}/${reward}_retry/episodes.csv" \
        --retry_on_failure \
        "${extra[@]}"
}

run_train_and_eval() {
    local scenario="$1" obs="$2" reward="$3"
    shift 3
    local extra=("$@")
    train "$scenario" "$obs" "$reward" "${extra[@]}"
    eval_ "$scenario" "$obs" "$reward" "${extra[@]}"
}

run_eval_only() {
    local scenario="$1" obs="$2" reward="$3"
    shift 3
    local extra=("$@")
    eval_ "$scenario" "$obs" "$reward" "${extra[@]}"
}

# ---------------------------------------------------------------------------
# Missing experiments as of current thesis snapshot
# ---------------------------------------------------------------------------
if [[ "$RUN_TRAIN_MISSING" == "1" ]]; then
    log "=== TRAIN + EVAL: models not trained yet ==="
    run_train_and_eval forward bev dense --encoder unet_unfrozen --n_envs "$N_ENVS_BEV"
    run_train_and_eval reverse bev dense --encoder ae_frozen --n_envs "$N_ENVS_BEV"
    run_train_and_eval reverse bev dense --encoder ae_unfrozen --n_envs "$N_ENVS_BEV"
    run_train_and_eval reverse bev dense --encoder unet_frozen --n_envs "$N_ENVS_BEV"
    run_train_and_eval reverse bev dense --encoder unet_unfrozen --n_envs "$N_ENVS_BEV"
    run_train_and_eval forward lidar dense --n_envs "$N_ENVS_FAST" --lidar_beams 4
    run_train_and_eval reverse_obs bev dense --n_envs "$N_ENVS_BEV" --encoder scratch
fi

if [[ "$RUN_EVAL_MISSING" == "1" ]]; then
    log "=== EVAL ONLY: trained models missing result CSVs ==="
    run_eval_only forward lidar dense --lidar_beams 8
    run_eval_only forward lidar dense --lidar_beams 24
    run_eval_only forward lidar dense --lidar_beams 32
    run_eval_only forward lidar tractor_focus --lidar_beams 24
    run_eval_only forward lidar multiplicative --lidar_beams 24
    run_eval_only forward lidar guided --lidar_beams 24
    run_eval_only reverse lidar dense --lidar_beams 24
    run_eval_only reverse lidar no_hitch --lidar_beams 24
    run_eval_only reverse lidar multiplicative --lidar_beams 24
    run_eval_only reverse lidar guided --lidar_beams 24
    eval_retry reverse lidar dense --lidar_beams 24
fi

log "=== DONE ==="
