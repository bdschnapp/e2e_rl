#!/usr/bin/env bash
# run_ablations.sh — Sequential thesis ablation runner
#
# Safe defaults:
# - Always runs observation-space ablations.
# - Reward, failure-replay, obstacle, and benchmark phases are gated on
#   explicitly chosen best configurations.
# - Obstacle state runs are kept as a deliberate sensing ablation.

set -euo pipefail

if [[ -f "venv/bin/activate" ]]; then
    # Ensure the overnight run uses the project environment even in tmux/SSH.
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

# Selected best configurations for later phases.
# Valid obs tags:
#   state | lidar | lidar_32 | bev | bev_ae_frozen | bev_unet_unfrozen
BEST_OBS_FORWARD="lidar_24"
BEST_OBS_REVERSE="lidar_24"
BEST_REWARD_FORWARD="dense"
BEST_REWARD_REVERSE="dense"

LOG_DIR="logs"
FIGS="thesis/figures/experiment_results"
mkdir -p "$LOG_DIR" "$FIGS"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

is_set() { [[ -n "${1:-}" ]]; }

model_path_for() {
    local scenario="$1" obs_tag="$2" reward="$3" retry="${4:-0}"
    local suffix=""
    [[ "$retry" == "1" ]] && suffix="_retry"
    printf 'models/%s/%s/%s%s/best_model.zip' "$scenario" "$obs_tag" "$reward" "$suffix"
}

obs_tag_from_cli() {
    local obs="$1"
    shift
    local encoder="scratch"
    local lidar_beams="$LIDAR_BEAMS_DEFAULT"
    local i=1
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
        i=$((i + 1))
    done

    if [[ "$obs" == "bev" && "$encoder" != "scratch" ]]; then
        printf 'bev_%s' "$encoder"
    elif [[ "$obs" == "lidar" && "$lidar_beams" != "$LIDAR_BEAMS_DEFAULT" ]]; then
        printf 'lidar_%s' "$lidar_beams"
    else
        printf '%s' "$obs"
    fi
}

obs_tag_to_cli() {
    local obs_tag="$1"
    local -n out_ref="$2"
    out_ref=()
    case "$obs_tag" in
        state)
            out_ref+=(--obs state)
            ;;
        lidar)
            out_ref+=(--obs lidar --lidar_beams "$LIDAR_BEAMS_DEFAULT")
            ;;
        lidar_*)
            out_ref+=(--obs lidar --lidar_beams "${obs_tag#lidar_}")
            ;;
        bev)
            out_ref+=(--obs bev --encoder scratch)
            ;;
        bev_ae_frozen|bev_ae_unfrozen|bev_unet_frozen|bev_unet_unfrozen)
            out_ref+=(--obs bev --encoder "${obs_tag#bev_}")
            ;;
        *)
            log "ERROR: unsupported obs tag '$obs_tag'"
            exit 1
            ;;
    esac
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
        "${extra[@]}" || {
            log "WARN: train failed for scenario=$scenario obs=$obs_tag reward=$reward"
            return 0
        }
}

eval_() {
    local scenario="$1" obs="$2" reward="$3"
    shift 3
    log "EVAL   scenario=$scenario  obs=$obs  reward=$reward  extra=${*:-}"
    python3 eval.py \
        --scenario "$scenario" \
        --obs "$obs" \
        --reward "$reward" \
        --episodes "$EVAL_EPISODES" \
        "$@" || log "WARN: eval failed for $scenario/$obs/$reward"
}

eval_retry() {
    local scenario="$1" obs_tag="$2" reward="$3"
    local retry_model
    retry_model="$(model_path_for "$scenario" "$obs_tag" "$reward" 1)"
    if [[ ! -f "$retry_model" ]]; then
        log "SKIP eval retry (missing): $retry_model"
        return 0
    fi

    local obs_args=()
    obs_tag_to_cli "$obs_tag" obs_args
    log "EVAL   scenario=$scenario  obs=$obs_tag  reward=${reward}_retry"
    python3 eval.py \
        --scenario "$scenario" \
        "${obs_args[@]}" \
        --reward "$reward" \
        --model "$retry_model" \
        --episodes "$EVAL_EPISODES" \
        --output_csv "results/${scenario}/${obs_tag}/${reward}_retry/episodes.csv" \
        --retry_on_failure || log "WARN: eval retry failed for $scenario/$obs_tag/$reward"
}

plot_curves() {
    local out="$1" title="$2"
    shift 2
    local run_args=("$@")
    local any_found=0
    local pair path
    for pair in "${run_args[@]}"; do
        path="${pair#*=}"
        [[ -f "$path" ]] && any_found=1 && break
    done
    [[ "$any_found" -eq 0 ]] && { log "SKIP plot (no data): $out"; return 0; }

    log "PLOT   $out"
    python3 plot_learning_curves.py \
        --title "$title" \
        --out "$out" \
        --smooth 3 \
        --runs "${run_args[@]}" || log "WARN: plot failed for $out"
}

benchmark_task() {
    local task="$1" controllers="$2" model_path="$3" out_dir="$4"
    [[ ! -d test_scenarios ]] && { log "SKIP benchmark (missing test_scenarios/)"; return 0; }
    [[ ! -f "$model_path" ]] && { log "SKIP benchmark (missing model): $model_path"; return 0; }

    log "BENCH  task=$task  controllers=$controllers"
    python3 benchmark.py \
        --task "$task" \
        --controllers "$controllers" \
        --model "$model_path" \
        --scenarios test_scenarios \
        --output "$out_dir" \
        --tuned_params controllers/tuned_params.json || log "WARN: benchmark failed for $task"
}

# ---------------------------------------------------------------------------
# PHASE 1A/B/C: observation ablations
# ---------------------------------------------------------------------------
log "=== PHASE 1A: Forward obs ablation ==="
# train forward state dense --n_envs "$N_ENVS_FAST"
# train forward lidar dense --n_envs "$N_ENVS_FAST" --lidar_beams "$LIDAR_BEAMS_DEFAULT"
# train forward bev dense --n_envs "$N_ENVS_BEV" --encoder scratch
# train forward bev dense --n_envs "$N_ENVS_BEV" --encoder ae_frozen
# train forward bev dense --n_envs "$N_ENVS_BEV" --encoder ae_unfrozen
# train forward bev dense --n_envs "$N_ENVS_BEV" --encoder unet_frozen
# train forward bev dense --n_envs "$N_ENVS_BEV" --encoder unet_unfrozen

log "=== PHASE 1B: Reverse obs ablation ==="
# train reverse state dense --n_envs "$N_ENVS_FAST"
# train reverse lidar dense --n_envs "$N_ENVS_FAST" --lidar_beams "$LIDAR_BEAMS_DEFAULT"
# train reverse bev dense --n_envs "$N_ENVS_BEV" --encoder scratch
# train reverse bev dense --n_envs "$N_ENVS_BEV" --encoder ae_frozen
# train reverse bev dense --n_envs "$N_ENVS_BEV" --encoder ae_unfrozen
# train reverse bev dense --n_envs "$N_ENVS_BEV" --encoder unet_frozen
# train reverse bev dense --n_envs "$N_ENVS_BEV" --encoder unet_unfrozen

log "=== PHASE 1C: Lidar beams ablation ==="
# train forward lidar dense --n_envs "$N_ENVS_FAST" --lidar_beams 8
# train forward lidar dense --n_envs "$N_ENVS_FAST" --lidar_beams 16
# train forward lidar dense --n_envs "$N_ENVS_FAST" --lidar_beams 24
# train forward lidar dense --n_envs "$N_ENVS_FAST" --lidar_beams 32

# ---------------------------------------------------------------------------
# PHASE 2: reward ablations
# ---------------------------------------------------------------------------
if is_set "$BEST_OBS_FORWARD"; then
    log "=== PHASE 2A: Forward reward ablation on $BEST_OBS_FORWARD ==="
    obs_args=()
    obs_tag_to_cli "$BEST_OBS_FORWARD" obs_args
    # train forward "${obs_args[1]}" dense "${obs_args[@]:2}" --n_envs "$([[ "${obs_args[1]}" == "bev" ]] && echo "$N_ENVS_BEV" || echo "$N_ENVS_FAST")"
    # train forward "${obs_args[1]}" tractor_focus "${obs_args[@]:2}" --n_envs "$([[ "${obs_args[1]}" == "bev" ]] && echo "$N_ENVS_BEV" || echo "$N_ENVS_FAST")"
    # train forward "${obs_args[1]}" multiplicative "${obs_args[@]:2}" --n_envs "$([[ "${obs_args[1]}" == "bev" ]] && echo "$N_ENVS_BEV" || echo "$N_ENVS_FAST")"
    # train forward "${obs_args[1]}" guided "${obs_args[@]:2}" --n_envs "$([[ "${obs_args[1]}" == "bev" ]] && echo "$N_ENVS_BEV" || echo "$N_ENVS_FAST")"
else
    log "SKIP Phase 2A: set BEST_OBS_FORWARD first"
fi

if is_set "$BEST_OBS_REVERSE"; then
    log "=== PHASE 2B: Reverse reward ablation on $BEST_OBS_REVERSE ==="
    obs_args=()
    obs_tag_to_cli "$BEST_OBS_REVERSE" obs_args
    # train reverse "${obs_args[1]}" dense "${obs_args[@]:2}" --n_envs "$([[ "${obs_args[1]}" == "bev" ]] && echo "$N_ENVS_BEV" || echo "$N_ENVS_FAST")"
    # train reverse "${obs_args[1]}" no_hitch "${obs_args[@]:2}" --n_envs "$([[ "${obs_args[1]}" == "bev" ]] && echo "$N_ENVS_BEV" || echo "$N_ENVS_FAST")"
    # train reverse "${obs_args[1]}" multiplicative "${obs_args[@]:2}" --n_envs "$([[ "${obs_args[1]}" == "bev" ]] && echo "$N_ENVS_BEV" || echo "$N_ENVS_FAST")"
    # train reverse "${obs_args[1]}" guided "${obs_args[@]:2}" --n_envs "$([[ "${obs_args[1]}" == "bev" ]] && echo "$N_ENVS_BEV" || echo "$N_ENVS_FAST")"
else
    log "SKIP Phase 2B: set BEST_OBS_REVERSE first"
fi

# ---------------------------------------------------------------------------
# PHASE 3: failure replay
# ---------------------------------------------------------------------------
if is_set "$BEST_OBS_REVERSE" && is_set "$BEST_REWARD_REVERSE"; then
    log "=== PHASE 3: Failure replay on reverse/$BEST_OBS_REVERSE/$BEST_REWARD_REVERSE ==="
    obs_args=()
    obs_tag_to_cli "$BEST_OBS_REVERSE" obs_args
    # train reverse "${obs_args[1]}" "$BEST_REWARD_REVERSE" "${obs_args[@]:2}" \
    #     --n_envs "$([[ "${obs_args[1]}" == "bev" ]] && echo "$N_ENVS_BEV" || echo "$N_ENVS_FAST")" \
    #     --retry_on_failure
else
    log "SKIP Phase 3: set BEST_OBS_REVERSE and BEST_REWARD_REVERSE first"
fi

# ---------------------------------------------------------------------------
# PHASE 4: obstacle studies
# ---------------------------------------------------------------------------
if is_set "$BEST_REWARD_FORWARD" && is_set "$BEST_REWARD_REVERSE"; then
    log "=== PHASE 4: Obstacle studies ==="
    # train forward_obs state "$BEST_REWARD_FORWARD" --n_envs "$N_ENVS_FAST"
    # train forward_obs lidar "$BEST_REWARD_FORWARD" --n_envs "$N_ENVS_FAST" --lidar_beams "$LIDAR_BEAMS_DEFAULT"
    # train forward_obs bev "$BEST_REWARD_FORWARD" --n_envs "$N_ENVS_BEV" --encoder scratch

    train reverse_obs state "$BEST_REWARD_REVERSE" --n_envs "$N_ENVS_FAST"
    train reverse_obs lidar "$BEST_REWARD_REVERSE" --n_envs "$N_ENVS_FAST" --lidar_beams "$LIDAR_BEAMS_DEFAULT"
    # train reverse_obs bev "$BEST_REWARD_REVERSE" --n_envs "$N_ENVS_BEV" --encoder scratch
else
    log "SKIP Phase 4: set BEST_REWARD_FORWARD and BEST_REWARD_REVERSE first"
fi

# ---------------------------------------------------------------------------
# EVAL: completed phases
# ---------------------------------------------------------------------------
log "=== EVAL ==="

eval_ forward state dense
eval_ forward lidar dense --lidar_beams "$LIDAR_BEAMS_DEFAULT"
eval_ forward bev dense --encoder scratch
eval_ forward bev dense --encoder ae_frozen
eval_ forward bev dense --encoder ae_unfrozen
eval_ forward bev dense --encoder unet_frozen
eval_ forward bev dense --encoder unet_unfrozen

eval_ reverse state dense
eval_ reverse lidar dense --lidar_beams "$LIDAR_BEAMS_DEFAULT"
eval_ reverse bev dense --encoder scratch
eval_ reverse bev dense --encoder ae_frozen
eval_ reverse bev dense --encoder ae_unfrozen
eval_ reverse bev dense --encoder unet_frozen
eval_ reverse bev dense --encoder unet_unfrozen

eval_ forward lidar dense --lidar_beams 4
eval_ forward lidar dense --lidar_beams 8
eval_ forward lidar dense --lidar_beams 24
eval_ forward lidar dense --lidar_beams 32

if is_set "$BEST_OBS_FORWARD"; then
    obs_args=()
    obs_tag_to_cli "$BEST_OBS_FORWARD" obs_args
    eval_ forward "${obs_args[1]}" dense "${obs_args[@]:2}"
    eval_ forward "${obs_args[1]}" tractor_focus "${obs_args[@]:2}"
    eval_ forward "${obs_args[1]}" multiplicative "${obs_args[@]:2}"
    eval_ forward "${obs_args[1]}" guided "${obs_args[@]:2}"
fi

if is_set "$BEST_OBS_REVERSE"; then
    obs_args=()
    obs_tag_to_cli "$BEST_OBS_REVERSE" obs_args
    eval_ reverse "${obs_args[1]}" dense "${obs_args[@]:2}"
    eval_ reverse "${obs_args[1]}" no_hitch "${obs_args[@]:2}"
    eval_ reverse "${obs_args[1]}" multiplicative "${obs_args[@]:2}"
    eval_ reverse "${obs_args[1]}" guided "${obs_args[@]:2}"
fi

if is_set "$BEST_OBS_REVERSE" && is_set "$BEST_REWARD_REVERSE"; then
    eval_retry reverse "$BEST_OBS_REVERSE" "$BEST_REWARD_REVERSE"
fi

if is_set "$BEST_REWARD_FORWARD" && is_set "$BEST_REWARD_REVERSE"; then
    eval_ forward_obs state "$BEST_REWARD_FORWARD"
    eval_ forward_obs lidar "$BEST_REWARD_FORWARD" --lidar_beams "$LIDAR_BEAMS_DEFAULT"
    eval_ forward_obs bev "$BEST_REWARD_FORWARD" --encoder scratch
    eval_ reverse_obs state "$BEST_REWARD_REVERSE"
    eval_ reverse_obs lidar "$BEST_REWARD_REVERSE" --lidar_beams "$LIDAR_BEAMS_DEFAULT"
    eval_ reverse_obs bev "$BEST_REWARD_REVERSE" --encoder scratch
fi

# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------
log "=== PLOTS ==="

plot_curves "$FIGS/forward_obs_ablation.pdf" "Forward Obs Ablation (dense reward)" \
    "State=models/forward/state/dense/logs/evaluations.npz" \
    "Lidar-16=models/forward/lidar/dense/logs/evaluations.npz"

    "BEV CNN=models/forward/bev/dense/logs/evaluations.npz" \
    "BEV AE frozen=models/forward/bev_ae_frozen/dense/logs/evaluations.npz" \
    "BEV AE unfrozen=models/forward/bev_ae_unfrozen/dense/logs/evaluations.npz" \
    "BEV UNet frozen=models/forward/bev_unet_frozen/dense/logs/evaluations.npz" \
    "BEV UNet unfrozen=models/forward/bev_unet_unfrozen/dense/logs/evaluations.npz"

plot_curves "$FIGS/reverse_obs_ablation.pdf" "Reverse Obs Ablation (dense reward)" \
    "State=models/reverse/state/dense/logs/evaluations.npz" \
    "Lidar-16=models/reverse/lidar/dense/logs/evaluations.npz"
    # "BEV CNN=models/reverse/bev/dense/logs/evaluations.npz" \
    # "BEV AE frozen=models/reverse/bev_ae_frozen/dense/logs/evaluations.npz" \
    # "BEV AE unfrozen=models/reverse/bev_ae_unfrozen/dense/logs/evaluations.npz" \
    # "BEV UNet frozen=models/reverse/bev_unet_frozen/dense/logs/evaluations.npz" \
    # "BEV UNet unfrozen=models/reverse/bev_unet_unfrozen/dense/logs/evaluations.npz"

plot_curves "$FIGS/lidar_beams_ablation.pdf" "Lidar Beam-Count Ablation (forward, dense)" \
    "4 beams=models/forward/lidar_4/dense/logs/evaluations.npz" \
    "8 beams=models/forward/lidar_8/dense/logs/evaluations.npz" \
    "16 beams=models/forward/lidar/dense/logs/evaluations.npz" \
    "32 beams=models/forward/lidar_32/dense/logs/evaluations.npz"

if is_set "$BEST_OBS_FORWARD"; then
    plot_curves "$FIGS/forward_reward_ablation.pdf" "Forward Reward Ablation ($BEST_OBS_FORWARD)" \
        "Dense=models/forward/${BEST_OBS_FORWARD}/dense/logs/evaluations.npz" \
        "Tractor focus=models/forward/${BEST_OBS_FORWARD}/tractor_focus/logs/evaluations.npz" \
        "Multiplicative=models/forward/${BEST_OBS_FORWARD}/multiplicative/logs/evaluations.npz" \
        "Guided=models/forward/${BEST_OBS_FORWARD}/guided/logs/evaluations.npz"
fi

if is_set "$BEST_OBS_REVERSE"; then
    plot_curves "$FIGS/reverse_reward_ablation.pdf" "Reverse Reward Ablation ($BEST_OBS_REVERSE)" \
        "Dense=models/reverse/${BEST_OBS_REVERSE}/dense/logs/evaluations.npz" \
        "No hitch=models/reverse/${BEST_OBS_REVERSE}/no_hitch/logs/evaluations.npz" \
        "Multiplicative=models/reverse/${BEST_OBS_REVERSE}/multiplicative/logs/evaluations.npz" \
        "Guided=models/reverse/${BEST_OBS_REVERSE}/guided/logs/evaluations.npz"
fi

if is_set "$BEST_OBS_REVERSE" && is_set "$BEST_REWARD_REVERSE"; then
    plot_curves "$FIGS/failure_replay_curve.pdf" "Failure Replay Curriculum ($BEST_OBS_REVERSE, $BEST_REWARD_REVERSE)" \
        "Random reset=models/reverse/${BEST_OBS_REVERSE}/${BEST_REWARD_REVERSE}/logs/evaluations.npz" \
        "Failure replay=models/reverse/${BEST_OBS_REVERSE}/${BEST_REWARD_REVERSE}_retry/logs/evaluations.npz"
fi

# ---------------------------------------------------------------------------
# BENCHMARKS
# ---------------------------------------------------------------------------
if is_set "$BEST_OBS_FORWARD" && is_set "$BEST_REWARD_FORWARD"; then
    benchmark_task \
        forward \
        td3,fpp,pid,mpc \
        "$(model_path_for forward "$BEST_OBS_FORWARD" "$BEST_REWARD_FORWARD")" \
        results/benchmark_forward
fi

if is_set "$BEST_OBS_REVERSE" && is_set "$BEST_REWARD_REVERSE"; then
    benchmark_task \
        reverse \
        td3,fpp_rev,pid_rev,mpc_rev \
        "$(model_path_for reverse "$BEST_OBS_REVERSE" "$BEST_REWARD_REVERSE")" \
        results/benchmark_reverse
fi

log "=== ALL DONE ==="
