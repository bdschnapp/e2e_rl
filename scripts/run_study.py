#!/usr/bin/env python3
"""
run_study.py — unattended, resource-aware experiment runner for the e2e_rl thesis study.

Wraps the existing train.py / eval.py / benchmark.py / generate_results_tables.py
(no edits to them). Runs a declarative job queue sequentially, idempotently, and
politely on a SHARED workstation: it sizes --n_envs from *measured CPU idle* (not the
misleading load-average), defers launching when the box is busy, nices children, caps
their BLAS/torch threads, keeps GPU (bev) jobs serialized, and regenerates the thesis
tables as results land.

Usage
-----
    venv/bin/python scripts/run_study.py --plan              # dry-run: show resolved queue
    venv/bin/python scripts/run_study.py --only fwd_lidar4   # run a single named job (validation)
    venv/bin/python scripts/run_study.py --profile balanced  # run the whole queue
    venv/bin/python scripts/run_study.py --stop              # stop a detached runner cleanly

State / logs live under the gitignored logs/study/:
    queue.json  progress.csv  runner.log  runner.pid  jobs/<name>.{train,eval}.log
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO = Path(__file__).resolve().parent.parent          # .../e2e_rl
PY = REPO / "venv" / "bin" / "python"
STUDY_DIR = REPO / "logs" / "study"
JOBS_DIR = STUDY_DIR / "jobs"
QUEUE_JSON = STUDY_DIR / "queue.json"
PROGRESS_CSV = STUDY_DIR / "progress.csv"
RUNNER_LOG = STUDY_DIR / "runner.log"
RUNNER_PID = STUDY_DIR / "runner.pid"
GEN_TABLES = REPO / "thesis" / "scripts" / "generate_results_tables.py"

# --------------------------------------------------------------------------- #
# Resource profiles (governor is keyed off MEASURED cpu-idle, not load average)
# --------------------------------------------------------------------------- #
PROFILES = {
    "conservative": dict(min_free_cores=6, margin=6, cap_cpu=8,  cap_gpu=8,
                         nice=15, threads=1, ram_floor_gb=6, vram_floor_mib=6000,
                         backoff0=300, backoff_max=1800),
    "balanced":     dict(min_free_cores=4, margin=3, cap_cpu=16, cap_gpu=12,
                         nice=10, threads=2, ram_floor_gb=4, vram_floor_mib=4500,
                         backoff0=120, backoff_max=900),
    "aggressive":   dict(min_free_cores=2, margin=1, cap_cpu=16, cap_gpu=12,
                         nice=5,  threads=4, ram_floor_gb=3, vram_floor_mib=3500,
                         backoff0=60,  backoff_max=600),
}
DEFER_TIMEOUT_S = 8 * 3600   # give up deferring a single job after 8h

# --------------------------------------------------------------------------- #
# The job queue (1 seed each; missing thesis-cell experiments).
#   kind=train_eval -> train (if model missing) then eval (if aggregate missing)
#   resource class is derived: obs==bev -> gpu, else cpu.
# Defaults: timesteps=100000, eval_episodes=30 (match the existing populated cells).
# --------------------------------------------------------------------------- #
DEFAULT_TIMESTEPS = 100_000
DEFAULT_EVAL_EPISODES = 30

# Feature-scale schedule for the scaled_cnn (state->vision swap) runs at 300k steps
# (matches the prior working forward run, which reached image=1/state=0 by 300k):
#   vision fades in 30k-120k, state holds, then fades to 0 over 140k-230k,
#   leaving ~70k steps fully vision-only.
SCALED_CNN_SCHEDULE = [
    "--image_scale_start", "0.0", "--image_scale_end", "1.0",
    "--image_scale_warmup_steps", "30000", "--image_scale_ramp_steps", "90000",
    "--state_scale_start", "1.0", "--state_scale_end", "0.0",
    "--state_scale_warmup_steps", "140000", "--state_scale_ramp_steps", "90000",
]

JOBS = [
    # --- quick CPU job first (fast end-to-end validation) ---
    # lidar-beams ablation: 4-beam variant (forward, dense). obs_tag -> lidar_4
    dict(name="fwd_lidar4",        kind="train_eval", scenario="forward",
         obs="lidar", lidar_beams=4, reward="dense"),

    # --- obs-ablation REVERSE: the 4 missing BEV encoder variants (dense, to
    #     match the rest of that fixed-reward observation-comparison table) ---
    dict(name="rev_bev_ae_frozen",    kind="train_eval", scenario="reverse",
         obs="bev", encoder="ae_frozen",    reward="dense"),
    dict(name="rev_bev_ae_unfrozen",  kind="train_eval", scenario="reverse",
         obs="bev", encoder="ae_unfrozen",  reward="dense"),
    dict(name="rev_bev_unet_frozen",  kind="train_eval", scenario="reverse",
         obs="bev", encoder="unet_frozen",  reward="dense"),
    dict(name="rev_bev_unet_unfrozen",kind="train_eval", scenario="reverse",
         obs="bev", encoder="unet_unfrozen",reward="dense"),

    # --- obs-ablation FORWARD: missing unet_unfrozen cell (dense) ---
    dict(name="fwd_bev_unet_unfrozen",kind="train_eval", scenario="forward",
         obs="bev", encoder="unet_unfrozen",reward="dense"),

    # --- obstacle REVERSE row at the deployed `multiplicative` reward
    #     (dense reverse is degenerate; see run_ablations.sh note). Fills the
    #     reverse obstacle table at state / lidar(16) / bev(scratch). ---
    dict(name="rev_obs_state_mult",   kind="train_eval", scenario="reverse_obs",
         obs="state",                 reward="multiplicative"),
    dict(name="rev_obs_lidar_mult",   kind="train_eval", scenario="reverse_obs",
         obs="lidar", lidar_beams=16, reward="multiplicative"),
    dict(name="rev_obs_bev_mult",     kind="train_eval", scenario="reverse_obs",
         obs="bev", encoder="scratch",reward="multiplicative"),

    # --- failure-replay curriculum on the deployed reverse config
    #     (lidar_24, multiplicative). Eval writes to <reward>_retry/. ---
    dict(name="rev_failreplay_mult",  kind="train_eval", scenario="reverse",
         obs="lidar", lidar_beams=24, reward="multiplicative", retry_on_failure=True),

    # --- gradual state->vision swap (scaled_cnn): "start on state, fade vision in,
    #     then fade state out to vision-only." 300k schedule (matches the prior
    #     working forward run, which reached image=1/state=0 by 300k). See
    #     SCALED_CNN_SCHEDULE above for the timeline. ---
    dict(name="fwd_bev_scaled_cnn", kind="train_eval", scenario="forward",
         obs="bev", encoder="scaled_cnn", reward="dense", timesteps=300_000,
         train_args=SCALED_CNN_SCHEDULE),
    dict(name="rev_bev_scaled_cnn", kind="train_eval", scenario="reverse",
         obs="bev", encoder="scaled_cnn", reward="dense", timesteps=300_000,
         train_args=SCALED_CNN_SCHEDULE),

    # --- retrain the bad 16-beam forward lidar checkpoint (was 0% completion,
    #     an outlier vs 4/8/24/32-beam which all complete 100%). ---
    dict(name="fwd_lidar16_retrain", kind="train_eval", scenario="forward",
         obs="lidar", lidar_beams=16, reward="dense"),
]


# --------------------------------------------------------------------------- #
# Path / idempotency helpers (mirror train.py:576-584 and eval.py:54-59,247 EXACTLY)
# --------------------------------------------------------------------------- #
def obs_tag(obs: str, encoder: str = "scratch", lidar_beams: int = 16) -> str:
    if obs == "bev" and encoder != "scratch":
        return f"{obs}_{encoder}"
    if obs == "lidar" and lidar_beams != 16:
        return f"lidar_{lidar_beams}"
    return obs


def _tag(job: dict) -> str:
    return obs_tag(job["obs"], job.get("encoder", "scratch"), job.get("lidar_beams", 16))


def resource_class(job: dict) -> str:
    return "gpu" if job.get("obs") == "bev" else "cpu"


def model_zip(job: dict) -> Path:
    retry = "_retry" if job.get("retry_on_failure") else ""
    return REPO / f"models/{job['scenario']}/{_tag(job)}/{job['reward']}{retry}/best_model.zip"


def aggregate_csv(job: dict) -> Path:
    # eval.py's default label has NO _retry suffix; for retry jobs we pass an
    # explicit --output_csv into <reward>_retry/ (see eval_cmd), so the path
    # carries the suffix either way.
    retry = "_retry" if job.get("retry_on_failure") else ""
    return REPO / f"results/{job['scenario']}/{_tag(job)}/{job['reward']}{retry}/aggregate.csv"


def summary_csv(job: dict) -> Path:
    out = REPO / job.get("output", "results")
    return out / f"summary_{job['task']}.csv"


def job_fully_done(job: dict) -> bool:
    if job["kind"] == "benchmark":
        return summary_csv(job).exists()
    return aggregate_csv(job).exists()


def needs_train(job: dict) -> bool:
    return job["kind"] == "train_eval" and not model_zip(job).exists()


def needs_eval(job: dict) -> bool:
    return job["kind"] in ("train_eval", "eval_only") and not aggregate_csv(job).exists()


# --------------------------------------------------------------------------- #
# Command builders
# --------------------------------------------------------------------------- #
def _common_obs_flags(job: dict) -> list[str]:
    flags = []
    if job.get("encoder", "scratch") != "scratch":
        flags += ["--encoder", job["encoder"]]
    if job.get("lidar_beams", 16) != 16:
        flags += ["--lidar_beams", str(job["lidar_beams"])]
    return flags


def train_cmd(job: dict, n_envs: int) -> list[str]:
    cmd = [str(PY), "train.py",
           "--scenario", job["scenario"], "--obs", job["obs"],
           "--reward", job["reward"], "--n_envs", str(n_envs),
           "--timesteps", str(job.get("timesteps", DEFAULT_TIMESTEPS)),
           "--eval_episodes", str(job.get("eval_episodes", DEFAULT_EVAL_EPISODES))]
    cmd += _common_obs_flags(job)
    cmd += ["--device", "auto" if resource_class(job) == "gpu" else "cpu"]
    if job.get("retry_on_failure"):
        cmd += ["--retry_on_failure"]
    cmd += [str(a) for a in job.get("train_args", [])]   # extra train flags (e.g. feature-scale schedule)
    return cmd


def eval_cmd(job: dict) -> list[str]:
    cmd = [str(PY), "eval.py",
           "--scenario", job["scenario"], "--obs", job["obs"],
           "--reward", job["reward"],
           "--episodes", str(job.get("eval_episodes", DEFAULT_EVAL_EPISODES))]
    cmd += _common_obs_flags(job)
    # Always pass --model explicitly: eval.py's _default_model_path() does NOT
    # thread lidar_beams into the obs_tag, so a --lidar_beams!=16 eval would
    # otherwise load the wrong (16-beam) checkpoint and crash on a space mismatch.
    cmd += ["--model", str(model_zip(job))]
    if job.get("retry_on_failure"):
        # The retry curriculum is a TRAINING strategy; the retry-trained model is
        # evaluated on the STANDARD protocol (no --retry_on_failure at eval) so it
        # is comparable to the random-reset baseline. We only (a) point --model at
        # the _retry checkpoint and (b) route outputs into the _retry/ results dir
        # (eval.py's auto label drops the _retry suffix). NOTE: passing
        # --retry_on_failure to eval.py wraps the env in RetryOnFailureWrapper,
        # which doesn't proxy get_vehicle_errors() and crashes the metrics logger.
        out_dir = REPO / f"results/{job['scenario']}/{_tag(job)}/{job['reward']}_retry"
        cmd += ["--output_csv", str(out_dir / "episodes.csv")]
    return cmd


def benchmark_cmd(job: dict) -> list[str]:
    cmd = [str(PY), "benchmark.py",
           "--task", job["task"], "--controllers", job["controllers"],
           "--obs", job["obs"], "--reward", job["reward"],
           "--output", job.get("output", "results")]
    cmd += _common_obs_flags(job)
    if "model" in job:
        cmd += ["--model", str(job["model"])]
    if job.get("tuned_params"):
        cmd += ["--tuned_params", str(job["tuned_params"])]
    return cmd


# --------------------------------------------------------------------------- #
# Telemetry
# --------------------------------------------------------------------------- #
def cpu_idle_cores(sample_s: float = 1.0) -> float:
    """Free cores measured from /proc/stat (idle+iowait), NOT load average."""
    def snap():
        with open("/proc/stat") as f:
            v = list(map(int, f.readline().split()[1:]))
        idle = v[3] + (v[4] if len(v) > 4 else 0)   # idle + iowait
        return idle, sum(v)
    i0, t0 = snap()
    time.sleep(sample_s)
    i1, t1 = snap()
    dt = t1 - t0
    frac = (i1 - i0) / dt if dt > 0 else 0.0
    return frac * (os.cpu_count() or 1)


def mem_available_gb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    return 0.0


def gpu_status() -> tuple[int | None, int | None]:
    """(free_mib, util_pct) via nvidia-smi; (None, None) if unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        line = out.stdout.strip().splitlines()[0]
        free_mib, util = (x.strip() for x in line.split(","))
        return int(free_mib), int(util)
    except Exception:
        return None, None


def choose_n_envs(rclass: str, prof: dict) -> int:
    free = cpu_idle_cores()
    cap = prof["cap_gpu"] if rclass == "gpu" else prof["cap_cpu"]
    return max(2, min(int(free - prof["margin"]), cap))


# --------------------------------------------------------------------------- #
# Logging + state
# --------------------------------------------------------------------------- #
def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def log(msg: str) -> None:
    line = f"[{now_str()}] {msg}"
    print(line, flush=True)
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUNNER_LOG, "a") as f:
        f.write(line + "\n")


def load_state() -> dict:
    if QUEUE_JSON.exists():
        try:
            return json.loads(QUEUE_JSON.read_text())
        except Exception:
            return {}
    return {}


def save_state(state: dict) -> None:
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    tmp = QUEUE_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, QUEUE_JSON)
    # compact progress.csv for quick external reads
    cols = ["name", "kind", "rclass", "status", "n_envs", "elapsed_s", "rc", "defer_reason"]
    with open(PROGRESS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for name, s in state.items():
            w.writerow({**{"name": name}, **s})


def mark(state: dict, job: dict, status: str, **kw) -> None:
    s = state.setdefault(job["name"], {})
    s["kind"] = job["kind"]
    s["rclass"] = resource_class(job)
    s["status"] = status
    s.update(kw)
    save_state(state)


# --------------------------------------------------------------------------- #
# Child execution
# --------------------------------------------------------------------------- #
_current_child: subprocess.Popen | None = None


def _spawn(cmd: list[str], logfile: Path, prof: dict, rclass: str) -> subprocess.Popen:
    env = os.environ.copy()
    if rclass == "cpu":
        for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                  "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[v] = str(prof["threads"])
    nice = prof["nice"]

    def _preexec():
        os.setsid()           # own process group -> killpg tears down the whole tree
        os.nice(nice)

    logfile.parent.mkdir(parents=True, exist_ok=True)
    lf = open(logfile, "a")
    lf.write(f"\n{'='*70}\n[{now_str()}] $ {' '.join(cmd)}\n{'='*70}\n")
    lf.flush()
    return subprocess.Popen(cmd, cwd=str(REPO), env=env,
                            stdout=lf, stderr=subprocess.STDOUT, preexec_fn=_preexec)


def run_and_wait(cmd: list[str], logfile: Path, prof: dict, rclass: str) -> int:
    global _current_child
    p = _spawn(cmd, logfile, prof, rclass)
    _current_child = p
    rc = p.wait()
    _current_child = None
    return rc


# --------------------------------------------------------------------------- #
# Governor: wait until the machine has room for this job
# --------------------------------------------------------------------------- #
class DeferTimeout(Exception):
    pass


def wait_for_slot(job: dict, prof: dict, state: dict) -> None:
    rclass = resource_class(job)
    backoff = prof["backoff0"]
    waited = 0
    while True:
        free = cpu_idle_cores()
        ram = mem_available_gb()
        reasons = []
        if free < prof["min_free_cores"]:
            reasons.append(f"cores={free:.1f}")
        if ram < prof["ram_floor_gb"]:
            reasons.append(f"ram={ram:.1f}G")
        if rclass == "gpu":
            gfree, _ = gpu_status()
            if gfree is not None and gfree < prof["vram_floor_mib"]:
                reasons.append(f"vram={gfree}MiB")
        if not reasons:
            return
        if waited >= DEFER_TIMEOUT_S:
            raise DeferTimeout(",".join(reasons))
        mark(state, job, "deferred", defer_reason=",".join(reasons))
        log(f"defer {job['name']} ({','.join(reasons)}); sleep {backoff}s")
        time.sleep(backoff)
        waited += backoff
        backoff = min(backoff * 2, prof["backoff_max"])


# --------------------------------------------------------------------------- #
# Table regeneration (cheap, pure-python; refreshes thesis/tables/*.tex)
# --------------------------------------------------------------------------- #
def regen_tables() -> None:
    try:
        subprocess.run([str(PY), str(GEN_TABLES)], cwd=str(REPO),
                       capture_output=True, text=True, timeout=300)
        log("regenerated thesis tables")
    except Exception as e:
        log(f"WARN table regen failed: {e}")


# --------------------------------------------------------------------------- #
# Run one job end-to-end
# --------------------------------------------------------------------------- #
def run_job(job: dict, prof: dict, state: dict) -> str:
    if job_fully_done(job):
        mark(state, job, "skipped")
        log(f"skip {job['name']} (outputs exist)")
        return "skipped"

    wait_for_slot(job, prof, state)
    rclass = resource_class(job)
    n_envs = choose_n_envs(rclass, prof)
    t0 = time.time()
    mark(state, job, "running", n_envs=n_envs, started=now_str(), defer_reason="")
    log(f"START {job['name']} [{rclass}] n_envs={n_envs}")

    ok = True
    rc = 0
    if needs_train(job):
        rc = run_and_wait(train_cmd(job, n_envs), JOBS_DIR / f"{job['name']}.train.log", prof, rclass)
        if rc != 0 or not model_zip(job).exists():
            ok = False
            log(f"FAIL train {job['name']} rc={rc} model={model_zip(job).exists()}")

    if ok and job["kind"] == "benchmark":
        rc = run_and_wait(benchmark_cmd(job), JOBS_DIR / f"{job['name']}.bench.log", prof, rclass)
        ok = rc == 0
    elif ok and needs_eval(job):
        # eval is light; keep it off the GPU contention path by treating as cpu nice
        rc = run_and_wait(eval_cmd(job), JOBS_DIR / f"{job['name']}.eval.log", prof, rclass)
        if rc != 0 or not aggregate_csv(job).exists():
            ok = False
            log(f"FAIL eval {job['name']} rc={rc}")

    elapsed = int(time.time() - t0)
    status = "done" if ok else "failed"
    mark(state, job, status, ended=now_str(), elapsed_s=elapsed, rc=rc)
    log(f"{status.upper()} {job['name']} in {elapsed}s")
    regen_tables()
    return status


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def reconcile(state: dict) -> None:
    """On startup, sync recorded state with what's actually on disk."""
    for job in JOBS:
        if job_fully_done(job):
            mark(state, job, "skipped")


def cmd_plan() -> None:
    print(f"Repo: {REPO}")
    print(f"Python: {PY}  (exists={PY.exists()})")
    gfree, gutil = gpu_status()
    print(f"Telemetry: free_cores={cpu_idle_cores():.1f}/{os.cpu_count()}  "
          f"ram_avail={mem_available_gb():.1f}G  gpu_free={gfree}MiB util={gutil}%")
    print(f"\n{'JOB':<24}{'CLASS':<6}{'STATUS':<10}{'ACTION'}")
    print("-" * 78)
    for job in JOBS:
        if job_fully_done(job):
            action = "skip (done)"
        elif needs_train(job):
            action = "train + eval"
        elif needs_eval(job):
            action = "eval only (model exists)"
        else:
            action = "run"
        status = "done" if job_fully_done(job) else "pending"
        print(f"{job['name']:<24}{resource_class(job):<6}{status:<10}{action}")
        print(f"{'':<24}model  -> {model_zip(job).relative_to(REPO)}")
        print(f"{'':<24}result -> {aggregate_csv(job).relative_to(REPO)}")
    n_gpu = sum(resource_class(j) == "gpu" for j in JOBS if not job_fully_done(j))
    n_cpu = sum(resource_class(j) == "cpu" for j in JOBS if not job_fully_done(j))
    print("-" * 78)
    print(f"To run: {n_gpu} gpu(bev) + {n_cpu} cpu jobs.  "
          f"Rough estimate: ~{n_gpu*3 + n_cpu*0.6:.0f}h (gpu jobs serialize).")


def cmd_stop() -> None:
    if not RUNNER_PID.exists():
        print("No runner.pid found.")
        return
    pid = int(RUNNER_PID.read_text().strip())
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        print(f"Sent SIGTERM to runner pgid of pid {pid}.")
    except ProcessLookupError:
        print(f"Runner pid {pid} not running.")
    RUNNER_PID.unlink(missing_ok=True)


def _install_signal_handlers(state: dict) -> None:
    def handler(signum, frame):
        log(f"received signal {signum}; tearing down current child and exiting")
        if _current_child is not None:
            try:
                os.killpg(os.getpgid(_current_child.pid), signal.SIGTERM)
            except Exception:
                pass
        # revert any running job to pending so a restart re-runs it
        for name, s in state.items():
            if s.get("status") == "running":
                s["status"] = "pending"
        save_state(state)
        sys.exit(143)
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def cmd_run(profile: str, only: str | None) -> None:
    prof = PROFILES[profile]
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    RUNNER_PID.write_text(str(os.getpid()))
    state = load_state()
    _install_signal_handlers(state)
    reconcile(state)

    jobs = [j for j in JOBS if (only is None or j["name"] == only)]
    if only and not jobs:
        log(f"no job named {only!r}")
        return
    log(f"runner start: profile={profile} jobs={len(jobs)} "
        f"(only={only}) free_cores={cpu_idle_cores():.1f}")

    counts = {"done": 0, "failed": 0, "skipped": 0}
    for job in jobs:
        try:
            status = run_job(job, prof, state)
        except DeferTimeout as e:
            mark(state, job, "deferred_timeout", defer_reason=str(e))
            log(f"GIVEUP {job['name']} deferred > {DEFER_TIMEOUT_S}s ({e})")
            status = "failed"
        counts[status] = counts.get(status, 0) + 1

    regen_tables()
    log(f"runner done: {counts}")
    RUNNER_PID.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=list(PROFILES), default="balanced")
    ap.add_argument("--plan", action="store_true", help="dry-run: show resolved queue")
    ap.add_argument("--only", default=None, help="run only the named job (validation)")
    ap.add_argument("--stop", action="store_true", help="stop a detached runner")
    args = ap.parse_args()

    if args.plan:
        cmd_plan()
    elif args.stop:
        cmd_stop()
    else:
        cmd_run(args.profile, args.only)


if __name__ == "__main__":
    main()
