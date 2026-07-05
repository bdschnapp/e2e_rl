# tractor_trailer_rl — GPU-parallel (CuPy) port

This is a copy of `tractor_trailer_rl` with an added `batched/` subpackage that
runs **N environments at once on the GPU** (CuPy) or CPU (numpy), for fast,
multi-seed ablation studies. The original scalar env is unchanged and is used as
the **parity oracle**.

Backend selection: `TTRL_BACKEND=cupy` (GPU) or `TTRL_BACKEND=numpy` (CPU), or
`tractor_trailer_rl.batched.backend.set_backend(...)`. The same batched code runs
on either — cupy is a drop-in for the numpy API used here.

## What was done (the five scalar blockers)

The scalar env is single-env with Python-float state; parallelism came only from
SB3 `SubprocVecEnv` over process copies. The batched env adds a leading `(N,)`
axis to every state array and replaces the five things that could not run batched
on a GPU:

| # | Scalar blocker | Batched replacement | File |
|---|---|---|---|
| 1 | per-step `scipy.signal.cont2discrete` ZOH | batched matrix-exponential ZOH (Padé-13 scaling-squaring) | `batched/expm.py` |
| 2 | per-reset `scipy.spatial.cKDTree` occupancy grid | analytic distance-to-centerline (no grid) + on-device path pool | `batched/corridor.py`, `batched/env.py` |
| 3 | sequential per-beam lidar ray-march (`break`) | vectorised fixed-length march, first-hit via `argmax` | `batched/lidar.py` |
| 4 | Python-loop collision / proximity | batched body-point gather + corridor test | `batched/corridor.py` |
| 5 | scalar termination / stop-signal / spawn branches | boolean masks + internal auto-reset | `batched/env.py` |

`batched/vehicle.py` is the vectorised tractor + kinematic trailer; `batched/
geometry.py` and `batched/reward.py` are the vectorised path-error/curvature and
reward composer. `batched/td3.py` is a compact GPU-resident TD3 (SB3 cannot consume
a single batched-GPU env); it bridges to the CuPy env via DLPack (zero-copy).

## Validation (parity vs the scalar oracle)

`pytest tests/test_batched_parity.py` (runs on numpy, no GPU needed) — **11 pass**:

- **ZOH == scipy** to ~1e-8 absolute / 1e-12 relative.
- **Vehicle dynamics == scalar** to ~1e-14 over a 200-step rollout.
- **Env observation == scalar EXACTLY (0.0)**; **reward == scalar to ~1e-14**
  wherever the proximity term is zero — i.e. the whole dynamics/geometry/
  observation/reward pipeline is bit-parity.
- **numpy and cupy backends produce identical results** (0.0 obs diff).

Two terms are *intentionally* not bit-identical because they replace a rasterised
grid with analytic geometry (both are approximations; the analytic form has no
rasterisation aliasing):

- **proximity penalty**: differs by ~0.1–0.5 only near the lane edge (0 when the
  rig is well inside the lane); irrelevant at termination (terminal reward used).
- **lidar**: analytic corridor ray-march within ~1 cell of the grid raycast on
  average; **collision decisions agree with the grid on 97–98.5%** of random
  states, the rest within one cell of the boundary.

## Throughput (RTX 4080 SUPER, `lab_config`)

Pure environment stepping, transitions/second (higher is better):

| Observation | Batched GPU (best N) | Scalar single-core | Scalar SubprocVecEnv |
|---|---|---|---|
| state (no lidar) | **~1.7 M** (N=16384) | ~2.1 k | (reset-bound) |
| lidar-24 | **~64 k** (N=1024–16384) | ~470 | ~300–350 |

Notes / honest caveats:
- The **state-obs env is the clean win** (~1.7M transitions/s): compute-bound,
  scales with N. That's ~50× an *ideal* 16-core scalar baseline (~800× one core).
- The **lidar-obs env (~64k tps)** is bounded by the vectorised ray-march (24
  beams × ~130 steps, no early exit). A fused CuPy `RawKernel` (one thread per
  ray, with early-exit) would close most of the gap to the state env — the clear
  next optimisation.
- The measured `SubprocVecEnv` baseline (~300 tps) is **artificially low**: (a)
  `build_occupancy_grid` calls `cKDTree(..., workers=-1)`, so 16 processes each
  spawn all-core threads → oversubscription (8/16 procs are *slower* than 1); and
  (b) a random policy causes frequent resets, and each scalar reset rebuilds the
  1500×900 occupancy grid. The batched env's **path pool** amortises exactly this
  cost — which is part of the port's contribution — so the ~180× headline is real
  for this workload but the per-step-vectorisation numbers above are the fairer
  apples-to-apples comparison.

Reproduce: `TTRL_BACKEND=cupy python scripts/benchmark_batched.py`

### numpy vs cupy — same code, import swap only (`scripts/compare_numpy_cupy.py`)

The batched env is written against `backend.xp`, so this is literally the identical
vectorised code with numpy or cupy behind it (`TTRL_BACKEND=numpy|cupy`). Same run,
RTX 4080 SUPER vs CPU, transitions/second:

| obs | N | numpy (CPU) | cupy (GPU) | GPU speedup |
|---|---|---|---|---|
| state | 256 | 87,060 | 30,821 | 0.35× (CPU wins) |
| state | 1024 | 81,668 | 122,464 | 1.5× |
| state | 4096 | 72,200 | 479,141 | 6.6× |
| lidar24 | 256 | 2,676 | 24,815 | 9.3× |
| lidar24 | 1024 | 2,334 | 71,394 | 30.6× |
| lidar24 | 4096 | 1,005 | 62,601 | 62.3× |

Reading: there is a **crossover** — at small N the GPU's per-kernel launch overhead
loses to CPU numpy (state N=256: CPU 2.8× faster); as N grows the GPU pulls ahead
(state N=4096: 6.6×). For the compute-heavy **lidar** obs the GPU wins at every N
(9–62×) and the gap widens with N, because each step does far more work per env
(24 beams × ~130 march steps). This is the expected batched-accelerator profile and
argues for large N (≥1024) and the lidar observation to exploit the GPU.

## How to run

```bash
pip install cupy-cuda12x                      # GPU backend (matches CUDA 12.x)

# smoke-train a policy on GPU (forward lab, state+lidar)
TTRL_BACKEND=cupy python scripts/train_batched_td3.py --steps 300000 --n_envs 1024

# multi-seed ablation cells with mean +/- 95% CI (fulfils Ch.5 "Statistical Rigor")
TTRL_BACKEND=cupy python scripts/run_ablations_gpu.py \
    --cells forward:lidar24:multiplicative reverse:lidar24:multiplicative \
    --seeds 0 1 2 --steps 300000 --n_envs 1024 --out results_gpu.csv

# parity + throughput
pytest tests/test_batched_parity.py -q
TTRL_BACKEND=cupy python scripts/benchmark_batched.py
```

## Trainer findings (what makes the GPU TD3 converge)

Validated on the **deployed best config — `lidar_24` + `multiplicative`** (from
`Electrans_project/lab_models/models/{forward,reverse}/lidar_24/multiplicative/`):

- **Reward conditioning matters most.** The `dense` reward has *unbounded*
  quadratic penalties (returns in the −1000s) and destabilises a from-scratch TD3
  (return diverged). The `multiplicative` reward is *bounded* (~[0, 4.5]/step); the
  same TD3 then trains cleanly: forward return **−43 → +1349**, completion rising
  through **56% @ 120k steps** (still climbing; ~100% expected with the SB3-scale
  ~200–300k budget). Use the multiplicative reward (it is also what is deployed).
- **Moderate N, not huge N, for off-policy TD3.** At matched update budget (UTD≈1),
  **N=16 learns fast (~945 return @ 96k)** while **N=256 crawls (~4 @ 102k)** —
  large batches make replay samples correlated and tank sample-efficiency. Keep
  `n_envs≈16–64` with `updates_per_step≈n_envs`.
- **Large N DOES work for TD3 if you also scale the update minibatch.** The initial
  N=256 failure was `batch_size` left at 256 (GPU-starved, sample-inefficient). With
  **N=256, minibatch 4096, lr≈2e-3** the run converges *faster in wall-clock* than
  N=16 (reached higher return in ~70s vs ~243s) — a batch-4096 gradient step costs
  ~the same as batch-256 on the GPU but does ~16× the work. So the recipe to exploit
  the batched env with off-policy TD3 is **big env-batch + big update-minibatch +
  higher LR**, not naive large-N with a tiny minibatch. (lr=3e-3 oscillates; 1–2e-3
  is steadier.)
- **To push N into the thousands, use on-policy PPO** (Brax / IsaacGym / rl_games).
  The thesis studies TD3, so large-minibatch TD3 at N≈256 is the sweet spot; the GPU
  also lets you run many (config×seed) jobs concurrently for the sweep.
- **Completion ceiling is a task property, not a trainer bug.** Forward lidar-24
  multiplicative plateaus at ~56% completion at **lab scale** for BOTH N=16 and
  N=256 — the lab paths have ~58% sharp/corner kinds that understeer at the small
  wheelbase (the `lab_deployment_findings.md` curvature-to-wheelbase effect,
  reproduced in sim). The SB3 tables' 100% completion was at **truck scale** (gentle
  paths vs. a large wheelbase). Match the scale when comparing to a known number.
- **Completion metric:** `env.step` now returns `info["success"]` captured *before*
  auto-reset (reading `env.success` afterwards is zero — the same auto-reset bug as
  the e2e_rl B1/B2 issue). The eval horizon must exceed one episode length (lab
  goal ≈127 m at low speed needs ~600+ steps).

## Validated ablation recipe (full-size, Stage-1)

Forward `lidar_24` + `multiplicative`, **full-size (truck) scale, fixed speed,
straight+gentle paths** → **100% completion in ~127 s** (500k steps) with:

    --preset truck --fixed_speed --mild_paths \
    --n_envs 256 --updates_per_step 32 --batch_size 4096 --lr 2e-3

i.e. **big env-batch + big update-minibatch + higher LR** converges to the known
SB3 result in ~2 min/run. This is the locked recipe for the multi-seed ablation
campaign. (Tesla vehicle params remain a placeholder — swap in real full-size
shunt-truck params before the final run; AgileX scale is Ch.6 only. See
`thesis/ABLATION_STUDY_DESIGN.md`.)

## Algorithm axis via SB3 (vetted, drop-in)

`batched/sb3_adapter.py` exposes the batched GPU env as an SB3 `VecEnv`, so SB3's
vetted **TD3 / SAC / PPO / DQN** run on it unchanged — the fair way to do the
algorithm ablation (the hand-rolled `batched/td3.py` stays as the throughput demo).
Run via `scripts/train_sb3.py --algo {td3,sac,ppo}`.

Head-to-head (truck fwd, lidar-24, multiplicative, fixed speed, gentle paths, 500k
steps, N=256, 3 jobs concurrent so ~1.5× contention-inflated):

| Algorithm | Wall-clock | Completion |
|---|---|---|
| **SB3 PPO** | **34 s** | 1.00 |
| custom TD3 (all-GPU, solo) | 127 s | 1.00 |
| SB3 TD3 | 220 s | 1.00 |
| SB3 SAC | 343 s | 1.00 |

Findings: (1) all algorithms drop in and solve Stage-1. (2) **PPO is ~6–10× faster**
because it is on-policy — no replay buffer, big rollout batches — so it *actually
exploits* the 1.7M-steps/s env (validates the "use PPO for N≫100" point). (3) On the
easy Stage-1-forward task **every algorithm saturates at completion 1.00**, so final
completion doesn't discriminate them — the algorithm ablation must be judged on
**sample-efficiency (steps-to-threshold), reverse, and the hard curriculum stages.**

## Status / next steps

- **Done & validated:** batched env (all 5 blockers), numpy/cupy parity, GPU TD3
  trains end-to-end to 100% completion (~2 min/run), throughput benchmark.
- **To dial in (needs GPU time):** TD3 hyperparameters + step budget so the GPU
  trainer reaches the same policy quality as the SB3 path, then run the full
  multi-seed ablation campaign (`run_ablations_gpu.py`) to regenerate the thesis
  ablation tables with confidence intervals.
- **Optimisation:** fused CuPy `RawKernel` lidar ray-march (biggest remaining
  throughput lever); optional vectorised path-generation (the pool already removes
  it from the hot path).
- **Obstacles:** the `STOP_SIGNAL` action mode is supported by the batched env;
  obstacle rasterisation into the corridor test is the remaining piece for the
  stop-gate obstacle study.
```
