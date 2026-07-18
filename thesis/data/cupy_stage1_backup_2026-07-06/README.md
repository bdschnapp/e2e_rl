# CuPy Stage-1 results backup (persisted 2026-07-06)

Snapshot copies of the definitive CuPy ablation artifacts, kept here so they
survive independently of the working dir in tractor_trailer_rl_cupy/.

- `results_stage1_v4/`        — original definitive lidar-24 Stage-1 sweep
                                (curves.csv, summary.csv, manifest.jsonl, thesis_export/).
- `results_stage1_v4_repro/`  — reproducibility rerun of v4 (incl. saved models).
- `classical_baselines_safepool.log` — PP/PID/LQR/MPC on the PP-validated safe
                                pool (shunt_mild_safe.npz), fwd+rev, 1500 steps.
                                Fwd: all 4 = 1.000 compl. Rev: PP 1.000, MPC 0.990,
                                LQR 0.967 (best CTE 0.568), PID 0.898.

Live working copies remain in tractor_trailer_rl_cupy/results_stage1_v4*,
and the in-progress V1 state-obs sweep is in results_stage1_V1/.
