# Thesis Context

Scope:
- MASc thesis on articulated tractor-trailer control with TD3 in a custom 2D Gymnasium/Pygame simulator.
- Thesis claims are simulation-only.

Core setup:
- Tasks: `forward`, `reverse`, `forward_obs`, `reverse_obs`
- Observations: `state`, `lidar`, `bev`
- No-obstacle tasks: fixed-speed path tracking
- Obstacle tasks: variable-speed end-to-end navigation
- Baselines: TD3, pure pursuit, PID, MPC

Key notation:
- `delta`: steering angle
- `delta_dot`: steering rate
- `gamma`: hitch angle
- `e_y`: cross-track error
- `e_psi`: heading error

Vehicle parameters:
- Tesla Model S defaults in simulation
- `m=1500`, `Iz=3000`, `Cf=Cr=80000`, `lf=1.2`, `lr=1.6`, `Cd=0.208`, `A=2.4`, `dt=0.1`

Read order:
1. This file
2. `thesis/summaries/ch0N.md`
3. Specific `thesis/chapters/*.tex` only when editing

Scripts tied to thesis experiments:
- `train.py`
- `tune_controllers.py`
- `benchmark.py`
- `scripts/generate_test_scenarios.py`

Maintenance:
- `thesis/pending_updates.md` tracks code edits that may require thesis updates.
