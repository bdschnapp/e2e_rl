# Thesis Context (AI Reference — Read This Instead of Raw .tex Files)

**Title**: Autonomous Articulated Vehicle Control via Deep Reinforcement Learning
**Degree**: MASc, Mechanical & Mechatronics Engineering, University of Waterloo
**Status**: Simulation only. No Gazebo, ROS2, Agilex hardware, or LIDAR — all future work.

## Prior Work (Foundations)
| Report | Authors | Result |
|---|---|---|
| ME 780 (Fall 2024) | Tymkow & Schnapp | DDPG: 0.505m CTE, DT: 1.717m CTE, PID: 0.784m CTE (single vehicle) |
| ECE 682 | Schnapp | LQR for vehicle bicycle model + tractor-trailer kinematics |

## Implementation Stack
- **Algorithm**: TD3 (Stable Baselines3) — NOT DDPG
- **Simulation**: Custom Gymnasium + Pygame 2D environment — NOT Gazebo
- **Observation**: state-only, state+lidar, or `{vector, image}` BEV dict observations
- **Action**: `[δ̇, v]` — steering rate ∈ ±15°/s, speed ∈ [0, 2] m/s
- **Collision**: Pygame mask-based; episode ends on collision or |γ| > 90°
- **Tasks**: forward/reverse `LineFollowing`, forward/reverse `ObstacleAvoidance`

## Key Notation
| Symbol | Meaning |
|---|---|
| δ | Steering angle |
| δ̇ | Steering rate (control input / action dim 0) |
| γ | Hitch articulation angle (tractor–trailer) |
| e_y | Cross-track error |
| e_ψ | Heading error |
| v | Speed (action dim 1) |
| l_f, l_r | Front/rear axle distance to CoM |
| l_t | Trailer axle length |
| C_f, C_r | Front/rear tire cornering stiffness |
| ψ | Yaw angle |

## Vehicle Parameters (Tesla Model S — used in ME780, ECE682, and this thesis)
`m=1500kg, Iz=3000 kg·m², Cf=Cr=80000 N/rad, lf=1.2m, lr=1.6m, Cd=0.208, A=2.4m², dt=0.1s`

## Experiment Baselines & Constraints
- All experiments: 2D Pygame simulation
- Observation ablations: `state_only`, `lidar_8`, `lidar_16`, `lidar_24`, `cnn_bev`, `ae_bev`, `unet_bev`
- Primary metrics: mean absolute CTE (tractor + trailer separately), max |γ|, completion, collision/jackknife rate, inference latency
- Reference: ME780 DDPG = 0.505m, PID = 0.784m (single vehicle, not tractor-trailer)
- Comparisons: tuned pure pursuit, PID, MPC, and ME780 baselines where appropriate

## Current Experiment Scripts
- `train_phase1.py`: trains matched TD3 models across observation variants
- `eval_phase1.py`: evaluates saved Phase 1 models over repeated episodes
- `benchmark.py`: runs TD3 and classical controllers on fixed scenario sets
- `tune_controllers.py`: tunes forward/reverse classical controllers with differential evolution
- `scripts/generate_test_scenarios.py`: creates reusable benchmark scenarios

## Chapter Map — When to Read Each File
| Chapter | Summary file | .tex file | Read .tex only when… |
|---|---|---|---|
| Ch1 Introduction | `summaries/ch01.md` | `chapters/01_introduction.tex` | Changing contributions or scope |
| Ch2 Literature | `summaries/ch02.md` | `chapters/02_literature_review.tex` | Adding/removing related work citations |
| Ch3 Modeling | `summaries/ch03.md` | `chapters/03_modeling.tex` | Vehicle model equations change |
| Ch4 Perception | `summaries/ch04.md` | `chapters/04_perception.tex` | CNN/BEV/autoencoder design changes |
| Ch5 RL Framework | `summaries/ch05.md` | `chapters/05_rl_framework.tex` | Reward function or algorithm changes |
| Ch6 Simulation | `summaries/ch06.md` | `chapters/06_simulation.tex` | Environment, tasks, or params change |
| Ch7 Results | `summaries/ch07.md` | `chapters/07_results.tex` | Adding/updating experiment results |
| Ch8 Conclusion | `summaries/ch08.md` | `chapters/08_conclusion.tex` | Scope, limitations, future work changes |

## Pending Code Changes Needing Thesis Updates
See `thesis/pending_updates.md` — check at session start.
