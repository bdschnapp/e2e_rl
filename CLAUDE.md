# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

End-to-End Reinforcement Learning (E2E-RL) research project for autonomous tractor-trailer vehicle control. Implements Gymnasium-based RL environments with Stable Baselines3 TD3 agents combined with classical control methods.

## Running Training

```bash
python train.py --scenario forward --obs state
python tune_controllers.py --controllers fpp,pid,mpc
python scripts/generate_test_scenarios.py
python benchmark.py --task forward --controllers td3,fpp,pid,mpc
```

## Dependencies

Core dependencies (no requirements.txt exists):
- stable-baselines3
- gymnasium
- numpy
- scipy
- torch (CUDA)
- pygame
- osqp

## Architecture

```
Environments/           # Gymnasium RL environments
├── TractorTrailer.py   # Base env with vehicle dynamics + pygame rendering
├── LineFollowing.py    # Lane-following task (extends TractorTrailer)
├── ObstacleAvoidance.py # Obstacle-aware task variants
└── wrappers.py         # Training/evaluation wrappers

VehicleModels/          # Vehicle dynamics
├── vehicle_model.py    # Base kinematic/dynamic models
└── tractor_trailer.py  # State-space tractor-trailer with hitch kinematics

Models/                 # Neural network components for SB3
├── CNNFeatureExtractor.py  # CNN feature extractor wrapping ImageEncoder
├── UNetFeatureExtractor.py # UNet-based feature extractor
├── AutoEncoder.py      # CNN autoencoder for representation learning
└── UNet.py             # UNet encoder implementation

controllers/            # Classical control
├── mpc.py              # Model Predictive Control (OSQP-based)
└── mpc_traj_gen.py     # Trajectory generation for MPC

e2erl_utils/config.py   # Global configuration (vehicle params, scales, BEV settings)
```

## Key Concepts

**Action Space**:
- no-obstacle tasks: steering-rate control with fixed speed
- obstacle tasks: `[steering_rate, speed]`

**Observation Space**: Dictionary with:
- `vector`: State variables (steering angle, hitch angle, tracking errors)
- `image`: 84x84 grayscale BEV (Bird's Eye View) image

**Vehicle Model**: `StateSpaceTractorTrailer` combines dynamic tractor model with kinematic trailer. Trailer state computed via hitch kinematics from tractor rear axle.

**BEV Camera**: Configurable anchor point (tractor_rear_axle by default), renders vehicle and environment from above. Settings in `e2erl_utils/config.py`.

**Collision Detection**: Pygame mask-based. Episode terminates on collision or jackknife (hitch angle > 90°).

## Configuration

All tunable parameters are in `e2erl_utils/config.py`:
- Vehicle parameters (Tesla Model S defaults)
- Action/observation scaling factors
- BEV camera settings (anchor, offset, zoom)
- Lane geometry parameters

## Thesis Integration

### Default: Use Summaries, Not Raw .tex Files

**Do NOT read `.tex` files by default.** Use this three-level escalation instead:

1. **Level 1 — always available**: `THESIS_CONTEXT.md` (root). Contains scope, notation, parameters, chapter map, baseline numbers. Read this for any thesis-related question.
2. **Level 2 — chapter detail**: `thesis/summaries/ch0N.md`. One per chapter, ~20 lines. Read before editing a chapter.
3. **Level 3 — edit target only**: Read the specific `.tex` file (or the relevant section with a line offset) only when you are about to edit it.

**Never read a `.tex` file just for context.** Never read multiple chapters in one session unless each is being edited.

### When to Update the Thesis

Only update thesis content when a code change affects one of:
- **Notation or equations** (vehicle model, reward, observation)
- **Architecture or design** (CNN structure, action space, environment behaviour)
- **Experiment results** (new benchmark numbers, training curves)
- **Scope or assumptions** (what is/isn't implemented)

Cosmetic refactors, bug fixes that don't change behaviour, and internal implementation details do not need thesis updates.

### Code → Thesis Chapter (use Level 2 summary first)

| Code area | Summary | .tex (edit only) |
|---|---|---|
| `VehicleModels/` | `summaries/ch03.md` | `chapters/03_modeling.tex` |
| `Environments/TractorTrailer.py` | `summaries/ch06.md` | `chapters/06_simulation.tex` |
| `Environments/LineFollowing.py` (reward) | `summaries/ch05.md` + `ch06.md` | `chapters/05_rl_framework.tex` |
| `Environments/ObstacleAvoidance.py` | `summaries/ch06.md` | `chapters/06_simulation.tex` |
| `Models/CNN*.py`, `Models/UNet*.py` | `summaries/ch04.md` | `chapters/04_perception.tex` |
| `Models/AutoEncoder.py` | `summaries/ch04.md` | `chapters/04_perception.tex` |
| `controllers/` | `summaries/ch06.md` | `chapters/06_simulation.tex` |
| `e2erl_utils/config.py` | `summaries/ch06.md` | `chapters/06_simulation.tex` |
| Reward functions | `summaries/ch05.md` | `chapters/05_rl_framework.tex` |
| Experiment results | `summaries/ch07.md` | `chapters/07_results.tex` |

### Pending Thesis Updates

`thesis/pending_updates.md` — auto-populated by a hook when code files are edited. Check at session start; clear entries once the thesis has been updated.
