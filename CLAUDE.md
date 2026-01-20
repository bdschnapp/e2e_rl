# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

End-to-End Reinforcement Learning (E2E-RL) research project for autonomous tractor-trailer vehicle control. Implements Gymnasium-based RL environments with Stable Baselines3 TD3 agents combined with classical control methods.

## Running Training

```bash
# Line-following task (lane tracking with tractor-trailer)
python line_following_main.py

# Distribution center task (warehouse parking/navigation)
python distribution_center_main.py
```

Both scripts train TD3 agents with 200k timesteps. Rendering is enabled by default (`render_mode='human'`).

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
├── DistributionCenter.py  # Warehouse parking task
└── LatticePlanner.py   # Ring-sweep motion planning

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

**Action Space**: `[steering_rate, speed]` - steering rate in rad/s (±15°/s), speed in m/s (0-2)

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
