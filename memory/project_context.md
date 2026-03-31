---
name: Project Context - E2E RL Thesis
description: MASc thesis on articulated tractor-trailer control with TD3 in a custom Gymnasium/Pygame simulator.
type: project
---

Implemented:
- TD3 training for `forward`, `reverse`, `forward_obs`, `reverse_obs`
- Observation modes: `state`, `lidar`, `bev`
- Classical baselines: pure pursuit, PID, MPC
- Benchmarking on fixed scenario sets

Experimental framing:
- No-obstacle tasks: fixed-speed path tracking
- Obstacle tasks: variable-speed navigation

Do not claim:
- ROS2 deployment
- sim-to-real transfer
- hardware validation
