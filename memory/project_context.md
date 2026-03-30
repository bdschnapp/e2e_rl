---
name: Project Context - E2E RL Thesis
description: The thesis is a MASc on RL for tractor-trailer control. Prior work includes ME780 (DDPG/DT for single vehicle path following) and ECE682 (LQR for vehicle bicycle model with tractor-trailer kinematics). Actual implementation uses TD3 (not DDPG), Pygame/Gymnasium environments (not Gazebo), no real hardware yet.
type: project
---

The thesis work builds on two prior course projects:
- **ME 780 Final Report** (Fall 2024, Ryan Tymkow & Benjamin Schnapp): Applied DDPG and Decision Transformer RL for single vehicle path following. Used bicycle model with Pacejka tire model, Gymnasium environment, Pygame visualization. DDPG achieved 0.505m mean CTE, DT achieved 1.717m, PID baseline 0.784m.
- **ECE 682 Final Report** (Benjamin Schnapp): LQR control of vehicle bicycle model, including kinematic tractor-trailer equations. Used same Tesla Model S parameters, same bicycle model dynamics.

**Actual implementation (what exists in the repo):**
- Environment: Custom Gymnasium environments with Pygame rendering (NOT Gazebo)
- Algorithm: TD3 from Stable Baselines3 (NOT DDPG)
- Vehicle: StateSpaceTractorTrailer (dynamic tractor + kinematic trailer with hitch kinematics)
- Observation: Dictionary with `vector` (state vars: steering angle, hitch angle, tracking errors) and `image` (84x84 grayscale BEV)
- BEV camera: Bird's-eye view rendered via Pygame, configurable anchor at tractor_rear_axle
- Collision detection: Pygame mask-based
- Tasks: LineFollowing (lane tracking) and ObstacleAvoidance
- No Gazebo, no ROS2/Autoware, no Agilex hardware, no LIDAR, no Basler cameras
- Tesla Model S vehicle parameters (same as prior work)

**Why:** The thesis incorrectly contained Gazebo, ROS2, Agilex hardware, LIDAR sensor descriptions - none of which are implemented. These were corrected in March 2026 to reflect the Pygame/Gymnasium simulation only.

**How to apply:** When discussing the simulation environment, always describe the custom Pygame/Gymnasium environment. Gazebo/ROS2/Agilex/LIDAR are future work items, not current implementations.
