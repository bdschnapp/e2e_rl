"""tractor_trailer_rl — clean-room tractor-trailer RL package.

Self-contained successor to e2e_rl: config-driven envs, a pure observation
builder shared by training and deployment, TD3 training, and portable export.
No monkeypatching required by consumers.
"""

from .config import (
    Config, Direction, VehicleKind, ActionMode, LookaheadMode,
    VehicleConfig, WorldConfig, ObsConfig, LookaheadConfig, ActionConfig,
    PathConfig, SpeedRandomConfig, SpawnConfig, RewardConfig, TargetSpeedConfig,
    truck_config, lab_config, lab_chicane_config,
)
from .observation.builder import build_observation, EgoState, TrailerState, ObsResult
from .observation.occupancy import build_occupancy_grid, GridMeta
from .observation.layouts import observation_bounds, state_bounds
from .vehicle.bicycle import StateSpaceVehicleModel
from .vehicle.tractor_trailer import StateSpaceTractorTrailer, TrailerModel

__all__ = [
    "Config", "Direction", "VehicleKind", "ActionMode", "LookaheadMode",
    "VehicleConfig", "WorldConfig", "ObsConfig", "LookaheadConfig", "ActionConfig",
    "PathConfig", "SpeedRandomConfig", "SpawnConfig", "RewardConfig", "TargetSpeedConfig",
    "truck_config", "lab_config",
    "build_observation", "EgoState", "TrailerState", "ObsResult",
    "build_occupancy_grid", "GridMeta", "observation_bounds", "state_bounds",
    "StateSpaceVehicleModel", "StateSpaceTractorTrailer", "TrailerModel",
]
