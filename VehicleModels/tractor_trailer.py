import numpy as np
from VehicleModels.vehicle_model import StateSpaceVehicleModel


class TrailerModel:
    """
    Simple kinematic trailer model coupled to a tractor.
    Assumes hitch at tractor rear axle and trailer axle behind hitch.
    """
    def __init__(self, length, dt=0.1):
        self.L = length                # Distance from hitch to trailer axle (m)
        self.dt = dt                   # Time step (s)
        self.x = 0.0                   # Trailer axle x-position (m)
        self.y = 0.0                   # Trailer axle y-position (m)
        self.yaw = 0.0                 # Trailer orientation (rad)
        self.yaw_rate = 0.0            # Trailer yaw rate (rad/s)

    def reset(self, tractor_x, tractor_y, tractor_yaw):
        # Initialize trailer so its axle is behind the hitch by L along tractor yaw
        self.yaw = tractor_yaw
        self.x = tractor_x - self.L * np.cos(self.yaw)
        self.y = tractor_y - self.L * np.sin(self.yaw)
        self.yaw_rate = 0.0

    def update(self, tractor_x, tractor_y, tractor_yaw, tractor_speed, tractor_lr):
        # Compute hitch point at rear axle of tractor
        hitch_x = tractor_x - tractor_lr * np.cos(tractor_yaw)
        hitch_y = tractor_y - tractor_lr * np.sin(tractor_yaw)
        # Yaw difference between tractor hitch orientation and trailer
        angle_diff = tractor_yaw - self.yaw
        # Kinematic yaw rate of trailer
        self.yaw_rate = (tractor_speed / self.L) * np.sin(angle_diff)
        # Update trailer yaw
        self.yaw += self.yaw_rate * self.dt
        # Update trailer axle position behind hitch by L along trailer yaw
        self.x = hitch_x - self.L * np.cos(self.yaw)
        self.y = hitch_y - self.L * np.sin(self.yaw)
        return np.array([self.x, self.y, self.yaw, self.yaw_rate], dtype=float)


class StateSpaceTractorTrailer(StateSpaceVehicleModel):
    """
    Combines a StateSpaceVehicleModel tractor with a simple kinematic trailer.
    State is extended to include trailer axle position and yaw.
    """
    def __init__(self, args=None, trailer_length=3.0):
        super().__init__(args)
        # Initialize trailer model
        self.trailer = TrailerModel(trailer_length, self.dt)

    def reset(self, xd, x=0, y=0, p=0):
        # Reset tractor state
        super().reset(xd, x, y, p)
        # Reset trailer based on tractor's current pose
        self.trailer.reset(self.x, self.y, self.p)

    def loop(self, action):
        # Update tractor using parent method
        tractor_state = super().loop(action)
        tx, ty, t_vx, t_vy, t_yaw, t_yaw_rate, t_steer = tractor_state
        # Update trailer kinematics
        trailer_state = self.trailer.update(tx, ty, t_yaw, t_vx, self.lr)
        # Concatenate tractor and trailer states
        return np.concatenate((tractor_state, trailer_state), axis=0)
