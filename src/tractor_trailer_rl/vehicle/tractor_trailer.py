"""Tractor + simple kinematic trailer. Ported from e2e_rl
VehicleModels/tractor_trailer.py (logic unchanged; params from VehicleConfig)."""

from __future__ import annotations

import numpy as np

from .bicycle import StateSpaceVehicleModel


class TrailerModel:
    """Kinematic trailer: hitch at tractor rear axle, axle L behind the hitch."""

    def __init__(self, length, dt=0.1):
        self.L = length
        self.dt = dt
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.yaw_rate = 0.0

    def reset(self, tractor_x, tractor_y, tractor_yaw):
        self.yaw = tractor_yaw
        self.x = tractor_x - self.L * np.cos(self.yaw)
        self.y = tractor_y - self.L * np.sin(self.yaw)
        self.yaw_rate = 0.0

    def update(self, tractor_x, tractor_y, tractor_yaw, tractor_speed, tractor_lr):
        hitch_x = tractor_x - tractor_lr * np.cos(tractor_yaw)
        hitch_y = tractor_y - tractor_lr * np.sin(tractor_yaw)
        angle_diff = tractor_yaw - self.yaw
        self.yaw_rate = (tractor_speed / self.L) * np.sin(angle_diff)
        self.yaw += self.yaw_rate * self.dt
        self.x = hitch_x - self.L * np.cos(self.yaw)
        self.y = hitch_y - self.L * np.sin(self.yaw)
        return np.array([self.x, self.y, self.yaw, self.yaw_rate], dtype=float)


class StateSpaceTractorTrailer(StateSpaceVehicleModel):
    def __init__(self, vehicle_config=None, *, args=None, trailer_length=None):
        super().__init__(vehicle_config, args=args)
        if trailer_length is None:
            trailer_length = (vehicle_config.trailer_length_m
                              if vehicle_config is not None else 3.0)
        self.trailer = TrailerModel(trailer_length, self.dt)

    def reset(self, xd, x=0.0, y=0.0, p=0.0):
        super().reset(xd, x, y, p)
        self.trailer.reset(self.x, self.y, self.p)

    def loop(self, action):
        tractor_state = super().loop(action)
        tx, ty, t_vx, t_vy, t_yaw, t_yaw_rate, t_steer = tractor_state
        trailer_state = self.trailer.update(tx, ty, t_yaw, t_vx, self.lr)
        return np.concatenate((tractor_state, trailer_state), axis=0)
