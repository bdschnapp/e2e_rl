"""Action modes (first-class; replaces _patch_variable_speed_action + stop_signal_patch).

  FIXED_SPEED    1-D [steer_rate]; speed = the per-episode fixed_speed_command.
  VARIABLE_SPEED 2-D [steer_rate, velocity]; velocity bounded (reverse mirrored).
  STOP_SIGNAL    2-D [steer_rate, stop in [-1,1]]; constant speed. stop>threshold
                 => stop + terminate + -stop_penalty; else drive normally. The
                 stop is kept dormant in no-obstacle training by a high threshold
                 + low action noise (NOT a per-step penalty, which would mask the
                 positive driving reward).

Each handler exposes: action_space(cfg), noise_sigma(cfg, n), and resolve(action,
cfg, episode_speed) -> ActionResult(steer_rate, velocity_cmd, stop_now).
The env owns stepping; the handler owns the semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from gymnasium import spaces

from ..config import ActionMode, Direction


@dataclass(frozen=True)
class ActionResult:
    steer_rate: float
    velocity_cmd: float       # signed target speed handed to the vehicle PID
    stop_now: bool


def _max_steer_rate(cfg) -> float:
    return float(np.deg2rad(cfg.action.steering_action_deg))


def action_space(cfg) -> spaces.Box:
    r = _max_steer_rate(cfg)
    m = cfg.action.mode
    if m == ActionMode.FIXED_SPEED:
        return spaces.Box(low=np.array([-r], np.float32),
                          high=np.array([r], np.float32), dtype=np.float32)
    if m == ActionMode.VARIABLE_SPEED:
        if cfg.direction == Direction.REVERSE:
            lo = np.array([-r, -cfg.action.v_max], np.float32)
            hi = np.array([r, -cfg.action.v_min], np.float32)
        else:
            lo = np.array([-r, cfg.action.v_min], np.float32)
            hi = np.array([r, cfg.action.v_max], np.float32)
        return spaces.Box(low=lo, high=hi, dtype=np.float32)
    if m == ActionMode.STOP_SIGNAL:
        return spaces.Box(low=np.array([-r, -1.0], np.float32),
                          high=np.array([r, 1.0], np.float32), dtype=np.float32)
    if m == ActionMode.DISCRETE:
        # flat Discrete(n_steer_bins * 2) so it works for both DQN (Discrete-only)
        # and PPO. index = steer_bin * 2 + speed_bin; speed_bin 0=go, 1=stop.
        return spaces.Discrete(cfg.action.n_steer_bins * 2)
    raise ValueError(m)


def noise_sigma(cfg, n_actions: int) -> np.ndarray:
    sigma = np.full(n_actions, cfg.action.steer_noise_sigma, dtype=np.float32)
    if cfg.action.mode == ActionMode.STOP_SIGNAL and n_actions > 1:
        sigma[1:] = cfg.action.stop_noise_sigma
    elif n_actions > 1:
        sigma[1:] = 0.5  # variable-speed velocity exploration (e2e_rl default)
    return sigma


def action_dim(cfg) -> int:
    if cfg.action.mode in (ActionMode.FIXED_SPEED, ActionMode.DISCRETE):
        return 1
    return 2


def resolve(action, cfg, episode_speed: float) -> ActionResult:
    """Map a raw policy action to (steer_rate, velocity_cmd, stop_now, penalty).

    episode_speed: the signed per-episode constant speed (forward +, reverse -)
    used by FIXED_SPEED and STOP_SIGNAL.
    """
    a = np.asarray(action, dtype=np.float32).flatten()
    steer_rate = float(a[0])
    m = cfg.action.mode
    if m == ActionMode.FIXED_SPEED:
        return ActionResult(steer_rate, float(episode_speed), False)
    if m == ActionMode.VARIABLE_SPEED:
        return ActionResult(steer_rate, float(a[1]), False)
    # STOP_SIGNAL: drive at constant speed unless the policy fires the stop
    # (stop > threshold), which zeroes speed + terminates with -stop_penalty.
    # No per-step penalty on stop intent — the stop is kept dormant by the high
    # threshold + low action noise, and the positive driving reward is left intact.
    stop = float(a[1]) if a.size > 1 else -1.0
    if stop > cfg.action.stop_threshold:
        return ActionResult(steer_rate, 0.0, True)
    return ActionResult(steer_rate, float(episode_speed), False)
