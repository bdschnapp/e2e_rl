"""Reward composition — pure ports of the e2e_rl get_reward bodies, selected by
Config (direction x vehicle_kind x mode) instead of a class hierarchy.

Fixes baked in:
  * B: the multiplicative reward keeps an additive +0.5*clip(|xd|) floor (was the
       lone fully-multiplicative reward; the floor prevents the suicidal/stop-early
       collapse).
  * C: the proximity penalty is passed in already computed with the derived
       threshold (see proximity.py).

Reward terms (per-step, non-terminal):

  forward trailer multiplicative:   4*pt*ht + 0.5*clip(xd,0,1)        - prox
  reverse trailer multiplicative:   5*pt*ht + 0.5*clip(-xd,0,1)       - prox
  forward tractor-only mult:        4*pt    + 0.5*clip(xd,0,1)        - prox
  reverse tractor-only mult:        5*pt    + 0.5*clip(-xd,0,1)       - prox
    pt = exp(-(|e| + 0.5|e_th| [+ 0.75|e_t| + 0.5|e_th_t| for trailer]) * tightness)
    ht = exp(-1.5|hitch|) fwd, exp(-2.0|hitch|) rev  (trailer only)

Terminal: fwd +/-100; rev +200/-500.
"""

from __future__ import annotations

import numpy as np

from ..config import Direction, VehicleKind


def terminal_reward(cfg, success: bool) -> float:
    # Direction defaults; overridable via RewardConfig.terminal_success/_fail so a
    # crash/jackknife can be penalised harder than a deliberate stop (see config).
    if cfg.direction == Direction.REVERSE:
        succ, fail = 200.0, -500.0
    else:
        succ, fail = 100.0, -100.0
    rc = cfg.reward
    if rc.terminal_success is not None:
        succ = rc.terminal_success
    if rc.terminal_fail is not None:
        fail = rc.terminal_fail
    return succ if success else fail


def _path_term(cfg, e_y, e_psi, e_y_t, e_psi_t) -> float:
    tight = cfg.reward.path_tightness
    if cfg.is_tractor_only:
        s = abs(e_y) + 0.5 * abs(e_psi)
    else:
        s = abs(e_y) + 0.5 * abs(e_psi) + 0.75 * abs(e_y_t) + 0.5 * abs(e_psi_t)
    return float(np.exp(-tight * s))


def running_reward(cfg, *, e_y, e_psi, e_y_t=0.0, e_psi_t=0.0, hitch=0.0, xd,
                   proximity=0.0) -> float:
    """Per-step (non-terminal) reward for the configured mode/direction/kind."""
    mode = cfg.reward.mode
    reverse = cfg.direction == Direction.REVERSE
    tractor_only = cfg.is_tractor_only

    if not reverse:
        progress = 0.5 * xd
        tractor_pen = e_y ** 2 + e_psi ** 2
        trailer_pen = (0.5 * e_y_t) ** 2 + (0.5 * e_psi_t) ** 2
        if mode == "dense":
            base = progress - tractor_pen - (0.0 if tractor_only else trailer_pen)
            return base - proximity
        if mode == "tractor_focus":
            return progress - tractor_pen - proximity
        if mode == "multiplicative":
            pt = _path_term(cfg, e_y, e_psi, e_y_t, e_psi_t)
            scale = 4.0
            val = scale * pt
            if not tractor_only:
                val *= float(np.exp(-1.5 * abs(hitch)))
            if cfg.reward.multiplicative_floor:
                val += 0.5 * np.clip(xd, 0.0, 1.0)
            return float(val - proximity)
        raise ValueError(f"forward reward mode {mode!r} not supported")

    # reverse
    path_pen = (e_y ** 2 + 0.5 * e_psi ** 2
                + (0.0 if tractor_only else (0.5 * e_y_t ** 2 + 0.25 * e_psi_t ** 2)))
    jackknife_pen = 0.0 if tractor_only else 10.0 * max(0.0, abs(hitch) - 0.4) ** 2
    reverse_reward = 3.0 * np.clip(-xd, 0.0, 1.0)
    if mode in ("dense",):
        base = reverse_reward - path_pen - jackknife_pen
        return float(base - proximity)
    if mode == "corridor":
        # For deliberately un-trackable sharp corners: an ADDITIVE, BOUNDED
        # center-seeking bonus (not a factor, not a quadratic penalty) so it is
        # steep/centering near e=0 (gentle turns) yet harmlessly fades to 0 when
        # the centerline is unreachable (sharp turns) -- there progress + the
        # anti-jackknife term keep the gradient, and the collision terminal (-500)
        # enforces staying in the lane. NO proximity term (it ramps to -5 near the
        # wall and would re-punish the unavoidable close pass through the corner).
        center = 3.0 * _path_term(cfg, e_y, e_psi, e_y_t, e_psi_t)
        return float(reverse_reward + center - jackknife_pen)
    if mode == "no_hitch":
        return float(reverse_reward - path_pen - proximity)
    if mode == "multiplicative":
        pt = _path_term(cfg, e_y, e_psi, e_y_t, e_psi_t)
        val = 5.0 * pt
        if not tractor_only:
            val *= float(np.exp(-2.0 * abs(hitch)))
        if cfg.reward.multiplicative_floor:
            val += 0.5 * np.clip(-xd, 0.0, 1.0)
        return float(val - proximity)
    raise ValueError(f"reverse reward mode {mode!r} not supported")
