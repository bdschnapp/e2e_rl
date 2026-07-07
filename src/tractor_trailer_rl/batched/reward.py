"""Batched reward (vectorised port of ``reward.composer``).

Returns a shape-(N,) running reward for the configured direction/kind/mode. The
per-env scalar ``if`` on the reward mode is a single Python branch (the mode is
constant across the batch), so no masking is needed there.
"""

from __future__ import annotations

from .backend import xp
from ..config import Direction


def terminal_reward(cfg, success_mask):
    """(N,) terminal reward: success value where success_mask, else fail value."""
    if cfg.direction == Direction.REVERSE:
        succ, fail = 200.0, -500.0
    else:
        succ, fail = 100.0, -100.0
    rc = cfg.reward
    if rc.terminal_success is not None:
        succ = rc.terminal_success
    if rc.terminal_fail is not None:
        fail = rc.terminal_fail
    return xp.where(success_mask, succ, fail)


def _path_term(cfg, e_y, e_psi, e_y_t, e_psi_t):
    tight = cfg.reward.path_tightness
    if cfg.is_tractor_only:
        s = xp.abs(e_y) + 0.5 * xp.abs(e_psi)
    else:
        s = (xp.abs(e_y) + 0.5 * xp.abs(e_psi)
             + 0.75 * xp.abs(e_y_t) + 0.5 * xp.abs(e_psi_t))
    return xp.exp(-tight * s)


def running_reward(cfg, *, e_y, e_psi, e_y_t=0.0, e_psi_t=0.0, hitch=0.0, xd, proximity=0.0,
                   steer_diff=None, alpha=None):
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
            val = 4.0 * pt
            if not tractor_only:
                val = val * xp.exp(-1.5 * xp.abs(hitch))
            if cfg.reward.multiplicative_floor:
                val = val + 0.5 * xp.clip(xd, 0.0, 1.0)
            return val - proximity
        if mode == "guided":
            # PP-imitation warmup (alpha 1) decaying to the MULTIPLICATIVE reward (alpha 0)
            pt = _path_term(cfg, e_y, e_psi, e_y_t, e_psi_t)
            mult = 4.0 * pt
            if not tractor_only:
                mult = mult * xp.exp(-1.5 * xp.abs(hitch))
            if cfg.reward.multiplicative_floor:
                mult = mult + 0.5 * xp.clip(xd, 0.0, 1.0)
            # Stage-1 clone: PURE PP imitation (no task/progress term). alpha=1 => clone PP;
            # alpha decay blends in the multiplicative task reward to fine-tune PAST PP.
            guide = -5.0 * steer_diff ** 2
            return alpha * guide + (1.0 - alpha) * mult - proximity
        raise ValueError(f"forward reward mode {mode!r} not supported")

    # reverse
    path_pen = (e_y ** 2 + 0.5 * e_psi ** 2
                + (0.0 if tractor_only else (0.5 * e_y_t ** 2 + 0.25 * e_psi_t ** 2)))
    jackknife_pen = 0.0 if tractor_only else 10.0 * xp.clip(xp.abs(hitch) - 0.4, 0.0, None) ** 2
    reverse_reward = 3.0 * xp.clip(-xd, 0.0, 1.0)
    if mode == "dense":
        return reverse_reward - path_pen - jackknife_pen - proximity
    if mode == "corridor":
        center = 3.0 * _path_term(cfg, e_y, e_psi, e_y_t, e_psi_t)
        return reverse_reward + center - jackknife_pen
    if mode == "no_hitch":
        return reverse_reward - path_pen - proximity
    if mode == "multiplicative":
        pt = _path_term(cfg, e_y, e_psi, e_y_t, e_psi_t)
        val = 5.0 * pt
        if not tractor_only:
            val = val * xp.exp(-2.0 * xp.abs(hitch))
        if cfg.reward.multiplicative_floor:
            val = val + 0.5 * xp.clip(-xd, 0.0, 1.0)
        return val - proximity
    if mode == "guided":
        # PP-imitation warmup decaying to the MULTIPLICATIVE reverse reward
        pt = _path_term(cfg, e_y, e_psi, e_y_t, e_psi_t)
        mult = 5.0 * pt
        if not tractor_only:
            mult = mult * xp.exp(-2.0 * xp.abs(hitch))
        if cfg.reward.multiplicative_floor:
            mult = mult + 0.5 * xp.clip(-xd, 0.0, 1.0)
        # Stage-1 clone: PURE PP imitation (no progress/jackknife task terms). alpha=1 =>
        # clone PP; alpha decay blends in the multiplicative task reward to fine-tune PAST PP.
        guide = -5.0 * steer_diff ** 2
        return alpha * guide + (1.0 - alpha) * mult - proximity
    raise ValueError(f"reverse reward mode {mode!r} not supported")
