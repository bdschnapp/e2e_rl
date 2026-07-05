"""Portable policy export: SB3 model -> .pth (torch state_dict) + meta.json.

The bridge loads the .pth + JSON (no SB3 .zip cloudpickle, no env class paths, no
numpy-version coupling). meta records an obs_spec descriptor (not an env class) so
deployment never imports a specific env class.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


def obs_spec_from_cfg(cfg) -> dict:
    return {
        "kind": "lidar" if cfg.obs.lidar_beams > 0 else "state",
        "state_dim": 5 if cfg.is_tractor_only else 8,
        "lidar_beams": int(cfg.obs.lidar_beams),
        "reverse": bool(cfg.is_reverse),
        "tractor_only": bool(cfg.is_tractor_only),
    }


def export_policy(model, out_path, cfg, *, action_mode: str | None = None) -> dict:
    """Write <out_path>.pth + <out_path>.meta.json. Returns the meta dict."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.policy.state_dict(), str(out_path) + ".pth")

    actor_keys = [k for k in model.policy.state_dict() if k.startswith("actor.mu")]
    last_w = model.policy.state_dict()[sorted(actor_keys)[-2]] if actor_keys else None
    action_dim = int(last_w.shape[0]) if last_w is not None else None

    # policy_kwargs as primitives (net_arch etc.); MlpPolicy has no class refs.
    pk = dict(getattr(model, "policy_kwargs", {}) or {})

    meta = {
        "obs_spec": obs_spec_from_cfg(cfg),
        "action_dim": action_dim,
        "action_mode": action_mode or cfg.action.mode.value,
        "policy_kwargs": pk,
        "stop_threshold": cfg.action.stop_threshold,
        "max_steer_rate": float(__import__("numpy").deg2rad(cfg.action.steering_action_deg)),
    }
    with open(str(out_path) + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return meta
