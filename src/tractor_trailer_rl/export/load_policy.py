"""Load a portable policy (.pth + meta.json) for inference, with no env and no
SB3 .zip. Rebuilds the SB3 TD3 policy from the recorded obs_spec + policy_kwargs
against tiny dummy spaces, loads the state_dict, and returns a predict callable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _spaces_from_meta(meta):
    import gymnasium as gym
    spec = meta["obs_spec"]
    obs_dim = spec["state_dim"] + spec["lidar_beams"]
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    n = int(meta["action_dim"])
    r = float(meta["max_steer_rate"])
    if n == 1:
        low, high = np.array([-r], np.float32), np.array([r], np.float32)
    else:
        low, high = np.array([-r, -1.0], np.float32), np.array([r, 1.0], np.float32)
    act_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
    return obs_space, act_space


def _make_dummy_env(obs_space, act_space):
    import gymnasium as gym

    class _DummyEnv(gym.Env):
        """Exposes only the spaces; never actually stepped (used to construct TD3
        for state_dict loading)."""
        def __init__(self):
            super().__init__()
            self.observation_space = obs_space
            self.action_space = act_space

        def reset(self, *, seed=None, options=None):
            return self.observation_space.sample() * 0, {}

        def step(self, action):
            return self.observation_space.sample() * 0, 0.0, True, False, {}

    return _DummyEnv()


def load_policy(path, device="cpu"):
    """`path` is the prefix used at export (so <path>.pth and <path>.meta.json
    exist). Returns (predict_fn, meta) where predict_fn(obs)->action (deterministic)."""
    import torch
    from stable_baselines3 import TD3

    path = str(path)
    with open(path + ".meta.json") as f:
        meta = json.load(f)
    obs_space, act_space = _spaces_from_meta(meta)
    model = TD3("MlpPolicy", _make_dummy_env(obs_space, act_space),
                policy_kwargs=meta.get("policy_kwargs") or {},
                device=device, buffer_size=1, learning_starts=0)
    state = torch.load(path + ".pth", map_location=device)
    model.policy.load_state_dict(state)

    def predict_fn(obs):
        a, _ = model.predict(np.asarray(obs, np.float32), deterministic=True)
        return a
    return predict_fn, meta
