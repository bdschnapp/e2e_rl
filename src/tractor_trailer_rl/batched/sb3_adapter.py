"""Expose the batched GPU env through SB3's VecEnv interface.

SB3 owns vectorisation (it expects a VecEnv of N envs and manages the batch), so
it can't natively drive one big on-GPU batched env. This thin adapter presents the
already-batched env AS a VecEnv, so SB3's vetted TD3 / SAC / PPO / DQN run on it
unchanged (the same pattern envpool/Brax use for SB3). Rollouts happen on the GPU;
SB3 handles the learning. Vetted implementations = a fair, defensible algorithm
ablation (vs. the hand-rolled `batched/td3.py`, which stays as the throughput demo).

Notes:
- obs/reward/done are moved to host numpy (tiny for low-dim state+lidar obs).
- auto-reset is handled by the batched env; we surface `terminal_observation` and
  `TimeLimit.truncated` in per-env infos so SB3 bootstraps correctly on truncation.
- SB3's ReplayBuffer divides buffer_size by n_envs internally, so large n_envs does
  not blow up memory.
"""

from __future__ import annotations

import warnings
import numpy as np

warnings.filterwarnings("ignore")

from stable_baselines3.common.vec_env.base_vec_env import VecEnv  # noqa: E402

from .backend import to_numpy  # noqa: E402
from .env import BatchedLaneFollowingEnv  # noqa: E402


def _obs_to_np(obs):
    """Move a batched obs to host float32. Dict obs (BEV: {vector, image}) is
    converted per key; a plain array is converted directly."""
    if isinstance(obs, dict):
        return {k: to_numpy(v).astype(np.float32) for k, v in obs.items()}
    return to_numpy(obs).astype(np.float32)


def _index_obs(obs, i):
    """Slice env i out of a batched host obs (array or Dict)."""
    if isinstance(obs, dict):
        return {k: v[i] for k, v in obs.items()}
    return obs[i]


def make_batched_env(cfg, num_envs, path_pool_size=4096):
    """Env factory: the obstacle subclass iff ``cfg.obstacle`` is set, else the
    pure lane-following env. Keeps the two paths fully separate so the ablation
    (obstacle=None) and the Track-D obstacle study can both run unchanged."""
    if getattr(cfg, "obstacle", None) is not None:
        from .obstacle_env import BatchedObstacleAvoidanceEnv
        return BatchedObstacleAvoidanceEnv(cfg, num_envs, path_pool_size=path_pool_size)
    return BatchedLaneFollowingEnv(cfg, num_envs, path_pool_size=path_pool_size)


class SB3BatchedVecEnv(VecEnv):
    def __init__(self, cfg, num_envs, path_pool_size=4096):
        self.env = make_batched_env(cfg, num_envs, path_pool_size=path_pool_size)
        super().__init__(num_envs, self.env.single_observation_space,
                         self.env.single_action_space)
        self._actions = None
        self._seed = None

    def reset(self):
        obs = self.env.reset(seed=self._seed)
        return _obs_to_np(obs)

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        obs, rew, term, trunc, info = self.env.step(self._actions)
        obs = _obs_to_np(obs)
        rew = to_numpy(rew).astype(np.float32)
        term = to_numpy(term).astype(bool)
        trunc = to_numpy(trunc).astype(bool)
        dones = term | trunc
        infos = [{} for _ in range(self.num_envs)]
        if dones.any():
            final = info.get("final_observation")
            succ = info.get("success")
            fo = _obs_to_np(final) if final is not None else None
            cause_keys = [k for k in info if k.startswith("cause_")]
            for i in np.nonzero(dones)[0]:
                src = fo if fo is not None else obs
                infos[i]["terminal_observation"] = _index_obs(src, i)
                if trunc[i] and not term[i]:
                    infos[i]["TimeLimit.truncated"] = True
                if succ is not None:
                    infos[i]["is_success"] = bool(succ[i])
                for k in cause_keys:
                    infos[i][k] = bool(info[k][i])
        return obs, rew, dones, infos

    def seed(self, seed=None):
        self._seed = seed
        return [seed] * self.num_envs

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        n = self.num_envs if indices is None else len(self._get_indices(indices))
        return [getattr(self.env, attr_name, None)] * n

    def set_attr(self, attr_name, value, indices=None):
        setattr(self.env, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [getattr(self.env, method_name)(*args, **kwargs)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def render(self, mode=None):
        return None
