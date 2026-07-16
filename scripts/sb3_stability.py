"""Stability features extractor + SDF vec-env + TD3 builder (Phase-2B — NEW, additive).

Targets the diagnosed BEV failure (a random CNN branch emits high-variance features
that corrupt TD3's critic Q-fit -> collapse). Levers, cheapest-first:
  - LayerNorm on the image-CNN features so their scale cannot dominate the fused
    critic input (modality-dominance fix).
  - separate actor/critic encoders (share_features_extractor=False) so the actor's
    noisy encoder gradients don't corrupt the critic's encoder.
Optional (added only if the above is insufficient): stop-gradient on the image
features in the actor path, and a smaller LR on the CNN encoder.

Nothing here touches the frozen training path (train_sb3 / run_stage1_sweep).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import gymnasium as gym  # noqa: E402
from stable_baselines3.common.vec_env.base_vec_env import VecEnv  # noqa: E402

from tractor_trailer_rl.batched.sb3_adapter import SB3BatchedVecEnv  # noqa: E402
from tractor_trailer_rl.batched.bev_sdf_env import SdfBevEnv, SdfObsCfg  # noqa: E402
from train_sb3 import _td3_action_noise, _bev_extractor_class  # noqa: E402


def stability_extractor_class(layernorm_vector=False, stop_grad_image=False):
    """BaseFeaturesExtractor: CNN -> Linear -> LayerNorm(image feats), concat vector.

    layernorm_vector: also LayerNorm the state vector (default off — the state signal
    already works, don't rescale it). stop_grad_image: detach image feats (use with a
    SHARED encoder so the critic still trains it; pointless with separate encoders)."""
    import torch as th
    import torch.nn as nn
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    class StabilityBevExtractor(BaseFeaturesExtractor):
        def __init__(self, observation_space, cnn_features=128):
            img = observation_space.spaces["image"]
            vec_dim = int(observation_space.spaces["vector"].shape[0])
            super().__init__(observation_space, features_dim=cnn_features + vec_dim)
            c = int(img.shape[0])
            self.cnn = nn.Sequential(
                nn.Conv2d(c, 16, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Flatten(),
            )
            with th.no_grad():
                n_flat = self.cnn(th.zeros(1, *img.shape)).shape[1]
            self.linear = nn.Sequential(nn.Linear(n_flat, cnn_features), nn.ReLU())
            self.img_ln = nn.LayerNorm(cnn_features)
            self.vec_ln = nn.LayerNorm(vec_dim) if layernorm_vector else None
            self._stop_grad = stop_grad_image

        def forward(self, obs):
            f = self.img_ln(self.linear(self.cnn(obs["image"])))
            if self._stop_grad:
                f = f.detach()
            v = obs["vector"]
            if self.vec_ln is not None:
                v = self.vec_ln(v)
            return th.cat([f, v], dim=1)

    return StabilityBevExtractor


class SdfVecEnv(SB3BatchedVecEnv):
    """SB3 VecEnv over the SDF-BEV env (bypasses make_batched_env's occupancy path)."""

    def __init__(self, cfg, num_envs, sdf_cfg: SdfObsCfg = SdfObsCfg(), path_pool_size=4096):
        self.env = SdfBevEnv(cfg, num_envs, sdf_cfg=sdf_cfg, path_pool_size=path_pool_size)
        VecEnv.__init__(self, num_envs, self.env.single_observation_space,
                        self.env.single_action_space)
        self._actions = None
        self._seed = None


def build_td3(env, seed, lr=1e-3, extractor_class=None, share_features_extractor=None,
              cnn_features=128, encoder_lr_scale=None):
    """TD3 mirroring train_sb3.build_model's hyperparameters (lr 1e-3, learning_starts
    10k, per-dim action noise, SB3 defaults otherwise). extractor_class/share_* let the
    2B variants swap the extractor; encoder_lr_scale (e.g. 0.1) puts the CNN encoder on
    a smaller LR via a custom optimizer param-group."""
    from stable_baselines3 import TD3
    is_dict = isinstance(env.observation_space, gym.spaces.Dict)
    pk = {}
    if is_dict:
        ec = extractor_class or _bev_extractor_class()
        pk["features_extractor_class"] = ec
        pk["features_extractor_kwargs"] = {"cnn_features": cnn_features}
        if share_features_extractor is not None:
            pk["share_features_extractor"] = share_features_extractor
    policy = "MultiInputPolicy" if is_dict else "MlpPolicy"
    model = TD3(policy, env, learning_rate=lr, learning_starts=10_000,
                action_noise=_td3_action_noise(env), policy_kwargs=pk or None,
                seed=seed, device="cuda", verbose=0)
    if encoder_lr_scale is not None and is_dict:
        _apply_encoder_lr(model, encoder_lr_scale, lr)
    return model


def _apply_encoder_lr(model, scale, base_lr):
    """Rebuild actor/critic optimizers with the CNN-encoder params on base_lr*scale."""
    import torch as th

    def split(module):
        enc, rest = [], []
        for name, p in module.named_parameters():
            (enc if ("features_extractor" in name and ".cnn" in name) or
             ("features_extractor" in name and ".linear" in name) else rest).append(p)
        return enc, rest

    for net in (model.actor, model.critic):
        enc, rest = split(net)
        net.optimizer = th.optim.Adam(
            [{"params": rest, "lr": base_lr},
             {"params": enc, "lr": base_lr * scale}])
    # keep targets in sync (they are not optimized directly)
    return model
