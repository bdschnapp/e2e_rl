"""Geometry-observation env (Phase-2B; NEW, additive). Subclass of SdfBevEnv that
replaces the Dict{vector,image} observation with a plain vector
    [true proprio (delta, gamma)] + [cupy-encoder-predicted exteroceptive geometry]
computed ENTIRELY on the batched backend (xp = cupy) — no torch anywhere in the env
loop (SB3's policy is the only torch user). This is the stable, GPU-resident form of
the geometry-distillation recipe (fixes the torch/cupy-interleave segfault).

The frozen occupancy path (base env / run_stage1_sweep) is untouched.
"""

from __future__ import annotations

from .backend import xp
from .bev_sdf_env import SdfBevEnv, SdfObsCfg
from .geom_encoder_cupy import GeomEncoderCupy


class GeomObsEnv(SdfBevEnv):
    def __init__(self, cfg, num_envs, encoder_ckpt, sdf_cfg: SdfObsCfg = SdfObsCfg(coord=True),
                 proprio=(0, 1), path_pool_size: int = 4096):
        self._geom_proprio = tuple(proprio)     # dims taken from the TRUE state (measurable: steer, hitch)
        super().__init__(cfg, num_envs, sdf_cfg=sdf_cfg, path_pool_size=path_pool_size)
        self._geom_enc = GeomEncoderCupy(encoder_ckpt)
        self._last_true_vec = None              # stashed each step so eval can use ground-truth metrics
        if self._bev_on:                        # obs space -> the vector Box (drop the Dict/image)
            self.single_observation_space = self.single_observation_space.spaces["vector"]

    def _observe(self):
        obs, errors = super()._observe()        # {"vector": true_vec, "image": sdf_img}, errors
        true_vec = obs["vector"]
        pred = self._geom_enc.predict(obs["image"]).astype(xp.float32)   # (N, out_dim), all cupy
        for d in self._geom_proprio:
            pred[:, d] = true_vec[:, d]          # override proprio dims with the true (sensor-measurable) values
        self._last_true_vec = true_vec
        return pred, errors
