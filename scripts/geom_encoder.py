"""Geometry-prediction encoder for Phase-2B (NEW file, additive; touches nothing that
run_stage1_sweep depends on).

A CNN over the SDF-BEV image that regresses PHYSICAL GEOMETRY (the exteroceptive state
components the ego-centric image actually contains: tractor cross-track / heading error,
path curvature; later, obstacle distances). Trained supervised against the simulator's
ground-truth (privileged teacher), then FROZEN and used as a fixed perception module so
RL runs on the *estimated* geometry (deployable, non-privileged) — the "learning by
cheating" / state-estimation paradigm.

Exposes both:
  - features(img): the penultimate embedding (for a raw-feature RL variant), and
  - forward(img):  the predicted geometry vector (the primary, interpretable obs).
"""

from __future__ import annotations

import torch as th
import torch.nn as nn


class GeomEncoder(nn.Module):
    def __init__(self, in_ch: int, out_dim: int, img_size: int = 32, feat: int = 128):
        super().__init__()
        self.in_ch = in_ch
        self.out_dim = out_dim
        self.feat_dim = feat
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            n_flat = self.cnn(th.zeros(1, in_ch, img_size, img_size)).shape[1]
        self.trunk = nn.Sequential(nn.Linear(n_flat, feat), nn.ReLU(), nn.LayerNorm(feat))
        self.head = nn.Linear(feat, out_dim)   # regresses standardized geometry labels

    def features(self, img):
        return self.trunk(self.cnn(img))

    def forward(self, img):
        return self.head(self.features(img))
