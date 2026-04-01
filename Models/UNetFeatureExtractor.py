import torch
from torch import nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import is_image_space_channels_first

from Models.UNet import UNetEncoder


class UNetFeatureExtractor(BaseFeaturesExtractor):
    """
    Combined feature extractor for Dict observation spaces using a UNet encoder.

    Image branch
    ------------
    UNetEncoder (depth=3, base_ch=32) + GlobalAveragePool → 128-dim.
    Supports optional pretrained weights (from Models.UNet.pretrain_unet) and
    freezing.

    Vector branch
    -------------
    2-layer MLP: vec_dim → 64 → 64.

    Combined output dim = 128 + 64 = 192 (for default depth=3, base_ch=32).

    Fixes vs old UNetFeatureExtractor
    -----------------------------------
    1. GlobalAveragePool replaces Flatten: 12,800 → 128 features
    2. Processes the 'vector' key instead of discarding it
    3. Correct channels-last image obs detection
    4. Pretrained weight loading support
    """

    VECTOR_HIDDEN = 64

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        encoder_state_dict_path: str | None = None,
        freeze_encoder: bool = False,
        base_ch: int = 32,
        depth: int = 3,
    ):
        super().__init__(observation_space, features_dim=1)

        total_dim = 0
        self._has_image = False
        self._has_vector = False
        self._freeze_encoder = freeze_encoder

        # --- Image branch ---
        if "image" in observation_space.spaces:
            img_space = observation_space.spaces["image"]
            if is_image_space_channels_first(img_space):
                n_ch = img_space.shape[0]
            else:
                n_ch = img_space.shape[2]   # channels-last: (H, W, C)

            encoder = UNetEncoder(
                in_channels=n_ch,
                base_ch=base_ch,
                depth=depth,
            )

            if encoder_state_dict_path is not None:
                state = torch.load(encoder_state_dict_path, map_location="cpu")
                encoder.load_state_dict(state)
                print(f"[UNetFeatureExtractor] Loaded encoder from {encoder_state_dict_path}")

            if freeze_encoder:
                for p in encoder.parameters():
                    p.requires_grad = False
                encoder.eval()

            self.image_encoder = encoder
            total_dim += encoder.output_dim   # 128 with default args
            self._has_image = True

        # --- Vector branch ---
        if "vector" in observation_space.spaces:
            vec_dim = int(observation_space.spaces["vector"].shape[0])
            self.vector_mlp = nn.Sequential(
                nn.Linear(vec_dim, self.VECTOR_HIDDEN),
                nn.ReLU(),
                nn.Linear(self.VECTOR_HIDDEN, self.VECTOR_HIDDEN),
                nn.ReLU(),
            )
            total_dim += self.VECTOR_HIDDEN
            self._has_vector = True

        self._features_dim = total_dim

    def forward(self, observations) -> torch.Tensor:
        parts = []
        if self._has_image:
            parts.append(self.image_encoder(observations["image"]))
        if self._has_vector:
            parts.append(self.vector_mlp(observations["vector"]))
        return torch.cat(parts, dim=1)

    def train(self, mode: bool = True):
        super().train(mode)
        if self._freeze_encoder and self._has_image:
            # Keep BatchNorm running statistics fixed in frozen mode.
            self.image_encoder.eval()
        return self
