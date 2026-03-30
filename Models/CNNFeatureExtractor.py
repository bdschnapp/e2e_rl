import torch
from torch import nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import is_image_space_channels_first

from Models.AutoEncoder import ImageEncoder


class CNNFeatureExtractor(BaseFeaturesExtractor):
    """
    Combined feature extractor for Dict observation spaces with 'image' and 'vector' keys.

    Image branch
    ------------
    Uses ImageEncoder (NatureCNN-style: 32→64→64 filters, 3136-dim for 84×84).
    Supports optional pretrained weights and freezing — set encoder_state_dict_path
    to the file saved by Models.AutoEncoder.pretrain_autoencoder() and freeze_encoder=True
    to get the pretrained-AE behaviour.  With no path, the encoder trains end-to-end.

    Vector branch
    -------------
    2-layer MLP: vec_dim → 64 → 64.

    Combined output dim = 3136 + 64 = 3200 (for 84×84 input).

    Fixes vs old CNNFeatureExtractor
    ---------------------------------
    1. Properly detects n_input_channels for channels-last observation spaces
    2. Processes the 'vector' key instead of discarding it
    3. 6× more CNN capacity (NatureCNN vs 5-filter CNN)
    """

    VECTOR_HIDDEN = 64

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        encoder_state_dict_path: str | None = None,
        freeze_encoder: bool = False,
    ):
        super().__init__(observation_space, features_dim=1)

        total_concat_size = 0
        self._has_image = False
        self._has_vector = False

        # --- Image branch ---
        if "image" in observation_space.spaces:
            img_space = observation_space.spaces["image"]
            if is_image_space_channels_first(img_space):
                n_ch = img_space.shape[0]
                h, w = img_space.shape[1], img_space.shape[2]
            else:
                # Channels-last: (H, W, C)
                h, w = img_space.shape[0], img_space.shape[1]
                n_ch = img_space.shape[2]

            encoder = ImageEncoder(n_ch, h, w)

            if encoder_state_dict_path is not None:
                state_dict = torch.load(encoder_state_dict_path, map_location="cpu")
                encoder.load_state_dict(state_dict)
                print(f"[CNNFeatureExtractor] Loaded encoder from {encoder_state_dict_path}")

            if freeze_encoder:
                for p in encoder.parameters():
                    p.requires_grad = False

            self.image_encoder = encoder
            total_concat_size += encoder.output_dim
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
            total_concat_size += self.VECTOR_HIDDEN
            self._has_vector = True

        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        parts = []
        if self._has_image:
            parts.append(self.image_encoder(observations["image"]))
        if self._has_vector:
            parts.append(self.vector_mlp(observations["vector"]))
        return torch.cat(parts, dim=1)
