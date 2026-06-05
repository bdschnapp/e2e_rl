import torch
from torch import nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import is_image_space_channels_first

from Models.AutoEncoder import ImageEncoder


class DictStateOnlyFeatureExtractor(BaseFeaturesExtractor):
    """
    Ablation extractor for Dict observation spaces.

    Keeps the BEV/MultiInputPolicy environment path intact, but returns only the
    state vector. This should learn like the state-only baseline if the dict
    policy path is healthy.
    """

    def __init__(self, observation_space: gym.spaces.Dict):
        if "vector" not in observation_space.spaces:
            raise ValueError("DictStateOnlyFeatureExtractor requires a 'vector' observation")
        vec_dim = int(observation_space.spaces["vector"].shape[0])
        super().__init__(observation_space, features_dim=vec_dim)

    def forward(self, observations) -> torch.Tensor:
        return observations["vector"]


class CNNFeatureExtractor(BaseFeaturesExtractor):
    """
    Combined feature extractor for Dict observation spaces with 'image' and 'vector' keys.

    Image branch
    ------------
    Uses ImageEncoder (compact 1→5→5 filters, 80-dim for 32×32),
    followed by a small projection head so the policy consumes a compact visual
    embedding instead of the raw flattened convolutional map.
    Supports optional pretrained weights and freezing — set encoder_state_dict_path
    to the file saved by Models.AutoEncoder.pretrain_autoencoder() and freeze_encoder=True
    to get the pretrained-AE behaviour.  With no path, the encoder trains end-to-end.

    Vector branch
    -------------
    2-layer MLP: vec_dim → 64 → 64.

    Combined output dim = 64 + 64 = 128 (for 32×32 input and default settings).

    Fixes vs old CNNFeatureExtractor
    ---------------------------------
    1. Properly detects n_input_channels for channels-last observation spaces
    2. Processes the 'vector' key instead of discarding it
    3. Compact image branch matching the prior 32×32 vision baseline
    """

    VECTOR_HIDDEN = 64
    IMAGE_FEATURES_DIM = 64

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        encoder_state_dict_path: str | None = None,
        freeze_encoder: bool = False,
        image_features_dim: int = IMAGE_FEATURES_DIM,
        image_scale: float = 1.0,
        state_scale: float = 1.0,
    ):
        super().__init__(observation_space, features_dim=1)

        total_concat_size = 0
        self._has_image = False
        self._has_vector = False
        self.register_buffer("image_scale", torch.tensor(float(image_scale), dtype=torch.float32))
        self.register_buffer("state_scale", torch.tensor(float(state_scale), dtype=torch.float32))
        self.register_load_state_dict_pre_hook(self._fill_missing_scale_buffers)

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
            self.image_projection = nn.Sequential(
                nn.Linear(encoder.output_dim, image_features_dim),
                nn.ReLU(),
            )
            total_concat_size += image_features_dim
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

    def _fill_missing_scale_buffers(
        self,
        module,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        for name in ("image_scale", "state_scale"):
            key = prefix + name
            if key not in state_dict:
                state_dict[key] = getattr(self, name).detach().clone()

    def set_image_scale(self, image_scale: float) -> None:
        self.image_scale.fill_(float(image_scale))

    def set_state_scale(self, state_scale: float) -> None:
        self.state_scale.fill_(float(state_scale))

    def _scale_state_observation(self, vector_obs: torch.Tensor) -> torch.Tensor:
        if vector_obs.shape[-1] <= 1:
            return vector_obs
        scaled = vector_obs.clone()
        scaled[..., 1:] = scaled[..., 1:] * self.state_scale
        return scaled

    def forward(self, observations) -> torch.Tensor:
        parts = []
        if self._has_image:
            image_features = self.image_projection(self.image_encoder(observations["image"]))
            parts.append(image_features * self.image_scale)
        if self._has_vector:
            vector_obs = self._scale_state_observation(observations["vector"])
            parts.append(self.vector_mlp(vector_obs))
        return torch.cat(parts, dim=1)
