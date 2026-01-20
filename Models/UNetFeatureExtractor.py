import torch
from torch import nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from Models.UNet import UNetEncoder


class UNetFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        encoder_state_dict_path: str | None = None,
        freeze_encoder: bool = False,
    ):
        super().__init__(observation_space, features_dim=1)

        extractors = {}
        total_concat_size = 0

        for key, subspace in observation_space.spaces.items():
            if key == "image":
                c, h, w = subspace.shape
                encoder = UNetEncoder(in_channels=c, base_ch=32, depth=3, height=h, width=w)

                if encoder_state_dict_path is not None:
                    state = torch.load(encoder_state_dict_path, map_location="cpu")
                    encoder.load_state_dict(state)

                if freeze_encoder:
                    for p in encoder.parameters():
                        p.requires_grad = False

                self.unet_encoder = encoder
                extractors[key] = nn.Sequential(
                    self.unet_encoder,
                    nn.Flatten(),  # or global pooling first
                )
                total_concat_size += encoder.output_dim

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        encoded = []
        for key, extractor in self.extractors.items():
            encoded.append(extractor(observations[key]))
        return torch.cat(encoded, dim=1)