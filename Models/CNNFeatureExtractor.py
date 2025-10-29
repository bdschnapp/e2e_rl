import torch
from torch import nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if key == "image":
                # Simple CNN feature extractor for the image
                n_input_channels = 1  # Assuming single-channel image
                self.image_height, self.image_width = subspace.shape[1], subspace.shape[2]
                extractors[key] = nn.Sequential(
                    nn.BatchNorm2d(n_input_channels),
                    nn.Conv2d(n_input_channels, 5, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2),  # Downsample to (8, H/2, W/2)
                    nn.Conv2d(5, 5, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2),  # Downsample to (16, H/4, W/4)
                    nn.Flatten()
                )
                # Compute the output size of the CNN
                with torch.no_grad():
                    sample_input = torch.zeros(1, n_input_channels, self.image_height, self.image_width)
                    cnn_output_size = extractors[key](sample_input).shape[1]
                total_concat_size += cnn_output_size

            elif key == "vector":
                # Run through a simple MLP
                extractors[key] = nn.Linear(subspace.shape[0], 1)
                total_concat_size += 1

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return torch.cat(encoded_tensor_list, dim=1)
