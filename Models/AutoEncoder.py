"""
AutoEncoder models for BEV image representation learning.

Architecture
------------
ImageEncoder   — NatureCNN-style encoder (32→64→64 filters) producing 3136-dim
                 features for 84×84 input. Shared with CNNFeatureExtractor so
                 pretrained weights transfer directly.

ImageDecoder   — Transposed NatureCNN, mirrors the encoder exactly:
                 (64,7,7) → ConvT → (64,9,9) → ConvT → (32,20,20) → ConvT → (1,84,84)

ImageAutoencoder — Encoder + Decoder, trained on BEV images for representation
                 pretraining. After training, save encoder.state_dict() and load
                 it into CNNFeatureExtractor(encoder_state_dict_path=..., freeze_encoder=True).

Utilities
---------
collect_bev_images(env, n_steps)            — roll out env, return image tensor
pretrain_autoencoder(env, save_path, ...)   — collect → train → save encoder weights
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import is_image_space_channels_first


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class ImageEncoder(nn.Module):
    """
    NatureCNN-style encoder for 84×84 grayscale BEV images.

    Produces 3136-dim features for 84×84 input:
      Conv(1→32, k=8, s=4) → (32,20,20)
      Conv(32→64, k=4, s=2) → (64,9,9)
      Conv(64→64, k=3, s=1) → (64,7,7)
      Flatten → 3136

    This is identical to the NatureCNNEncoder in CNNFeatureExtractor, so
    pretrained weights from ImageAutoencoder transfer directly.
    """

    def __init__(self, n_input_channels: int = 1, height: int = 84, width: int = 84):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.zeros(1, n_input_channels, height, width)
            self.output_dim = self.net(sample).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Decoder  (transpose of the encoder, for pretraining only)
# ---------------------------------------------------------------------------

class ImageDecoder(nn.Module):
    """
    Transpose-NatureCNN decoder.  Reconstructs 84×84 from 3136 features.

    (batch, 3136) → Unflatten(64,7,7)
      ConvT(64→64, k=3, s=1) → (64,9,9)
      ConvT(64→32, k=4, s=2) → (32,20,20)
      ConvT(32→C,  k=8, s=4) → (C,84,84)
      Sigmoid
    """

    def __init__(self, n_output_channels: int = 1, encoded_dim: int = 3136):
        super().__init__()

        self.unflatten = nn.Unflatten(1, (64, 7, 7))

        self.net = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, n_output_channels, kernel_size=8, stride=4, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(self.unflatten(z))


# ---------------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------------

class ImageAutoencoder(nn.Module):
    """
    Full autoencoder = ImageEncoder + ImageDecoder.

    Train with MSE loss on normalised [0,1] BEV images, then save encoder
    weights for use in downstream RL:

        torch.save(ae.encoder.state_dict(), 'ae_encoder.pt')

    And load in CNNFeatureExtractor:

        CNNFeatureExtractor(
            obs_space,
            encoder_state_dict_path='ae_encoder.pt',
            freeze_encoder=True,
        )
    """

    def __init__(
        self,
        n_input_channels: int = 1,
        height: int = 84,
        width: int = 84,
    ):
        super().__init__()
        self.encoder = ImageEncoder(n_input_channels, height, width)
        self.decoder = ImageDecoder(n_input_channels, self.encoder.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def _pure_pursuit_action(obs, env) -> np.ndarray:
    """Simple feed-forward lane controller. Falls back to random on failure."""
    try:
        import e2erl_utils.config as config

        e_y    = float(obs["vector"][2])
        e_psi  = float(obs["vector"][3])
        kappa  = float(obs["vector"][6])

        wheelbase = env.vehicle.lf + env.vehicle.lr
        u = 0.85 * np.arctan(wheelbase * kappa) + 0.15 * (-1.2 * e_y - 2.4 * e_psi)

        steer_rate = float(np.clip(
            (u - env.vehicle.s) / env.vehicle.dt,
            -np.deg2rad(config.steering_action),
            np.deg2rad(config.steering_action),
        ))
        return np.array([steer_rate, float(config.initial_xd)], dtype=np.float32)
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        return env.action_space.sample()


def collect_bev_images(
    env,
    n_steps: int = 5_000,
    use_pure_pursuit: bool = True,
    driver_models=None,
) -> torch.Tensor:
    """
    Roll out `env` for `n_steps` steps and collect BEV images.

    Returns a float32 tensor of shape (N, 1, H, W) normalised to [0, 1],
    ready for DataLoader.  Only the 'image' key is used.

    Parameters
    ----------
    env            : BevObservationLineFollowingEnv (or any Dict env with 'image')
    n_steps        : number of environment steps to collect
    use_pure_pursuit : include a pure-pursuit controller as one of the drivers
    driver_models  : list of (model, obs_fn) pairs.  `model` is any SB3 policy
                     with a .predict(obs, deterministic) method; `obs_fn` maps the
                     BEV Dict obs to the observation the model expects.  Episodes
                     are assigned to drivers in round-robin order so every driver
                     contributes roughly equally to the dataset.
    """
    from tqdm import tqdm

    # Build driver list — each entry is a callable(obs) -> action
    drivers = []
    if use_pure_pursuit:
        drivers.append(lambda obs: _pure_pursuit_action(obs, env))
    if driver_models:
        for model, obs_fn in driver_models:
            drivers.append(lambda obs, m=model, f=obs_fn: m.predict(f(obs), deterministic=True)[0])
    if not drivers:
        drivers.append(lambda obs: env.action_space.sample())

    images = []
    obs, _ = env.reset()
    steps_done = 0
    episode_idx = 0
    pbar = tqdm(total=n_steps, desc="Collecting BEV images", unit="step")

    while steps_done < n_steps:
        driver = drivers[episode_idx % len(drivers)]
        action = driver(obs)

        obs, _, term, trunc, _ = env.step(action)
        img = obs["image"]  # (H, W, 1) uint8
        img_t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0  # (1, H, W)
        images.append(img_t)
        steps_done += 1
        pbar.update(1)

        if term or trunc:
            obs, _ = env.reset()
            episode_idx += 1

    pbar.close()
    return torch.stack(images)  # (N, 1, H, W)


# ---------------------------------------------------------------------------
# Pretraining
# ---------------------------------------------------------------------------

def pretrain_autoencoder(
    env,
    save_path: str,
    n_collect_steps: int = 10_000,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "auto",
    driver_models=None,
) -> str:
    """
    Collect BEV images, train an ImageAutoencoder, and save the encoder weights.

    Parameters
    ----------
    env             : BevObservationLineFollowingEnv
    save_path       : path to save encoder state dict (e.g. 'models/Phase1/ae_encoder.pt')
    n_collect_steps : env steps to collect training images
    epochs          : AE training epochs
    batch_size      : mini-batch size
    lr              : Adam learning rate
    device          : 'auto', 'cuda', or 'cpu'
    driver_models   : list of (model, obs_fn) pairs — see collect_bev_images

    Returns
    -------
    save_path  (string, for chaining)
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    n_drivers = len(driver_models) if driver_models else 0
    print(f"[AE pretrain] Collecting {n_collect_steps} BEV images "
          f"(pure-pursuit + {n_drivers} trained driver(s)) …")
    images = collect_bev_images(
        env, n_steps=n_collect_steps, use_pure_pursuit=True, driver_models=driver_models
    )
    print(f"[AE pretrain] Dataset: {images.shape}  device={device}")

    h, w = images.shape[2], images.shape[3]
    ae = ImageAutoencoder(n_input_channels=1, height=h, width=w).to(device)
    optimiser = torch.optim.Adam(ae.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataset = TensorDataset(images)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    ae.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimiser.zero_grad()
            recon = ae(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * len(batch)

        mean_loss = total_loss / len(images)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"[AE pretrain] epoch {epoch+1:3d}/{epochs}  recon_loss={mean_loss:.5f}")

    torch.save(ae.encoder.state_dict(), save_path)
    print(f"[AE pretrain] Encoder weights saved → {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# SB3 Feature Extractor  (used when encoder_state_dict_path is given)
# ---------------------------------------------------------------------------
# Note: the primary extractor for end-to-end and pretrained AE is
# CNNFeatureExtractor in Models/CNNFeatureExtractor.py, which imports
# ImageEncoder from here.  This class is kept for direct instantiation.

class AEFeatureExtractor(BaseFeaturesExtractor):
    """
    SB3 feature extractor that uses a (optionally pretrained) ImageEncoder
    together with a small MLP for the 'vector' key.

    Identical in behaviour to CNNFeatureExtractor with an encoder_state_dict_path,
    but imports ImageEncoder directly so no circular dependency arises.
    """

    VECTOR_HIDDEN = 64

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        encoder_state_dict_path: str | None = None,
        freeze_encoder: bool = True,
    ):
        super().__init__(observation_space, features_dim=1)

        total_dim = 0
        self._has_image = False
        self._has_vector = False

        if "image" in observation_space.spaces:
            img_space = observation_space.spaces["image"]
            if is_image_space_channels_first(img_space):
                n_ch, h, w = img_space.shape
            else:
                h, w, n_ch = img_space.shape

            encoder = ImageEncoder(n_ch, h, w)

            if encoder_state_dict_path is not None:
                state = torch.load(encoder_state_dict_path, map_location="cpu")
                encoder.load_state_dict(state)
                print(f"[AEFeatureExtractor] Loaded encoder weights from {encoder_state_dict_path}")

            if freeze_encoder:
                for p in encoder.parameters():
                    p.requires_grad = False

            self.image_encoder = encoder
            total_dim += encoder.output_dim
            self._has_image = True

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
