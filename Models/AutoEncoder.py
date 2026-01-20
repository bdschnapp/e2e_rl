import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ImageEncoder(nn.Module):
    def __init__(self, n_input_channels: int, height: int, width: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.BatchNorm2d(n_input_channels),
            nn.Conv2d(n_input_channels, 5, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(5, 5, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )

        # compute output dim
        with torch.no_grad():
            sample = torch.zeros(1, n_input_channels, height, width)
            self.output_dim = self.net(sample).shape[1]

    def forward(self, x):
        return self.net(x)


class ImageAutoencoder(nn.Module):
    def __init__(self, n_input_channels: int, height: int, width: int):
        super().__init__()

        # shared encoder
        self.encoder = ImageEncoder(n_input_channels, height, width)

        self.height = height
        self.width = width
        self.n_input_channels = n_input_channels

        # simple decoder: fully-connected + reshape + convs
        enc_dim = self.encoder.output_dim

        self.decoder_fc = nn.Linear(enc_dim, 32 * (height // 4) * (width // 4))

        self.decoder_conv = nn.Sequential(
            # start from (32, H/4, W/4), upsample back to (C, H, W)
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, n_input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # if you normalize images to [0,1]
        )

    def forward(self, x):
        z = self.encoder(x)
        x_dec = self.decoder_fc(z)
        x_dec = x_dec.view(
            x_dec.size(0),
            32,
            self.height // 4,
            self.width // 4
        )
        x_rec = self.decoder_conv(x_dec)
        return x_rec


def train_autoencoder(images):
    dataset = TensorDataset(images)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    height, width = images.shape[-2], images.shape[-1]
    ae = ImageAutoencoder(n_input_channels=1, height=height, width=width).to(device)
    optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(20):
        ae.train()
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = ae(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        print(f"Epoch {epoch}: recon loss = {total_loss / len(dataset):.6f}")

    # save only the encoder weights
    torch.save(ae.encoder.state_dict(), "pretrained_image_encoder.pt")