"""
UNet encoder for BEV image representation learning.

Architecture (depth=3, base_ch=32, input 84×84)
------------------------------------------------
DoubleConv(1→32)  + MaxPool(2)  → (32, 42, 42)
DoubleConv(32→64) + MaxPool(2)  → (64, 21, 21)
DoubleConv(64→128)+ MaxPool(2)  → (128, 10, 10)
GlobalAveragePool               → (128,)           ← output_dim

Global average pooling replaces the old Flatten, which produced 12,800 features
(128×10×10) — far too large for a policy network hidden layer.

Pretraining
-----------
UNetAutoencoder uses skip connections for faithful reconstruction.  After
pretraining, the encoder weights (down_blocks + pools) are transferred to a
standalone UNetEncoder.  Use pretrain_unet(env, save_path) for the full pipeline.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Encoder (standalone, used inside SB3 feature extractor)
# ---------------------------------------------------------------------------

class UNetEncoder(nn.Module):
    """
    Encoder-only path (no skip connections returned).

    output_dim = base_ch * 2^(depth-1)  (= 128 for base_ch=32, depth=3)
    after GlobalAveragePool.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_ch: int = 32,
        depth: int = 3,
        height: int = None,
        width: int = None,
    ):
        super().__init__()

        chans = [base_ch * (2 ** i) for i in range(depth)]
        self.down_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()

        prev_ch = in_channels
        for c in chans:
            self.down_blocks.append(DoubleConv(prev_ch, c))
            self.pools.append(nn.MaxPool2d(2))
            prev_ch = c

        self.depth = depth
        self.gap = nn.AdaptiveAvgPool2d(1)   # (C, H', W') → (C, 1, 1)
        self.output_dim = chans[-1]           # e.g. 128 for base_ch=32, depth=3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block, pool in zip(self.down_blocks, self.pools):
            x = block(x)
            x = pool(x)
        x = self.gap(x)          # (B, C, 1, 1)
        return x.flatten(1)      # (B, C)


# ---------------------------------------------------------------------------
# Full UNet autoencoder (for pretraining only)
# ---------------------------------------------------------------------------

class _UpBlock(nn.Module):
    """Single up-sampling step: bilinear upsample to skip size + concat + DoubleConv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Upsample x to exactly the skip spatial size (handles non-power-of-2 inputs)
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetAutoencoder(nn.Module):
    """
    Full UNet autoencoder with skip connections for faithful reconstruction.

    Encoder path  ↓  (same as UNetEncoder)
    Bottleneck    —
    Decoder path  ↑  (bilinear upsample + skip + DoubleConv at each level)
    Final conv    →  n_input_channels, Sigmoid

    After pretraining call transfer_encoder_weights(ae, encoder) to copy
    the encoder weights into a standalone UNetEncoder.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_ch: int = 32,
        depth: int = 3,
    ):
        super().__init__()

        chans = [base_ch * (2 ** i) for i in range(depth)]

        # Encoder
        self.down_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_ch = in_channels
        for c in chans:
            self.down_blocks.append(DoubleConv(prev_ch, c))
            self.pools.append(nn.MaxPool2d(2))
            prev_ch = c

        # Bottleneck (same channel width as last encoder stage)
        self.bottleneck = DoubleConv(chans[-1], chans[-1])

        # Decoder
        self.up_blocks = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            out_ch = chans[i - 1] if i > 0 else base_ch
            self.up_blocks.append(_UpBlock(chans[i], chans[i], out_ch))

        # Final 1×1 conv
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_ch, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def _encode(self, x: torch.Tensor):
        skips = []
        for block, pool in zip(self.down_blocks, self.pools):
            x = block(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        return x, skips

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, skips = self._encode(x)
        for up, skip in zip(self.up_blocks, reversed(skips)):
            z = up(z, skip)
        return self.final_conv(z)


# ---------------------------------------------------------------------------
# Weight transfer helper
# ---------------------------------------------------------------------------

def transfer_encoder_weights(ae: UNetAutoencoder, encoder: UNetEncoder):
    """
    Copy matching weights from a trained UNetAutoencoder into a UNetEncoder.
    Only down_blocks and pools are transferred; GAP has no learnable params.
    """
    ae_state = ae.state_dict()
    enc_state = encoder.state_dict()

    matched = {
        k: v for k, v in ae_state.items()
        if k in enc_state and enc_state[k].shape == v.shape
    }
    enc_state.update(matched)
    encoder.load_state_dict(enc_state)
    return len(matched)


# ---------------------------------------------------------------------------
# Pretraining pipeline
# ---------------------------------------------------------------------------

def pretrain_unet(
    env,
    save_path: str,
    n_collect_steps: int = 10_000,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    base_ch: int = 32,
    depth: int = 3,
    device: str = "auto",
    driver_models=None,
) -> str:
    """
    Collect BEV images, train UNetAutoencoder, transfer weights to UNetEncoder,
    and save the encoder state dict.

    Parameters
    ----------
    env             : BevObservationLineFollowingEnv
    save_path       : where to save UNetEncoder state dict (.pt)
    n_collect_steps : env steps to collect training images
    epochs / batch_size / lr : training hyper-parameters
    base_ch / depth : UNet geometry  (should match UNetFeatureExtractor kwargs)
    device          : 'auto', 'cuda', or 'cpu'
    driver_models   : list of (model, obs_fn) pairs — see collect_bev_images

    Returns
    -------
    save_path
    """
    from Models.AutoEncoder import collect_bev_images  # avoid circular import

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    n_drivers = len(driver_models) if driver_models else 0
    print(f"[UNet pretrain] Collecting {n_collect_steps} BEV images "
          f"(pure-pursuit + {n_drivers} trained driver(s)) …")
    images = collect_bev_images(
        env, n_steps=n_collect_steps, use_pure_pursuit=True, driver_models=driver_models
    )
    print(f"[UNet pretrain] Dataset: {images.shape}  device={device}")

    ae = UNetAutoencoder(in_channels=1, base_ch=base_ch, depth=depth).to(device)
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
            print(f"[UNet pretrain] epoch {epoch+1:3d}/{epochs}  recon_loss={mean_loss:.5f}")

    # Transfer weights to standalone encoder and save
    ae.eval()
    encoder = UNetEncoder(in_channels=1, base_ch=base_ch, depth=depth).to(device)
    n_matched = transfer_encoder_weights(ae, encoder)
    print(f"[UNet pretrain] Transferred {n_matched} weight tensors to UNetEncoder")

    torch.save(encoder.state_dict(), save_path)
    print(f"[UNet pretrain] Encoder weights saved → {save_path}")
    return save_path
