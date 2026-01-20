import torch
from torch import nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetEncoder(nn.Module):
    def __init__(self, in_channels, base_ch=32, depth=3, height=None, width=None):
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
        self.out_channels = chans[-1]

        # compute flattened feature dim
        if height is not None and width is not None:
            with torch.no_grad():
                x = torch.zeros(1, in_channels, height, width)
                feats = self.forward(x)  # [B, C, H', W']
                self.output_dim = feats.view(1, -1).shape[1]
        else:
            self.output_dim = None

    def forward(self, x):
        # encoder path only, no skips returned here
        for i in range(self.depth):
            x = self.down_blocks[i](x)
            x = self.pools[i](x)
        return x  # [B, C, H', W']


class UNetDecoder(nn.Module):
    def __init__(self, chans, out_channels):
        super().__init__()
        # chans: list like [c0, c1, c2, ...] from encoder
        self.up_convs = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        for i in range(len(chans) - 1, 0, -1):
            up = nn.ConvTranspose2d(chans[i], chans[i-1], kernel_size=2, stride=2)
            self.up_convs.append(up)
            self.up_blocks.append(DoubleConv(chans[i], chans[i-1]))

        self.final_conv = nn.Conv2d(chans[0], out_channels, kernel_size=1)

    def forward(self, x, skips):
        # skips: list of feature maps from encoder stages
        #       [skip0, skip1, ..., skip_{L-1}]
        for up, block, skip in zip(self.up_convs, self.up_blocks, reversed(skips)):
            x = up(x)
            # pad/crop if shapes mismatch, omitted here for brevity
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        return self.final_conv(x)


class UNetAutoencoder(nn.Module):
    def __init__(self, in_channels, height, width, base_ch=32, depth=3):
        super().__init__()
        self.height = height
        self.width = width

        # full encoder with skip outputs
        chans = [base_ch * (2 ** i) for i in range(depth)]
        self.down_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_ch = in_channels
        for c in chans:
            self.down_blocks.append(DoubleConv(prev_ch, c))
            self.pools.append(nn.MaxPool2d(2))
            prev_ch = c

        self.bottleneck = DoubleConv(chans[-1], chans[-1])

        self.decoder = UNetDecoder(chans, out_channels=in_channels)

    def encode_with_skips(self, x):
        skips = []
        for block, pool in zip(self.down_blocks, self.pools):
            x = block(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        return x, skips

    def forward(self, x):
        z, skips = self.encode_with_skips(x)
        x_rec = self.decoder(z, skips)
        return x_rec


def train_autoencoder(images):
    ae = UNetAutoencoder(in_channels=1, height=H, width=W)
    # load pre-trained weights...
    state_dict = torch.load("unet_ae.pt", map_location="cpu")
    ae.load_state_dict(state_dict)

    # build a UNetEncoder that matches the encoder part of ae
    encoder = UNetEncoder(
        in_channels=1,
        base_ch=32,
        depth=3,
        height=H,
        width=W,
    )
    # copy matching weights from ae.down_blocks / ae.pools into encoder
    encoder.load_state_dict(
        {
            **encoder.state_dict(),
            **{k.replace("net.", "net."): v for k, v in ae.state_dict().items()
               if k.startswith("down_blocks") or k.startswith("pools")}
        }
    )
    torch.save(encoder.state_dict(), "unet_encoder_pretrained.pt")