"""Person C — NAFNet (Nonlinear Activation Free Network) implementation.

Based on: "Simple Baselines for Image Restoration" (Chen et al., ECCV 2022)
Modified to accept in_channels != 3 so we can concat noise/illum maps.

Key innovation: first conv accepts `in_channels` (default 5 = 3 RGB + 1 noise + 1 illum).

Run forward pass sanity check:
  python model/nafnet.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class LayerNorm2d(nn.Module):
    """Channel-last layer norm adapted for NCHW tensors."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / (s + self.eps).sqrt()
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class SimpleGate(nn.Module):
    """Split channels in half and multiply — replaces nonlinear activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """Core NAFNet block: DW-conv + SimpleGate + channel attention."""

    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        dw_ch = c * dw_expand
        ffn_ch = c * ffn_expand

        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw_ch, 1)
        self.conv2 = nn.Conv2d(dw_ch, dw_ch, 3, padding=1, groups=dw_ch)
        self.conv3 = nn.Conv2d(dw_ch // 2, c, 1)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_ch // 2, dw_ch // 2, 1),
        )
        self.sg1 = SimpleGate()

        self.norm2 = LayerNorm2d(c)
        self.conv4 = nn.Conv2d(c, ffn_ch, 1)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_ch // 2, c, 1)

        self.beta = nn.Parameter(torch.ones(1, c, 1, 1) * 1e-3)
        self.gamma = nn.Parameter(torch.ones(1, c, 1, 1) * 1e-3)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg1(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg2(x)
        x = self.conv5(x)
        return y + x * self.gamma


# ---------------------------------------------------------------------------
# NAFNet encoder-decoder
# ---------------------------------------------------------------------------

class NAFNet(nn.Module):
    """U-Net style NAFNet.

    Args:
        in_channels:  number of input channels (default 5: RGB + noise + illum)
        width:        base feature width
        enc_blocks:   blocks per encoder stage
        dec_blocks:   blocks per decoder stage
        middle_blocks: blocks in the bottleneck
    """

    def __init__(
        self,
        in_channels: int = 5,
        width: int = 32,
        enc_blocks: tuple = (2, 2, 4, 8),
        dec_blocks: tuple = (2, 2, 2, 2),
        middle_blocks: int = 4,
    ):
        super().__init__()

        self.intro = nn.Conv2d(in_channels, width, 3, padding=1)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for num in enc_blocks:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, chan * 2, 2, 2))
            chan *= 2

        self.middle = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blocks)])

        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        for num in dec_blocks:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1),
                nn.PixelShuffle(2),
            ))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.outro = nn.Conv2d(chan, 3, 3, padding=1)
        self.padder = 2 ** len(enc_blocks)  # must be multiple of this

    def check_image_size(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        ph = (self.padder - h % self.padder) % self.padder
        pw = (self.padder - w % self.padder) % self.padder
        return F.pad(x, (0, pw, 0, ph))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        x = self.check_image_size(x)

        x = self.intro(x)

        enc_skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            enc_skips.append(x)
            x = down(x)

        x = self.middle(x)

        for dec, up, skip in zip(self.decoders, self.ups, reversed(enc_skips)):
            x = up(x)
            x = x + skip
            x = dec(x)

        x = self.outro(x)
        return x[:, :, :H, :W]


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = NAFNet(in_channels=5, width=32)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"NAFNet  params: {n_params:.2f}M")

    x = torch.randn(2, 5, 256, 256)
    with torch.no_grad():
        y = model(x)
    print(f"input {tuple(x.shape)} → output {tuple(y.shape)}")
    assert y.shape == (2, 3, 256, 256), f"unexpected shape {y.shape}"
    print("Forward pass OK")
