"""Person C — RestorationNet wrapper.

Accepts (img, noise_map, illum_map), concatenates them into a 5-channel
tensor, and feeds it through NAFNet.

This is the main model interface used by train.py.

Run sanity check (works even before B's maps are ready — uses dummy zeros):
  python model/restoration.py
"""

import torch
import torch.nn as nn

from .nafnet import NAFNet


class RestorationNet(nn.Module):
    """Joint denoising + low-light enhancement model.

    Inputs:
      img:       [B, 3, H, W]  — low-light noisy image, float32 [0,1]
      noise_map: [B, 1, H, W]  — σ map from noise_map.py (or zeros)
      illum_map: [B, 1, H, W]  — L map from illum_map.py (or zeros)

    Output:
      restored:  [B, 3, H, W]  — clean enhanced image, float32 (unclamped)
    """

    def __init__(
        self,
        width: int = 32,
        enc_blocks: tuple = (2, 2, 4, 8),
        dec_blocks: tuple = (2, 2, 2, 2),
        middle_blocks: int = 4,
    ):
        super().__init__()
        self.net = NAFNet(
            in_channels=5,
            width=width,
            enc_blocks=enc_blocks,
            dec_blocks=dec_blocks,
            middle_blocks=middle_blocks,
        )

    def forward(
        self,
        img: torch.Tensor,
        noise_map: torch.Tensor | None = None,
        illum_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, _, H, W = img.shape
        device = img.device

        if noise_map is None:
            noise_map = torch.zeros(B, 1, H, W, device=device, dtype=img.dtype)
        if illum_map is None:
            illum_map = torch.zeros(B, 1, H, W, device=device, dtype=img.dtype)

        x = torch.cat([img, noise_map, illum_map], dim=1)  # [B, 5, H, W]
        out = self.net(x)                                   # [B, 3, H, W]
        # residual learning: predict the residual added to the input
        return (img + out).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Quick sanity check — runs without B's maps
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

    model = RestorationNet(width=32)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"RestorationNet  params: {n_params:.2f}M")

    img   = torch.randn(2, 3, 256, 256).clamp(0, 1)
    noise = torch.zeros(2, 1, 256, 256)   # dummy — B can swap in real maps
    illum = torch.zeros(2, 1, 256, 256)   # dummy

    with torch.no_grad():
        out = model(img, noise, illum)

    print(f"img {tuple(img.shape)} + maps → restored {tuple(out.shape)}")
    assert out.shape == (2, 3, 256, 256)
    assert out.min() >= 0.0 and out.max() <= 1.0
    print("Forward pass OK — maps are zero (dummy), swap in real ones from B.")
