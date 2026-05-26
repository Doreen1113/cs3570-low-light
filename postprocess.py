"""Person D (part 3) — Post-processing.

Two operations applied after the restoration network:
  1. Luminance-aware chroma compensation in YCbCr space
     — corrects colour drift caused by large brightness increases
  2. Unsharp masking
     — recovers fine edge detail that gets softened during denoising

Both are differentiable (pure PyTorch ops) so they could be embedded in
training later if needed.

Run standalone:
  python postprocess.py [--root H:/low-light-data/low-light]
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image


# ---------------------------------------------------------------------------
# RGB ↔ YCbCr helpers (BT.601 coefficients, matching OpenCV default)
# ---------------------------------------------------------------------------

def rgb_to_ycbcr(img: torch.Tensor) -> torch.Tensor:
    """img: [B,3,H,W] or [3,H,W] in [0,1]. Returns YCbCr in same range."""
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    Y  =  0.299 * r + 0.587 * g + 0.114 * b
    Cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    Cr =  0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return torch.cat([Y, Cb, Cr], dim=1)


def ycbcr_to_rgb(img: torch.Tensor) -> torch.Tensor:
    """img: [B,3,H,W] YCbCr in [0,1]. Returns RGB in [0,1]."""
    Y  = img[:, 0:1]
    Cb = img[:, 1:2] - 0.5
    Cr = img[:, 2:3] - 0.5
    r = (Y + 1.402 * Cr).clamp(0, 1)
    g = (Y - 0.344136 * Cb - 0.714136 * Cr).clamp(0, 1)
    b = (Y + 1.772 * Cb).clamp(0, 1)
    return torch.cat([r, g, b], dim=1)


# ---------------------------------------------------------------------------
# Luminance-aware chroma compensation
# ---------------------------------------------------------------------------

def chroma_compensation(
    restored: torch.Tensor,
    alpha: float = 0.3,
) -> torch.Tensor:
    """Reduce colour saturation proportionally to luminance increase.

    When the network brightens a very dark image, the Cb/Cr channels can
    drift. This nudges them back toward the neutral (0.5) axis in regions
    that were significantly brightened.

    Args:
        restored: [B, 3, H, W] restored image in [0, 1]
        alpha:    compensation strength (0 = no change, 1 = full desaturation)
    """
    ycbcr = rgb_to_ycbcr(restored)
    Y, Cb, Cr = ycbcr[:, 0:1], ycbcr[:, 1:2], ycbcr[:, 2:3]

    # Weight compensation by luminance: brighter regions get more correction
    weight = (alpha * Y).clamp(0.0, 1.0)
    Cb_comp = Cb * (1 - weight) + 0.5 * weight
    Cr_comp = Cr * (1 - weight) + 0.5 * weight

    compensated = torch.cat([Y, Cb_comp, Cr_comp], dim=1)
    return ycbcr_to_rgb(compensated)


# ---------------------------------------------------------------------------
# Unsharp masking
# ---------------------------------------------------------------------------

def _gaussian_blur(img: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    coords = torch.arange(kernel_size, dtype=img.dtype, device=img.device) - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = g.unsqueeze(0) * g.unsqueeze(1)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # [1,1,k,k]
    kernel_2d = kernel_2d.expand(img.shape[1], 1, kernel_size, kernel_size)
    pad = kernel_size // 2
    return F.conv2d(img, kernel_2d, padding=pad, groups=img.shape[1])


def unsharp_mask(
    img: torch.Tensor,
    strength: float = 0.3,
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Apply unsharp masking to recover fine detail after denoising.

    sharpened = img + strength * (img - blur(img))

    Args:
        img:         [B, 3, H, W] in [0, 1]
        strength:    amount of sharpening (0 = none, higher = more)
        kernel_size: Gaussian blur kernel size
        sigma:       Gaussian blur sigma
    """
    blurred = _gaussian_blur(img, kernel_size, sigma)
    detail = img - blurred
    return (img + strength * detail).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Combined post-processor
# ---------------------------------------------------------------------------

def postprocess(
    restored: torch.Tensor,
    chroma_alpha: float = 0.3,
    sharpen_strength: float = 0.3,
) -> torch.Tensor:
    """Apply chroma compensation then unsharp masking.

    Both can be disabled by setting their strength to 0.
    """
    out = restored
    if chroma_alpha > 0:
        out = chroma_compensation(out, alpha=chroma_alpha)
    if sharpen_strength > 0:
        out = unsharp_mask(out, strength=sharpen_strength)
    return out


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="H:/low-light-data/low-light")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--out", default="H:/low-light-data/postproc_check")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    stems = sorted((root / "val").glob("*-in.webp"))[:args.n]
    for p in stems:
        img = TF.to_tensor(Image.open(p).convert("RGB")).unsqueeze(0)
        name = p.stem.replace("-in", "")

        # Simulate a "restored" image by brightening the input
        brightened = (img * 3.0).clamp(0, 1)

        processed = postprocess(brightened, chroma_alpha=0.3, sharpen_strength=0.3)

        TF.to_pil_image(brightened.squeeze(0)).save(out_dir / f"{name}_bright.png")
        TF.to_pil_image(processed.squeeze(0)).save(out_dir / f"{name}_postproc.png")
        print(f"  {name}: bright_mean={brightened.mean():.3f}  proc_mean={processed.mean():.3f}")

    print(f"\nSaved to {out_dir} — compare _bright vs _postproc")
