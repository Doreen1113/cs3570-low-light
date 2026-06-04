"""Person B (part 1) — Illumination map estimation.

Two estimators are provided:
  MaxChannelIllum  — fast, zero-param: L = max(R, G, B) per pixel
  GuidedIllum      — uses an optional guided filter to smooth the map

Both return tensors of shape [B, 1, H, W] in [0, 1].

Run standalone to visualise illumination maps on a few training images:
  python illum_map.py [--root H:/low-light-data/low-light] [--guided]
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

class MaxChannelIllum:
    """Retinex-style illumination: L = max(R, G, B).

    Optional smoothing can reduce pixel-level noise while preserving the
    original low-light brightness scale.
    """

    def __init__(self, smooth_kernel: int = 7, smooth_alpha: float = 0.0):
        self.smooth_kernel = smooth_kernel
        self.smooth_alpha = smooth_alpha

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """img: [B, 3, H, W] or [3, H, W], float32 in [0,1]
        returns: [B, 1, H, W] or [1, H, W]
        """
        batched = img.dim() == 4
        if not batched:
            img = img.unsqueeze(0)

        illum = img.max(dim=1, keepdim=True).values
        if self.smooth_kernel > 1 and self.smooth_alpha > 0:
            pad = self.smooth_kernel // 2
            smooth = F.avg_pool2d(
                illum,
                kernel_size=self.smooth_kernel,
                stride=1,
                padding=pad,
            )
            illum = (1 - self.smooth_alpha) * illum + self.smooth_alpha * smooth

        illum = illum.clamp(0.0, 1.0)
        return illum if batched else illum.squeeze(0)


class GuidedIllum:
    """Guided-filter smoothed illumination map.

    Uses the grayscale input as both guide and source with a simple
    box-filter approximation (no extra dependencies needed).

    Args:
        radius: half-size of the box filter window
        eps:    regularisation term (prevents over-smoothing flat regions)
    """

    def __init__(self, radius: int = 15, eps: float = 0.01):
        self.r = radius
        self.eps = eps

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """img: [B, 3, H, W] float32 in [0,1] — returns [B, 1, H, W]"""
        batched = img.dim() == 4
        if not batched:
            img = img.unsqueeze(0)

        L = img.max(dim=1, keepdim=True).values  # [B,1,H,W]
        L_smooth = self._guided_filter(L, L)

        return L_smooth if batched else L_smooth.squeeze(0)

    def _box(self, x: torch.Tensor) -> torch.Tensor:
        k = 2 * self.r + 1
        return F.avg_pool2d(x, kernel_size=k, stride=1, padding=self.r)

    def _guided_filter(self, guide: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        mean_g = self._box(guide)
        mean_s = self._box(src)
        mean_gs = self._box(guide * src)
        cov_gs = mean_gs - mean_g * mean_s

        mean_gg = self._box(guide * guide)
        var_g = mean_gg - mean_g * mean_g

        a = cov_gs / (var_g + self.eps)
        b = mean_s - a * mean_g

        mean_a = self._box(a)
        mean_b = self._box(b)
        return (mean_a * guide + mean_b).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Convenience function used by noise_map.py and model/restoration.py
# ---------------------------------------------------------------------------

_default_illum = MaxChannelIllum()
_guided_illum = GuidedIllum()


def compute_illum(img: torch.Tensor, guided: bool = True) -> torch.Tensor:
    """Compute illumination map for a batch or single image tensor.

    Default is now guided filter (Retinex-style smooth illumination).
    Max-channel only is available with guided=False, but tends to leak
    texture/noise into the map.
    """
    if guided:
        return _guided_illum(img)
    return _default_illum(img)


# ---------------------------------------------------------------------------
# Standalone visualisation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="H:/low-light-data/low-light")
    parser.add_argument("--guided", action="store_true")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--out", default="H:/low-light-data/illum_check")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    estimator = GuidedIllum() if args.guided else MaxChannelIllum()

    stems = sorted((root / "train").glob("*-in.webp"))[:args.n]
    for p in stems:
        img = TF.to_tensor(Image.open(p).convert("RGB")).unsqueeze(0)
        L = estimator(img).squeeze(0)  # [1,H,W]
        L_pil = TF.to_pil_image(L)
        name = p.stem.replace("-in", "")
        L_pil.save(out_dir / f"{name}_illum.png")
        print(f"  {p.name}: L mean={L.mean():.3f}  min={L.min():.3f}  max={L.max():.3f}")

    print(f"\nSaved {len(stems)} illumination maps to {out_dir}")
    method = "guided" if args.guided else "max-channel"
    print(f"Method: {method}")
