"""Person D (part 1) — Multi-loss functions.

Total loss:
  L_total = λ1 * L_pixel + λ2 * SSIM [+ λ3 * Adversarial]

Pixel loss can be either L1 or Charbonnier (Charbonnier = smooth L1, used
by NAFNet/Restormer SOTA papers, typically +0.2-0.5 dB vs plain L1).

Classes:
  CharbonnierLoss  — sqrt((x-y)^2 + eps^2), differentiable + robust
  SSIMLoss         — differentiable SSIM loss (1 - SSIM)
  PatchGANLoss     — adversarial loss with a small PatchGAN discriminator
  TotalLoss        — combines all of the above with configurable λ weights

Run standalone to check all losses work on dummy tensors:
  python losses.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


# ---------------------------------------------------------------------------
# Charbonnier loss (smooth L1, NAFNet/Restormer standard)
# ---------------------------------------------------------------------------

class CharbonnierLoss(nn.Module):
    """L_char = mean(sqrt((pred - target)^2 + eps^2))

    More robust than L1 around zero (smooth gradient), typically gives
    +0.2-0.5 dB PSNR over plain L1 on image restoration tasks.
    """

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


# ---------------------------------------------------------------------------
# SSIM loss
# ---------------------------------------------------------------------------

def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)


class SSIMLoss(nn.Module):
    """1 - SSIM loss, computed per-channel and averaged.

    Args:
        window_size: Gaussian window size (default 11)
        sigma:       Gaussian sigma (default 1.5)
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        kernel = _gaussian_kernel(window_size, sigma)
        self.register_buffer("kernel", kernel)
        self.window_size = window_size
        self.pad = window_size // 2
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred, target: [B, 3, H, W] in [0, 1]. Returns scalar."""
        B, C, H, W = pred.shape
        ssim_vals = []
        for c in range(C):
            p = pred[:, c:c+1]
            t = target[:, c:c+1]
            mu_p = F.conv2d(p, self.kernel, padding=self.pad)
            mu_t = F.conv2d(t, self.kernel, padding=self.pad)
            mu_pp = mu_p ** 2
            mu_tt = mu_t ** 2
            mu_pt = mu_p * mu_t
            sigma_pp = F.conv2d(p * p, self.kernel, padding=self.pad) - mu_pp
            sigma_tt = F.conv2d(t * t, self.kernel, padding=self.pad) - mu_tt
            sigma_pt = F.conv2d(p * t, self.kernel, padding=self.pad) - mu_pt
            num = (2 * mu_pt + self.C1) * (2 * sigma_pt + self.C2)
            den = (mu_pp + mu_tt + self.C1) * (sigma_pp + sigma_tt + self.C2)
            ssim_vals.append((num / den.clamp(min=1e-8)).mean())
        return 1.0 - torch.stack(ssim_vals).mean()


# ---------------------------------------------------------------------------
# PatchGAN discriminator (optional adversarial loss)
# ---------------------------------------------------------------------------

class PatchDiscriminator(nn.Module):
    """70×70 PatchGAN discriminator (from pix2pix)."""

    def __init__(self, in_channels: int = 3, ndf: int = 64):
        super().__init__()

        def block(ic, oc, norm=True):
            layers = [nn.Conv2d(ic, oc, 4, stride=2, padding=1, bias=not norm)]
            if norm:
                layers.append(nn.BatchNorm2d(oc))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *block(in_channels, ndf, norm=False),
            *block(ndf, ndf * 2),
            *block(ndf * 2, ndf * 4),
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=1, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PatchGANLoss(nn.Module):
    """LSGAN-style adversarial loss (MSE on patch predictions)."""

    def __init__(self):
        super().__init__()

    def generator_loss(self, disc: PatchDiscriminator, fake: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(disc(fake), torch.ones_like(disc(fake)))

    def discriminator_loss(
        self,
        disc: PatchDiscriminator,
        real: torch.Tensor,
        fake: torch.Tensor,
    ) -> torch.Tensor:
        loss_real = F.mse_loss(disc(real), torch.ones_like(disc(real)))
        loss_fake = F.mse_loss(disc(fake.detach()), torch.zeros_like(disc(fake.detach())))
        return (loss_real + loss_fake) * 0.5


# ---------------------------------------------------------------------------
# Combined total loss
# ---------------------------------------------------------------------------

class TotalLoss(nn.Module):
    """Weighted combination of Pixel + SSIM [+ Adversarial].

    Args:
        lambda_l1:     weight for pixel loss (default 1.0)
        lambda_ssim:   weight for SSIM loss (default 0.5)
        lambda_adv:    weight for adversarial generator loss (0 = disabled)
        pixel_loss:    "charbonnier" (default, NAFNet/Restormer SOTA) or "l1"
        disc:          PatchDiscriminator instance (required if lambda_adv > 0)
    """

    def __init__(
        self,
        lambda_l1: float = 1.0,
        lambda_ssim: float = 0.5,
        lambda_adv: float = 0.0,
        pixel_loss: str = "charbonnier",
        disc: Optional[PatchDiscriminator] = None,
    ):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_adv = lambda_adv
        self.pixel_loss_name = pixel_loss

        self.charbonnier = CharbonnierLoss() if pixel_loss == "charbonnier" else None
        self.ssim = SSIMLoss()
        self.adv = PatchGANLoss() if lambda_adv > 0 else None
        self.disc = disc

    def _pixel_loss(self, pred, target):
        if self.charbonnier is not None:
            return self.charbonnier(pred, target)
        return F.l1_loss(pred, target)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        losses = {}
        total = torch.tensor(0.0, device=pred.device)

        if self.lambda_l1 > 0:
            losses["pixel"] = self._pixel_loss(pred, target)
            total = total + self.lambda_l1 * losses["pixel"]

        if self.lambda_ssim > 0:
            losses["ssim"] = self.ssim(pred, target)
            total = total + self.lambda_ssim * losses["ssim"]

        if self.lambda_adv > 0 and self.adv is not None and self.disc is not None:
            losses["adv"] = self.adv.generator_loss(self.disc, pred)
            total = total + self.lambda_adv * losses["adv"]

        losses["total"] = total
        return total, losses


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pred   = torch.rand(2, 3, 128, 128)
    target = torch.rand(2, 3, 128, 128)

    print("=== SSIM loss ===")
    ssim_loss = SSIMLoss()
    print(f"  {ssim_loss(pred, target).item():.4f}  (expect ~0.5 for random)")

    print("=== TotalLoss (no adv) ===")
    loss_fn = TotalLoss(lambda_l1=1.0, lambda_ssim=0.5)
    total, breakdown = loss_fn(pred, target)
    for k, v in breakdown.items():
        print(f"  {k}: {v.item():.4f}")

    print("=== PatchGANLoss ===")
    disc = PatchDiscriminator()
    adv = PatchGANLoss()
    g_loss = adv.generator_loss(disc, pred)
    d_loss = adv.discriminator_loss(disc, target, pred)
    print(f"  generator={g_loss.item():.4f}  discriminator={d_loss.item():.4f}")

    print("\nAll losses OK")
