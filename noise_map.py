"""Person B (part 2) — Noise map estimation.

Two estimators:
  LocalStdNoise   — sliding-window local std (fast, zero-param)
  CNNNoiseEstim   — tiny CNN trained from GT images + synthetic noise

Both output [B, 1, H, W] tensors (σ map, values roughly in [0, 1]).

Usage:
  # Use the fast estimator (no training needed)
  from noise_map import compute_noise, LocalStdNoise

  # Train the CNN estimator
  python noise_map.py --train --root H:/low-light-data/low-light

  # Visualise on a few images
  python noise_map.py [--cnn] [--root ...]
"""

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Fast estimator: local standard deviation
# ---------------------------------------------------------------------------

class LocalStdNoise:
    """Estimate per-pixel noise level as local standard deviation.

    Converts image to grayscale, computes std over a (2r+1)^2 window, and
    can optionally suppress strong edges so texture boundaries are less likely
    to be mistaken for sensor noise.
    """

    def __init__(self, radius: int = 3, edge_suppression: float = 0.0):
        self.r = radius
        self.edge_suppression = edge_suppression

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """img: [B,3,H,W] or [3,H,W], returns [B,1,H,W] or [1,H,W]"""
        batched = img.dim() == 4
        if not batched:
            img = img.unsqueeze(0)

        # Grayscale  [B,1,H,W]
        gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]

        k = 2 * self.r + 1
        mean = F.avg_pool2d(gray, k, stride=1, padding=self.r)
        mean_sq = F.avg_pool2d(gray ** 2, k, stride=1, padding=self.r)
        var = (mean_sq - mean ** 2).clamp(min=0.0)
        sigma = var.sqrt()

        if self.edge_suppression > 0:
            edge = self._edge_strength(gray)
            edge = F.avg_pool2d(edge, k, stride=1, padding=self.r)
            sigma = (sigma - self.edge_suppression * edge).clamp(min=0.0)

        sigma = sigma.clamp(0.0, 1.0)

        return sigma if batched else sigma.squeeze(0)

    def _edge_strength(self, gray: torch.Tensor) -> torch.Tensor:
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            device=gray.device,
            dtype=gray.dtype,
        ).view(1, 1, 3, 3) / 8.0
        sobel_y = sobel_x.transpose(-1, -2)
        gx = F.conv2d(gray, sobel_x, padding=1)
        gy = F.conv2d(gray, sobel_y, padding=1)
        return (gx.square() + gy.square()).sqrt()


# ---------------------------------------------------------------------------
# CNN estimator
# ---------------------------------------------------------------------------

class _ConvBnRelu(nn.Sequential):
    def __init__(self, in_c, out_c, k=3, p=1):
        super().__init__(
            nn.Conv2d(in_c, out_c, k, padding=p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )


class CNNNoiseEstim(nn.Module):
    """Tiny DnCNN-style network that maps RGB image → σ map [B,1,H,W].

    Train by feeding (GT + synthetic noise) pairs; target is the known σ map.
    """

    def __init__(self, depth: int = 6, channels: int = 32):
        super().__init__()
        layers = [_ConvBnRelu(3, channels)]
        for _ in range(depth - 2):
            layers.append(_ConvBnRelu(channels, channels))
        layers.append(nn.Conv2d(channels, 1, 3, padding=1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.net(img)


# ---------------------------------------------------------------------------
# Training dataset for CNNNoiseEstim
# ---------------------------------------------------------------------------

class _SyntheticNoiseDS(Dataset):
    """Yields (noisy_gt, sigma_map) pairs for training CNNNoiseEstim."""

    def __init__(self, root: Path, crop: int = 128, n_samples: int = 4000):
        self.paths = sorted(root.glob("*-gt.webp"))
        self.crop = crop
        self.n = n_samples

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        p = random.choice(self.paths)
        gt = TF.to_tensor(Image.open(p).convert("RGB"))
        C, H, W = gt.shape
        x = random.randint(0, W - self.crop)
        y = random.randint(0, H - self.crop)
        gt = gt[:, y:y + self.crop, x:x + self.crop]

        sigma_val = random.uniform(0.01, 0.20)
        noise = torch.randn_like(gt) * sigma_val
        noisy = (gt + noise).clamp(0.0, 1.0)
        sigma_map = torch.full((1, self.crop, self.crop), sigma_val)
        return noisy, sigma_map


def train_cnn_estimator(
    root: Path,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3,
    save_path: Path = Path("noise_estimator.pth"),
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNNoiseEstim().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ds = _SyntheticNoiseDS(root / "train")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    for epoch in range(1, epochs + 1):
        total = 0.0
        for noisy, sigma_gt in loader:
            noisy, sigma_gt = noisy.to(device), sigma_gt.to(device)
            pred = model(noisy)
            loss = F.l1_loss(pred, sigma_gt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"epoch {epoch}/{epochs}  L1={total/len(loader):.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Saved to {save_path}")
    return model


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

_fast_estimator = LocalStdNoise()


def compute_noise(img: torch.Tensor, model: CNNNoiseEstim | None = None) -> torch.Tensor:
    """Compute noise map. Uses CNN if model provided, else LocalStd."""
    if model is not None:
        model.eval()
        with torch.no_grad():
            return model(img)
    return _fast_estimator(img)


# ---------------------------------------------------------------------------
# Standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="H:/low-light-data/low-light")
    parser.add_argument("--train", action="store_true", help="train CNN estimator")
    parser.add_argument("--cnn", action="store_true", help="use CNN for visualisation")
    parser.add_argument("--weights", default="noise_estimator.pth")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--out", default="H:/low-light-data/noise_check")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.train:
        train_cnn_estimator(root, save_path=Path(args.weights))

    cnn_model = None
    if args.cnn:
        cnn_model = CNNNoiseEstim()
        cnn_model.load_state_dict(torch.load(args.weights, map_location="cpu"))
        cnn_model.eval()

    stems = sorted((root / "train").glob("*-in.webp"))[:args.n]
    for p in stems:
        img = TF.to_tensor(Image.open(p).convert("RGB")).unsqueeze(0)
        sigma = compute_noise(img, cnn_model).squeeze(0)
        sigma_pil = TF.to_pil_image((sigma * 5).clamp(0, 1))  # boost for visibility
        name = p.stem.replace("-in", "")
        sigma_pil.save(out_dir / f"{name}_noise.png")
        print(f"  {p.name}: σ mean={sigma.mean():.4f}  max={sigma.max():.4f}")

    print(f"\nSaved {len(stems)} noise maps to {out_dir}")
