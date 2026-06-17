"""
Measure model parameters (M) and FLOPs (GFLOPs) for the full inference pipeline.

Usage:
    python measure_flops.py
    python measure_flops.py --num-models 3   # if you ensemble 3 checkpoints
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile, clever_format

from model.restoration import RestorationNet
from noise_map import LocalStdNoise
from illum_map import MaxChannelIllum


# ---------------------------------------------------------------------------
# Wrap the FULL pipeline into a single nn.Module so thop can trace it cleanly
# ---------------------------------------------------------------------------

class FullPipeline(nn.Module):
    """Wraps illum_map + noise_map + RestorationNet for FLOPs measurement."""

    def __init__(self, width=32):
        super().__init__()
        self.restoration = RestorationNet(width=width)
        # These have zero params but non-zero FLOPs
        self._noise_estim = LocalStdNoise(radius=3)
        self._illum_estim = MaxChannelIllum()

    def forward(self, img):
        noise = self._noise_map(img)
        illum = self._illum_map(img)
        return self.restoration(img, noise, illum)

    def _noise_map(self, img):
        """LocalStdNoise — local std via avg_pool."""
        gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        k = 2 * self._noise_estim.r + 1
        mean = F.avg_pool2d(gray, k, stride=1, padding=self._noise_estim.r)
        mean_sq = F.avg_pool2d(gray ** 2, k, stride=1, padding=self._noise_estim.r)
        var = (mean_sq - mean ** 2).clamp(min=0.0)
        return var.sqrt().clamp(0.0, 1.0)

    def _illum_map(self, img):
        """MaxChannelIllum — max(R, G, B) per pixel."""
        return img.max(dim=1, keepdim=True).values.clamp(0.0, 1.0)


def measure(num_models: int = 1, width: int = 32):
    device = torch.device("cpu")  # CPU for consistent FLOPs measurement

    pipeline = FullPipeline(width=width).to(device).eval()

    # Standard input size as required: 256x256x3
    dummy = torch.randn(1, 3, 256, 256).to(device)

    # --- Parameters ---
    total_params = sum(p.numel() for p in pipeline.parameters())
    trainable_params = sum(p.numel() for p in pipeline.parameters() if p.requires_grad)
    params_M = total_params / 1e6

    # --- FLOPs via thop ---
    macs, params = profile(pipeline, inputs=(dummy,), verbose=False)
    # thop reports MACs; convention: FLOPs ≈ 2 × MACs
    flops = macs * 2
    gflops = flops / 1e9

    print("=" * 55)
    print("  INFERENCE PIPELINE METRICS  (single model)")
    print("=" * 55)
    print(f"  Input size   : 1 × 3 × 256 × 256")
    print(f"  Total params : {params_M:.4f} M")
    print(f"  MACs         : {macs/1e9:.4f} GMACs")
    print(f"  FLOPs (2×MACs): {gflops:.4f} GFLOPs")
    print()

    if num_models > 1:
        total_params_M = params_M * num_models
        total_gflops = gflops * num_models
        print(f"  Ensemble of {num_models} models:")
        print(f"  Total params : {total_params_M:.4f} M  ({params_M:.4f} × {num_models})")
        print(f"  Total FLOPs  : {total_gflops:.4f} GFLOPs  ({gflops:.4f} × {num_models})")
        print()
        print("  *** SUBMIT THESE VALUES ***")
        print(f"  Model size : {total_params_M:.2f} M params")
        print(f"  FLOPs      : {total_gflops:.2f} GFLOPs")
    else:
        print("  *** SUBMIT THESE VALUES ***")
        print(f"  Model size : {params_M:.2f} M params")
        print(f"  FLOPs      : {gflops:.2f} GFLOPs")

    print("=" * 55)

    # Breakdown by component
    print("\n  Component breakdown:")
    restoration_params = sum(p.numel() for p in pipeline.restoration.parameters()) / 1e6
    print(f"  RestorationNet  : {restoration_params:.4f} M params")
    print(f"  LocalStdNoise   : 0 params  (pure math, but counted in FLOPs)")
    print(f"  MaxChannelIllum : 0 params  (pure math, but counted in FLOPs)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-models", type=int, default=1,
                        help="Number of checkpoints in your ensemble (default: 1)")
    parser.add_argument("--width", type=int, default=32,
                        help="NAFNet width (default: 32)")
    args = parser.parse_args()

    measure(num_models=args.num_models, width=args.width)
