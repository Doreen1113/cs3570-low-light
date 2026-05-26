"""Person A — Advanced augmentation for low-light paired dataset.

Extra pair-aware transforms beyond dataset.py:
  PairedRandomRotate90   — 0/90/180/270 rotation, applied identically to both
  PairedColorJitter      — brightness/contrast jitter on input only (gt unchanged)
  SyntheticNoisePair     — generate extra (noisy, clean) pairs from GT images
  make_train_transform   — convenience builder used by train.py

Run standalone to sanity-check augmented pairs:
  python augment.py [--root H:/low-light-data/low-light] [--n 8]
"""

import argparse
import random
from pathlib import Path
from typing import Tuple

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from dataset import (
    PairedCompose,
    PairedLowLightDataset,
    PairedRandomCrop,
    PairedRandomFlip,
    PairedToTensor,
    _default_to_tensor,
)


# ---------------------------------------------------------------------------
# New pair-aware transforms
# ---------------------------------------------------------------------------

class PairedRandomRotate90:
    """Rotate both images by a random multiple of 90 degrees."""
    paired = True

    def __call__(self, a: Image.Image, b: Image.Image):
        k = random.choice([0, 1, 2, 3])
        if k:
            a = a.rotate(k * 90, expand=True)
            b = b.rotate(k * 90, expand=True)
        return a, b


class PairedColorJitter:
    """Apply brightness/contrast jitter to the input image only.

    The GT is left unchanged because we do not want to shift the target
    distribution — only simulate variation in the degraded input.
    """
    paired = True

    def __init__(self, brightness: float = 0.2, contrast: float = 0.2):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, inp: Image.Image, gt: Image.Image):
        bf = 1.0 + random.uniform(-self.brightness, self.brightness)
        cf = 1.0 + random.uniform(-self.contrast, self.contrast)
        inp = TF.adjust_brightness(inp, bf)
        inp = TF.adjust_contrast(inp, cf)
        return inp, gt


# ---------------------------------------------------------------------------
# Synthetic noise augmentation dataset
# ---------------------------------------------------------------------------

class SyntheticNoisePair(Dataset):
    """Wraps a PairedLowLightDataset and adds synthetic Gaussian noise to GT
    images to produce extra (noisy_input, clean_gt) training pairs.

    This effectively doubles (or N-tuples) the dataset size without violating
    the 'no extra dataset' rule — we only use the provided GT images.
    """

    def __init__(
        self,
        base_dataset: PairedLowLightDataset,
        sigma_range: Tuple[float, float] = (0.02, 0.15),
        repeat: int = 1,
    ):
        self.base = base_dataset
        self.sigma_lo, self.sigma_hi = sigma_range
        self.repeat = repeat  # how many synthetic copies per real sample

    def __len__(self) -> int:
        return len(self.base) * (1 + self.repeat)

    def __getitem__(self, idx: int):
        real_len = len(self.base)
        if idx < real_len:
            return self.base[idx]
        # synthetic: take gt from a base sample and corrupt it
        base_idx = (idx - real_len) % real_len
        _, gt = self.base[base_idx]
        sigma = random.uniform(self.sigma_lo, self.sigma_hi)
        noisy = (gt + torch.randn_like(gt) * sigma).clamp(0.0, 1.0)
        return noisy, gt


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def make_train_transform(crop_size: int = 256) -> PairedCompose:
    """Standard transform chain used by train.py."""
    return PairedCompose([
        PairedRandomCrop(crop_size),
        PairedRandomFlip(p_h=0.5, p_v=0.0),
        PairedRandomRotate90(),
        PairedColorJitter(brightness=0.15, contrast=0.15),
        PairedToTensor(),
    ])


# ---------------------------------------------------------------------------
# Standalone sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="H:/low-light-data/low-light")
    parser.add_argument("--n", type=int, default=8, help="number of pairs to save")
    parser.add_argument("--out", default="H:/low-light-data/augment_check")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tf = make_train_transform(256)
    base_ds = PairedLowLightDataset(root / "train", transform=tf)
    syn_ds = SyntheticNoisePair(base_ds, repeat=1)

    print(f"base train: {len(base_ds)}  with synthetic: {len(syn_ds)}")

    for i in range(min(args.n, len(syn_ds))):
        inp, gt = syn_ds[i]
        inp_pil = TF.to_pil_image(inp)
        gt_pil = TF.to_pil_image(gt)
        inp_pil.save(out_dir / f"{i:03d}_in.png")
        gt_pil.save(out_dir / f"{i:03d}_gt.png")
        print(f"  [{i}] in={tuple(inp.shape)} gt={tuple(gt.shape)} "
              f"in_mean={inp.mean():.3f} gt_mean={gt.mean():.3f}")

    print(f"\nSaved {min(args.n, len(syn_ds))} pairs to {out_dir}")
    print("Check that in/gt are geometrically aligned (same crop/rotation).")
