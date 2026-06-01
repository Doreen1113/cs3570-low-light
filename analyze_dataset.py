"""Person A — Dataset statistics analysis.

Computes and plots:
  - Sample counts per split (train / val / test)
  - Brightness histogram (input vs GT)
  - Noise level histogram (input vs GT, MAD estimator)

Usage:
  python analyze_dataset.py --root H:/low-light-data/low-light --n 500 --out stats/
"""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


IMG_EXT = ".webp"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_stems(split_dir: Path, require_gt: bool) -> list[Path]:
    suffix = f"-in{IMG_EXT}"
    stems = []
    for p in sorted(split_dir.glob(f"*{suffix}")):
        stem = p.name[: -len(suffix)]
        if require_gt and not (split_dir / f"{stem}-gt{IMG_EXT}").exists():
            continue
        stems.append(stem)
    return stems


def _to_gray_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L")).astype(np.float32) / 255.0


def _mean_brightness(img: Image.Image) -> float:
    return float(np.mean(_to_gray_array(img)))


def _noise_sigma(img: Image.Image) -> float:
    """MAD-based noise estimator on horizontal differences."""
    gray = _to_gray_array(img)
    diff = np.abs(np.diff(gray, axis=1))
    return float(np.median(diff) / 0.6745)


# ---------------------------------------------------------------------------
# Analysis routines
# ---------------------------------------------------------------------------

def count_splits(root: Path) -> dict[str, int]:
    counts = {}
    for split in ("train", "val", "test"):
        d = root / split
        if not d.is_dir():
            counts[split] = 0
            continue
        require_gt = split != "test"
        counts[split] = len(_collect_stems(d, require_gt=require_gt))
    return counts


def sample_stats(root: Path, n: int) -> dict:
    """Sample up to n paired images from train and collect brightness + noise."""
    train_dir = root / "train"
    stems = _collect_stems(train_dir, require_gt=True)
    sample = random.sample(stems, min(n, len(stems)))

    in_brightness, gt_brightness = [], []
    in_noise, gt_noise = [], []

    for stem in sample:
        img_in = Image.open(train_dir / f"{stem}-in{IMG_EXT}")
        img_gt = Image.open(train_dir / f"{stem}-gt{IMG_EXT}")
        in_brightness.append(_mean_brightness(img_in))
        gt_brightness.append(_mean_brightness(img_gt))
        in_noise.append(_noise_sigma(img_in))
        gt_noise.append(_noise_sigma(img_gt))

    return {
        "in_brightness": np.array(in_brightness),
        "gt_brightness": np.array(gt_brightness),
        "in_noise": np.array(in_noise),
        "gt_noise": np.array(gt_noise),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_brightness(stats: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 51)
    ax.hist(stats["in_brightness"], bins=bins, alpha=0.6, label="input (low-light)", color="steelblue")
    ax.hist(stats["gt_brightness"], bins=bins, alpha=0.6, label="GT (clean)", color="darkorange")
    ax.set_xlabel("Mean brightness")
    ax.set_ylabel("Count")
    ax.set_title("Brightness distribution — train split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_noise(stats: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    hi = max(stats["in_noise"].max(), stats["gt_noise"].max())
    bins = np.linspace(0, hi * 1.05, 51)
    ax.hist(stats["in_noise"], bins=bins, alpha=0.6, label="input", color="steelblue")
    ax.hist(stats["gt_noise"], bins=bins, alpha=0.6, label="GT", color="darkorange")
    ax.set_xlabel("Estimated noise σ (MAD)")
    ax.set_ylabel("Count")
    ax.set_title("Noise level distribution — train split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {out_path}")


def print_summary(counts: dict, stats: dict) -> None:
    print("\n=== Dataset Summary ===")
    for split, n in counts.items():
        print(f"  {split:5s}: {n:>6d} samples")
    total = counts.get("train", 0) + counts.get("val", 0)
    print(f"  train+val total: {total}")

    def row(label, arr):
        return (f"  {label:<22s}  "
                f"min={arr.min():.4f}  median={np.median(arr):.4f}  max={arr.max():.4f}")

    print("\n=== Image Statistics (train sample) ===")
    print(row("input brightness", stats["in_brightness"]))
    print(row("GT brightness", stats["gt_brightness"]))
    print(row("input noise σ", stats["in_noise"]))
    print(row("GT noise σ", stats["gt_noise"]))
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent / "dataset"))
    parser.add_argument("--n", type=int, default=500, help="images to sample for stats")
    parser.add_argument("--out", default="stats", help="output directory for plots")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = count_splits(root)
    stats = sample_stats(root, args.n)

    print_summary(counts, stats)
    plot_brightness(stats, out_dir / "brightness_hist.png")
    plot_noise(stats, out_dir / "noise_hist.png")
