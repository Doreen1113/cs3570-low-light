"""eval.py — 載入訓練好的 checkpoint，在 val set 上算 PSNR/SSIM/LPIPS。

用法：
  python eval.py --root /kaggle/input/datasets/doreen071/cs3570-lowlight \
                 --ckpt /kaggle/working/checkpoints/best.pth

輸出範例：
  Loaded: best.pth (epoch=10, train best_psnr=18.34)
  Eval on 1000 val images...
  PSNR  = 18.34 dB
  SSIM  = 0.6512
  LPIPS = 0.2845
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PairedLowLightDataset
from illum_map import compute_illum
from noise_map import compute_noise
from metrics import MetricTracker, evaluate_batch
from model.restoration import RestorationNet
from postprocess import postprocess


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="dataset root (contains val/)")
    p.add_argument("--ckpt", required=True, help="path to .pth checkpoint")
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--guided-illum", action="store_true")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- model ----
    model = RestorationNet(width=args.width).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded: {Path(args.ckpt).name}  "
          f"(epoch={ckpt.get('epoch', '?')}, "
          f"train best_psnr={ckpt.get('best_psnr', 0):.2f})")

    # ---- data ----
    val_ds = PairedLowLightDataset(Path(args.root) / "val")
    loader = DataLoader(val_ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=0, pin_memory=True)
    print(f"Eval on {len(val_ds)} val images...")

    # ---- run ----
    tracker = MetricTracker()
    for inp, gt in tqdm(loader, desc="Eval", ncols=100):
        inp, gt = inp.to(device), gt.to(device)
        noise_map = compute_noise(inp)
        illum_map = compute_illum(inp, guided=args.guided_illum)
        pred = model(inp, noise_map, illum_map)
        pred = postprocess(pred, chroma_alpha=0.3, sharpen_strength=0.3)
        tracker.update(evaluate_batch(pred, gt))

    s = tracker.summary()
    print()
    print(f"PSNR  = {s.get('psnr', 0):.4f} dB")
    print(f"SSIM  = {s.get('ssim', 0):.4f}")
    if "lpips" in s:
        print(f"LPIPS = {s['lpips']:.4f}")
    else:
        print("LPIPS = (lpips not installed, run: pip install lpips)")


if __name__ == "__main__":
    main()
