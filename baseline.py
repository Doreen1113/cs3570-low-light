"""Gamma correction baseline — 不用任何 model。

用最簡單的 gamma correction 亮化圖片，計算 val PSNR。
這是最低標準，你們的 model 一定要比這個好。

執行：
  python baseline.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from dataset import PairedLowLightDataset
from metrics import MetricTracker, evaluate_batch

parser = argparse.ArgumentParser()
parser.add_argument("--root", default="H:/low-light-data/low-light")
args = parser.parse_args()

DATA_ROOT = Path(args.root)
GAMMA_VALUES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]  # 試幾個 gamma 找最好的

val_ds = PairedLowLightDataset(DATA_ROOT / "val")

loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

print(f"Val set: {len(val_ds)} images\n")
print(f"{'Gamma':>8} | {'PSNR':>8} | {'SSIM':>8}")
print("-" * 32)

best_psnr = 0
best_gamma = 0

for gamma in GAMMA_VALUES:
    tracker = MetricTracker()
    for i, (inp, gt) in enumerate(loader):
        pred = inp.pow(gamma).clamp(0, 1)
        tracker.update(evaluate_batch(pred, gt))
        print(f"  γ={gamma}  [{(i+1)*4}/{len(val_ds)}]", end="\r")

    summary = tracker.summary()
    psnr = summary.get("psnr", 0)
    ssim = summary.get("ssim", 0)
    print(f"  γ={gamma:.1f}  | {psnr:>8.2f} | {ssim:>8.4f}")

    if psnr > best_psnr:
        best_psnr = psnr
        best_gamma = gamma

print(f"\nBest baseline: γ={best_gamma}  PSNR={best_psnr:.2f} dB")
print(f"\n你們的 model 目標：PSNR > {best_psnr:.2f} dB（baseline）")
print("好的結果：PSNR > baseline + 3~5 dB")
