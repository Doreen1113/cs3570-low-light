"""C 的自測腳本 — 不需要其他人完成就能跑

測試方式：拿 val 裡的 1 張圖，讓 model 對這 1 張訓練 300 步。
PSNR 能從 ~15dB 衝到 35dB+ 就代表架構沒問題。

執行：
  python model/overfit_test.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from dataset import PairedLowLightDataset
from model import RestorationNet

# ---- 設定 ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "val"
STEPS     = 300
LR        = 1e-3
# --------------

total_start = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
print(f"Device: {device}")

# 載入 1 張圖並裁成 256x256（加快速度）
t0 = time.time()
ds  = PairedLowLightDataset(DATA_ROOT)
inp, gt = ds[0]
inp = inp[:, :256, :256].unsqueeze(0).to(device)
gt  = gt[:, :256, :256].unsqueeze(0).to(device)
print(f"Image loaded: input={tuple(inp.shape)}  gt={tuple(gt.shape)}  ({time.time()-t0:.1f}s)")

# 建立 model
t0 = time.time()
model = RestorationNet(width=32).to(device)
n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Model params: {n_params:.2f}M  ({time.time()-t0:.1f}s)\n")

opt = torch.optim.Adam(model.parameters(), lr=LR)

# 計算 B 的 map（已寫好，不需等 B）
from illum_map import compute_illum
from noise_map import compute_noise
with torch.no_grad():
    noise_map = compute_noise(inp)
    illum_map = compute_illum(inp, guided=False)
print(f"noise_map: {tuple(noise_map.shape)}  mean={noise_map.mean():.4f}")
print(f"illum_map: {tuple(illum_map.shape)}  mean={illum_map.mean():.4f}\n")

print("step  |  L1 loss  |  PSNR     |  elapsed  |  step/s")
print("-" * 55)

step_start = time.time()
step_times = []

for step in range(STEPS + 1):
    t_step = time.time()

    pred = model(inp, noise_map, illum_map)  # 使用真實 map
    loss = F.l1_loss(pred, gt)
    opt.zero_grad()
    loss.backward()
    opt.step()

    step_times.append(time.time() - t_step)

    if step % 50 == 0:
        with torch.no_grad():
            mse  = ((pred - gt) ** 2).mean()
            psnr = 10 * torch.log10(1.0 / mse.clamp(min=1e-10))
        elapsed = time.time() - step_start
        avg_step_time = sum(step_times[-50:]) / len(step_times[-50:])
        steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0
        remaining = (STEPS - step) * avg_step_time
        print(f" {step:4d}  |  {loss.item():.4f}     |  {psnr.item():.2f} dB"
              f"  |  {elapsed:5.1f}s     |  {steps_per_sec:.1f}  "
              f"(剩 ~{remaining:.0f}s)")

total_elapsed = time.time() - total_start
print(f"\nTotal elapsed: {total_elapsed:.1f}s")
print("\nResult check:")
print("  PSNR step 300 > 35 dB  -> architecture looks correct")
print("  PSNR step 300 < 20 dB  -> likely has a bug")
