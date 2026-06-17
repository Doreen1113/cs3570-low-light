import torch
from metrics import psnr

# 測試 1：完全相同的圖 → 應該接近 100 dB
x = torch.rand(2, 3, 256, 256)
print(f"Perfect: {psnr(x, x).mean().item():.1f} dB")  # 期望 ~100

# 測試 2：完全隨機 → 應該接近 7-8 dB
y = torch.rand(2, 3, 256, 256)
print(f"Random: {psnr(x, y).mean().item():.2f} dB")   # 期望 ~7-8

# 測試 3：接近但有小誤差 → 應該在 30-40 dB
z = (x + torch.randn_like(x) * 0.01).clamp(0, 1)
print(f"Small noise: {psnr(x, z).mean().item():.2f} dB")  # 期望 ~40