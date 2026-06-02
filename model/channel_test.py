"""測試 1-channel map vs 3-channel map 哪個效果比較好。

1ch：noise/illum 各一個灰階 map（目前版本）
3ch：noise/illum 各三個 per-channel map（R/G/B 分開算）

執行：
  python model/channel_test.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from dataset import PairedLowLightDataset
from model.nafnet import NAFNet
from model.restoration import RestorationNet

DATA_ROOT = Path("H:/low-light-data/low-light/val")
STEPS = 300
LR    = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ds = PairedLowLightDataset(DATA_ROOT)
inp, gt = ds[0]
inp = inp[:, :256, :256].unsqueeze(0).to(device)
gt  = gt[:, :256, :256].unsqueeze(0).to(device)


def run_overfit(model, inp, gt, label):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    t0 = time.time()
    for step in range(STEPS + 1):
        pred = model(inp)
        loss = F.l1_loss(pred, gt)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = model(inp)
        mse  = ((pred - gt)**2).mean()
        psnr = 10 * torch.log10(1.0 / mse.clamp(min=1e-10))
    elapsed = time.time() - t0
    print(f"  {label:30s}  PSNR={psnr.item():.2f} dB  ({elapsed:.0f}s)")
    return psnr.item()


# ---------------------------------------------------------------------------
# 版本 1：1ch map（現有版本，in_channels=5）
# ---------------------------------------------------------------------------

class RestorationNet1ch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = NAFNet(in_channels=5, width=32)

    def forward(self, inp):
        from illum_map import compute_illum
        from noise_map import compute_noise
        with torch.no_grad():
            noise = compute_noise(inp)       # [B,1,H,W]
            illum = compute_illum(inp)       # [B,1,H,W]
        x = torch.cat([inp, noise, illum], dim=1)  # [B,5,H,W]
        out = self.net(x)
        return (inp + out).clamp(0, 1)


# ---------------------------------------------------------------------------
# 版本 2：3ch map（per-channel，in_channels=9）
# ---------------------------------------------------------------------------

def compute_noise_3ch(img):
    """LocalStd per R/G/B channel → [B,3,H,W]"""
    import torch.nn.functional as F
    k = 7
    maps = []
    for c in range(3):
        ch = img[:, c:c+1]
        mean = F.avg_pool2d(ch, k, stride=1, padding=k//2)
        mean_sq = F.avg_pool2d(ch**2, k, stride=1, padding=k//2)
        var = (mean_sq - mean**2).clamp(min=0)
        maps.append(var.sqrt())
    return torch.cat(maps, dim=1)  # [B,3,H,W]


def compute_illum_3ch(img):
    """Per-channel brightness map → [B,3,H,W]"""
    return img.clone()  # 每個 channel 本身就是亮度


class RestorationNet3ch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = NAFNet(in_channels=9, width=32)

    def forward(self, inp):
        with torch.no_grad():
            noise = compute_noise_3ch(inp)   # [B,3,H,W]
            illum = compute_illum_3ch(inp)   # [B,3,H,W]
        x = torch.cat([inp, noise, illum], dim=1)  # [B,9,H,W]
        out = self.net(x)
        return (inp + out).clamp(0, 1)


# ---------------------------------------------------------------------------
# 跑比較
# ---------------------------------------------------------------------------

print(f"Device: {device}")
print(f"Image: {tuple(inp.shape)}\n")
print("比較 1ch vs 3ch map（各跑 300 steps overfit test）")
print("-" * 55)

psnr_1ch = run_overfit(RestorationNet1ch().to(device), inp, gt, "1ch map (noise+illum, 5ch input)")
psnr_3ch = run_overfit(RestorationNet3ch().to(device), inp, gt, "3ch map (per-channel, 9ch input)")

print("-" * 55)
diff = psnr_3ch - psnr_1ch
print(f"\n結果：3ch - 1ch = {diff:+.2f} dB")
if diff > 0.5:
    print("→ 3ch 明顯更好，考慮換成 9-channel 輸入")
elif diff > 0:
    print("→ 3ch 略好，但差距小，1ch 已夠用")
else:
    print("→ 1ch 已足夠，不需要改")
