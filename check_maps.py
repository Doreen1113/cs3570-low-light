"""視覺化檢查 illum map 和 noise map 是否合理。

輸出一張拼圖：原圖 | illum map | noise map
用肉眼確認：
  - illum map：亮的地方白、暗的地方黑 → 應該幾乎全黑（low-light 圖）
  - noise map：噪的地方白、乾淨的地方黑 → 暗部邊緣應該比較白

執行：
  python check_maps.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont

from illum_map import compute_illum
from noise_map import compute_noise

# ---- 設定 ----
DATA_ROOT = Path("H:/low-light-data/low-light/val")
OUT_DIR   = Path("H:/low-light-data/map_check")
N         = 4   # 幾張圖
# --------------

OUT_DIR.mkdir(parents=True, exist_ok=True)

stems = sorted(DATA_ROOT.glob("*-in.webp"))[:N]

for p in stems:
    name = p.stem.replace("-in", "")

    # 讀原圖
    img_pil = Image.open(p).convert("RGB")
    img = TF.to_tensor(img_pil).unsqueeze(0)

    # 計算 map
    with torch.no_grad():
        illum = compute_illum(img, guided=False).squeeze()   # [1,H,W]
        noise = compute_noise(img).squeeze()                  # [1,H,W]

    # 轉成 PIL
    illum_pil = TF.to_pil_image(illum)
    noise_pil = TF.to_pil_image((noise * 5).clamp(0, 1))  # 放大 5 倍才看得清楚

    # 縮小到 256x256 方便看
    size = (256, 256)
    img_small   = img_pil.resize(size)
    illum_small = illum_pil.resize(size)
    noise_small = noise_pil.resize(size)

    # 拼成一排：原圖 | illum | noise
    W, H = size
    canvas = Image.new("RGB", (W * 3 + 20, H + 30), (30, 30, 30))
    canvas.paste(img_small,   (0,       30))
    canvas.paste(illum_small.convert("RGB"), (W + 10,  30))
    canvas.paste(noise_small.convert("RGB"), (W * 2 + 20, 30))

    # 加標籤
    draw = ImageDraw.Draw(canvas)
    draw.text((10,   5), "原圖 (input)",    fill=(255,255,255))
    draw.text((W+20, 5), "Illum Map",       fill=(255,255,255))
    draw.text((W*2+30,5),"Noise Map (x5)",  fill=(255,255,255))

    out_path = OUT_DIR / f"{name}_compare.png"
    canvas.save(out_path)
    print(f"  saved: {out_path.name}  illum_mean={illum.mean():.3f}  noise_mean={noise.mean():.4f}")

print(f"\n開啟資料夾查看：{OUT_DIR}")
