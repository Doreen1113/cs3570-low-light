# Person C — Restoration Network

## 負責檔案
- `model/nafnet.py` — NAFNet 架構
- `model/restoration.py` — RestorationNet wrapper
- `model/overfit_test.py` — 自測腳本
- `check_maps.py` — B 的 map 視覺驗證
- `baseline.py` — Gamma correction baseline

---

## 完成項目

### 1. NAFNet 架構（5-channel 輸入）
標準 NAFNet 修改第一層 conv，從 3-channel（RGB）改為 5-channel（RGB + noise map + illum map），讓 model 能接收 B 的 conditioning 資訊。

### 2. Forward Pass 驗證
```
input: [B, 3, H, W] + noise_map [B, 1, H, W] + illum_map [B, 1, H, W]
output: [B, 3, H, W]  restored image in [0, 1]
```
Shape 驗證通過，模型參數：**14.35M**

### 3. Width 比較實驗
| 版本 | 參數量 | 時間 | PSNR @300步 |
|------|--------|------|-------------|
| width=16 | 3.64M | 43s | 42.75 dB |
| width=32 | 14.35M | 52s | 49.58 dB |

**結論：使用 width=32**（速度只慢 20%，PSNR 差 6.8 dB）

### 4. B 的 Map 視覺驗證
- Illum map：正確捕捉亮度分布（低光圖幾乎全黑）
- Noise map：正確顯示 noise 分布（暗部邊緣偏白）

### 5. 真實 Map 整合測試
| 版本 | PSNR @300步 |
|------|-------------|
| Dummy map（全零）| 49.58 dB |
| 真實 map（B 的輸出）| **49.99 dB (+0.41 dB)** |

Map conditioning 在 overfit test 已有正向效果，完整訓練預期差距更大。

### 6. Baseline（Gamma Correction，不用任何 model）
在 1000 張 val set 上測試所有 gamma 值：

| Gamma | PSNR | SSIM |
|-------|------|------|
| γ=0.2 | 11.65 dB | 0.2205 |
| γ=0.3 | 13.92 dB | 0.2263 |
| γ=0.4 | 14.87 dB | 0.2352 |
| **γ=0.5** | **15.02 dB** | **0.2446** ← Best |
| γ=0.6 | 14.74 dB | 0.2520 |
| γ=0.7 | 14.18 dB | 0.2549 |

**Best baseline：γ=0.5，PSNR = 15.02 dB**

這是我們的最低標準，model 必須超過這個數字。
好的結果：PSNR > 15.02 + 3~5 dB（約 18~20 dB 以上）

---

## 交接給 B

`illum_map.py` 和 `noise_map.py` 已有可以跑的簡單版本（scaffold），B 的工作是改進裡面的演算法讓 map 更準。

**唯一限制：output shape 不能改**
```python
noise_map: [B, 1, H, W]  # float32, 值域 [0, 1]
illum_map: [B, 1, H, W]  # float32, 值域 [0, 1]
```
C 的 model 已經寫死接這個 shape，改了會壞掉。

**現在 scaffold 用的是簡單版：**
- `illum_map.py`：`L = max(R, G, B)`（MaxChannel），可改成 guided filter 版本
- `noise_map.py`：局部標準差（LocalStd），可改成訓練 CNN estimator（裡面已有框架）

**B 改完後：**
1. 跑 `python check_maps.py` 確認視覺上合理
2. 告知 C，C 會重跑 `python model/overfit_test.py` 確認 PSNR 有提升
3. 目前真實 map 的 overfit PSNR = 49.99 dB，改進後應該要更高

---

## 待完成（等 D 整合後）

### Ablation Study（4 個版本）
等 D 跑完整訓練後，需要比較：

| 版本 | 輸入 channel |
|------|-------------|
| Baseline（無 map）| RGB（3ch）|
| + noise map only | RGB + noise（4ch）|
| + illum map only | RGB + illum（4ch）|
| Full（兩個都有）| RGB + noise + illum（5ch）|

這四個數字是 novelty 的核心證據，必須放進 report 和 presentation。

### 架構圖
需要畫一張 pipeline 圖給 presentation 用：
```
Input → [concat: noise map, illum map] → NAFNet(5ch) → Restored Output
```

---

## 使用方式

```powershell
# Forward pass 驗證
python model/restoration.py

# Overfit 自測（確認架構正確）
python model/overfit_test.py

# 視覺驗證 B 的 map
python check_maps.py

# Gamma baseline
python baseline.py
```

---

## 接入 train.py 的方式

```python
from model import RestorationNet
from illum_map import compute_illum
from noise_map import compute_noise

model = RestorationNet(width=32)

# training loop 裡
noise_map = compute_noise(inp)
illum_map = compute_illum(inp)
pred = model(inp, noise_map, illum_map)
```
