# Final Project 分工計畫：Joint Denoising + Low Light Enhancement

## Context

CS3570 Track 1，NVIDIA 出題。Pipeline 已在 proposal 確定（5 stage），dataset.py 已寫好。
目標：讓四個人**同時開工**，Week 2 整合。不依賴上游完成才能動。
評分重點：PSNR（主指標）、Novelty（創新設計）、Completeness、Report & Presentation。
Deadline：發表 6/12、報告 6/17。

## 模組介面（提前約好，讓大家用 mock 先跑）

```
DataLoader → (input [B,3,H,W], gt [B,3,H,W])  # float32 [0,1]
B 產出：noise_map [B,1,H,W], illum_map [B,1,H,W]
C 輸入：concat → [B,5,H,W]，輸出：restored [B,3,H,W]
D 輸入：restored + gt → loss scalar；post-process → final [B,3,H,W]
```

每個模組都有獨立的 `__main__` 可以 standalone 測試（不需其他人完成）。

---

## 人員分工

### Person A — Data Pipeline + Augmentation
**檔案：** `augment.py`（新建），修改 `dataset.py`

工作清單（可立刻開始，只需要 dataset）：
1. 分析 dataset 統計：亮度 histogram、noise level 估算、train/val 筆數確認
2. 擴充 augmentation：
   - `PairedRandomRotate90`（0/90/180/270，幾何一致）
   - `PairedColorJitter`（對 input 微調 brightness/contrast，gt 不動）
   - Synthetic noise augmentation：對 gt 加 Gaussian noise 擴充 noisy 樣本
3. 測試所有 transform 不會造成 input/gt 不同步

**Novelty 貢獻：** synthetic augmentation 利用 gt 自製更多 noisy pair，有效擴充不到 32k 的小 dataset

### Person B — Noise Map + Illumination Map
**檔案：** `noise_map.py`、`illum_map.py`（新建）

工作清單（只需要 raw 圖片即可開始）：
1. **Illumination map**（`illum_map.py`）：
   - Retinex decomposition：`L = max(R,G,B)` 或 guided filter 版本
   - 輸出 L map [B,1,H,W]，值域 [0,1]
   - standalone 測試：讀一張圖，視覺化 L map
2. **Noise map**（`noise_map.py`）：
   - 方法1（快）：局部標準差估算 σ(x,y)（sliding window）
   - 方法2（強）：小型 CNN（3→1 channel），用 gt 加不同程度 noise 訓練
   - 輸出 σ map [B,1,H,W]
3. 確認兩個 map 的輸出 shape 符合介面定義

**Novelty 貢獻：** 把 noise map + illumination map 作為 conditioning 明確告訴 network「哪裡噪、哪裡暗」，非標準 end-to-end 做法

### Person C — Restoration Network
**檔案：** `model/nafnet.py`（或 `model/scunet.py`）、`model/restoration.py`（新建）

工作清單（用 dummy tensor 就能跑，不需等 B）：
1. 實作或移植 NAFNet（優先，輕量易訓練）
2. 修改 encoder 第一層：`conv(3→C)` 改為 `conv(5→C)`，接受 concat 後的 5-channel 輸入
3. `restoration.py`：wrapper，`forward(img, noise_map, illum_map)` → concat → network
4. 用 dummy tensor `torch.zeros(2,5,256,256)` 跑通 forward pass，確認輸出形狀

```python
# 快速驗證（不需 B 完成）
img = torch.randn(2,3,256,256)
noise = torch.zeros(2,1,256,256)
illum = torch.zeros(2,1,256,256)
out = model(img, noise, illum)  # 應得 [2,3,256,256]
```

**Novelty 貢獻：** 修改 NAFNet 架構接受額外 conditioning channel（noise + illumination），結構本身就是創新

### Person D — Loss + Training Loop + Evaluation + Post-processing
**檔案：** `losses.py`、`metrics.py`、`postprocess.py`、`train.py`（新建）

工作清單（不需 model 訓練完就能寫）：
1. **`losses.py`**：
   - L1 loss
   - SSIM loss（`pytorch-msssim` 或手寫）
   - Perceptual loss（VGG16 feature map L1，pretrained backbone 非 restoration model，合規）
   - Adversarial loss（PatchGAN discriminator，optional，用 flag 開關）
   - `total_loss = λ1*L1 + λ2*SSIM + λ3*perceptual [+ λ4*adv]`
2. **`metrics.py`**：
   - PSNR（主要）、SSIM、LPIPS 計算
   - 支援 batch 輸入，輸出 dict
3. **`postprocess.py`**：
   - YCbCr chroma compensation：把 restored 的 Cb/Cr 從 gt 方向微調
   - Unsharp masking：細節銳化
4. **`train.py`**：training loop、checkpoint、val 監控
   - 用 random tensor 先跑通整個 loop，等 C 的 model 接入即可

---

## 整合時序

```
Week 1（現在）：四人同時動工，各自 standalone 測試通過
Week 2 初：
  1. A 的 DataLoader + B 的 maps 接在一起測試
  2. C 的 model 接 B 的 output 跑 forward
  3. D 的 train.py 整合全部，開始訓練
Week 2 中：
  4. 超參數調整（λ 權重、learning rate、batch size）
  5. 試不同 loss 組合
Week 2 末（6/10前）：
  6. 跑 private test set，後處理，提交
  7. 整理 report + 準備 slides
```

---

## 創新點（Novelty，評分重點）

1. **Multi-channel conditioning**：noise map + illumination map concat 進 network，明確 guide restoration（主要創新）
2. **Multi-loss training**：L1 + SSIM + Perceptual + 可選 Adversarial
3. **YCbCr post-processing**：色度空間補償防止亮度提升後色彩偏移
4. （加分）**Synthetic augmentation**：用 GT 反向生成 noisy pair 擴充資料

---

## 限制確認

- 不能用 extra dataset ✓（合規，只用 train/val 的 GT 生成 synthetic）
- 不能用 pretrained image restoration model 直接輸出 ✓（VGG 當 feature extractor 合規，NAFNet 從頭訓練）
- 可以 augment 資料 ✓

---

## 驗證方式

每個模組：
- A: `python augment.py` → 存幾張 augmented pair 圖確認一致
- B: `python illum_map.py` / `python noise_map.py` → 視覺化 heatmap
- C: `python model/restoration.py` → forward pass shape 正確
- D: `python train.py --dry-run` → 跑 2 個 batch，loss 有值，不 crash

整合後：
- 訓練 10 epoch，val PSNR 有上升趨勢即確認 pipeline 正確
- 對照 baseline（只用 gamma correction）確認 model 有改善