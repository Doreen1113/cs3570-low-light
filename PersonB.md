# Person B - Noise Map + Illumination Map

## 負責檔案

`illum_map.py` - illumination map / brightness map estimation

`noise_map.py` - noise map estimation

`check_maps.py` - map 視覺驗證，確認輸出的 map 是否合理

`train.py` - 透過 `compute_maps()` 將 B 的輸出接入 training pipeline

## 完成項目

### 1. Illumination Map Estimation

目標是從 low-light input image 中估計每個位置的亮度分布，輸出：

```python
illum_map: [B, 1, H, W]
```

目前 scaffold 使用 Retinex-style 的 max-channel 方法：

```python
L = max(R, G, B)
```

也就是每個 pixel 取 RGB 三個 channel 中的最大值作為 illumination estimate。

這張 map 可以告訴 model：

```text
哪些區域很暗，需要被增強
哪些區域已經夠亮，不應該過度提亮
```

### 2. Guided Illumination Map

除了簡單的 max-channel 版本，目前也支援 guided filter 平滑版本：

```python
compute_illum(img, guided=True)
```

Guided illumination map 會讓亮度圖更平滑，避免局部 pixel-level noise 讓 illumination map 太破碎。

輸出限制不變：

```python
illum_map: [B, 1, H, W]  # float32, value range [0, 1]
```

### 3. Noise Map Estimation

目標是從 low-light input image 中估計每個位置的雜訊強度，輸出：

```python
noise_map: [B, 1, H, W]
```

目前 scaffold 使用 local standard deviation 方法：

```text
RGB image
-> grayscale
-> local mean
-> local variance
-> local standard deviation
```

也就是用 sliding window 計算每個 pixel 附近區域的變化程度。局部變化越大，noise map 的值越高。

這張 map 可以告訴 model：

```text
哪些區域雜訊比較嚴重
哪些暗部或邊緣在提亮時需要特別去噪
```

### 4. CNN Noise Estimator Scaffold

`noise_map.py` 裡也保留了一個 CNN-based noise estimator 的框架：

```python
CNNNoiseEstim
```

訓練方式是用 GT image 加上 synthetic Gaussian noise，讓 CNN 學會預測 noise level map。

目前可用指令：

```cmd
python noise_map.py --train --root C:\multimedia\cs3570-low-light
```

不過目前主要整合版本仍使用 `LocalStdNoise`，因為它不需要額外訓練，穩定且可以直接接進 C 的 model。

### 5. Output Shape 驗證

B 的兩個輸出都必須維持固定 shape：

```python
noise_map: [B, 1, H, W]
illum_map: [B, 1, H, W]
```

這是 C 的 RestorationNet 已經寫死的輸入格式。

C 的 model 最終會接收：

```python
input RGB:  [B, 3, H, W]
noise_map:  [B, 1, H, W]
illum_map:  [B, 1, H, W]
```

concat 後變成：

```python
[B, 5, H, W]
```

然後輸入 NAFNet。

### 6. 與 Training Pipeline 的關係

在 training loop 中，資料流程是：

```text
A: augmentation
   ↓
B: compute noise map + illumination map
   ↓
C: RestorationNet
   ↓
D: loss / metrics / checkpoint
```

也就是 B 是在 augmentation 之後執行，因為 model 實際看到的是 augmented input，所以 B 產生的 map 也必須描述 augmented input。

目前在 `train.py` 中的接法：

```python
noise_map = LocalStdNoise()(img)
illum_map = compute_illum(img, guided=guided)
pred = model(inp, noise_map, illum_map)
```

## B 的核心貢獻

B 的工作不是直接產生 restored image，而是提供 model 額外的 condition guidance。

原本 model 只看到：

```text
RGB image
```

加入 B 後 model 看到：

```text
RGB image + noise map + illumination map
```

因此 model 可以更明確知道：

```text
哪裡暗
哪裡亮
哪裡雜訊多
哪裡提亮時需要小心
```

這讓 restoration network 不需要完全從 RGB 裡自己猜出 brightness / noise distribution。

## 視覺驗證

使用：

```cmd
python illum_map.py --root C:\multimedia\cs3570-low-light --guided --out C:\multimedia\cs3570-low-light\illum_check
python noise_map.py --root C:\multimedia\cs3570-low-light --out C:\multimedia\cs3570-low-light\noise_check
```

預設會產生前 4 張 map。若要更多張：

```cmd
python illum_map.py --root C:\multimedia\cs3570-low-light --guided --n 20 --out C:\multimedia\cs3570-low-light\illum_check
python noise_map.py --root C:\multimedia\cs3570-low-light --n 20 --out C:\multimedia\cs3570-low-light\noise_check
```

## 交接給 C

C 的 RestorationNet 已經支援 B 的輸出。

B 只需要保證：

```python
compute_noise(inp) -> [B, 1, H, W]
compute_illum(inp) -> [B, 1, H, W]
```

並且值域保持在：

```python
[0, 1]
```

C 會將它們與 RGB input concat：

```python
x = torch.cat([img, noise_map, illum_map], dim=1)
```

形成 5-channel input：

```python
[B, 5, H, W]
```

## 使用方式

```cmd
# 產生 illumination map
python illum_map.py --root C:\multimedia\cs3570-low-light --guided --out C:\multimedia\cs3570-low-light\illum_check

# 產生 noise map
python noise_map.py --root C:\multimedia\cs3570-low-light --out C:\multimedia\cs3570-low-light\noise_check

# 若有 check_maps.py，可視覺驗證兩種 map
python check_maps.py
```

## 接入 train.py 的方式

```python
from illum_map import compute_illum
from noise_map import compute_noise

# training loop 裡
noise_map = compute_noise(inp)
illum_map = compute_illum(inp, guided=True)

pred = model(inp, noise_map, illum_map)
```

## 待改進項目

1. 改進 illumination map

目前 max-channel 方法簡單穩定，但可能對 noise 敏感。可以比較：

```text
MaxChannel
GuidedIllum
Smoothed Retinex-style illumination
```

2. 改進 noise map

目前 LocalStdNoise 容易把 texture / edge 也視為 noise。可以嘗試：

```text
larger window
illumination-aware noise normalization
CNNNoiseEstim
```

3. 跑 C 的 overfit test

B 改完 map 後，交給 C 重跑：

```cmd
python model/overfit_test.py
```

4. 準備 ablation study

最後需要比較：

```text
RGB only
RGB + noise map
RGB + illum map
RGB + noise map + illum map
```

## 一句話總結

Person B 負責從 augmented low-light input 中估計 illumination map 和 noise map，將局部亮度與雜訊資訊作為額外 conditioning 提供給 RestorationNet，幫助模型更有效地進行低光增強與去雜訊。
