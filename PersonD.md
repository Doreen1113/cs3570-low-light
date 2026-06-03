### Person D — Loss + Training Loop + Evaluation + Post-processing

## 負責檔案
 `losses.py`、`metrics.py`、`postprocess.py`、`train.py`

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


## 正式訓練

基本訓練指令：
python train.py --root {path of your dataset} --epochs 30 --workers 0


## 常用參數

epochs: 控制訓練幾輪

batch size: 預設是 8，如果 GPU 記憶體不夠或太慢，可以改小

learning rate: 預設是 2e-4

workers: 我用4以上會卡住，供參

width: 控制 NAFNet 寬度，預設是 32，比較快但模型小一點：

synthetic augmentation: 預設1

代表會把 train set 擴充一倍
如果想加快訓練，可以關掉(0)

## Loss 設定

L1 loss + SSIM loss 

預設 --lambda-l1 1.0 --lambda-ssim 0.5

