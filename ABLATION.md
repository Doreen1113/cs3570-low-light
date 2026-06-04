# Ablation Study 跑法說明

> 每個人用自己的電腦或 Kaggle 帳號跑，比 val PSNR / SSIM / LPIPS 。

---

## 每個人要跑的設定

| 人 | 實驗 | 指令差異 |
|----|------|----------|
| **A** | + Data Augmentation | `--synthetic-repeat 1 --epochs 5` |
| **B** | 改進 map（先 push 新 map） | 其他不變，用新 map 的 repo |
| **C** | Full pipeline（base） | 現在跑的就是 |
| **D** | 高 SSIM weight | `--lambda-ssim 1.0` |

---

## 步驟

### Step 1 — Clone repo

```bash
git clone https://github.com/Doreen1113/cs3570-low-light.git
cd cs3570-low-light
pip install -r requirements.txt
```

### Step 2 — 產生 patched 版本

```bash
python patch_train.py --mode full
```

### Step 3 — 開始訓練（改成自己的路徑和設定）

```bash
# 本地跑（改 --root 成自己的 dataset 路徑）
python -u train_patched.py \
  --root <你的 dataset 路徑> \
  --batch-size 4 \
  --lambda-ssim 0 \
  --epochs 10 \
  --synthetic-repeat 0 \
  --workers 2 \
  --save-dir checkpoints
```

---

## 各人修改的參數

**A — 加 augmentation：**
```bash
--synthetic-repeat 1   # 改這個，其他不動
```

**B — 改進 map：**
```bash
git pull   # 先確認有 pull 最新的 map code
# 其他參數不動
```

**D — 高 SSIM weight（強調結構相似度）：**
```bash
--lambda-ssim 1.0   # 改這個（從 0.5 提到 1.0），其他不動
```

---

## Kaggle 的話

Cell 1：
```python
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
!git clone https://github.com/Doreen1113/cs3570-low-light.git
%cd cs3570-low-light
!pip install -r requirements.txt -q
```

Cell 2：
```python
%run patch_train.py --mode full
```

Cell 3：
```python
!python -u train_patched.py \
  --root /kaggle/input/datasets/doreen071/cs3570-lowlight \
  --batch-size 16 \
  --lambda-ssim 0 \
  --epochs 10 \
  --synthetic-repeat 0 \
  --workers 2 \
  --save-dir /kaggle/working/checkpoints
```

---

## 跑完後記錄這三個數字

val PSNR（越高越好）、SSIM（越高越好）、LPIPS（越低越好）

Epoch 5、10 各記一次，貼到群組。（train.py 每 5 epoch 跑一次 validate）

---

## 最終 Ablation Table（presentation 用）

| 設定 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|--------|--------|---------|
| Baseline (gamma γ=0.5) | 15.02 | 0.2446 | 0.7669 |
| C：Full pipeline | | | |
| A：+ Augmentation | | | |
| B：+ 改進 map | | | |
| D：高 SSIM weight | | | |
