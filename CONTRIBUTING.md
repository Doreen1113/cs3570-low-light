# 協作指南 — cs3570-low-light

## 分工

| 人員 | 負責檔案 |
|------|---------|
| A | `augment.py` |
| B | `illum_map.py`, `noise_map.py` |
| C | `model/nafnet.py`, `model/restoration.py` |
| D | `losses.py`, `metrics.py`, `postprocess.py`, `train.py` |

每個人只動自己的檔案，不會有 conflict。

---

## 環境安裝

```powershell
pip install -r requirements.txt
```

Dataset（不在 repo 裡）放到以下路徑：
```
low-light/
├── train/
├── val/
└── test/
```

---

## 第一次設定

```powershell
git clone https://github.com/Doreen1113/cs3570-low-light.git
cd cs3570-low-light
git checkout dev
```

---

## 每天的工作流程

```powershell
git checkout dev
git pull                        # 拉最新
# ... 做自己的事 ...
git add 自己的檔案
git commit -m "feat: 做了什麼"
git push
```

---

## Commit 訊息格式

| 前綴 | 範例 |
|------|------|
| `feat:` | `feat: add guided filter illumination` |
| `fix:` | `fix: shape mismatch in noise CNN` |
| `exp:` | `exp: lambda_ssim=0.8, PSNR +0.3dB` |

---

## 整合時間點

```
~6/1   各人 standalone 測試通過，告知 D
~6/3   D 跑 train.py --dry-run，沒問題後 merge dev → main，開始訓練
6/10   最終版 merge 到 main
6/12   Presentation
6/17   Report 截止
```

---

## 各模組獨立測試指令

```powershell
python augment.py                  # A
python illum_map.py --guided       # B
python noise_map.py                # B
python model/restoration.py        # C
python losses.py                   # D
python train.py --dry-run          # D（整合後）
```
