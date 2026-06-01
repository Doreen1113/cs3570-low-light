# Person A — Data Pipeline + Augmentation

## What Was Done

### `dataset.py`
Core dataset classes and pair-aware transform infrastructure.

- `PairedLowLightDataset` — loads `(input, gt)` pairs from `train/` or `val/`
- `TestLowLightDataset` — loads inputs only from `test/`
- `PairedCompose`, `PairedRandomCrop`, `PairedRandomFlip`, `PairedToTensor` — building blocks that apply identical transforms to both images to keep pairs in sync

### `augment.py`
Additional augmentations and synthetic data expansion.

- `PairedRandomRotate90` — randomly rotates both images by 0/90/180/270°
- `PairedColorJitter` — applies brightness/contrast jitter to the **input only** (GT unchanged, so the target distribution is preserved)
- `SyntheticNoisePair` — wraps a dataset and adds Gaussian-noised copies of GT images as extra `(noisy, clean)` pairs, expanding 30k → 60k training samples without using external data
- `make_train_transform(crop_size)` — convenience builder used by `train.py`

### `analyze_dataset.py`
One-shot statistics script. Outputs:
- Sample counts per split
- Brightness and noise distribution summary (min / median / max)
- `brightness_hist.png` and `noise_hist.png`

**Key findings from the dataset (500-sample analysis):**

| Metric | Input (low-light) | GT (clean) |
|---|---|---|
| Brightness median | 0.12 | 0.35 |
| Noise σ median | 0.041 | 0.023 |

---

## How to Run

Activate the environment first:
```bash
conda env create -f environment.yml
conda activate low_light
```

### Verify the dataset
```bash
python dataset.py
```
Expected output:
```
train: 30000  val: 1000  test: 1000
batch x=(4, 3, 256, 256) y=(4, 3, 256, 256) dtype=torch.float32
```

### Sanity-check augmented pairs
```bash
python augment.py
```
Saves 8 sample pairs to `augment/`. Inspect `augment/000_in.png` vs `augment/000_gt.png` to confirm geometric alignment (same crop and rotation) and that color differs only on the `_in` image.

### Analyze dataset statistics
```bash
python analyze_dataset.py
```
Saves histograms to `stats/brightness_hist.png` and `stats/noise_hist.png`.

Optional flags:
```bash
python analyze_dataset.py --n 1000 --out my_stats/
```

---

## How Augmentation Is Used in Training

Augmentation runs **on-the-fly** — no pre-generation needed. Person D's `train.py` should set up the DataLoader as follows:

```python
from augment import SyntheticNoisePair, make_train_transform
from dataset import PairedLowLightDataset
from torch.utils.data import DataLoader

train_tf = make_train_transform(crop_size=256)
base_ds  = PairedLowLightDataset("dataset/train", transform=train_tf)
train_ds = SyntheticNoisePair(base_ds, repeat=1)   # 30k → 60k pairs
loader   = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)
```

Each epoch applies fresh random augmentations, so the effective variety exceeds 60k.
