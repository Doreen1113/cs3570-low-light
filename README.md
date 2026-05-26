# Low-Light Restoration Dataset

Paired low-light / normal-light image patches for training and evaluation.

## Contents of `low-light.tar`

```
low-light/
├── train/        30,000 paired patches  (60,000 files)
├── val/          1,000 paired patches   (2,000 files)
├── test/         1,000 input patches    (no ground truth)
└── dataset.py    reference PyTorch Dataset (Python 3.10+)
```

All images are **lossless WebP** (`.webp`).

### File naming

Every image is named `<id>-<role>.webp` where `role ∈ {in, gt}`:

| Split | Files |
|-------|-------|
| `train/` | `<id>-in.webp` paired with `<id>-gt.webp` (30,000 pairs) |
| `val/`   | `<id>-in.webp` paired with `<id>-gt.webp` (1,000 pairs) |
| `test/`  | `<id>-in.webp` only — **no GT is provided** |

`<id>` is opaque; do not parse it. Pairing is by exact stem match.

## Quick start

```python
from pathlib import Path
from torch.utils.data import DataLoader
from dataset import (
    PairedLowLightDataset, TestLowLightDataset,
    PairedCompose, PairedRandomCrop, PairedRandomFlip, PairedToTensor,
)

root = Path("low-light")

train_tf = PairedCompose([
    PairedRandomCrop(256),
    PairedRandomFlip(p_h=0.5),
    PairedToTensor(),
])

train_set = PairedLowLightDataset(root / "train", transform=train_tf)
val_set   = PairedLowLightDataset(root / "val",   transform=None)
test_set  = TestLowLightDataset(root / "test",    transform=None)

train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_set,   batch_size=1,  shuffle=False, num_workers=2)
test_loader  = DataLoader(test_set,  batch_size=1,  shuffle=False, num_workers=2)

for x, y in train_loader:        # x, y are float32 CHW tensors in [0, 1]
    ...

for x, stem in test_loader:      # test yields (input, stem)
    pred = model(x)
    save_image(pred, f"submission/{stem[0]}-in.webp")
```

## Dataset classes (in `dataset.py`)

### `PairedLowLightDataset(root, transform=None)`
For `train/` and `val/`. Returns `(input_tensor, gt_tensor)`.

### `TestLowLightDataset(root, transform=None)`
For `test/`. Returns `(input_tensor, stem)` where `stem` lets you save predictions
under the original filename.

## Transform contract

A transform may be one of:

1. **`None`** — images are converted to `float32` CHW tensors in `[0, 1]`.
2. **A single-image torchvision-style callable** `fn(pil) -> tensor` — applied
   independently to input and GT. **Use only for deterministic ops**
   (`ToTensor`, `Normalize`). Random single-image transforms will desync the pair.
3. **A pair-aware callable** `fn(in_pil, gt_pil) -> (in_tensor, gt_tensor)`,
   marked by setting `fn.paired = True`. The callable owns randomness and must
   apply the same geometric augmentation to both images.

The provided `PairedCompose`, `PairedRandomCrop`, `PairedRandomFlip`,
`PairedToTensor` building blocks already follow contract #3.

## Submission format

For each `<id>-in.webp` in `test/`, produce a restored image and save it as
`<id>-in.webp` (or `.png` if preferred). Keep the original stem.

Evaluation pairs each prediction against the private ground truth held by the
organizers — do **not** attempt to obtain or infer test GTs.

## Requirements

- Python 3.10+
- `torch`, `torchvision`, `Pillow` (with WebP support; built into modern Pillow)
