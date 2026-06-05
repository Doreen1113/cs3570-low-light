"""eval_tta.py — Test-Time Augmentation + 可調 postprocess 的評估腳本。

支援兩種模式：
  --mode eval   : 跑 val set 算 PSNR/SSIM/LPIPS（有 GT）
  --mode infer  : 跑 test set 存修復圖（沒 GT）

TTA：把 input 翻轉旋轉 8 種，分別跑 model，反變換後平均。
可調 postprocess：透過 --chroma-alpha 和 --sharpen-strength 調整色彩補償強度。

Examples:
  # Val 上算 metric（帶 TTA）
  python eval_tta.py --mode eval \\
      --root /kaggle/input/.../cs3570-lowlight \\
      --ckpt checkpoints/best.pth \\
      --tta

  # Test 上跑 inference
  python eval_tta.py --mode infer \\
      --root /kaggle/input/.../cs3570-lowlight/test \\
      --ckpt checkpoints/best.pth \\
      --out-dir results --tta --chroma-alpha 0.1
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from PIL import Image

from dataset import PairedLowLightDataset
from illum_map import compute_illum
from noise_map import compute_noise
from metrics import MetricTracker, evaluate_batch
from model.restoration import RestorationNet
from postprocess import postprocess


# ---------------------------------------------------------------------------
# TTA helpers
# ---------------------------------------------------------------------------

def _augment(x: torch.Tensor, k: int) -> torch.Tensor:
    """Apply one of 8 D4 transforms (flips × 90-degree rotations)."""
    if k & 1:
        x = torch.flip(x, dims=[-1])
    if k & 2:
        x = torch.flip(x, dims=[-2])
    if k & 4:
        x = torch.rot90(x, k=1, dims=[-2, -1])
    return x


def _deaugment(x: torch.Tensor, k: int) -> torch.Tensor:
    """Inverse of _augment."""
    if k & 4:
        x = torch.rot90(x, k=-1, dims=[-2, -1])
    if k & 2:
        x = torch.flip(x, dims=[-2])
    if k & 1:
        x = torch.flip(x, dims=[-1])
    return x


@torch.no_grad()
def forward_with_tta(model, inp, guided=False, use_tta=True):
    """Forward pass with optional 8-way TTA averaging."""
    if not use_tta:
        noise_map = compute_noise(inp)
        illum_map = compute_illum(inp, guided=guided)
        return model(inp, noise_map, illum_map)

    preds = []
    for k in range(8):
        inp_k = _augment(inp, k)
        noise_k = compute_noise(inp_k)
        illum_k = compute_illum(inp_k, guided=guided)
        pred_k = model(inp_k, noise_k, illum_k)
        pred_k = _deaugment(pred_k, k)
        preds.append(pred_k)
    return torch.stack(preds, dim=0).mean(dim=0)


# ---------------------------------------------------------------------------
# Test-set dataset (no GT)
# ---------------------------------------------------------------------------

class TestDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        exts = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
        paths = []
        for e in exts:
            paths += list(self.root.glob(e))
        paths = sorted(paths)
        if not paths:
            raise FileNotFoundError(f"No images found in {self.root}")
        self.paths = paths
        print(f"Found {len(paths)} images in {self.root}")
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        # Strip "-in" suffix so output filename matches submission format <id>.png
        stem = self.paths[i].stem
        if stem.endswith("-in"):
            stem = stem[:-3]
        return self.to_tensor(img), stem


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["eval", "infer"], required=True)
    p.add_argument("--root", required=True, help="dataset root (eval) or test folder (infer)")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--tta", action="store_true", help="enable 8-way TTA")
    p.add_argument("--guided-illum", action="store_true")
    p.add_argument("--no-map", action="store_true", help="for no_map ckpt")
    p.add_argument("--no-postprocess", action="store_true", help="skip YCbCr/sharpen")
    p.add_argument("--chroma-alpha", type=float, default=0.1)   # tuned on val
    p.add_argument("--sharpen-strength", type=float, default=0.3)
    # infer-only
    p.add_argument("--out-dir", default="test_results")
    p.add_argument("--ext", default="png")
    return p.parse_args()


def apply_postprocess(pred, args):
    if args.no_postprocess:
        return pred.clamp(0, 1)
    return postprocess(pred, chroma_alpha=args.chroma_alpha,
                       sharpen_strength=args.sharpen_strength)


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  TTA: {args.tta}  |  postprocess: "
          f"{'off' if args.no_postprocess else f'chroma={args.chroma_alpha} sharpen={args.sharpen_strength}'}")

    # Load model
    if args.no_map:
        model = RestorationNet(width=args.width,
                              use_noise_map=False,
                              use_illum_map=False).to(device)
        print("Model: NO_MAP (3ch)")
    else:
        model = RestorationNet(width=args.width).to(device)
        print("Model: FULL (5ch)")

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded {Path(args.ckpt).name}  epoch={ckpt.get('epoch', '?')}  "
          f"train_best_psnr={ckpt.get('best_psnr', 0):.2f}")

    if args.mode == "eval":
        val_ds = PairedLowLightDataset(Path(args.root) / "val")
        loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)
        print(f"Eval on {len(val_ds)} val images...")

        tracker = MetricTracker()
        for inp, gt in tqdm(loader, desc="Eval", ncols=100):
            inp, gt = inp.to(device), gt.to(device)
            pred = forward_with_tta(model, inp, args.guided_illum, args.tta)
            pred = apply_postprocess(pred, args)
            tracker.update(evaluate_batch(pred, gt))

        s = tracker.summary()
        print()
        print(f"PSNR  = {s.get('psnr', 0):.4f} dB")
        print(f"SSIM  = {s.get('ssim', 0):.4f}")
        if "lpips" in s:
            print(f"LPIPS = {s['lpips']:.4f}")
        else:
            print("LPIPS = (lpips not installed)")

    else:  # infer
        ds = TestDataset(args.root)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving outputs to {out_dir}")

        for img, name in tqdm(ds, desc="Inference", ncols=100):
            img = img.unsqueeze(0).to(device)
            pred = forward_with_tta(model, img, args.guided_illum, args.tta)
            pred = apply_postprocess(pred, args)
            save_image(pred, out_dir / f"{name}.{args.ext}")

        print(f"\nDone! {len(ds)} images saved")


if __name__ == "__main__":
    main()
