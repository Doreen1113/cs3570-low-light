"""ensemble_predict.py — Prediction averaging (NOT weight averaging).

Loads multiple checkpoints SEPARATELY. For each input image:
  1. Run each model independently (with TTA optional)
  2. Average all predictions
  3. Apply postprocess
  4. Compute metric (eval mode) or save image (infer mode)

This works even when models were trained with very different losses,
because we never mix weights — we mix predictions.

Examples:
  # Eval ensemble on val
  python ensemble_predict.py --mode eval \\
      --root /path/to/dataset \\
      --ckpts kris.pth irene.pth lee.pth \\
      --tta --no-postprocess

  # Test inference
  python ensemble_predict.py --mode infer \\
      --root /path/to/test \\
      --ckpts kris.pth irene.pth lee.pth \\
      --tta --no-postprocess \\
      --out-dir submission/
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
from pathlib import Path
import torch
import torch.nn as nn
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


def _augment(x, k):
    if k & 1: x = torch.flip(x, dims=[-1])
    if k & 2: x = torch.flip(x, dims=[-2])
    if k & 4: x = torch.rot90(x, k=1, dims=[-2, -1])
    return x

def _deaugment(x, k):
    if k & 4: x = torch.rot90(x, k=-1, dims=[-2, -1])
    if k & 2: x = torch.flip(x, dims=[-2])
    if k & 1: x = torch.flip(x, dims=[-1])
    return x


@torch.no_grad()
def forward_with_tta(model, inp, guided=False, use_tta=True):
    if not use_tta:
        n = compute_noise(inp)
        i = compute_illum(inp, guided=guided)
        return model(inp, n, i)
    preds = []
    for k in range(8):
        ik = _augment(inp, k)
        nk = compute_noise(ik)
        lk = compute_illum(ik, guided=guided)
        pk = model(ik, nk, lk)
        preds.append(_deaugment(pk, k))
    return torch.stack(preds, dim=0).mean(dim=0)


@torch.no_grad()
def ensemble_forward(models, inp, weights, use_tta, guided):
    """Run each model, average their outputs (weighted)."""
    preds = []
    for model in models:
        p = forward_with_tta(model, inp, guided, use_tta)
        preds.append(p)
    out = torch.zeros_like(preds[0])
    for p, w in zip(preds, weights):
        out += w * p
    return out


class TestDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        exts = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
        paths = []
        for e in exts:
            paths += list(self.root.glob(e))
        paths = sorted(paths)
        if not paths:
            raise FileNotFoundError(f"No images in {self.root}")
        self.paths = paths
        print(f"Found {len(paths)} images")
        self.to_tensor = transforms.ToTensor()
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        return self.to_tensor(Image.open(self.paths[i]).convert("RGB")), self.paths[i].stem


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["eval", "infer"], required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--ckpts", nargs="+", required=True, help="checkpoint paths to ensemble")
    p.add_argument("--weights", nargs="+", type=float, default=None)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--tta", action="store_true")
    p.add_argument("--guided-illum", action="store_true")
    p.add_argument("--no-postprocess", action="store_true")
    p.add_argument("--chroma-alpha", type=float, default=0.1)
    p.add_argument("--sharpen-strength", type=float, default=0.3)
    p.add_argument("--out-dir", default="ensemble_results")
    p.add_argument("--ext", default="png")
    return p.parse_args()


def apply_pp(pred, args):
    if args.no_postprocess: return pred.clamp(0, 1)
    return postprocess(pred, chroma_alpha=args.chroma_alpha,
                       sharpen_strength=args.sharpen_strength)


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = len(args.ckpts)

    # Normalize weights
    if args.weights:
        assert len(args.weights) == n
        total = sum(args.weights)
        weights = [w / total for w in args.weights]
    else:
        weights = [1.0 / n] * n

    pp_str = 'off' if args.no_postprocess else f"chroma={args.chroma_alpha} sharpen={args.sharpen_strength}"
    print(f"Device: {device} | TTA: {args.tta} | postprocess: {pp_str}")
    print(f"Ensemble of {n} models (prediction averaging):")
    for c, w in zip(args.ckpts, weights):
        print(f"  weight={w:.3f}  {c}")

    # Load all models
    models = []
    for path in args.ckpts:
        m = RestorationNet(width=args.width).to(device)
        ckpt = torch.load(path, map_location=device)
        m.load_state_dict(ckpt["model"])
        m.eval()
        if torch.cuda.device_count() > 1:
            m = nn.DataParallel(m)
        models.append(m)
        print(f"  Loaded {Path(path).name} (epoch={ckpt.get('epoch')}, best_psnr={ckpt.get('best_psnr', 0):.2f})")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")

    if args.mode == "eval":
        val_ds = PairedLowLightDataset(Path(args.root) / "val")
        loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)
        print(f"Eval on {len(val_ds)} val images")

        tracker = MetricTracker()
        for inp, gt in tqdm(loader, desc="Ensemble Eval", ncols=100):
            inp, gt = inp.to(device), gt.to(device)
            pred = ensemble_forward(models, inp, weights, args.tta, args.guided_illum)
            pred = apply_pp(pred, args)
            tracker.update(evaluate_batch(pred, gt))

        s = tracker.summary()
        print(f"\nPSNR  = {s.get('psnr', 0):.4f} dB")
        print(f"SSIM  = {s.get('ssim', 0):.4f}")
        if "lpips" in s:
            print(f"LPIPS = {s['lpips']:.4f}")

    else:
        ds = TestDataset(args.root)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True,
                            collate_fn=lambda b: (torch.stack([x[0] for x in b]),
                                                  [x[1] for x in b]))
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for imgs, names in tqdm(loader, desc="Ensemble Infer", ncols=100):
            imgs = imgs.to(device)
            preds = ensemble_forward(models, imgs, weights, args.tta, args.guided_illum)
            preds = apply_pp(preds, args)
            for i, name in enumerate(names):
                save_image(preds[i], out_dir / f"{name}.{args.ext}")
        print(f"\nDone! {len(ds)} images saved")


if __name__ == "__main__":
    main()
