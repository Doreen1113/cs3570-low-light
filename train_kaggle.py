"""train_kaggle.py — Kaggle-ready training with all optimizations baked in.

Includes:
  * AMP (mixed precision) — 1.5-2x speedup, ~50% VRAM
  * DataParallel — uses all available GPUs
  * EMA (Exponential Moving Average) — +0.1-0.3 dB PSNR (NAFNet/Restormer standard)
  * Save every epoch (not every 5) — survives Kaggle 12-hour timeout
  * DataParallel-safe resume — load weights into wrapped model correctly
  * Charbonnier loss option (default) — SOTA pixel loss
  * --no-map flag — train 3-channel (no noise/illum) for ablation

Usage:
  python train_kaggle.py --root <path> --epochs 8 --batch-size 16 \
    --lambda-l1 1.0 --lambda-ssim 1.0 --pixel-loss charbonnier \
    --synthetic-repeat 1 --workers 0 --save-dir /kaggle/working/ckpts

  # Resume
  python train_kaggle.py ... --resume /path/to/last.pth

  # no_map ablation
  python train_kaggle.py ... --no-map
"""

import argparse
import copy
import time
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from augment import SyntheticNoisePair, make_train_transform
from dataset import PairedLowLightDataset
from illum_map import compute_illum
from losses import PatchDiscriminator, TotalLoss
from metrics import MetricTracker, evaluate_batch, psnr
from model.restoration import RestorationNet
from noise_map import LocalStdNoise, compute_noise
from postprocess import postprocess


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="H:/low-light-data/low-light")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--crop", type=int, default=256)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--lambda-l1", type=float, default=1.0)
    p.add_argument("--lambda-ssim", type=float, default=1.0)
    p.add_argument("--lambda-percep", type=float, default=0.0, help="VGG16 perceptual loss (TA approved)")
    p.add_argument("--lambda-adv", type=float, default=0.0)
    p.add_argument("--pixel-loss", choices=["charbonnier", "l1"], default="charbonnier")
    p.add_argument("--guided-illum", action="store_true")
    p.add_argument("--no-map", action="store_true", help="3-channel input (no maps)")
    p.add_argument("--synthetic-repeat", type=int, default=1)
    p.add_argument("--save-dir", default="checkpoints")
    p.add_argument("--resume", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--ema-decay", type=float, default=0.999,
                   help="EMA decay (0 = disable EMA, default 0.999 = NAFNet)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------

class EMA:
    """Exponential Moving Average of model parameters.

    Maintains shadow weights that get updated as:
      ema_w = decay * ema_w + (1 - decay) * model_w

    At validation/inference, swap in EMA weights via apply_shadow(), then
    restore() to get back the live weights for next training step.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in self._iter(model):
            self.shadow[name] = param.data.clone().detach()

    @staticmethod
    def _iter(model):
        m = model.module if isinstance(model, nn.DataParallel) else model
        for name, param in m.named_parameters():
            if param.requires_grad:
                yield name, param

    @torch.no_grad()
    def update(self, model):
        for name, param in self._iter(model):
            self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_shadow(self, model):
        for name, param in self._iter(model):
            self.backup[name] = param.data.clone()
            param.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model):
        for name, param in self._iter(model):
            param.data.copy_(self.backup[name])
        self.backup = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_loaders(args):
    root = Path(args.root)
    train_tf = make_train_transform(args.crop)
    base_ds = PairedLowLightDataset(root / "train", transform=train_tf)
    train_ds = SyntheticNoisePair(base_ds, repeat=args.synthetic_repeat)
    val_ds = PairedLowLightDataset(root / "val", transform=None)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False,
                            num_workers=0, pin_memory=True)
    return train_loader, val_loader


def compute_maps(img, guided, device, noise_model=None):
    with torch.no_grad():
        noise_map = compute_noise(img.to(device), model=noise_model)
        illum_map = compute_illum(img.to(device), guided=guided)
    return noise_map, illum_map


def save_checkpoint(path, model, opt, epoch, best_psnr, ema_shadow=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    m = model.module if isinstance(model, nn.DataParallel) else model
    state = {"epoch": epoch, "model": m.state_dict(),
             "opt": opt.state_dict(), "best_psnr": best_psnr}
    if ema_shadow is not None:
        state["ema"] = ema_shadow
    torch.save(state, path)


def load_checkpoint(path, model, opt=None, ema=None):
    ckpt = torch.load(path, map_location="cpu")
    target = model.module if isinstance(model, nn.DataParallel) else model
    target.load_state_dict(ckpt["model"])
    if opt is not None and "opt" in ckpt:
        opt.load_state_dict(ckpt["opt"])
    if ema is not None and "ema" in ckpt:
        for name, tensor in ckpt["ema"].items():
            if name in ema.shadow:
                ema.shadow[name] = tensor.to(ema.shadow[name].device)
        print(f"  + EMA weights restored")
    return ckpt.get("epoch", 0), ckpt.get("best_psnr", 0.0)


# ---------------------------------------------------------------------------
# Validation (uses EMA weights if available)
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, device, guided, ema=None):
    if ema is not None:
        ema.apply_shadow(model)

    model.eval()
    tracker = MetricTracker()
    for inp, gt in loader:
        inp, gt = inp.to(device), gt.to(device)
        noise_map, illum_map = compute_maps(inp, guided, device)
        pred = model(inp, noise_map, illum_map)
        pred = postprocess(pred, chroma_alpha=0.1, sharpen_strength=0.3)
        tracker.update(evaluate_batch(pred, gt))

    if ema is not None:
        ema.restore(model)

    return tracker.summary()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader = make_loaders(args)
    print(f"Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}")

    # Build model (no_map → 3ch, else 5ch)
    if args.no_map:
        model = RestorationNet(width=args.width,
                              use_noise_map=False,
                              use_illum_map=False).to(device)
        print("Model: NO_MAP (3ch input)")
    else:
        model = RestorationNet(width=args.width).to(device)
        print("Model: FULL (5ch input)")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model params: {n_params:.2f}M")

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")

    disc = PatchDiscriminator().to(device) if args.lambda_adv > 0 else None

    loss_fn = TotalLoss(lambda_l1=args.lambda_l1,
                        lambda_ssim=args.lambda_ssim,
                        lambda_percep=args.lambda_percep,
                        lambda_adv=args.lambda_adv,
                        pixel_loss=args.pixel_loss,
                        disc=disc).to(device)
    print(f"Pixel loss: {args.pixel_loss}  | SSIM: {args.lambda_ssim}  | Percep: {args.lambda_percep}")

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    opt_disc = optim.Adam(disc.parameters(), lr=args.lr) if disc else None
    scaler = GradScaler('cuda')

    # EMA
    ema = EMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None
    if ema is not None:
        print(f"EMA enabled (decay={args.ema_decay})")

    start_epoch, best_psnr = 0, 0.0
    if args.resume:
        start_epoch, best_psnr = load_checkpoint(Path(args.resume), model, opt, ema)
        print(f"Resumed from epoch {start_epoch}, best PSNR={best_psnr:.2f}")

    save_dir = Path(args.save_dir)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=120)

        for step, (inp, gt) in enumerate(pbar, 1):
            inp, gt = inp.to(device), gt.to(device)
            noise_map, illum_map = compute_maps(inp, args.guided_illum, device)

            opt.zero_grad()
            with autocast('cuda'):
                pred = model(inp, noise_map, illum_map)
            pred = pred.float()
            total, breakdown = loss_fn(pred, gt)

            with torch.no_grad():
                batch_psnr = psnr(pred.clamp(0, 1), gt).mean().item()

            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            # EMA update after each optimizer step
            if ema is not None:
                ema.update(model)

            if disc and opt_disc and "adv" in breakdown:
                from losses import PatchGANLoss
                adv = PatchGANLoss()
                with autocast('cuda'):
                    d_loss = adv.discriminator_loss(disc, gt, pred.detach())
                opt_disc.zero_grad()
                scaler.scale(d_loss).backward()
                scaler.step(opt_disc)
                scaler.update()

            epoch_loss += total.item()
            pbar.set_postfix({"loss": f"{total.item():.4f}",
                              "psnr": f"{batch_psnr:.2f}",
                              "avg": f"{epoch_loss/step:.4f}",
                              "lr": f"{opt.param_groups[0]['lr']:.1e}"})

            if step % args.log_every == 0:
                lr = opt.param_groups[0]["lr"]
                print(f"  [{epoch}/{args.epochs}] step={step}  "
                      f"loss={total.item():.4f}  lr={lr:.2e}")

            if args.dry_run and step >= 2:
                print("Dry run complete — pipeline OK")
                return

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - t0

        # Validate every epoch (with EMA weights if enabled)
        metrics = validate(model, val_loader, device, args.guided_illum, ema)
        psnr_val = metrics.get("psnr", 0.0)
        ssim_val = metrics.get("ssim", 0.0)
        lpips_val = metrics.get("lpips", float("nan"))
        print(f"Epoch {epoch}/{args.epochs}  loss={avg_loss:.4f}  "
              f"PSNR={psnr_val:.2f}  SSIM={ssim_val:.4f}  LPIPS={lpips_val:.4f}  ({elapsed:.0f}s)")

        ema_shadow = ema.shadow if ema is not None else None
        save_checkpoint(save_dir / "last.pth", model, opt, epoch, best_psnr, ema_shadow)
        if psnr_val > best_psnr:
            best_psnr = psnr_val
            save_checkpoint(save_dir / "best.pth", model, opt, epoch, best_psnr, ema_shadow)
            print(f"  → new best PSNR: {best_psnr:.2f}")

    print(f"\nTraining complete. Best val PSNR: {best_psnr:.2f} dB")


if __name__ == "__main__":
    train(parse_args())
