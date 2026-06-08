"""Person D (part 4) — Training loop.

Integrates all modules:
  A: DataLoader (dataset.py + augment.py)
  B: noise_map.py + illum_map.py
  C: model/restoration.py
  D: losses.py + metrics.py + postprocess.py

Quick-start (dry run — verifies pipeline without real training):
  python train.py --dry-run

Full training:
  python train.py --root H:/low-light-data/low-light --epochs 100

Resume from checkpoint:
  python train.py --resume checkpoints/best.pth
"""

import argparse
import time
from pathlib import Path
from tqdm import tqdm

import torch
import torch.optim as optim
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
    p = argparse.ArgumentParser(description="Low-light restoration training")
    p.add_argument("--root", default="H:/low-light-data/low-light")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--crop", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--width", type=int, default=32, help="NAFNet base channel width")
    p.add_argument("--lambda-l1", type=float, default=1.0)
    p.add_argument("--lambda-ssim", type=float, default=0.5)
    p.add_argument("--lambda-percep", type=float, default=0.0, help="VGG16 perceptual loss (0=disabled, TA approved)")
    p.add_argument("--lambda-adv", type=float, default=0.0, help="0=disabled")
    p.add_argument("--pixel-loss", choices=["charbonnier", "l1"], default="charbonnier",
                   help="pixel loss type (charbonnier = NAFNet/Restormer default, +0.2-0.5 dB)")
    p.add_argument("--gamma", type=float, default=2.2,
                   help="gamma for input pre-brightening (inp = inp^(1/gamma)); 0 = disabled")
    p.add_argument("--guided-illum", action="store_true", help="use guided-filter illumination")
    p.add_argument("--synthetic-repeat", type=int, default=1)
    p.add_argument("--save-dir", default="checkpoints")
    p.add_argument("--resume", default=None)
    p.add_argument("--dry-run", action="store_true", help="run 2 batches then exit")
    p.add_argument("--log-every", type=int, default=50)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_loaders(args):
    root = Path(args.root)
    train_tf = make_train_transform(args.crop)
    base_ds = PairedLowLightDataset(root / "train", transform=train_tf)
    train_ds = SyntheticNoisePair(base_ds, repeat=args.synthetic_repeat)
    val_ds = PairedLowLightDataset(root / "val", transform=None)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return train_loader, val_loader


def compute_maps(img: torch.Tensor, guided: bool, device: torch.device, noise_model=None):
    with torch.no_grad():
        noise_map = compute_noise(img.to(device), model=noise_model)
        illum_map = compute_illum(img.to(device), guided=guided)
    return noise_map, illum_map

def save_checkpoint(path: Path, model, opt, epoch: int, best_psnr: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "best_psnr": best_psnr,
    }, path)


def load_checkpoint(path: Path, model, opt=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if opt is not None and "opt" in ckpt:
        opt.load_state_dict(ckpt["opt"])
    return ckpt.get("epoch", 0), ckpt.get("best_psnr", 0.0)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, device, guided: bool, gamma: float = 0.0) -> dict:
    model.eval()
    tracker = MetricTracker()
    for inp, gt in loader:
        inp, gt = inp.to(device), gt.to(device)
        if gamma > 0:
            inp = inp.pow(1.0 / gamma)
        noise_map, illum_map = compute_maps(inp, guided, device)
        pred = model(inp, noise_map, illum_map)
        pred = postprocess(pred, chroma_alpha=0.1, sharpen_strength=0.3)
        tracker.update(evaluate_batch(pred, gt))
    return tracker.summary()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader = make_loaders(args)
    print(f"Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}")

    model = RestorationNet(width=args.width).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model params: {n_params:.2f}M")

    disc = None
    if args.lambda_adv > 0:
        disc = PatchDiscriminator().to(device)

    loss_fn = TotalLoss(
        lambda_l1=args.lambda_l1,
        lambda_ssim=args.lambda_ssim,
        lambda_percep=args.lambda_percep,
        lambda_adv=args.lambda_adv,
        pixel_loss=args.pixel_loss,
        disc=disc,
    ).to(device)
    print(f"Pixel loss: {args.pixel_loss}  | SSIM: {args.lambda_ssim}  | Percep: {args.lambda_percep}")

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    opt_disc = optim.Adam(disc.parameters(), lr=args.lr) if disc else None

    start_epoch, best_psnr = 0, 0.0
    if args.resume:
        start_epoch, best_psnr = load_checkpoint(Path(args.resume), model, opt)
        print(f"Resumed from epoch {start_epoch}, best PSNR={best_psnr:.2f}")

    save_dir = Path(args.save_dir)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=120)

        for step, (inp, gt) in enumerate(pbar, 1):
            inp, gt = inp.to(device), gt.to(device)
            if args.gamma > 0:
                inp = inp.pow(1.0 / args.gamma)
            noise_map, illum_map = compute_maps(inp, args.guided_illum, device)

            pred = model(inp, noise_map, illum_map)
            total, breakdown = loss_fn(pred, gt)

            with torch.no_grad():
                batch_psnr = psnr(pred.clamp(0, 1), gt).mean().item()

            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            # Optional discriminator update
            if disc and opt_disc and "adv" in breakdown:
                from losses import PatchGANLoss
                adv = PatchGANLoss()
                d_loss = adv.discriminator_loss(disc, gt, pred.detach())
                opt_disc.zero_grad()
                d_loss.backward()
                opt_disc.step()

            epoch_loss += total.item()
            pbar.set_postfix({
                "loss": f"{total.item():.4f}",
                "psnr": f"{batch_psnr:.2f}",
                "avg": f"{epoch_loss / step:.4f}",
                "lr": f"{opt.param_groups[0]['lr']:.1e}",
            })

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

        # Validate and save every epoch
        metrics = validate(model, val_loader, device, args.guided_illum, args.gamma)
        psnr_val = metrics.get("psnr", 0.0)
        ssim_val = metrics.get("ssim", 0.0)
        print(f"Epoch {epoch}/{args.epochs}  loss={avg_loss:.4f}  "
              f"PSNR={psnr_val:.2f}  SSIM={ssim_val:.4f}  ({elapsed:.0f}s)")

        save_checkpoint(save_dir / "last.pth", model, opt, epoch, best_psnr)
        if psnr_val > best_psnr:
            best_psnr = psnr_val
            save_checkpoint(save_dir / "best.pth", model, opt, epoch, best_psnr)
            print(f"  → new best PSNR: {best_psnr:.2f}")

    print(f"\nTraining complete. Best val PSNR: {best_psnr:.2f} dB")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    train(args)
