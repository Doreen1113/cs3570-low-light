"""Full ablation study for presentation slides."""
import os, torch, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, ".")

from model.restoration import RestorationNet
from model.nafnet import NAFNet
from postprocess import postprocess
from noise_map import compute_noise
from illum_map import compute_illum
from metrics import MetricTracker, evaluate_batch
from dataset import PairedLowLightDataset
from torch.utils.data import DataLoader
from pathlib import Path
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load main checkpoint
ckpt = torch.load("H:/low-light-data/kris/kris.pth", map_location=device, weights_only=False)

ds = PairedLowLightDataset(Path("H:/low-light-data/low-light/val"))
loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

results = {}

# =====================================================================
# Ablation 1: Full model (no postprocess) — baseline
# =====================================================================
print("\n[1/7] Full model (no PP)...")
model = RestorationNet(width=32).to(device)
model.load_state_dict(ckpt["model"])
model.eval()
tracker = MetricTracker()
with torch.no_grad():
    for inp, gt in loader:
        inp, gt = inp.to(device), gt.to(device)
        noise = compute_noise(inp)
        illum = compute_illum(inp)
        pred = model(inp, noise, illum)
        tracker.update(evaluate_batch(pred, gt))
results["Full (no PP)"] = tracker.summary()

# =====================================================================
# Ablation 2: No noise map (illum only → 4ch)
# =====================================================================
print("[2/7] No noise map...")
model_no_noise = RestorationNet(width=32, use_noise_map=False, use_illum_map=True).to(device)
# Can't load checkpoint directly (different in_channels), skip weight loading
# Instead, run with zeros for noise map in full model
model.eval()
tracker = MetricTracker()
with torch.no_grad():
    for inp, gt in loader:
        inp, gt = inp.to(device), gt.to(device)
        noise = torch.zeros(inp.size(0), 1, inp.size(2), inp.size(3), device=device)
        illum = compute_illum(inp)
        pred = model(inp, noise, illum)
        tracker.update(evaluate_batch(pred, gt))
results["No noise map"] = tracker.summary()

# =====================================================================
# Ablation 3: No illum map (noise only)
# =====================================================================
print("[3/7] No illum map...")
tracker = MetricTracker()
with torch.no_grad():
    for inp, gt in loader:
        inp, gt = inp.to(device), gt.to(device)
        noise = compute_noise(inp)
        illum = torch.zeros(inp.size(0), 1, inp.size(2), inp.size(3), device=device)
        pred = model(inp, noise, illum)
        tracker.update(evaluate_batch(pred, gt))
results["No illum map"] = tracker.summary()

# =====================================================================
# Ablation 4: No maps at all (zeros for both)
# =====================================================================
print("[4/7] No maps...")
tracker = MetricTracker()
with torch.no_grad():
    for inp, gt in loader:
        inp, gt = inp.to(device), gt.to(device)
        noise = torch.zeros(inp.size(0), 1, inp.size(2), inp.size(3), device=device)
        illum = torch.zeros(inp.size(0), 1, inp.size(2), inp.size(3), device=device)
        pred = model(inp, noise, illum)
        tracker.update(evaluate_batch(pred, gt))
results["No maps (zeros)"] = tracker.summary()

# =====================================================================
# Ablation 5: Chroma only
# =====================================================================
print("[5/7] Chroma compensation only...")
tracker = MetricTracker()
with torch.no_grad():
    for inp, gt in loader:
        inp, gt = inp.to(device), gt.to(device)
        noise = compute_noise(inp)
        illum = compute_illum(inp)
        pred = model(inp, noise, illum)
        pred = postprocess(pred, chroma_alpha=0.1, sharpen_strength=0.0)
        tracker.update(evaluate_batch(pred, gt))
results["+ Chroma only"] = tracker.summary()

# =====================================================================
# Ablation 6: Sharpen only
# =====================================================================
print("[6/7] Unsharp masking only...")
tracker = MetricTracker()
with torch.no_grad():
    for inp, gt in loader:
        inp, gt = inp.to(device), gt.to(device)
        noise = compute_noise(inp)
        illum = compute_illum(inp)
        pred = model(inp, noise, illum)
        pred = postprocess(pred, chroma_alpha=0.0, sharpen_strength=0.3)
        tracker.update(evaluate_batch(pred, gt))
results["+ Sharpen only"] = tracker.summary()

# =====================================================================
# Ablation 7: Both PP
# =====================================================================
print("[7/7] Both postprocess...")
tracker = MetricTracker()
with torch.no_grad():
    for inp, gt in loader:
        inp, gt = inp.to(device), gt.to(device)
        noise = compute_noise(inp)
        illum = compute_illum(inp)
        pred = model(inp, noise, illum)
        pred = postprocess(pred, chroma_alpha=0.1, sharpen_strength=0.3)
        tracker.update(evaluate_batch(pred, gt))
results["+ Both PP"] = tracker.summary()


# =====================================================================
# Print results table
# =====================================================================
print("\n" + "=" * 60)
print("  ABLATION STUDY RESULTS")
print("=" * 60)
print(f"{'Config':<20} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}")
print("-" * 60)
for name, s in results.items():
    lpips_str = f"{s['lpips']:.4f}" if 'lpips' in s else "  N/A "
    print(f"{name:<20} {s['psnr']:>8.4f} {s['ssim']:>8.4f} {lpips_str:>8}")
print("=" * 60)
