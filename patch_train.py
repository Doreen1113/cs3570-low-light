"""
patch_train.py — 產生 train_patched.py，在不動 D 的 train.py 情況下
套上 AMP、DataParallel、以及 ablation mode。

【所有人通用】本地或 Kaggle 都可以跑。

Step 1 — 產生 patched 版本：
    python patch_train.py --mode full

Step 2 — 跑訓練（自己指定 --root 路徑）：
    python -u train_patched.py --root <你的 dataset 路徑> --batch-size 4 --epochs 20 ...

Ablation modes（四人各跑一個）：
    full        → 5ch (RGB + noise + illum)  ← C 跑這個
    noise_only  → 4ch (RGB + noise)
    illum_only  → 4ch (RGB + illum)
    no_map      → 3ch (RGB only，pure baseline)
"""

import os, re, pathlib, subprocess, sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── 讀取 --mode 參數（不傳給 train.py）────────────────────────────────────────
_mode = "full"
_argv = sys.argv[1:]
if "--mode" in _argv:
    _i = _argv.index("--mode")
    _mode = _argv[_i + 1]
    _argv = _argv[:_i] + _argv[_i + 2:]

_MODE_FLAGS = {
    "full":       (True,  True),
    "noise_only": (True,  False),
    "illum_only": (False, True),
    "no_map":     (False, False),
}
assert _mode in _MODE_FLAGS, f"--mode 必須是 {list(_MODE_FLAGS)}"
_use_noise, _use_illum = _MODE_FLAGS[_mode]
print(f"Ablation mode: {_mode}  (use_noise={_use_noise}, use_illum={_use_illum})")

# ── 讀取 train.py ─────────────────────────────────────────────────────────────
SRC = pathlib.Path("train.py").read_text(encoding="utf-8")

# ── Patch 1: imports ──────────────────────────────────────────────────────────
SRC = SRC.replace(
    "import torch\nimport torch.optim as optim\nfrom torch.utils.data import DataLoader",
    "import torch\nimport torch.nn as nn\nimport torch.optim as optim\nfrom torch.amp import GradScaler, autocast\nfrom torch.utils.data import DataLoader",
)

# ── Patch 2: save_checkpoint — unwrap DataParallel ───────────────────────────
SRC = SRC.replace(
    '    torch.save({\n        "epoch": epoch,\n        "model": model.state_dict(),\n',
    '    m = model.module if isinstance(model, nn.DataParallel) else model\n'
    '    torch.save({\n        "epoch": epoch,\n        "model": m.state_dict(),\n',
)

# ── Patch 3: ablation mode → RestorationNet ───────────────────────────────────
SRC = SRC.replace(
    "    model = RestorationNet(width=args.width).to(device)",
    f"    model = RestorationNet(width=args.width,\n"
    f"                          use_noise_map={_use_noise},\n"
    f"                          use_illum_map={_use_illum}).to(device)",
)

# ── Patch 4: DataParallel ─────────────────────────────────────────────────────
SRC = SRC.replace(
    '    print(f"Model params: {n_params:.2f}M")\n'
    "\n"
    "    disc = None",

    '    print(f"Model params: {n_params:.2f}M")\n'
    "    if torch.cuda.device_count() > 1:\n"
    "        model = nn.DataParallel(model)\n"
    '        print(f"Using {torch.cuda.device_count()} GPUs")\n'
    "\n"
    "    disc = None",
)

# ── Patch 5: GradScaler ───────────────────────────────────────────────────────
SRC = SRC.replace(
    "    opt_disc = optim.Adam(disc.parameters(), lr=args.lr) if disc else None\n"
    "\n"
    "    start_epoch, best_psnr = 0, 0.0",

    "    opt_disc = optim.Adam(disc.parameters(), lr=args.lr) if disc else None\n"
    "    scaler = GradScaler('cuda')\n"
    "\n"
    "    start_epoch, best_psnr = 0, 0.0",
)

# ── Patch 6: autocast forward pass (D 新版有 batch_psnr block) ───────────────
SRC = SRC.replace(
    "            pred = model(inp, noise_map, illum_map)\n"
    "            total, breakdown = loss_fn(pred, gt)\n"
    "\n"
    "            with torch.no_grad():\n"
    "                batch_psnr = psnr(pred.clamp(0, 1), gt).mean().item()\n"
    "\n"
    "            opt.zero_grad()\n"
    "            total.backward()\n"
    "            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n"
    "            opt.step()",

    "            opt.zero_grad()\n"
    "            with autocast('cuda'):\n"
    "                pred = model(inp, noise_map, illum_map)\n"
    "                total, breakdown = loss_fn(pred, gt)\n"
    "            with torch.no_grad():\n"
    "                batch_psnr = psnr(pred.clamp(0, 1), gt).mean().item()\n"
    "            scaler.scale(total).backward()\n"
    "            scaler.unscale_(opt)\n"
    "            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n"
    "            scaler.step(opt)\n"
    "            scaler.update()",
)

# ── Patch 6.5: load_checkpoint 支援 DataParallel ─────────────────────────────
SRC = SRC.replace(
    "def load_checkpoint(path: Path, model, opt=None):\n"
    "    ckpt = torch.load(path, map_location=\"cpu\")\n"
    "    model.load_state_dict(ckpt[\"model\"])",
    "def load_checkpoint(path: Path, model, opt=None):\n"
    "    ckpt = torch.load(path, map_location=\"cpu\")\n"
    "    state = ckpt[\"model\"]\n"
    "    target = model.module if isinstance(model, nn.DataParallel) else model\n"
    "    target.load_state_dict(state)",
)

# ── Patch 7: 拿掉 tqdm 進度條（Kaggle 顯示會變多行，改回純 step= print）─────
SRC = SRC.replace(
    "        pbar = tqdm(train_loader, desc=f\"Epoch {epoch}/{args.epochs}\", ncols=120)\n"
    "\n"
    "        for step, (inp, gt) in enumerate(pbar, 1):",
    "        for step, (inp, gt) in enumerate(train_loader, 1):",
)
# 移除 pbar.set_postfix 那段
SRC = SRC.replace(
    "            epoch_loss += total.item()\n"
    "            pbar.set_postfix({\n"
    "                \"loss\": f\"{total.item():.4f}\",\n"
    "                \"psnr\": f\"{batch_psnr:.2f}\",\n"
    "                \"avg\": f\"{epoch_loss / step:.4f}\",\n"
    "                \"lr\": f\"{opt.param_groups[0]['lr']:.1e}\",\n"
    "            })\n",
    "            epoch_loss += total.item()\n",
)

# ── 寫出 patched 版本 ─────────────────────────────────────────────────────────
out = pathlib.Path("train_patched.py")
out.write_text(SRC, encoding="utf-8")
print(f"[OK] train_patched.py ready  [{_mode}]")
print(f"  GPUs available: {__import__('torch').cuda.device_count()}")
