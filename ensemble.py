"""ensemble.py — 把多個 checkpoint 的權重平均，產生 ensemble model。

通常拿同個訓練最後幾個 epoch 的 ckpt 平均（SWA），或不同訓練的 best.pth 平均。
這比單一 model 通常多 +0.3-0.5 dB PSNR。

Examples:
  # Average best.pth from two trainings
  python ensemble.py \\
      --ckpts ckpt_a/best.pth ckpt_b/best.pth \\
      --out ensemble.pth

  # 然後用 eval_tta.py 跑這個 ensemble.pth
"""

import argparse
from pathlib import Path
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True, help="checkpoint paths to average")
    p.add_argument("--weights", nargs="+", type=float, default=None,
                   help="optional weights (same count as --ckpts, will be normalized to sum=1)")
    p.add_argument("--out", required=True, help="output ckpt path")
    return p.parse_args()


def main():
    args = parse_args()
    n = len(args.ckpts)

    # Normalize weights
    if args.weights is not None:
        assert len(args.weights) == n, f"weights count ({len(args.weights)}) must match ckpts ({n})"
        total = sum(args.weights)
        weights = [w / total for w in args.weights]
    else:
        weights = [1.0 / n] * n   # equal weights

    print(f"Averaging {n} checkpoints:")
    for c, w in zip(args.ckpts, weights):
        print(f"  weight={w:.3f}  - {c}")

    avg_state = None
    epochs = []
    best_psnrs = []

    for path, w in zip(args.ckpts, weights):
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt["model"]
        epochs.append(ckpt.get("epoch", -1))
        best_psnrs.append(ckpt.get("best_psnr", 0.0))

        if avg_state is None:
            avg_state = {k: v.float().clone() * w for k, v in state.items()}
        else:
            for k in avg_state:
                avg_state[k] += state[k].float() * w

    out_ckpt = {
        "epoch": max(epochs),
        "best_psnr": max(best_psnrs),
        "model": avg_state,
        "ensemble_of": args.ckpts,
        "ensemble_weights": weights,
    }
    torch.save(out_ckpt, args.out)
    print(f"\n[OK] Ensemble saved to {args.out}")
    print(f"  Source best_psnrs: {[f'{p:.2f}' for p in best_psnrs]}")
    print(f"  Weights: {[f'{w:.3f}' for w in weights]}")


if __name__ == "__main__":
    main()
