"""Person D (part 2) — Evaluation metrics.

Computes PSNR (primary), SSIM, and LPIPS for restored images.

Usage:
  from metrics import evaluate_batch, MetricTracker

  tracker = MetricTracker()
  for pred, gt in val_loader:
      tracker.update(evaluate_batch(pred, gt))
  print(tracker.summary())

Run standalone to verify on random tensors:
  python metrics.py
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------

def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """Per-image PSNR. pred/target: [B,3,H,W] float32 in [0,1].
    Returns tensor of shape [B].
    """
    mse = ((pred - target) ** 2).flatten(1).mean(dim=1)
    return 10.0 * torch.log10(max_val ** 2 / mse.clamp(min=1e-10))


# ---------------------------------------------------------------------------
# SSIM (same kernel as losses.py but evaluation-only, no gradients needed)
# ---------------------------------------------------------------------------

def _gauss_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)


def ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-image SSIM averaged over channels. Returns tensor of shape [B]."""
    kernel = _gauss_kernel().to(pred.device)
    pad = 5
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    vals = []
    for c in range(pred.shape[1]):
        p = pred[:, c:c+1]
        t = target[:, c:c+1]
        mu_p = F.conv2d(p, kernel, padding=pad)
        mu_t = F.conv2d(t, kernel, padding=pad)
        sig_pp = F.conv2d(p * p, kernel, padding=pad) - mu_p ** 2
        sig_tt = F.conv2d(t * t, kernel, padding=pad) - mu_t ** 2
        sig_pt = F.conv2d(p * t, kernel, padding=pad) - mu_p * mu_t
        num = (2 * mu_p * mu_t + C1) * (2 * sig_pt + C2)
        den = (mu_p ** 2 + mu_t ** 2 + C1) * (sig_pp + sig_tt + C2)
        vals.append((num / den.clamp(min=1e-8)).flatten(1).mean(dim=1))
    return torch.stack(vals, dim=1).mean(dim=1)


# ---------------------------------------------------------------------------
# LPIPS (lazy-load to avoid import errors when lpips not installed)
# ---------------------------------------------------------------------------

_lpips_fn = None


def _get_lpips(device: torch.device):
    global _lpips_fn
    if _lpips_fn is None:
        try:
            import lpips
            _lpips_fn = lpips.LPIPS(net="alex").to(device)
        except ImportError:
            return None
    return _lpips_fn.to(device)


def lpips_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
    """LPIPS distance per image. Returns [B] or None if lpips not installed."""
    fn = _get_lpips(pred.device)
    if fn is None:
        return None
    # lpips expects inputs in [-1, 1]
    p = pred * 2.0 - 1.0
    t = target * 2.0 - 1.0
    with torch.no_grad():
        return fn(p, t).squeeze()


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_batch(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Compute metrics for one batch. Returns dict with per-image tensors."""
    with torch.no_grad():
        result = {
            "psnr": psnr(pred, target),
            "ssim": ssim(pred, target),
        }
        lp = lpips_score(pred, target)
        if lp is not None:
            result["lpips"] = lp
    return result


# ---------------------------------------------------------------------------
# Running average tracker
# ---------------------------------------------------------------------------

class MetricTracker:
    """Accumulates per-batch metric dicts and reports averages."""

    def __init__(self):
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def update(self, metrics: dict):
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                vals = v.detach().cpu()
                self._sums[k] = self._sums.get(k, 0.0) + vals.sum().item()
                self._counts[k] = self._counts.get(k, 0) + vals.numel()

    def summary(self) -> dict[str, float]:
        return {k: self._sums[k] / self._counts[k] for k in self._sums}

    def reset(self):
        self._sums.clear()
        self._counts.clear()


# ---------------------------------------------------------------------------
# Standalone check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    # Perfect prediction (same tensor) → PSNR should be very high
    x = torch.rand(4, 3, 256, 256)
    p_perfect = psnr(x, x)
    print(f"PSNR (perfect):  {p_perfect.mean().item():.1f} dB  (expect ~100)")

    # Random prediction
    y = torch.rand(4, 3, 256, 256)
    p_rand = psnr(x, y)
    s_rand = ssim(x, y)
    print(f"PSNR (random):   {p_rand.mean().item():.2f} dB")
    print(f"SSIM (random):   {s_rand.mean().item():.4f}  (expect ~0.0)")

    # MetricTracker
    tracker = MetricTracker()
    for _ in range(3):
        tracker.update(evaluate_batch(y, x))
    print(f"Tracker summary: {tracker.summary()}")

    lp = lpips_score(x, y)
    if lp is not None:
        print(f"LPIPS (random):  {lp.mean().item():.4f}")
    else:
        print("LPIPS: lpips package not installed — install with: pip install lpips")
