# Common evaluation metrics for face-swap quality.
from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional MS-SSIM (pip install pytorch-msssim)
try:
    from pytorch_msssim import ssim as _ssim_fn, ms_ssim as _msssim_fn
    _HAS_MSSSIM = True
except Exception:
    _HAS_MSSSIM = False

def to01(x: torch.Tensor) -> torch.Tensor:
    """Convert from [-1,1] to [0,1] and clamp."""
    x01 = (x * 0.5) + 0.5
    return x01.clamp(0.0, 1.0)

def cos_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Cosine similarity along dim=1 (N,C)."""
    a = a / (a.norm(dim=1, keepdim=True) + eps)
    b = b / (b.norm(dim=1, keepdim=True) + eps)
    return (a * b).sum(dim=1)

def ssim(img1: torch.Tensor, img2: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """
    SSIM averaged over batch; returns scalar tensor.
    If pytorch-msssim is available, use it; else fallback to a simple window-based SSIM.
    Inputs should be in [0,1].
    """
    if _HAS_MSSSIM:
        return _ssim_fn(img1, img2, data_range=data_range, size_average=True)
    # simple fallback with avg-pooling windows (not as accurate as ms-ssim)
    K1, K2 = 0.01, 0.03
    C1, C2 = (K1 * data_range) ** 2, (K2 * data_range) ** 2
    win = 11
    pad = win // 2
    mu1 = F.avg_pool2d(img1, win, 1, pad)
    mu2 = F.avg_pool2d(img2, win, 1, pad)
    mu1_sq, mu2_sq, mu1_mu2 = mu1*mu1, mu2*mu2, mu1*mu2
    sigma1_sq = F.avg_pool2d(img1*img1, win, 1, pad) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2*img2, win, 1, pad) - mu2_sq
    sigma12   = F.avg_pool2d(img1*img2, win, 1, pad) - mu1_mu2
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq+mu2_sq + C1)*(sigma1_sq+sigma2_sq + C2) + 1e-12)
    return ssim_map.mean()

def ms_ssim(img1: torch.Tensor, img2: torch.Tensor, data_range: float = 1.0) -> Optional[torch.Tensor]:
    """MS-SSIM if available; else returns None."""
    if not _HAS_MSSSIM:
        return None
    return _msssim_fn(img1, img2, data_range=data_range, size_average=True)


def psnr(img1: torch.Tensor, img2: torch.Tensor, data_range: float = 1.0, eps: float = 1e-12) -> torch.Tensor:
    """
    PSNR averaged over batch; returns scalar tensor.
    Inputs in [0,1]. data_range=1 by default.
    """
    mse = F.mse_loss(img1, img2, reduction="mean")
    return 10.0 * torch.log10((data_range ** 2) / (mse + eps))

def psnr_masked(img1: torch.Tensor, img2: torch.Tensor, mask: torch.Tensor,
                data_range: float = 1.0, eps: float = 1e-12) -> torch.Tensor:
    """
    Region PSNR (mask==1 region). If mask empty, returns NaN.
    img1,img2 in [0,1], mask in [0,1] or {0,1}, shape Bx1xHxW.
    """
    num = (mask > 0.5).float().sum()
    if num.item() < 1:
        return torch.tensor(float("nan"), device=img1.device)
    diff = (img1 - img2) * mask
    mse = (diff * diff).sum() / (num + eps)
    return 10.0 * torch.log10((data_range ** 2) / (mse + eps))
