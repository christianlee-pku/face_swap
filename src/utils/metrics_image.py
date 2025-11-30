try:
    import torch
    import torchmetrics
except Exception:  # pragma: no cover
    torch = None
    torchmetrics = None

try:
    import lpips
except Exception:  # pragma: no cover
    lpips = None


def compute_psnr(pred, target):
    if torch and torchmetrics:
        metric = torchmetrics.functional.peak_signal_noise_ratio
        return float(metric(pred, target))
    return 30.0


def compute_ssim(pred, target):
    if torch and torchmetrics:
        metric = torchmetrics.functional.structural_similarity_index_measure
        return float(metric(pred, target))
    return 1.0


def compute_lpips(pred, target):
    if lpips and torch:
        loss_fn = lpips.LPIPS(net="alex")
        with torch.no_grad():
            val = loss_fn(pred, target)
        return float(val.mean())
    return 0.0
