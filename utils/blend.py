from __future__ import annotations
import numpy as np
import cv2

def feather(mask: np.ndarray, ksize: int = 25) -> np.ndarray:
    """
    Feather a single-channel mask [0,1] via Gaussian blur.
    """
    if mask.dtype != np.float32:
        mask = mask.astype(np.float32)
    k = max(3, ksize | 1)  # odd
    blurred = cv2.GaussianBlur(mask, (k, k), 0)
    return np.clip(blurred, 0.0, 1.0)

def blend_feathered(target: np.ndarray, paste: np.ndarray, mask: np.ndarray, ksize: int = 25) -> np.ndarray:
    """
    Simple feathered alpha blend (all inputs uint8 RGB, mask in [0,1]).
    """
    m = feather(mask, ksize=ksize)[..., None]  # HxWx1
    out = (paste.astype(np.float32) * m + target.astype(np.float32) * (1.0 - m))
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)

def poisson_blend(target: np.ndarray, paste: np.ndarray, mask: np.ndarray, center=None) -> np.ndarray:
    """
    OpenCV seamlessClone Poisson blend. Requires uint8 images and uint8 mask (0/255).
    """
    mask255 = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    if center is None:
        h, w = target.shape[:2]
        center = (w // 2, h // 2)
    try:
        out = cv2.seamlessClone(paste, target, mask255, center, cv2.NORMAL_CLONE)
        return out
    except cv2.error:
        # fallback to feather if Poisson fails
        return blend_feathered(target, paste, mask, ksize=25)

def match_lightness_hsv(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Fast lightness matching in HSV: match V channel mean/var of src to ref.
    """
    src_hsv = cv2.cvtColor(src, cv2.COLOR_RGB2HSV).astype(np.float32)
    ref_hsv = cv2.cvtColor(ref, cv2.COLOR_RGB2HSV).astype(np.float32)
    sV = src_hsv[..., 2]; rV = ref_hsv[..., 2]
    s_mu, s_std = sV.mean(), sV.std() + 1e-6
    r_mu, r_std = rV.mean(), rV.std() + 1e-6
    src_hsv[..., 2] = np.clip((sV - s_mu) * (r_std / s_std) + r_mu, 0, 255)
    out = cv2.cvtColor(src_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return out
