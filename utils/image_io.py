from __future__ import annotations
from typing import Tuple
from PIL import Image, ImageOps
import numpy as np
import cv2

def load_rgb(path: str) -> Image.Image:
    """
    Load an image as RGB with EXIF orientation corrected.
    """
    im = Image.open(path).convert("RGB")
    im = ImageOps.exif_transpose(im)  # respect device orientation
    return im

def pil_to_numpy_rgb(im: Image.Image) -> np.ndarray:
    """
    Convert PIL Image (RGB) -> numpy array uint8 HxWx3.
    """
    return np.array(im, dtype=np.uint8)

def numpy_to_pil_rgb(arr: np.ndarray) -> Image.Image:
    """
    Convert numpy uint8 HxWx3 (RGB) -> PIL Image.
    """
    return Image.fromarray(arr, mode="RGB")

def save_rgb(arr: np.ndarray, path: str) -> None:
    """
    Save numpy RGB image to disk (PNG/JPEG decided by extension).
    """
    Image.fromarray(arr, mode="RGB").save(path)

def ensure_3c(arr: np.ndarray) -> np.ndarray:
    """
    Ensure RGB 3 channels.
    """
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return arr

def resize_max_side(im: np.ndarray, max_side: int = 1400) -> Tuple[np.ndarray, float]:
    """
    Resize image so that max(H, W) <= max_side. Returns resized image and scale factor.
    """
    h, w = im.shape[:2]
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        im = cv2.resize(im, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return im, scale
