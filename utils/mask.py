import numpy as np
import cv2

def soft_ellipse_mask(h=256, w=256, scale_x=0.9, scale_y=1.0, blur=25):
    """
    Create a soft elliptical mask in [0..255] for aligned face images.
    - The ellipse is centered at (w/2, h/2) with radii scaled by scale_x, scale_y.
    - blur (odd) controls edge softness (Gaussian).
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w//2, h//2)
    axes = (int(w*scale_x*0.45), int(h*scale_y*0.52))  # empirically reasonable coverage
    cv2.ellipse(mask, center, axes, angle=0, startAngle=0, endAngle=360, color=255, thickness=-1)
    if blur > 0:
        if blur % 2 == 0: blur += 1
        mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    return mask  # uint8 0..255
