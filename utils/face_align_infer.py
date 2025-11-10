from __future__ import annotations
from typing import Tuple, Optional, List
import numpy as np
import cv2
import torch
from facenet_pytorch import MTCNN

# ArcFace 5-point template in 112x112; scale to 256x256
_ARCFACE_5PTS_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)
_SCALE = 256.0 / 112.0
_ARCFACE_5PTS_256 = _ARCFACE_5PTS_112 * _SCALE

def estimate_affine(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """
    Estimate 2x3 similarity transform from src_pts(5x2) -> dst_pts(5x2).
    """
    M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
    if M is None:
        # fallback: identity-like
        M = np.array([[1,0,0],[0,1,0]], dtype=np.float32)
    return M.astype(np.float32)

def warp_by_M(img: np.ndarray, M: np.ndarray, out_size: int = 256) -> np.ndarray:
    return cv2.warpAffine(img, M, (out_size, out_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def apply_M(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Apply 2x3 affine to Nx2 points.
    """
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
    out = (M @ pts_h.T).T
    return out

def mtcnn_detector(device: str = "cpu") -> MTCNN:
    return MTCNN(keep_all=True, device=device, post_process=False)

def detect_first_face(im_rgb: np.ndarray, mtcnn: MTCNN, prefer_index: Optional[int] = None):
    """
    Detect faces and return (bbox, landmarks, prob, index). Pick highest prob or prefer_index.
    """
    import torch
    pil = torch.from_numpy(im_rgb).to(torch.uint8).cpu().numpy()  # placeholder for type check
    boxes, probs, landmarks = mtcnn.detect(im_rgb, landmarks=True)
    if boxes is None or len(boxes) == 0:
        return None
    if prefer_index is not None and 0 <= prefer_index < len(boxes):
        idx = prefer_index
    else:
        idx = int(np.argmax(probs))
    return boxes[idx], landmarks[idx], probs[idx], idx

def build_face_mask_256(land5_256: np.ndarray, scale_x: float = 1.0, scale_y: float = 1.0) -> np.ndarray:
    """
    Quick oval mask in aligned space (256x256), derived from eye & mouth geometry.
    scale_x/scale_y allow shrinking/expanding the ellipse.
    """
    le, re, nose, lm, rm = land5_256
    eye_center = (le + re) / 2.0
    mouth_center = (lm + rm) / 2.0
    center = (eye_center * 0.35 + mouth_center * 0.65)
    eye_dist = np.linalg.norm(re - le)
    mouth_span = np.linalg.norm(rm - lm)
    base = max(eye_dist, mouth_span)
    rx = 1.20 * base * 1.1 * float(scale_x)
    ry = 1.35 * base * 1.25 * float(scale_y)
    mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.ellipse(mask,
                center=tuple(center.astype(int)),
                axes=(int(rx), int(ry)),
                angle=0, startAngle=0, endAngle=360,
                color=255, thickness=-1)
    mask = mask.astype(np.float32) / 255.0
    return mask


def align_face_256(im_rgb: np.ndarray, mtcnn: MTCNN, face_index: Optional[int] = None):
    """
    Detect -> affine align to 256. Return:
      aligned_256, M(2x3), M_inv(2x3), aligned_landmarks_256, detection_info(dict)
    """
    det = detect_first_face(im_rgb, mtcnn, face_index)
    if det is None:
        return None, None, None, None, {"ok": False, "msg": "no_face"}
    box, land5, prob, idx = det
    src_pts = land5.astype(np.float32)
    dst_pts = _ARCFACE_5PTS_256
    M = estimate_affine(src_pts, dst_pts)
    aligned = warp_by_M(im_rgb, M, out_size=256)
    Minv = cv2.invertAffineTransform(M)
    aligned_land = apply_M(src_pts, M)
    info = {"ok": True, "prob": float(prob), "index": int(idx), "box": box.tolist()}
    return aligned, M, Minv, aligned_land, info
